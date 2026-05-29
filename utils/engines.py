"""Engine factory utilities for unified VASP/CP2K support."""

from typing import Dict, Any, Optional, Tuple

from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.orm import Dict


class EngineFactory:
    """Factory class for creating engine-specific inputs and configurations."""

    VASP_CALC = "vasp.vasp"
    VASP_RELAX = "vasp.relax"
    CP2K_CALC = "cp2k"
    CP2K_OPTIMIZE = "cp2k.optimize"

    @classmethod
    def get_calculation_class(cls, engine: str):
        """Get the calculation class for the specified engine."""
        engine = engine.lower().strip()

        if engine == "vasp":
            return CalculationFactory(cls.VASP_CALC)
        elif engine == "cp2k":
            return CalculationFactory(cls.CP2K_CALC)
        else:
            raise ValueError(f"Unknown engine: {engine}. Must be 'vasp' or 'cp2k'.")

    @classmethod
    def get_relax_workchain_class(cls, engine: str):
        """Get the relaxation workchain class for the specified engine."""
        engine = engine.lower().strip()

        if engine == "vasp":
            return WorkflowFactory(cls.VASP_RELAX)
        elif engine == "cp2k":
            return WorkflowFactory(cls.CP2K_OPTIMIZE)
        else:
            raise ValueError(f"Unknown engine: {engine}. Must be 'vasp' or 'cp2k'.")

    @classmethod
    def get_default_resources(cls, engine: str) -> Dict[str, Any]:
        """Get default resource requirements for the engine."""
        defaults = {
            "vasp": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 8,
            },
            "cp2k": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 16,
            },
        }
        return defaults.get(engine.lower().strip(), defaults["vasp"])

    @classmethod
    def build_generic_parameters(
        cls,
        engine: str,
        generic_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build engine-specific parameters from generic parameters.

        Args:
            engine: 'vasp' or 'cp2k'
            generic_params: Dictionary with generic parameter names:
                - encut: Plane-wave cutoff energy (eV for VASP, Ry for CP2K)
                - max_steps: Maximum optimization/SCF steps
                - energy_tolerance: Convergence tolerance
                - xc_functional: Exchange-correlation functional
                - scf_guess: Initial guess for SCF
                - charge: System charge
                - multiplicity: Spin multiplicity

        Returns:
            Dictionary in engine-specific format
        """
        engine = engine.lower().strip()

        if engine == "vasp":
            return cls._build_vasp_parameters(generic_params)
        elif engine == "cp2k":
            return cls._build_cp2k_parameters(generic_params)
        else:
            raise ValueError(f"Unknown engine: {engine}")

    @classmethod
    def _build_vasp_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build VASP INCAR parameters from generic params."""
        run_type = params.get("run_type", "energy")

        incar = {
            "ENCUT": params.get("encut", 400),
            "PREC": params.get("prec", "Normal"),
            "EDIFF": params.get("energy_tolerance", 1e-5),
            "ISMEAR": params.get("ismeart", 0),
            "SIGMA": params.get("sigma", 0.05),
            "NELM": params.get("max_scf", 200),
        }

        if run_type == "relax":
            incar.update({
                "IBRION": params.get("ibrion", 2),
                "ISIF": params.get("isif", 3),
                "NSW": params.get("max_steps", 50),
                "EDIFFG": params.get("force_tolerance", -0.01),
            })
        elif run_type == "scf":
            incar["NSW"] = 0

        return {"incar": incar}

    @classmethod
    def _build_cp2k_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build CP2K input parameters from generic params."""
        run_type = params.get("run_type", "energy")

        if run_type == "relax":
            run_type_str = "GEO_OPT"
        elif run_type == "cell_opt":
            run_type_str = "CELL_OPT"
        else:
            run_type_str = "ENERGY"

        cp2k_params = {
            "GLOBAL": {
                "RUN_TYPE": run_type_str,
                "PRINT_LEVEL": params.get("print_level", "MEDIUM"),
            },
            "FORCE_EVAL": {
                "METHOD": "Quickstep",
                "DFT": {
                    "BASIS_SET_FILE_NAME": "BASIS_MOLOPT",
                    "POTENTIAL_FILE_NAME": "GTH_POTENTIALS",
                    "CHARGE": params.get("charge", 0),
                    "MULTIPLICITY": params.get("multiplicity", 1),
                    "SCF": {
                        "SCF_GUESS": params.get("scf_guess", "ATOMIC"),
                        "EPS_SCF": params.get("energy_tolerance", 1.0e-6),
                        "MAX_SCF": params.get("max_scf", 200),
                        "DIAGONALIZATION": {
                            "ALGORITHM": "STANDARD",
                        },
                        "MIXING": {
                            "METHOD": "BROYDEN_MIXING",
                            "ALPHA": 0.2,
                            "BETA": 1.5,
                            "NBROYDEN": 8,
                        },
                    },
                    "XC": {
                        "XC_FUNCTIONAL": {"_": params.get("xc_functional", "PBE")},
                    },
                },
                "SUBSYS": {
                    "CELL": {
                        "PERIODIC": params.get("periodic", "XYZ"),
                    },
                },
            },
        }

        cutoff = params.get("cutoff") or params.get("mgrid_cutoff")
        if cutoff:
            cp2k_params["FORCE_EVAL"]["DFT"]["MGRID"] = {
                "CUTOFF": cutoff,
                "NGRIDS": params.get("ngrids", 4),
            }

        if run_type in ("relax", "cell_opt"):
            cp2k_params["MOTION"] = {
                "GEO_OPT": {
                    "MAX_ITER": params.get("max_steps", 50),
                    "OPTIMIZER": params.get("optimizer", "BFGS"),
                    "TYPE": "MINIMIZATION",
                }
            }

            if run_type == "cell_opt":
                cp2k_params["MOTION"]["CELL_OPT"] = {
                    "MAX_ITER": params.get("max_steps", 50),
                    "OPTIMIZER": params.get("optimizer", "BFGS"),
                }

        return cp2k_params

    @classmethod
    def get_kpoints_settings(
        cls,
        engine: str,
        mesh: Optional[list] = None
    ) -> Dict[str, Any]:
        """Get default k-point settings for the engine.

        Args:
            engine: 'vasp' or 'cp2k'
            mesh: K-point mesh [kx, ky, kz]. For CP2K, typically [4, 1, 4]
                  for slab calculations.

        Returns:
            Dictionary with k-points configuration
        """
        engine = engine.lower().strip()

        if mesh is None:
            if engine == "vasp":
                mesh = [4, 4, 4]
            elif engine == "cp2k":
                mesh = [4, 1, 4]
            else:
                mesh = [4, 4, 4]

        return {
            "vasp": {"mesh": mesh, "shift": [0.0, 0.0, 0.0]},
            "cp2k": {"mesh": mesh},
        }.get(engine, {"mesh": mesh})


def get_engine_launcher(engine_name: str) -> Dict[str, Any]:
    """Returns the correct WorkChain and dynamic input generator.

    DEPRECATED: Use EngineFactory class methods instead.
    """
    engine = engine_name.lower().strip()

    if engine == "vasp":
        return {
            "workchain": WorkflowFactory("vasp.vasp"),
            "builder_func": _build_vasp_inputs_legacy,
        }
    elif engine == "cp2k":
        return {
            "workchain": WorkflowFactory("cp2k.cp2k"),
            "builder_func": _build_cp2k_inputs_legacy,
        }
    else:
        raise ValueError(f"Unknown electronic structure engine: {engine_name}")


def _build_vasp_inputs_legacy(structure, parameters):
    """Legacy VASP input builder."""
    return {
        "parameters": Dict(
            dict={
                "incar": {
                    "IBRION": 2,
                    "ISIF": 3,
                    "NSW": parameters.get("max_steps", 50),
                    "EDIFF": parameters.get("energy_tolerance", 1e-5),
                }
            }
        )
    }


def _build_cp2k_inputs_legacy(structure, parameters):
    """Legacy CP2K input builder."""
    return {
        "parameters": Dict(
            dict={
                "GLOBAL": {"RUN_TYPE": "GEO_OPT"},
                "FORCE_EVAL": {
                    "METHOD": "Quickstep",
                    "DFT": {"XC": {"XC_FUNCTIONAL": {"PBE": {}}}},
                },
                "MOTION": {"GEO_OPT": {"MAX_ITER": parameters.get("max_steps", 50)}},
            }
        )
    }