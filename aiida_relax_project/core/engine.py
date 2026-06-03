"""Engine adapters for VASP and CP2K.

Each adapter provides a consistent interface for interacting with
specific DFT engines, handling engine-specific logic and translations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from aiida import orm
    from aiida.orm import StructureData, Dict, KpointsData, AbstractCode
    from aiida_relax_project.core.enums import EngineType, RunType, RelaxType

from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_relax_project.core.enums import RelaxType, RESOURCE_PRESETS
from aiida_relax_project.core.exceptions import EngineError, StructureValidationError

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into a copy of base."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class BaseEngineAdapter(ABC):
    """Abstract base class for engine adapters.

    Subclasses must implement engine-specific methods while providing
    a consistent interface for the rest of the package.
    """

    engine_type: Literal["vasp", "cp2k"]

    @abstractmethod
    def get_calculation_class(self) -> type:
        """Return the AiiDA calculation class for this engine."""
        raise NotImplementedError

    @abstractmethod
    def get_workflow_class(self, run_type: RunType) -> type:
        """Return the appropriate workchain class for the run type."""
        raise NotImplementedError

    @abstractmethod
    def build_parameters(
        self,
        generic_params: dict,
        run_type: RunType = "energy",
    ) -> "orm.Dict":
        """Translate generic parameters to engine format."""
        raise NotImplementedError

    def build_kpoints(self, mesh: list[int]) -> "orm.KpointsData":
        """Create k-points data with engine-appropriate settings."""
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh(mesh)
        return kpoints

    def validate_structure(self, structure: "orm.StructureData") -> None:
        """Validate structure is suitable for this engine."""
        if structure.pbc != (True, True, True):
            raise StructureValidationError(
                "Structure must be periodic in 3D for DFT calculations",
                structure_pk=structure.pk if hasattr(structure, "pk") else None,
            )

    def get_default_resources(self, preset: str = "default") -> dict:
        """Return default resource requirements."""
        return RESOURCE_PRESETS.get(preset, RESOURCE_PRESETS["default"]).copy()

    def get_relaxation_settings(self, relax_type: RelaxType) -> dict:
        """Return engine-specific relaxation settings."""
        return {}


class VaspAdapter(BaseEngineAdapter):
    """Adapter for VASP calculations."""

    engine_type: Literal["vasp"] = "vasp"

    def get_calculation_class(self) -> type:
        """Return the VASP calculation class."""
        return CalculationFactory("vasp.vasp")

    def get_workflow_class(self, run_type: RunType) -> type:
        """Return the appropriate VASP workchain class."""
        if run_type in ("relax", "cell_opt"):
            return WorkflowFactory("vasp.relax")
        return WorkflowFactory("vasp.vasp")

    def build_parameters(
        self,
        generic_params: dict,
        run_type: RunType = "energy",
    ) -> "orm.Dict":
        """Build VASP INCAR parameters from generic params."""
        from aiida.orm import Dict

        encut = generic_params.get("encut", 400)
        prec = generic_params.get("prec", "Normal")
        ediff = generic_params.get("ediff", 1e-5)
        ismear = generic_params.get("ismear", 0)
        sigma = generic_params.get("sigma", 0.05)
        max_scf = generic_params.get("max_scf", generic_params.get("max_steps", 200))
        xc = generic_params.get("xc_functional", "PBE")

        incar: dict[str, Any] = {
            "ENCUT": encut,
            "PREC": prec,
            "EDIFF": ediff,
            "ISMEAR": ismear,
            "SIGMA": sigma,
            "NELM": max_scf,
            "LCHARG": False,
            "LWAVE": False,
        }

        if xc.upper() != "PBE":
            incar["GGA"] = xc.upper()

        if run_type == "relax":
            incar.update({
                "IBRION": generic_params.get("ibrion", 2),
                "ISIF": generic_params.get("isif", 3),
                "NSW": generic_params.get("max_steps", 50),
                "EDIFFG": generic_params.get("force_tolerance", -0.01),
                "ISYM": 0,
            })
        elif run_type == "scf":
            incar["NSW"] = 0

        raw_incar = generic_params.get("raw_incar", {})
        if raw_incar:
            incar.update(raw_incar)
            logger.debug(f"Merged raw INCAR overrides: {raw_incar}")

        logger.debug(f"Built VASP INCAR: {incar}")
        return Dict(dict={"incar": incar})

    def get_relaxation_settings(self, relax_type: RelaxType) -> dict:
        """Return VASP-specific ISIF settings for relaxation types."""
        isif_map = {
            "volume": 3,
            "shape": 5,
            "positions": 2,
            "cell": 4,
        }
        ibrion_map = {
            "volume": 2,
            "shape": 2,
            "positions": 1,
            "cell": 2,
        }
        return {
            "isif": isif_map.get(relax_type, 3),
            "ibrion": ibrion_map.get(relax_type, 2),
        }


class Cp2kAdapter(BaseEngineAdapter):
    """Adapter for CP2K calculations."""

    engine_type: Literal["cp2k"] = "cp2k"

    def get_calculation_class(self) -> type:
        """Return the CP2K calculation class."""
        return CalculationFactory("cp2k")

    def get_workflow_class(self, run_type: RunType) -> type:
        """Return the appropriate CP2K workchain class."""
        if run_type in ("relax", "cell_opt"):
            return WorkflowFactory("cp2k.optimize")
        return WorkflowFactory("cp2k")

    def build_parameters(
        self,
        generic_params: dict,
        run_type: RunType = "energy",
    ) -> "orm.Dict":
        """Build CP2K input parameters from generic params."""
        from aiida.orm import Dict

        if run_type == "relax":
            run_type_str = "GEO_OPT"
        elif run_type == "cell_opt":
            run_type_str = "CELL_OPT"
        else:
            run_type_str = "ENERGY"

        charge = generic_params.get("charge", 0)
        multiplicity = generic_params.get("multiplicity", 1)
        xc = generic_params.get("xc_functional", "PBE")
        cutoff = generic_params.get("cutoff", generic_params.get("mgrid_cutoff", 400))
        eps_scf = generic_params.get("eps_scf", generic_params.get("energy_tolerance", 1e-6))
        max_scf = generic_params.get("max_scf", generic_params.get("max_steps", 200))

        basis_set_mapping = generic_params.get("basis_set_mapping", {})
        potential_mapping = generic_params.get("potential_mapping", {})

        subsys: dict[str, Any] = {
            "CELL": {
                "PERIODIC": generic_params.get("periodic", "XYZ"),
            },
        }

        for element, basis in basis_set_mapping.items():
            potential = potential_mapping.get(element, "GTH-PBE")
            subsys[f"KIND {element}"] = {
                "BASIS_SET": basis,
                "POTENTIAL": potential,
            }

        cp2k_params: dict[str, Any] = {
            "GLOBAL": {
                "RUN_TYPE": run_type_str,
                "PRINT_LEVEL": generic_params.get("print_level", "MEDIUM"),
                "WFN_RESTART_FILE_NAME": "__UNDEFINED__",
            },
            "FORCE_EVAL": {
                "METHOD": "Quickstep",
                "DFT": {
                    "BASIS_SET_FILE_NAME": generic_params.get("basis_set_file", "BASIS_MOLOPT"),
                    "BASIS_SET_FILE_NAME": generic_params.get("basis_set_file", "BASIS_RI_MOLOPT"),
                    "POTENTIAL_FILE_NAME": generic_params.get("potential_file", "GTH_POTENTIALS"),
                    "CHARGE": charge,
                    "MULTIPLICITY": multiplicity,
                    "QS": {
                        "METHOD": "GPW",
                        "EPS_DEFAULT": 1.0e-10,
                    },
                    "MGRID": {
                        "CUTOFF": cutoff,
                        "NGRIDS": generic_params.get("ngrids", 4),
                    },
                    "SCF": {
                        "SCF_GUESS": generic_params.get("scf_guess", "ATOMIC"),
                        "EPS_SCF": eps_scf,
                        "MAX_SCF": max_scf,
                        "DIAGONALIZATION": {"ALGORITHM": "STANDARD"},
                        "MIXING": {
                            "METHOD": "BROYDEN_MIXING",
                            "ALPHA": 0.2,
                            "BETA": 1.5,
                            "NBROYDEN": 8,
                        },
                    },
                    "XC": {
                        "XC_FUNCTIONAL": {"_": xc},
                    },
                },
                "SUBSYS": subsys,
            },
        }

        if run_type in ("relax", "cell_opt"):
            max_iter = generic_params.get("max_steps", 50)
            optimizer = generic_params.get("optimizer", "BFGS")

            if run_type == "relax":
                cp2k_params["MOTION"] = {
                    "GEO_OPT": {
                        "MAX_ITER": max_iter,
                        "OPTIMIZER": optimizer,
                        "TYPE": "MINIMIZATION",
                    }
                }
            else:
                cp2k_params["MOTION"] = {
                    "CELL_OPT": {
                        "MAX_ITER": max_iter,
                        "OPTIMIZER": optimizer,
                    },
                    "GEO_OPT": {
                        "MAX_ITER": max_iter,
                        "OPTIMIZER": optimizer,
                    },
                }

        raw_params = generic_params.get("raw_parameters", {})
        if raw_params:
            cp2k_params = _deep_merge(cp2k_params, raw_params)
            logger.debug(f"Merged raw CP2K parameter overrides")

        logger.debug(f"Built CP2K parameters for run_type={run_type}")
        return Dict(dict=cp2k_params)

    def get_relaxation_settings(self, relax_type: RelaxType) -> dict:
        """Return CP2K-specific cell_opt settings."""
        cell_opt_map = {
            "volume": "FULL",
            "shape": "ABC",
            "positions": "NONE",
            "cell": "FULL",
        }
        return {
            "cell_opt": cell_opt_map.get(relax_type, "FULL"),
        }

    def validate_structure(self, structure: "orm.StructureData") -> None:
        """Validate structure for CP2K, including element support."""
        super().validate_structure(structure)

        symbols = [site.kind_name for site in structure.sites]
        unique_elements = list(dict.fromkeys(symbols))

        logger.info(f"Validating structure with elements: {unique_elements}")
        return None


class EngineFactory:
    """Factory for creating engine adapters.

    Usage:
        adapter = EngineFactory.create("vasp")
        params = adapter.build_parameters({"encut": 500})
    """

    _adapters: dict[Literal["vasp", "cp2k"], type[BaseEngineAdapter]] = {
        "vasp": VaspAdapter,
        "cp2k": Cp2kAdapter,
    }

    @classmethod
    def create(cls, engine: EngineType) -> BaseEngineAdapter:
        """Create an engine adapter instance.

        Args:
            engine: The engine type ('vasp' or 'cp2k')

        Returns:
            Adapter instance for the specified engine

        Raises:
            EngineError: If the engine is not supported
        """
        engine = engine.lower().strip()

        if engine not in cls._adapters:
            raise EngineError(
                engine,
                supported=list(cls._adapters.keys()),
            )

        return cls._adapters[engine]()

    @classmethod
    def register_adapter(
        cls,
        engine: Literal["vasp", "cp2k"],
        adapter_class: type[BaseEngineAdapter],
    ) -> None:
        """Register a custom adapter for an engine.

        Args:
            engine: The engine type to register
            adapter_class: The adapter class to use
        """
        cls._adapters[engine] = adapter_class

    @classmethod
    def supported_engines(cls) -> list[str]:
        """Return list of supported engine names."""
        return list(cls._adapters.keys())
