"""Base workflow classes and engine factory for unified VASP/CP2K support."""

from enum import Enum
from typing import Optional, Dict, Any, Type

from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory


class EngineType(Enum):
    VASP = "vasp"
    CP2K = "cp2k"


def get_calculation_class(engine: EngineType) -> Any:
    """Get the calculation class for the specified engine."""
    if engine == EngineType.VASP:
        return CalculationFactory("vasp.vasp")
    elif engine == EngineType.CP2K:
        return CalculationFactory("cp2k")
    raise ValueError(f"Unknown engine: {engine}")


def get_relax_workchain_class(engine: EngineType) -> Any:
    """Get the relax workchain class for the specified engine."""
    if engine == EngineType.VASP:
        return WorkflowFactory("vasp.relax")
    elif engine == EngineType.CP2K:
        return WorkflowFactory("cp2k.optimize")
    raise ValueError(f"Unknown engine: {engine}")


def create_base_builder(
    engine: EngineType,
    code: orm.AbstractCode,
    structure: orm.StructureData,
    parameters: orm.Dict,
    metadata_options: Optional[orm.Dict] = None,
    kpoints: Optional[orm.KpointsData] = None,
    **extra_inputs
) -> Dict[str, Any]:
    """Create a base builder dict for the specified engine."""
    
    base_inputs = {
        "code": code,
        "structure": structure,
        "parameters": parameters,
        **extra_inputs,
    }
    
    if kpoints is not None and engine == EngineType.VASP:
        base_inputs["kpoints"] = kpoints
    
    if metadata_options is not None:
        base_inputs["metadata"] = {"options": metadata_options.get_dict()}
    
    return base_inputs


def build_engine_parameters(
    engine: EngineType,
    generic_params: Dict[str, Any]
) -> orm.Dict:
    """Build engine-specific parameters from generic parameters."""
    
    if engine == EngineType.VASP:
        return _build_vasp_parameters(generic_params)
    elif engine == EngineType.CP2K:
        return _build_cp2k_parameters(generic_params)
    raise ValueError(f"Unknown engine: {engine}")


def _build_vasp_parameters(params: Dict[str, Any]) -> orm.Dict:
    """Build VASP INCAR parameters from generic params."""
    
    incar = {
        "ENCUT": params.get("encut", 400),
        "PREC": params.get("prec", "Normal"),
        "EDIFF": params.get("ediff", 1e-5),
        "ISMEAR": params.get("ismeart", 0),
        "SIGMA": params.get("sigma", 0.05),
    }
    
    if params.get("run_type") == "relax":
        incar.update({
            "IBRION": params.get("ibrion", 2),
            "ISIF": params.get("isif", 3),
            "NSW": params.get("max_steps", 50),
        })
    
    return orm.Dict(dict={"incar": incar})


def _build_cp2k_parameters(params: Dict[str, Any]) -> orm.Dict:
    """Build CP2K input parameters from generic params."""
    
    run_type = params.get("run_type", "energy")
    
    if run_type == "relax":
        run_type = "geo_opt"
    
    cp2k_params = {
        "GLOBAL": {
            "RUN_TYPE": run_type.upper(),
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
                    "EPS_SCF": params.get("eps_scf", 1.0e-6),
                    "MAX_SCF": params.get("max_scf", 200),
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
    
    if params.get("run_type") == "relax":
        cp2k_params["MOTION"] = {
            "GEO_OPT": {
                "MAX_ITER": params.get("max_steps", 50),
                "OPTIMIZER": params.get("optimizer", "BFGS"),
            }
        }
    
    return orm.Dict(dict=cp2k_params)


def validate_structure(structure: orm.StructureData) -> None:
    """Validate structure for both VASP and CP2K."""
    if structure.pbc != (True, True, True):
        raise ValueError("Structure must be periodic in 3D for DFT calculations")


def get_default_metadata_options(
    engine: EngineType,
    custom_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get default metadata options for the specified engine."""
    
    defaults = {
        EngineType.VASP: {
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 8,
            },
            "max_wallclock_seconds": 3600,
            "withmpi": True,
        },
        EngineType.CP2K: {
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 16,
            },
            "max_wallclock_seconds": 3600,
            "withmpi": True,
        },
    }
    
    options = defaults.get(engine, defaults[EngineType.VASP]).copy()
    
    if custom_options:
        options.update(custom_options)
    
    return options