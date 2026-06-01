"""Configuration loader and validator for aiida-relax-project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class OptimadeConfig:
    """OPTIMADE API configuration."""

    endpoint: str = "https://optimade.materialscloud.org/main/mc2d/v1/structures"
    filter: Optional[str] = None
    max_structures: Optional[int] = None
    page_limit: int = 100


@dataclass
class TransformationsConfig:
    """Structure transformation configuration."""

    rotate_xy_to_xz: bool = False
    vacuum: float = 20.0
    make_supercell: Optional[list[int]] = None


@dataclass
class ResourcesConfig:
    """Computational resources configuration."""

    num_machines: int = 1
    num_mpiprocs_per_machine: int = 8
    withmpi: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""

    store_structures: bool = False
    save_cif: bool = False
    group_label: Optional[str] = None


@dataclass
class VaspConfig:
    """VASP-specific configuration."""

    code_label: str = "vasp@localhost"
    potential_family: str = "PBE.54"
    potential_mapping: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "ENCUT": 400,
        "EDIFF": 1e-5,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "NSW": 0,
    })


@dataclass
class Cp2kKindConfig:
    """CP2K KIND section configuration."""

    element: str
    basis_set: str
    potential: str


@dataclass
class Cp2kConfig:
    """CP2K-specific configuration."""

    code_label: str = "cp2k@localhost"
    basis_file: str = "BASIS_MOLOPT"
    potential_file: str = "GTH_POTENTIALS"
    kinds: list[Cp2kKindConfig] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "CUTOFF": 400,
        "EPS_SCF": 1e-6,
        "MAX_SCF": 200,
        "RUN_TYPE": "ENERGY",
        "CHARGE": 0,
        "MULTIPLICITY": 1,
    })


@dataclass
class StructureConfig:
    """Inline structure configuration."""

    lattice: list[list[float]]
    coords_are_fractional: bool = False
    atoms: list[list[Any]] = field(default_factory=list)


@dataclass
class CalculatorConfig:
    """Main configuration container."""

    engine: str
    description: str = ""
    optimade: OptimadeConfig = field(default_factory=OptimadeConfig)
    structure_group: Optional[str] = None
    structure: Optional[StructureConfig] = None
    transformations: TransformationsConfig = field(default_factory=TransformationsConfig)
    kpoints_mesh: Optional[list[int]] = None
    vasp: Optional[VaspConfig] = None
    cp2k: Optional[Cp2kConfig] = None
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    max_wallclock_seconds: int = 3600
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.engine not in ("vasp", "cp2k"):
            raise ValueError(f"engine must be 'vasp' or 'cp2k', got '{self.engine}'")

        if self.engine == "vasp" and self.vasp is None:
            raise ValueError("engine=vasp requires vasp section in config")

        if self.engine == "cp2k" and self.cp2k is None:
            raise ValueError("engine=cp2k requires cp2k section in config")

        has_optimade = self.optimade.filter or self.optimade.max_structures
        has_group = self.structure_group is not None
        has_inline = self.structure is not None

        if not (has_optimade or has_group or has_inline):
            raise ValueError(
                "Config must specify one of: optimade.filter, structure_group, or structure"
            )


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in config values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, value)
    return value


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path: str | Path) -> CalculatorConfig:
    """
    Load configuration from YAML file and merge with defaults.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        CalculatorConfig instance with merged configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    defaults_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
    defaults_data = {}

    if defaults_path.exists():
        with open(defaults_path) as f:
            defaults_data = yaml.safe_load(f) or {}

    with open(config_path) as f:
        config_data = yaml.safe_load(f) or {}

    merged = _deep_merge(defaults_data, config_data)

    return _dict_to_config(merged)


def _dict_to_config(data: dict[str, Any]) -> CalculatorConfig:
    """Convert dictionary to CalculatorConfig."""
    engine = data.get("engine", "vasp")

    optimade_data = data.get("optimade")
    optimade = _dict_to_optimade_config(optimade_data) if optimade_data else OptimadeConfig()

    transformations_data = data.get("transformations")
    transformations = (
        _dict_to_transformations_config(transformations_data)
        if transformations_data
        else TransformationsConfig()
    )

    resources_data = data.get("resources")
    resources = (
        _dict_to_resources_config(resources_data)
        if resources_data
        else ResourcesConfig()
    )

    output_data = data.get("output")
    output = _dict_to_output_config(output_data) if output_data else OutputConfig()

    kpoints_mesh = data.get("kpoints_mesh")
    structure_group = data.get("structure_group")
    structure = None
    if data.get("structure"):
        structure = _dict_to_struct_config(data["structure"])

    vasp = None
    if data.get("vasp"):
        vasp = _dict_to_vasp_config(data["vasp"])

    cp2k = None
    if data.get("cp2k"):
        cp2k = _dict_to_cp2k_config(data["cp2k"])

    return CalculatorConfig(
        engine=engine,
        description=data.get("description", ""),
        optimade=optimade,
        structure_group=structure_group,
        structure=structure,
        transformations=transformations,
        kpoints_mesh=kpoints_mesh,
        vasp=vasp,
        cp2k=cp2k,
        resources=resources,
        max_wallclock_seconds=data.get("max_wallclock_seconds", 3600),
        output=output,
    )


def _dict_to_optimade_config(data: dict | None) -> OptimadeConfig:
    """Convert dict to OptimadeConfig."""
    if data is None:
        return OptimadeConfig()
    return OptimadeConfig(
        endpoint=data.get("endpoint", "https://optimade.materialscloud.org/main/mc2d/v1/structures"),
        filter=data.get("filter"),
        max_structures=data.get("max_structures"),
        page_limit=data.get("page_limit", 100),
    )


def _dict_to_transformations_config(data: dict | None) -> TransformationsConfig:
    """Convert dict to TransformationsConfig."""
    if data is None:
        return TransformationsConfig()
    return TransformationsConfig(
        rotate_xy_to_xz=data.get("rotate_xy_to_xz", False),
        vacuum=data.get("vacuum", 20.0),
        make_supercell=data.get("make_supercell"),
    )


def _dict_to_resources_config(data: dict | None) -> ResourcesConfig:
    """Convert dict to ResourcesConfig."""
    if data is None:
        return ResourcesConfig()
    return ResourcesConfig(
        num_machines=data.get("num_machines", 1),
        num_mpiprocs_per_machine=data.get("num_mpiprocs_per_machine", 8),
        withmpi=data.get("withmpi", True),
    )


def _dict_to_output_config(data: dict | None) -> OutputConfig:
    """Convert dict to OutputConfig."""
    if data is None:
        return OutputConfig()
    return OutputConfig(
        store_structures=data.get("store_structures", False),
        save_cif=data.get("save_cif", False),
        group_label=data.get("group_label"),
    )


def _dict_to_struct_config(data: dict) -> StructureConfig:
    """Convert dict to StructureConfig."""
    return StructureConfig(
        lattice=data.get("lattice", []),
        coords_are_fractional=data.get("coords_are_fractional", False),
        atoms=data.get("atoms", []),
    )


def _dict_to_vasp_config(data: dict) -> VaspConfig:
    """Convert dict to VaspConfig."""
    return VaspConfig(
        code_label=data.get("code_label", "vasp@localhost"),
        potential_family=data.get("potential_family", "PBE.54"),
        potential_mapping=data.get("potential_mapping", {}),
        parameters=data.get("parameters", {}),
    )


def _dict_to_cp2k_config(data: dict) -> Cp2kConfig:
    """Convert dict to Cp2kConfig."""
    kinds = []
    for kind_data in data.get("kinds", []):
        kinds.append(
            Cp2kKindConfig(
                element=kind_data["element"],
                basis_set=kind_data["basis_set"],
                potential=kind_data["potential"],
            )
        )

    return Cp2kConfig(
        code_label=data.get("code_label", "cp2k@localhost"),
        basis_file=data.get("basis_file", "BASIS_MOLOPT"),
        potential_file=data.get("potential_file", "GTH_POTENTIALS"),
        kinds=kinds,
        parameters=data.get("parameters", {}),
    )