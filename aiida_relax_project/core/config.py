"""Configuration management using Pydantic for validation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings

from aiida_relax_project.core.enums import RESOURCE_PRESETS, RelaxType
from aiida_relax_project.utils.cp2k_parsers import (
    resolve_potential_name,
    resolve_ri_basis_name,
)

logger = logging.getLogger(__name__)


class MetadataOptions(BaseModel):
    """Metadata options for AiiDA calculations."""

    num_machines: int = Field(default=1, ge=1)
    num_mpiprocs_per_machine: int = Field(default=8, ge=1)
    max_wallclock_seconds: int = Field(default=3600, ge=60)
    withmpi: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for AiiDA."""
        return {
            "resources": {
                "num_machines": self.num_machines,
                "num_mpiprocs_per_machine": self.num_mpiprocs_per_machine,
            },
            "max_wallclock_seconds": self.max_wallclock_seconds,
            "withmpi": self.withmpi,
        }


class VaspConfig(BaseModel):
    """VASP-specific configuration."""

    potential_family: str = Field(default="PBE.54")
    potential_mapping: dict[str, str] = Field(default_factory=dict)
    kpoints_mesh: list[int] = Field(default_factory=lambda: [4, 4, 4])
    default_encut: int = Field(default=400, ge=0)
    default_isif: int = Field(default=3, ge=0, le=7)
    default_ibrion: int = Field(default=2, ge=-1, le=15)
    raw_incar: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kpoints_mesh")
    @classmethod
    def validate_kpoints_mesh(cls, v: list[int]) -> list[int]:
        if len(v) != 3:
            raise ValueError("kpoints_mesh must have exactly 3 elements [kx, ky, kz]")
        if any(x <= 0 for x in v):
            raise ValueError("kpoints_mesh elements must be positive")
        return v


class Cp2kConfig(BaseModel):
    """CP2K-specific configuration."""

    kpoints_mesh: list[int] = Field(default_factory=lambda: [4, 1, 4])
    default_cutoff: int = Field(default=400, ge=0)
    default_max_scf: int = Field(default=200, ge=1)
    default_eps_scf: float = Field(default=1e-6, gt=0)
    basis_set_file: str = "BASIS_MOLOPT"
    potential_file: str = "GTH_POTENTIALS"
    basis_set_mapping: dict[str, str] = Field(default_factory=dict)
    potential_mapping: dict[str, str] = Field(default_factory=dict)
    ri_basis_set_mapping: dict[str, str] = Field(default_factory=dict)
    ri_basis_set_file: str | None = Field(default=None)
    raw_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kpoints_mesh")
    @classmethod
    def validate_kpoints_mesh(cls, v: list[int]) -> list[int]:
        if len(v) != 3:
            raise ValueError("kpoints_mesh must have exactly 3 elements")
        if any(x <= 0 for x in v):
            raise ValueError("kpoints_mesh elements must be positive")
        return v


class RelaxConfig(BaseModel):
    """Relaxation-specific configuration."""

    relaxation_type: RelaxType = Field(default="volume")
    default_max_steps: int = Field(default=50, ge=1)
    default_energy_tolerance: float = Field(default=1e-5, gt=0)
    default_force_tolerance: float = Field(default=0.01, gt=0)


class VolumeScanConfig(BaseModel):
    """Volume scan-specific configuration."""

    max_structures: int = Field(default=5, ge=1)
    continue_on_failure: bool = False
    default_group: str = Field(default="default_scan_group")


class GenericParamsConfig(BaseModel):
    """Default generic parameters."""

    encut: int = Field(default=400, ge=0)
    max_steps: int = Field(default=50, ge=1)
    energy_tolerance: float = Field(default=1e-5, gt=0)
    xc_functional: str = Field(default="PBE")
    scf_guess: str = "ATOMIC"


class ElementGwConfig(BaseModel):
    """Per-element GW settings (RI basis and potential)."""

    ri_basis: str
    potential: str


class GwConfig(BaseModel):
    """GW-specific configuration for CP2K bandstructure calculations.

    Per-element settings (RI basis + potential) can be:
      - Set explicitly via ``element_settings`` in config.toml, OR
      - Resolved automatically by setting ``auto_resolve = true``
        (the code reads *basis_set_file*, *ri_basis_set_file*, and
        *potential_file* and extracts the correct name for each element).
    """

    auto_resolve: bool = False
    ri_basis_accuracy_target: float | None = Field(
        default=None,
        description="Target relative accuracy for RI basis selection "
                    "(e.g. 1e-5). Picks the best basis not exceeding this value.",
    )
    basis_set_file: str = Field(
        default="/home/tahmas41/work/GW_2D/BASIS_AUG_MOLOPT/BASIS_GTH_MOLOPT_AUG_for_excited_states"
    )
    ri_basis_set_file: str = Field(
        default="/home/tahmas41/work/GW_2D/BASIS_AUG_MOLOPT/BASIS_GTH_MOLOPT_AUG_for_excited_states_RI"
    )
    potential_file: str = Field(
        default="/home/tahmas41/work/GW_2D/cp2k/data/POTENTIAL_UZH"
    )
    kpoints_mesh: list[int] = Field(default_factory=lambda: [12, 1, 12])
    kpoints_w: list[int] = Field(default_factory=lambda: [12, 1, 12])
    periodic: str = Field(default="XZ")
    poisson_solver: str = Field(default="WAVELET")
    cutoff: int = Field(default=400, ge=0)
    rel_cutoff: int = Field(default=50, ge=0)
    eps_default: float = Field(default=1.0e-12, gt=0)
    eps_pgf_orb: float = Field(default=1.0e-12, gt=0)
    eps_scf: float = Field(default=5.0e-7, gt=0)
    max_scf: int = Field(default=500, ge=1)
    mixing_alpha: float = Field(default=0.2)
    mixing_beta: float = Field(default=0.8)
    mixing_nbroyden: int = Field(default=10)
    num_time_freq: int = Field(default=10, ge=1)
    memory_per_proc: int = Field(default=600, ge=1)
    eps_filter: float = Field(default=1.0e-6, gt=0)
    cutoff_radius_ri: int = Field(default=5, ge=1)
    regularization_ri: float = Field(default=0.01, gt=0)
    orb_basis: str = Field(default="aug-SZV-MOLOPT-GTH-tier-1")
    vacuum: float = Field(default=20.0, ge=5.0,
                          description="Vacuum gap (A) along y after rotation.")
    supercell: list[int] = Field(default_factory=lambda: [3, 3, 1])
    element_settings: dict[str, ElementGwConfig] = Field(default_factory=lambda: {
        "B": ElementGwConfig(
            ri_basis="RI_aug-SZV-MOLOPT-GTH-tier-1_N_RI_009_s_p_d_f_g_h_i_3_2_0_0_0_0_0_error_1.1e-06",
            potential="GTH-PBE-q3",
        ),
        "N": ElementGwConfig(
            ri_basis="RI_aug-SZV-MOLOPT-GTH-tier-1_N_RI_025_s_p_d_f_g_h_i_6_3_2_0_0_0_0_error_2.9e-06",
            potential="GTH-PBE-q5",
        ),
    })
    bs_npoints: int = Field(default=20, ge=1)
    special_points: list[str] | None = Field(
        default=None,
        description="Optional override for the bandstructure k-point path. "
        "When set to None (default), the path is auto-computed from the "
        "structure's space group via pymatgen's HighSymmKpath.",
    )

    def resolve_elements(self, elements: set[str]) -> dict[str, ElementGwConfig]:
        """Auto-resolve per-element settings from the configured files.

        *elements* — set of element symbols (e.g. ``{"B", "N"}``).
        Requires the three file paths to be readable locally.
        """
        result: dict[str, ElementGwConfig] = {}
        errors: list[str] = []

        for el in sorted(elements):
            try:
                potential = resolve_potential_name(self.potential_file, el)
                if potential is None:
                    errors.append(
                        f"{el}: no potential found in {self.potential_file}"
                    )
                    continue
                ri_basis = resolve_ri_basis_name(
                    self.ri_basis_set_file, el,
                    accuracy_target=self.ri_basis_accuracy_target,
                    orb_basis=self.orb_basis,
                )
                if ri_basis is None:
                    errors.append(
                        f"{el}: no RI basis found in {self.ri_basis_set_file}"
                    )
                    continue
                result[el] = ElementGwConfig(ri_basis=ri_basis, potential=potential)
            except FileNotFoundError as exc:
                errors.append(f"Cannot read {exc.filename} — file not found locally")
            except PermissionError as exc:
                errors.append(f"Cannot read {exc.filename} — permission denied")
            except OSError as exc:
                errors.append(f"File error reading {getattr(exc, 'filename', 'unknown')}: {exc}")
            except ValueError as exc:
                errors.append(f"Parse error for {el}: {exc}")

        if errors:
            raise RuntimeError(
                "Auto-resolve failed for some elements:\n  "
                + "\n  ".join(errors)
                + "\n\nEither:\n"
                "  1. Make the files accessible locally (mount, copy, or sshfs)\n"
                "  2. Set element_settings manually in config.toml (see [gw.element_settings])\n"
                "  3. Use the --resolve-remote flag to parse via SSH"
            )

        return result


class ProjectConfig(BaseSettings):
    """Main configuration class for aiida-relax-project.

    Loads configuration from:
    1. Environment variables (highest priority)
    2. config.toml file
    3. .env file
    4. Defaults (lowest priority)

    Environment variables follow the pattern: SECTION_KEY (e.g., ENGINE, VASP_ENCODUT)
    """

    model_config = {"env_nested_delimiter": "__", "extra": "ignore"}

    engine: Literal["vasp", "cp2k"] = Field(default="vasp")
    code_label: str = Field(default="localhost")
    aiida_profile: str | None = None

    resource_preset: str = Field(default="default")

    vasp: VaspConfig = Field(default_factory=VaspConfig)
    cp2k: Cp2kConfig = Field(default_factory=Cp2kConfig)
    gw: GwConfig = Field(default_factory=GwConfig)
    relax: RelaxConfig = Field(default_factory=RelaxConfig)
    volume_scan: VolumeScanConfig = Field(default_factory=VolumeScanConfig)
    generic: GenericParamsConfig = Field(default_factory=GenericParamsConfig)

    @field_validator("resource_preset")
    @classmethod
    def validate_resource_preset(cls, v: str) -> str:
        if v not in RESOURCE_PRESETS:
            logger.warning(
                f"Unknown resource preset '{v}', using 'default'. "
                f"Available: {list(RESOURCE_PRESETS.keys())}"
            )
            return "default"
        return v

    @model_validator(mode="after")
    def setup_environment(self) -> ProjectConfig:
        if self.aiida_profile:
            os.environ["AIIDA_PROFILE"] = self.aiida_profile
        return self

    @computed_field
    @property
    def metadata_options(self) -> MetadataOptions:
        """Get metadata options based on resource preset and engine."""
        preset = RESOURCE_PRESETS.get(self.resource_preset, RESOURCE_PRESETS["default"])

        return MetadataOptions(
            num_machines=preset.get("num_machines", 1),
            num_mpiprocs_per_machine=preset.get("num_mpiprocs_per_machine", 8),
        )

    def get_kpoints_mesh(self) -> list[int]:
        """Get k-points mesh for the current engine."""
        if self.engine == "cp2k":
            return self.cp2k.kpoints_mesh
        return self.vasp.kpoints_mesh

    def get_potential_mapping(self, elements: list[str] | None = None) -> dict[str, str]:
        """Get VASP potential mapping for given elements."""
        if elements:
            return {el: el for el in elements}
        if self.vasp.potential_mapping:
            return self.vasp.potential_mapping
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "engine": self.engine,
            "code_label": self.code_label,
            "resource_preset": self.resource_preset,
            "metadata_options": self.metadata_options.to_dict(),
            "vasp": self.vasp.model_dump(),
            "cp2k": self.cp2k.model_dump(),
            "gw": self.gw.model_dump(),
            "relax": self.relax.model_dump(),
            "volume_scan": self.volume_scan.model_dump(),
        }


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load configuration from file and environment.

    Args:
        config_path: Optional path to config.toml file.
                    If None, searches in standard locations.

    Returns:
        Validated ProjectConfig instance
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    config_data: dict[str, Any] = {}

    if config_path is None:
        search_paths = [
            Path.cwd() / "config.toml",
            Path(__file__).parent.parent.parent / "config.toml",
            Path.home() / ".config" / "aiida-relax" / "config.toml",
        ]

        for path in search_paths:
            if path.exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)

        # Unwrap [default] section into top-level keys (so engine, code_label
        # etc. are read by ProjectConfig, not lost under "default").
        if "default" in config_data:
            defaults = config_data.pop("default")
            for k, v in defaults.items():
                config_data.setdefault(k, v)

        logger.info(f"Loaded configuration from {config_path}")

    env_overrides = _parse_env_overrides()
    config_data = _merge_configs(config_data, env_overrides)

    return ProjectConfig(**config_data)


def _parse_env_overrides() -> dict[str, Any]:
    """Parse environment variable overrides."""
    overrides: dict[str, Any] = {}

    if os.environ.get("ENGINE"):
        overrides["engine"] = os.environ["ENGINE"]

    if os.environ.get("CODE_LABEL"):
        overrides["code_label"] = os.environ["CODE_LABEL"]

    if os.environ.get("VASP_POTENTIAL_FAMILY"):
        overrides.setdefault("vasp", {})["potential_family"] = os.environ["VASP_POTENTIAL_FAMILY"]

    if os.environ.get("DEFAULT_ENCUT"):
        overrides.setdefault("vasp", {})["default_encut"] = int(os.environ["DEFAULT_ENCUT"])

    return overrides


def _merge_configs(base: dict, overrides: dict) -> dict:
    """Recursively merge override config into base config."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


_config: ProjectConfig | None = None


def get_config() -> ProjectConfig:
    """Get the global configuration singleton (lazy-loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the configuration singleton (useful for testing)."""
    global _config
    _config = None
