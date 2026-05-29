"""Configuration management for aiida-relax-project.

Loads settings from:
1. Environment variables (highest priority)
2. config.toml file
3. .env file (via python-dotenv)
4. Defaults (lowest priority)
"""

import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class VaspConfig:
    potential_family: str = "PBE.54"
    kpoints_mesh: List[int] = field(default_factory=lambda: [4, 4, 4])
    default_encut: int = 400
    num_mpiprocs_per_machine: int = 8


@dataclass
class Cp2kConfig:
    kpoints_mesh: List[int] = field(default_factory=lambda: [4, 1, 4])
    default_cutoff: int = 400
    num_mpiprocs_per_machine: int = 16


@dataclass
class RelaxConfig:
    relaxation_type: str = "volume"
    default_max_steps: int = 50
    default_energy_tolerance: float = 1e-5


@dataclass
class VolumeScanConfig:
    max_structures: int = 5
    continue_on_failure: bool = False
    default_group: str = "default_scan_group"


@dataclass
class Config:
    engine: str = "vasp"
    code_label: str = "localhost"
    max_wallclock_seconds: int = 3600
    num_machines: int = 1

    vasp: VaspConfig = field(default_factory=VaspConfig)
    cp2k: Cp2kConfig = field(default_factory=Cp2kConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    volume_scan: VolumeScanConfig = field(default_factory=VolumeScanConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from file and environment variables."""
        config_file = config_path or os.environ.get("AIIDA_RELAX_CONFIG")

        if config_file and Path(config_file).exists():
            with open(config_file, "rb") as f:
                data = tomllib.load(f)
        else:
            config_dir = Path(__file__).parent.parent
            default_config = config_dir / "config.toml"
            data = {}

            if default_config.exists():
                with open(default_config, "rb") as f:
                    data = tomllib.load(f)

        config = cls()

        if "default" in data:
            config.engine = os.environ.get("ENGINE", data["default"].get("engine", "vasp"))
            config.code_label = os.environ.get("CODE_LABEL", data["default"].get("code_label", "localhost"))
            config.max_wallclock_seconds = int(os.environ.get(
                "MAX_WALLCLOCK_SECONDS",
                data["default"].get("max_wallclock_seconds", 3600)
            ))

        if "vasp" in data:
            config.vasp = VaspConfig(
                potential_family=os.environ.get(
                    "VASP_POTENTIAL_FAMILY",
                    data["vasp"].get("potential_family", "PBE.54")
                ),
                kpoints_mesh=data["vasp"].get("kpoints_mesh", [4, 4, 4]),
                default_encut=int(os.environ.get(
                    "DEFAULT_ENCUT",
                    data["vasp"].get("default_encut", 400)
                )),
                num_mpiprocs_per_machine=int(os.environ.get(
                    "NUM_MPIPROCS_PER_MACHINE",
                    data["vasp"].get("num_mpiprocs_per_machine", 8)
                )),
            )

        if "cp2k" in data:
            config.cp2k = Cp2kConfig(
                kpoints_mesh=data["cp2k"].get("kpoints_mesh", [4, 1, 4]),
                default_cutoff=int(os.environ.get(
                    "CP2K_CUTOFF",
                    data["cp2k"].get("default_cutoff", 400)
                )),
                num_mpiprocs_per_machine=int(os.environ.get(
                    "NUM_MPIPROCS_PER_MACHINE",
                    data["cp2k"].get("num_mpiprocs_per_machine", 16)
                )),
            )

        if "relax" in data:
            config.relax = RelaxConfig(
                relaxation_type=data["relax"].get("relaxation_type", "volume"),
                default_max_steps=int(os.environ.get(
                    "DEFAULT_MAX_STEPS",
                    data["relax"].get("default_max_steps", 50)
                )),
                default_energy_tolerance=float(os.environ.get(
                    "DEFAULT_ENERGY_TOLERANCE",
                    data["relax"].get("default_energy_tolerance", 1e-5)
                )),
            )

        if "volume_scan" in data:
            config.volume_scan = VolumeScanConfig(
                max_structures=int(os.environ.get(
                    "DEFAULT_MAX_STRUCTURES",
                    data["volume_scan"].get("max_structures", 5)
                )),
                continue_on_failure=data["volume_scan"].get("continue_on_failure", False),
                default_group=data["volume_scan"].get("default_group", "default_scan_group"),
            )

        return config

    def get_kpoints_mesh(self) -> List[int]:
        """Get k-points mesh for current engine."""
        if self.engine == "cp2k":
            return self.cp2k.kpoints_mesh
        return self.vasp.kpoints_mesh

    def get_metadata_options(self) -> dict:
        """Get default metadata options for current engine."""
        num_procs = (
            self.cp2k.num_mpiprocs_per_machine
            if self.engine == "cp2k"
            else self.vasp.num_mpiprocs_per_machine
        )

        return {
            "resources": {
                "num_machines": self.num_machines,
                "num_mpiprocs_per_machine": num_procs,
            },
            "max_wallclock_seconds": self.max_wallclock_seconds,
            "withmpi": True,
        }


_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration singleton."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reset_config() -> None:
    """Reset the configuration (useful for testing)."""
    global _config
    _config = None