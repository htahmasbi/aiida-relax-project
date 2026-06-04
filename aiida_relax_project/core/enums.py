"""Engine types and related enumerations."""

from __future__ import annotations

from enum import Enum
from typing import Literal

EngineType = Literal["vasp", "cp2k"]
"""Supported DFT calculation engines."""

RunType = Literal["energy", "scf", "relax", "cell_opt"]
"""Types of calculations that can be performed."""

RelaxType = Literal["volume", "shape", "positions", "cell"]
"""Types of relaxation degrees of freedom."""


class CalculationMode(str, Enum):
    """Calculation mode enumeration."""

    SINGLE_POINT = "single-point"
    RELAX = "relax"
    VOLUME_SCAN = "volume-scan"

    def __str__(self) -> str:
        return self.value


class ConvergenceStatus(str, Enum):
    """Status of a calculation or workflow."""

    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    UNCONVERGED = "unconverged"


# Resource presets for different HPC systems
RESOURCE_PRESETS: dict[str, dict] = {
    "default": {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 8,
    },
    "high_memory": {
        "num_machines": 2,
        "num_mpiprocs_per_machine": 8,
    },
    "gpu": {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 4,
        "num_cores_per_machine": 8,
    },
    "cp2k_large": {
        "num_machines": 4,
        "num_mpiprocs_per_machine": 16,
    },
}
