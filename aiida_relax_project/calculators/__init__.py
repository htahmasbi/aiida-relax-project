"""Calculator abstraction layer for VASP and CP2K."""

from aiida_relax_project.calculators.base import BaseCalculator
from aiida_relax_project.calculators.vasp import VaspCalculator
from aiida_relax_project.calculators.cp2k import Cp2kCalculator

__all__ = ["BaseCalculator", "VaspCalculator", "Cp2kCalculator"]


def get_calculator(engine: str) -> BaseCalculator:
    """Factory function to get the appropriate calculator."""
    if engine == "vasp":
        return VaspCalculator()
    elif engine == "cp2k":
        return Cp2kCalculator()
    else:
        raise ValueError(f"Unknown engine: {engine}. Must be 'vasp' or 'cp2k'")