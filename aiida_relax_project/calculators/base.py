"""Base calculator abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aiida import orm
from pymatgen.core import Structure


class BaseCalculator(ABC):
    """Abstract base class for DFT code calculators."""

    @abstractmethod
    def build_parameters(self, user_params: dict) -> orm.Dict:
        """
        Build code-specific parameters from user configuration.

        Args:
            user_params: User-provided parameters from config.

        Returns:
            AiiDA Dict with code-specific parameters.
        """
        pass

    @abstractmethod
    def build_kinds(self, structure: Structure) -> list[dict]:
        """
        Build KIND sections for the structure's elements.

        Args:
            structure: pymatgen Structure.

        Returns:
            List of KIND dictionaries for CP2K.
        """
        pass

    @abstractmethod
    def get_default_kpoints_mesh(self) -> list[int]:
        """Return default k-point mesh."""
        pass

    def build_metadata_options(self, resources: dict, max_wallclock: int) -> dict:
        """
        Build metadata options for the calculation.

        Args:
            resources: Resources configuration.
            max_wallclock: Maximum walltime in seconds.

        Returns:
            Dictionary with metadata options.
        """
        return {
            "resources": {
                "num_machines": resources.get("num_machines", 1),
                "num_mpiprocs_per_machine": resources.get(
                    "num_mpiprocs_per_machine", 8
                ),
            },
            "max_wallclock_seconds": max_wallclock,
            "withmpi": resources.get("withmpi", True),
        }