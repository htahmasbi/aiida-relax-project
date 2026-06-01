"""CP2K calculator implementation."""

from __future__ import annotations

from aiida import orm
from pymatgen.core import Structure

from aiida_relax_project.calculators.base import BaseCalculator
from aiida_relax_project.config import Cp2kConfig


class Cp2kCalculator(BaseCalculator):
    """Calculator for CP2K code."""

    def __init__(self, config: Cp2kConfig | None = None):
        """
        Initialize CP2K calculator.

        Args:
            config: CP2K configuration with KIND mappings.
        """
        self.config = config

    def build_parameters(self, user_params: dict) -> orm.Dict:
        """
        Build CP2K input blocks.

        Args:
            user_params: User-provided CP2K parameters.

        Returns:
            AiiDA Dict with CP2K input.
        """
        defaults = {
            "RUN_TYPE": "ENERGY",
            "CHARGE": 0,
            "MULTIPLICITY": 1,
            "EPS_SCF": 1e-6,
            "MAX_SCF": 200,
        }

        run_type = user_params.get("RUN_TYPE", defaults["RUN_TYPE"])
        charge = user_params.get("CHARGE", defaults["CHARGE"])
        multiplicity = user_params.get("MULTIPLICITY", defaults["MULTIPLICITY"])

        cp2k_input = {
            "GLOBAL": {
                "RUN_TYPE": run_type,
                "PRINT_LEVEL": "MEDIUM",
            },
            "FORCE_EVAL": {
                "METHOD": "Quickstep",
                "DFT": {
                    "BASIS_SET_FILE_NAME": (
                        self.config.basis_file if self.config else "BASIS_MOLOPT"
                    ),
                    "POTENTIAL_FILE_NAME": (
                        self.config.potential_file if self.config else "GTH_POTENTIALS"
                    ),
                    "CHARGE": charge,
                    "MULTIPLICITY": multiplicity,
                    "SCF": {
                        "SCF_GUESS": "ATOMIC",
                        "EPS_SCF": user_params.get("EPS_SCF", defaults["EPS_SCF"]),
                        "MAX_SCF": user_params.get("MAX_SCF", defaults["MAX_SCF"]),
                    },
                    "XC": {
                        "XC_FUNCTIONAL": {
                            "_": "PBE",
                        },
                    },
                },
                "MGRID": {
                    "CUTOFF": user_params.get("CUTOFF", 400),
                },
            },
        }

        return orm.Dict(dict=cp2k_input)

    def build_kinds(self, structure: Structure) -> list[dict]:
        """
        Build KIND sections from config for the structure's elements.

        Args:
            structure: pymatgen Structure.

        Returns:
            List of KIND dictionaries.
        """
        if not self.config or not self.config.kinds:
            return []

        structure_elements = {str(site.specie) for site in structure}

        kinds = []
        for kind_config in self.config.kinds:
            if kind_config.element in structure_elements:
                kinds.append(
                    {
                        "_": kind_config.element,
                        "BASIS_SET": kind_config.basis_set,
                        "POTENTIAL": kind_config.potential,
                    }
                )

        return kinds

    def get_default_kpoints_mesh(self) -> list[int]:
        """Return default CP2K k-point mesh (often 1 point for 2D)."""
        return [4, 1, 4]

    def get_required_inputs(self) -> list[str]:
        """Return list of required input keys for CP2K."""
        return ["code", "structure", "parameters", "kpoints"]