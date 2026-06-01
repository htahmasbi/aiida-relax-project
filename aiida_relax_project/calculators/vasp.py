"""VASP calculator implementation."""

from __future__ import annotations
from aiida import orm
from aiida_relax_project.calculators.base import BaseCalculator


class VaspCalculator(BaseCalculator):
    """Calculator for VASP code."""

    def build_parameters(self, user_params: dict) -> orm.Dict:
        """
        Build VASP INCAR parameters.

        Args:
            user_params: User-provided INCAR parameters.

        Returns:
            AiiDA Dict with INCAR parameters.
        """
        defaults = {
            "ENCUT": 400,
            "PREC": "Normal",
            "EDIFF": 1e-5,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "NSW": 0,
        }

        incar = {**defaults, **user_params}

        return orm.Dict(dict={"incar": incar})

    def build_kinds(self, structure) -> list[dict]:
        """VASP doesn't use KIND sections."""
        return []

    def get_default_kpoints_mesh(self) -> list[int]:
        """Return default VASP k-point mesh."""
        return [4, 4, 4]

    def get_required_inputs(self) -> list[str]:
        """Return list of required input keys for VASP."""
        return [
            "code",
            "structure",
            "parameters",
            "kpoints",
            "potential_family",
            "potential_mapping",
        ]
