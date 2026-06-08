"""Type protocols for aiida-relax-project.

These protocols define the interfaces that engine-specific implementations
must follow, enabling static type checking while maintaining flexibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from aiida.orm import AbstractCode, Dict, KpointsData, StructureData

    from aiida_relax_project.core.enums import EngineType, RelaxType, RunType


class ParameterBuilder(Protocol):
    """Protocol for building engine-specific parameters."""

    def __call__(
        self,
        engine: EngineType,
        generic_params: dict,
    ) -> Dict:
        """Build engine-specific parameters from generic ones.

        Args:
            engine: The target engine ('vasp' or 'cp2k')
            generic_params: Dictionary of generic parameters

        Returns:
            AiiDA Dict with engine-specific parameters
        """
        ...


class InputBuilder(Protocol):
    """Protocol for building engine-specific inputs."""

    def __call__(
        self,
        code: AbstractCode,
        structure: StructureData,
        parameters: Dict,
        kpoints: KpointsData | None = None,
        metadata_options: dict | None = None,
        **extra_inputs,
    ) -> dict:
        """Build complete input dictionary for a calculation.

        Args:
            code: AiiDA code for the calculation
            structure: Structure to calculate
            parameters: Calculation parameters
            kpoints: K-points (optional for some engines)
            metadata_options: Scheduler options
            **extra_inputs: Additional engine-specific inputs

        Returns:
            Complete inputs dictionary for the calculation
        """
        ...


class WorkflowBuilder(Protocol):
    """Protocol for building complete workflow inputs."""

    def __call__(
        self,
        structure: StructureData,
        code: AbstractCode,
        parameters: Dict,
        relaxation_type: RelaxType | None = None,
        kpoints: KpointsData | None = None,
        **workflow_options,
    ) -> dict:
        """Build complete inputs for a relaxation workflow.

        Args:
            structure: Structure to relax
            code: AiiDA code
            parameters: Calculation parameters
            relaxation_type: Type of relaxation to perform
            kpoints: K-points data
            **workflow_options: Additional workflow options

        Returns:
            Complete inputs dictionary for the workflow
        """
        ...


class EngineAdapter(Protocol):
    """Protocol for engine-specific adapters.

    An EngineAdapter provides a consistent interface for interacting
    with different DFT engines (VASP, CP2K, etc.).
    """

    engine_type: EngineType

    def get_calculation_class(self) -> type:  # noqa: N802
        """Return the AiiDA calculation class for this engine."""
        ...

    def get_workflow_class(self, run_type: RunType) -> type:  # noqa: N802
        """Return the appropriate workchain class for the run type."""
        ...

    def build_parameters(self, generic_params: dict) -> Dict:  # noqa: N802
        """Translate generic parameters to engine format."""
        ...

    def build_kpoints(self, mesh: list[int]) -> KpointsData:  # noqa: N802
        """Create k-points data with engine-appropriate settings."""
        ...

    def validate_structure(self, structure: StructureData) -> None:  # noqa: N802
        """Validate structure is suitable for this engine."""
        ...

    def get_default_resources(self) -> dict:  # noqa: N802
        """Return default resource requirements."""
        ...

    def get_relaxation_settings(self, relax_type: RelaxType) -> dict:  # noqa: N802
        """Return engine-specific relaxation settings."""
        ...
