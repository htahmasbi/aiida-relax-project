"""Custom exceptions for aiida-relax-project."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiida_relax_project.core.engine import EngineType


class AiidaRelaxError(Exception):
    """Base exception for all aiida-relax-project errors."""

    pass


class EngineError(AiidaRelaxError):
    """Raised when an unknown or unsupported engine is specified."""

    def __init__(self, engine: str, supported: list[str] | None = None) -> None:
        self.engine = engine
        self.supported = supported or ["vasp", "cp2k"]
        message = f"Unsupported engine: '{engine}'. Supported: {', '.join(self.supported)}"
        super().__init__(message)


class ConfigurationError(AiidaRelaxError):
    """Raised when configuration is invalid or missing."""

    pass


class StructureValidationError(AiidaRelaxError):
    """Raised when a structure fails validation."""

    def __init__(self, message: str, structure_pk: int | None = None) -> None:
        self.structure_pk = structure_pk
        detail = f" (structure_pk={structure_pk})" if structure_pk else ""
        super().__init__(f"{message}{detail}")


class WorkflowExecutionError(AiidaRelaxError):
    """Raised when a workflow execution fails."""

    def __init__(
        self,
        message: str,
        workflow_pk: int | None = None,
        exit_code: int | None = None,
        engine: EngineType | None = None,
    ) -> None:
        self.workflow_pk = workflow_pk
        self.exit_code = exit_code
        self.engine = engine
        detail_parts = []
        if workflow_pk:
            detail_parts.append(f"workflow_pk={workflow_pk}")
        if exit_code:
            detail_parts.append(f"exit_code={exit_code}")
        if engine:
            detail_parts.append(f"engine={engine}")
        detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
        super().__init__(f"{message}{detail}")
