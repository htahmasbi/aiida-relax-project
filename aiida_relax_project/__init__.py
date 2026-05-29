"""aiida-relax-project: Unified AiiDA workflows for VASP and CP2K.

A professional framework for running DFT calculations with AiiDA,
supporting both VASP and CP2K engines with a unified interface.

Basic Usage:
    >>> from aiida_relax_project.core.config import get_config
    >>> from aiida_relax_project.core.engine import EngineFactory
    >>> from aiida_relax_project.core.builders import create_example_structure
    >>>
    >>> # Load configuration
    >>> config = get_config()
    >>>
    >>> # Create an engine adapter
    >>> adapter = EngineFactory.create("vasp")
    >>> params = adapter.build_parameters({"encut": 500})
    >>>
    >>> # Create a test structure
    >>> structure = create_example_structure("Si")

CLI Usage:
    $ aiida-relax run --mode single-point --engine vasp
    $ aiida-relax config-show
    $ aiida-relax engines

Configuration:
    Configuration is loaded from (in order of priority):
    1. CLI arguments / environment variables
    2. config.toml file
    3. .env file
    4. Defaults

Environment Variables:
    ENGINE=vasp|cp2k           # Default engine
    CODE_LABEL=localhost       # Default code label
    AIIDA_PROFILE=default      # AiiDA profile to use
"""

__version__ = "0.2.0"
__author__ = "Hossein Tahmasbi"

from aiida_relax_project.core import (
    AiidaRelaxError,
    EngineError,
    ConfigurationError,
    StructureValidationError,
    WorkflowExecutionError,
    EngineType,
    RunType,
    RelaxType,
    CalculationMode,
    get_config,
    setup_logging,
    get_logger,
)

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "AiidaRelaxError",
    "EngineError",
    "ConfigurationError",
    "StructureValidationError",
    "WorkflowExecutionError",
    # Types
    "EngineType",
    "RunType",
    "RelaxType",
    "CalculationMode",
    # Config & Logging
    "get_config",
    "setup_logging",
    "get_logger",
]
