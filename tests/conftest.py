"""Pytest configuration and fixtures.

Module-level mocks for AiiDA, pymatgen, and requests are installed
immediately so that test imports succeed even when these packages
are not installed in the environment.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Module-level mocks – installed before any test module is imported
# ---------------------------------------------------------------------------

# --- AiiDA ---
_aiida = MagicMock()
sys.modules.setdefault("aiida", _aiida)

_orm = MagicMock()
_orm.Bool = MagicMock
_orm.Str = MagicMock
_orm.Dict = MagicMock
_orm.KpointsData = MagicMock
_orm.StructureData = MagicMock
_orm.AbstractCode = MagicMock
_orm.Group = MagicMock()
_orm.FolderData = MagicMock
_orm.load_code = MagicMock

class MockStructureData:
    pk = 1
    pbc = (True, True, True)
    sites = []

    @classmethod
    def get_or_create(cls, path):
        return cls(), False

_orm.StructureData = MockStructureData

class MockDict:
    def __init__(self, data=None, dict=None):
        self._data = data or dict or {}

    def get_dict(self):
        return self._data

_orm.Dict = MockDict

class MockKpointsData:
    def set_kpoints_mesh(self, mesh):
        pass

_orm.KpointsData = MockKpointsData

sys.modules.setdefault("aiida.orm", _orm)

_plugins = MagicMock()
_plugins.CalculationFactory = MagicMock(return_value=MagicMock)
_plugins.WorkflowFactory = MagicMock(return_value=MagicMock)
_plugins.DataFactory = MagicMock(return_value=MockStructureData)
sys.modules.setdefault("aiida.plugins", _plugins)

_engine = MagicMock()
_engine.WorkChain = type("WorkChain", (), {"__init__": lambda self: None})
_engine.ToContext = MagicMock
_engine.submit = MagicMock
_engine.ProcessState = MagicMock
sys.modules.setdefault("aiida.engine", _engine)

# --- pymatgen ---
_pymatgen = MagicMock()
sys.modules.setdefault("pymatgen", _pymatgen)
sys.modules.setdefault("pymatgen.core", MagicMock())
sys.modules.setdefault("pymatgen.core.lattice", MagicMock())
sys.modules.setdefault("pymatgen.core.structure", MagicMock())

# --- requests ---
sys.modules.setdefault("requests", MagicMock())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_aiida_orm():
    """Return the module-level AiiDA ORM mock for tests that need it."""
    return _orm


@pytest.fixture(autouse=True)
def mock_aiida_imports(mock_aiida_orm):
    """Ensure AiiDA mocks are in place for every test."""
    yield