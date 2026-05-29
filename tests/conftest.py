"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_aiida_orm():
    """Mock AiiDA orm module for testing without AiiDA installed."""
    mock_module = type(sys)('aiida.orm')

    class MockStructureData:
        pk = 1
        pbc = (True, True, True)
        sites = []

        @classmethod
        def get_or_create(cls, path):
            return cls(), False

    class MockDict:
        def __init__(self, data):
            self._data = data

        def get_dict(self):
            return self._data

    class MockKpointsData:
        def set_kpoints_mesh(self, mesh):
            pass

    class MockCode:
        pass

    mock_module.StructureData = MockStructureData
    mock_module.Dict = MockDict
    mock_module.KpointsData = MockKpointsData
    mock_module.AbstractCode = MockCode

    return mock_module


@pytest.fixture(autouse=True)
def mock_aiida_imports(mock_aiida_orm):
    """Automatically mock AiiDA imports in tests."""
    import sys
    from unittest.mock import MagicMock

    mock_orm = MagicMock()
    mock_orm.StructureData = mock_aiida_orm.StructureData
    mock_orm.Dict = mock_aiida_orm.Dict
    mock_orm.KpointsData = mock_aiida_orm.KpointsData
    mock_orm.AbstractCode = mock_aiida_orm.AbstractCode
    mock_orm.Bool = MagicMock
    mock_orm.Str = MagicMock
    mock_orm.Group = MagicMock()

    sys.modules['aiida'] = MagicMock()
    sys.modules['aiida.orm'] = mock_orm

    yield

    for module in ['aiida', 'aiida.orm']:
        if module in sys.modules:
            del sys.modules[module]