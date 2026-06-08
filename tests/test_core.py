"""Unit tests for aiida-relax-project core modules."""

import pytest

from aiida_relax_project.core.enums import (
    RESOURCE_PRESETS,
    CalculationMode,
    RelaxType,
)
from aiida_relax_project.core.exceptions import (
    AiidaRelaxError,
    ConfigurationError,
    EngineError,
    StructureValidationError,
)


class TestEnums:
    """Tests for enum definitions."""

    def test_relax_type_values(self):
        assert "volume" in RelaxType.__args__
        assert "shape" in RelaxType.__args__
        assert "positions" in RelaxType.__args__
        assert "cell" in RelaxType.__args__

    def test_calculation_mode_values(self):
        assert CalculationMode.SINGLE_POINT == "single-point"
        assert CalculationMode.RELAX == "relax"
        assert CalculationMode.VOLUME_SCAN == "volume-scan"

    def test_resource_presets_defined(self):
        assert "default" in RESOURCE_PRESETS
        assert "high_memory" in RESOURCE_PRESETS
        assert RESOURCE_PRESETS["default"]["num_machines"] == 1


class TestExceptions:
    """Tests for exception classes."""

    def test_engine_error_message(self):
        error = EngineError("quantum_espresso", ["vasp", "cp2k"])
        assert "quantum_espresso" in str(error)
        assert "vasp" in str(error)
        assert "cp2k" in str(error)

    def test_structure_validation_error_with_pk(self):
        error = StructureValidationError("Invalid structure", structure_pk=42)
        assert "42" in str(error)
        assert "Invalid structure" in str(error)

    def test_base_exception(self):
        error = AiidaRelaxError("base error")
        assert "base error" in str(error)

    def test_configuration_error(self):
        error = ConfigurationError("config error")
        assert "config error" in str(error)


class TestBuilders:
    """Tests for workflow builders helper functions."""

    def test_parse_generic_params(self):
        from aiida_relax_project.cli import _parse_params

        result = _parse_params("encut=500,max_steps=100")
        assert result["encut"] == 500
        assert result["max_steps"] == 100

        result = _parse_params("xc_functional=LDA")
        assert result["xc_functional"] == "LDA"

        result = _parse_params("verbose=true,converge=false")
        assert result["verbose"] is True
        assert result["converge"] is False

        result = _parse_params("tolerance=1e-6")
        assert result["tolerance"] == 1e-6

        assert _parse_params(None) == {}
        assert _parse_params("") == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
