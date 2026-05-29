"""Unit tests for aiida-relax-project core modules."""

import pytest
from unittest.mock import MagicMock, patch

from aiida_relax_project.core.enums import (
    EngineType,
    RelaxType,
    CalculationMode,
    ConvergenceStatus,
    RESOURCE_PRESETS,
)
from aiida_relax_project.core.engine import (
    EngineFactory,
    VaspAdapter,
    Cp2kAdapter,
    BaseEngineAdapter,
)
from aiida_relax_project.core.exceptions import (
    AiidaRelaxError,
    EngineError,
    ConfigurationError,
    StructureValidationError,
)


class TestEnums:
    """Tests for enum definitions."""

    def test_engine_type_values(self):
        """Test engine type values are correct."""
        assert "vasp" in EngineType.__args__
        assert "cp2k" in EngineType.__args__

    def test_relax_type_values(self):
        """Test relaxation type values."""
        assert "volume" in RelaxType.__args__
        assert "shape" in RelaxType.__args__
        assert "positions" in RelaxType.__args__
        assert "cell" in RelaxType.__args__

    def test_calculation_mode_values(self):
        """Test calculation mode values."""
        assert CalculationMode.SINGLE_POINT == "single-point"
        assert CalculationMode.RELAX == "relax"
        assert CalculationMode.VOLUME_SCAN == "volume-scan"

    def test_resource_presets_defined(self):
        """Test resource presets are defined."""
        assert "default" in RESOURCE_PRESETS
        assert "high_memory" in RESOURCE_PRESETS
        assert "cp2k_large" in RESOURCE_PRESETS
        assert RESOURCE_PRESETS["default"]["num_machines"] == 1


class TestEngineFactory:
    """Tests for EngineFactory class."""

    def test_create_vasp_adapter(self):
        """Test creating VASP adapter."""
        adapter = EngineFactory.create("vasp")
        assert isinstance(adapter, VaspAdapter)
        assert adapter.engine_type == "vasp"

    def test_create_cp2k_adapter(self):
        """Test creating CP2K adapter."""
        adapter = EngineFactory.create("cp2k")
        assert isinstance(adapter, Cp2kAdapter)
        assert adapter.engine_type == "cp2k"

    def test_create_unknown_engine(self):
        """Test creating adapter for unknown engine raises error."""
        with pytest.raises(EngineError) as exc_info:
            EngineFactory.create("unknown")
        assert "unknown" in str(exc_info.value)
        assert exc_info.value.supported == ["vasp", "cp2k"]

    def test_supported_engines(self):
        """Test listing supported engines."""
        engines = EngineFactory.supported_engines()
        assert "vasp" in engines
        assert "cp2k" in engines

    def test_case_insensitive(self):
        """Test engine creation is case insensitive."""
        adapter1 = EngineFactory.create("VASP")
        adapter2 = EngineFactory.create("vasp")
        assert type(adapter1) == type(adapter2)


class TestVaspAdapter:
    """Tests for VaspAdapter class."""

    @pytest.fixture
    def adapter(self):
        return VaspAdapter()

    def test_get_calculation_class(self, adapter):
        """Test getting VASP calculation class."""
        calc_class = adapter.get_calculation_class()
        assert calc_class is not None
        assert "vasp" in str(calc_class).lower()

    def test_get_workflow_class_scf(self, adapter):
        """Test getting VASP workflow class for SCF."""
        wf_class = adapter.get_workflow_class("energy")
        assert wf_class is not None

    def test_get_workflow_class_relax(self, adapter):
        """Test getting VASP workflow class for relaxation."""
        wf_class = adapter.get_workflow_class("relax")
        assert wf_class is not None

    def test_build_parameters_basic(self, adapter):
        """Test building basic VASP parameters."""
        params = adapter.build_parameters({"encut": 500})
        params_dict = params.get_dict()
        assert "incar" in params_dict
        assert params_dict["incar"]["ENCUT"] == 500

    def test_build_parameters_with_relax(self, adapter):
        """Test building VASP parameters for relaxation."""
        params = adapter.build_parameters(
            {"encut": 400, "max_steps": 100},
            run_type="relax"
        )
        params_dict = params.get_dict()
        assert "incar" in params_dict
        assert "IBRION" in params_dict["incar"]
        assert "NSW" in params_dict["incar"]
        assert params_dict["incar"]["NSW"] == 100

    def test_build_parameters_with_xc(self, adapter):
        """Test building VASP parameters with XC functional."""
        params = adapter.build_parameters({"xc_functional": "LDA"})
        params_dict = params.get_dict()
        assert "GGA" in params_dict["incar"]
        assert params_dict["incar"]["GGA"] == "LDA"

    def test_get_relaxation_settings(self, adapter):
        """Test getting VASP relaxation settings."""
        settings = adapter.get_relaxation_settings("volume")
        assert "isif" in settings
        assert settings["isif"] == 3

        settings = adapter.get_relaxation_settings("positions")
        assert settings["isif"] == 2
        assert settings["ibrion"] == 1

    def test_validate_structure_valid(self, adapter):
        """Test validation passes for valid structure."""
        mock_structure = MagicMock()
        mock_structure.pbc = (True, True, True)
        mock_structure.pk = 123

        adapter.validate_structure(mock_structure)

    def test_validate_structure_invalid_pbc(self, adapter):
        """Test validation fails for non-3D periodic structure."""
        mock_structure = MagicMock()
        mock_structure.pbc = (True, True, False)
        mock_structure.pk = 123

        with pytest.raises(StructureValidationError) as exc_info:
            adapter.validate_structure(mock_structure)
        assert "3D" in str(exc_info.value)


class TestCp2kAdapter:
    """Tests for Cp2kAdapter class."""

    @pytest.fixture
    def adapter(self):
        return Cp2kAdapter()

    def test_get_calculation_class(self, adapter):
        """Test getting CP2K calculation class."""
        calc_class = adapter.get_calculation_class()
        assert calc_class is not None
        assert "cp2k" in str(calc_class).lower()

    def test_get_workflow_class_energy(self, adapter):
        """Test getting CP2K workflow class for energy."""
        wf_class = adapter.get_workflow_class("energy")
        assert wf_class is not None

    def test_build_parameters_basic(self, adapter):
        """Test building basic CP2K parameters."""
        params = adapter.build_parameters({})
        params_dict = params.get_dict()
        assert "GLOBAL" in params_dict
        assert params_dict["GLOBAL"]["RUN_TYPE"] == "ENERGY"
        assert "FORCE_EVAL" in params_dict

    def test_build_parameters_with_cutoff(self, adapter):
        """Test building CP2K parameters with custom cutoff."""
        params = adapter.build_parameters({"cutoff": 600})
        params_dict = params.get_dict()
        assert params_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] == 600

    def test_build_parameters_relax(self, adapter):
        """Test building CP2K parameters for relaxation."""
        params = adapter.build_parameters({"max_steps": 200}, run_type="relax")
        params_dict = params.get_dict()
        assert params_dict["GLOBAL"]["RUN_TYPE"] == "GEO_OPT"
        assert "MOTION" in params_dict
        assert params_dict["MOTION"]["GEO_OPT"]["MAX_ITER"] == 200

    def test_build_parameters_cell_opt(self, adapter):
        """Test building CP2K parameters for cell optimization."""
        params = adapter.build_parameters({}, run_type="cell_opt")
        params_dict = params.get_dict()
        assert params_dict["GLOBAL"]["RUN_TYPE"] == "CELL_OPT"

    def test_get_relaxation_settings(self, adapter):
        """Test getting CP2K relaxation settings."""
        settings = adapter.get_relaxation_settings("volume")
        assert "cell_opt" in settings
        assert settings["cell_opt"] == "FULL"

        settings = adapter.get_relaxation_settings("shape")
        assert settings["cell_opt"] == "ABC"


class TestExceptions:
    """Tests for exception classes."""

    def test_engine_error_message(self):
        """Test EngineError message format."""
        error = EngineError("quantum_espresso", ["vasp", "cp2k"])
        assert "quantum_espresso" in str(error)
        assert "vasp" in str(error)
        assert "cp2k" in str(error)

    def test_structure_validation_error_with_pk(self):
        """Test StructureValidationError with structure pk."""
        error = StructureValidationError("Invalid structure", structure_pk=42)
        assert "42" in str(error)
        assert "Invalid structure" in str(error)


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_kpoints_mesh_validation(self):
        """Test k-points mesh validation."""
        from aiida_relax_project.core.config import VaspConfig

        valid_config = VaspConfig(kpoints_mesh=[4, 4, 4])
        assert valid_config.kpoints_mesh == [4, 4, 4]

        with pytest.raises(ValueError):
            VaspConfig(kpoints_mesh=[4, 4])

        with pytest.raises(ValueError):
            VaspConfig(kpoints_mesh=[4, 0, 4])

    def test_metadata_options_limits(self):
        """Test metadata options value limits."""
        from aiida_relax_project.core.config import MetadataOptions

        valid = MetadataOptions(num_machines=2, num_mpiprocs_per_machine=16)
        assert valid.num_machines == 2

        with pytest.raises(ValueError):
            MetadataOptions(num_machines=0)

        with pytest.raises(ValueError):
            MetadataOptions(max_wallclock_seconds=30)


class TestBuilders:
    """Tests for workflow builders."""

    def test_parse_generic_params(self):
        """Test parsing generic parameter string."""
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