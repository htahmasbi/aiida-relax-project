from __future__ import annotations

import pytest

from aiida_relax_project.core.engine import (
    EngineFactory,
    VaspAdapter,
    Cp2kAdapter,
)
from aiida_relax_project.core.exceptions import EngineError


class TestEngineFactory:
    """Tests for EngineFactory class."""

    def test_create_vasp_adapter(self):
        adapter = EngineFactory.create("vasp")
        assert isinstance(adapter, VaspAdapter)
        assert adapter.engine_type == "vasp"

    def test_create_cp2k_adapter(self):
        adapter = EngineFactory.create("cp2k")
        assert isinstance(adapter, Cp2kAdapter)
        assert adapter.engine_type == "cp2k"

    def test_create_unknown_engine(self):
        with pytest.raises(EngineError) as exc_info:
            EngineFactory.create("unknown")
        assert "unknown" in str(exc_info.value)
        assert exc_info.value.supported == ["vasp", "cp2k"]

    def test_supported_engines(self):
        engines = EngineFactory.supported_engines()
        assert "vasp" in engines
        assert "cp2k" in engines

    def test_case_insensitive(self):
        adapter1 = EngineFactory.create("VASP")
        adapter2 = EngineFactory.create("vasp")
        assert type(adapter1) == type(adapter2)


class TestVaspAdapter:
    """Tests for VaspAdapter class."""

    @pytest.fixture
    def adapter(self):
        return VaspAdapter()

    def test_engine_type(self, adapter):
        assert adapter.engine_type == "vasp"

    def test_build_parameters_basic(self, adapter):
        params = adapter.build_parameters({"encut": 500})
        params_dict = params.get_dict()
        assert "incar" in params_dict
        assert params_dict["incar"]["ENCUT"] == 500

    def test_build_parameters_with_ismear(self, adapter):
        params = adapter.build_parameters({"ismear": 1})
        params_dict = params.get_dict()
        assert params_dict["incar"]["ISMEAR"] == 1

    def test_build_parameters_with_relax(self, adapter):
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
        params = adapter.build_parameters({"xc_functional": "LDA"})
        params_dict = params.get_dict()
        assert "GGA" in params_dict["incar"]
        assert params_dict["incar"]["GGA"] == "LDA"

    def test_get_relaxation_settings(self, adapter):
        settings = adapter.get_relaxation_settings("volume")
        assert "isif" in settings
        assert settings["isif"] == 3

        settings = adapter.get_relaxation_settings("positions")
        assert settings["isif"] == 2
        assert settings["ibrion"] == 1

    def test_get_default_resources(self, adapter):
        resources = adapter.get_default_resources()
        assert "num_machines" in resources
        assert "num_mpiprocs_per_machine" in resources


class TestCp2kAdapter:
    """Tests for Cp2kAdapter class."""

    @pytest.fixture
    def adapter(self):
        return Cp2kAdapter()

    def test_engine_type(self, adapter):
        assert adapter.engine_type == "cp2k"

    def test_build_parameters_basic(self, adapter):
        params = adapter.build_parameters({})
        params_dict = params.get_dict()
        assert "GLOBAL" in params_dict
        assert params_dict["GLOBAL"]["RUN_TYPE"] == "ENERGY"
        assert "FORCE_EVAL" in params_dict

    def test_build_parameters_with_cutoff(self, adapter):
        params = adapter.build_parameters({"cutoff": 600})
        params_dict = params.get_dict()
        assert params_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] == 600

    def test_build_parameters_relax(self, adapter):
        params = adapter.build_parameters({"max_steps": 200}, run_type="relax")
        params_dict = params.get_dict()
        assert params_dict["GLOBAL"]["RUN_TYPE"] == "GEO_OPT"
        assert "MOTION" in params_dict
        assert params_dict["MOTION"]["GEO_OPT"]["MAX_ITER"] == 200

    def test_build_parameters_cell_opt(self, adapter):
        params = adapter.build_parameters({}, run_type="cell_opt")
        params_dict = params.get_dict()
        assert params_dict["GLOBAL"]["RUN_TYPE"] == "CELL_OPT"

    def test_get_relaxation_settings(self, adapter):
        settings = adapter.get_relaxation_settings("volume")
        assert "cell_opt" in settings
        assert settings["cell_opt"] == "FULL"

        settings = adapter.get_relaxation_settings("shape")
        assert settings["cell_opt"] == "ABC"

    def test_build_kpoints(self, adapter):
        kpoints = adapter.build_kpoints([4, 4, 4])
        assert kpoints is not None

    def test_build_parameters_with_ri_basis(self, adapter):
        params = adapter.build_parameters({
            "basis_set_mapping": {"B": "DZVP", "N": "DZVP"},
            "ri_basis_set_mapping": {"B": "RI_DZVP", "N": "RI_DZVP"},
        })
        params_dict = params.get_dict()
        subsys = params_dict["FORCE_EVAL"]["SUBSYS"]
        assert "KIND B" in subsys
        assert "KIND N" in subsys
        assert subsys["KIND B"]["BASIS_SET"] == "DZVP"
        assert subsys["KIND B"]["BASIS_SET RI_AUX"] == "RI_DZVP"
        assert subsys["KIND N"]["BASIS_SET RI_AUX"] == "RI_DZVP"

    def test_build_parameters_without_ri_basis(self, adapter):
        params = adapter.build_parameters({
            "basis_set_mapping": {"B": "DZVP"},
        })
        params_dict = params.get_dict()
        subsys = params_dict["FORCE_EVAL"]["SUBSYS"]
        assert "KIND B" in subsys
        assert "BASIS_SET" in subsys["KIND B"]
        assert "BASIS_SET RI_AUX" not in subsys["KIND B"]

    def test_build_parameters_with_ri_basis_set_file(self, adapter):
        params = adapter.build_parameters({
            "basis_set_file": "BASIS_MOLOPT",
            "ri_basis_set_file": "RI_BASIS",
        })
        params_dict = params.get_dict()
        dft = params_dict["FORCE_EVAL"]["DFT"]
        assert dft["BASIS_SET_FILE_NAME"] == ["BASIS_MOLOPT", "RI_BASIS"]

    def test_build_parameters_with_single_basis_set_file(self, adapter):
        params = adapter.build_parameters({
            "basis_set_file": "BASIS_MOLOPT",
        })
        params_dict = params.get_dict()
        dft = params_dict["FORCE_EVAL"]["DFT"]
        assert dft["BASIS_SET_FILE_NAME"] == "BASIS_MOLOPT"
