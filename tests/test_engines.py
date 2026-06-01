from __future__ import annotations

import pytest

try:
    from utils.engines import build_cp2k_inputs, build_vasp_inputs, get_engine_launcher

    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

pytestmark = pytest.mark.skipif(not HAS_UTILS, reason="aiida not installed")


class TestGetEngineLauncher:
    def test_vasp_engine(self):
        result = get_engine_launcher("vasp")
        assert result["workchain"].__name__ == "VaspWorkChain"
        assert result["builder_func"] == build_vasp_inputs

    def test_cp2k_engine(self):
        result = get_engine_launcher("cp2k")
        assert result["workchain"].__name__ == "Cp2kWorkChain"
        assert result["builder_func"] == build_cp2k_inputs

    def test_engine_case_insensitive(self):
        result = get_engine_launcher("VASP")
        assert result["workchain"].__name__ == "VaspWorkChain"

    def test_unknown_engine(self):
        with pytest.raises(ValueError, match="Unknown electronic structure engine"):
            get_engine_launcher("unknown")


class TestBuildVaspInputs:
    def test_default_parameters(self):
        result = build_vasp_inputs(None, {})
        incar = result["parameters"].get_dict()["incar"]
        assert incar["NSW"] == 50
        assert incar["EDIFF"] == 1e-5
        assert incar["ENCUT"] == 400

    def test_custom_parameters(self):
        result = build_vasp_inputs(
            None, {"max_steps": 100, "energy_tolerance": 1e-4, "encut": 500}
        )
        incar = result["parameters"].get_dict()["incar"]
        assert incar["NSW"] == 100
        assert incar["EDIFF"] == 1e-4
        assert incar["ENCUT"] == 500


class TestBuildCp2kInputs:
    def test_default_parameters(self):
        result = build_cp2k_inputs(None, {})
        params = result["parameters"].get_dict()
        assert params["MOTION"]["GEO_OPT"]["MAX_ITER"] == 50
        assert params["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] == 400

    def test_custom_parameters(self):
        result = build_cp2k_inputs(None, {"max_steps": 200, "cutoff": 600})
        params = result["parameters"].get_dict()
        assert params["MOTION"]["GEO_OPT"]["MAX_ITER"] == 200
        assert params["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] == 600
