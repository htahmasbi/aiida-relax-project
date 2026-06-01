from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Requires aiida to be installed")


class TestProjectImport:
    def test_import_workflows(self):
        from aiida_relax_project.workflows import (
            MyRelaxLearningWorkChain,
            VaspSinglePointWorkChain,
            VaspVolumeScanWorkChain,
        )

        assert MyRelaxLearningWorkChain is not None
        assert VaspSinglePointWorkChain is not None
        assert VaspVolumeScanWorkChain is not None

    def test_import_transformations(self):
        from aiida_relax_project.transformations import (
            make_supercell_3x3,
            rotate_xy_to_xz,
        )

        assert make_supercell_3x3 is not None
        assert rotate_xy_to_xz is not None

    def test_import_datasets(self):
        from aiida_relax_project.datasets import fetch_mc2d_structures

        assert fetch_mc2d_structures is not None

    def test_import_utils(self):
        from utils.engines import (
            build_cp2k_inputs,
            build_vasp_inputs,
            get_engine_launcher,
        )

        assert get_engine_launcher is not None
        assert build_vasp_inputs is not None
        assert build_cp2k_inputs is not None
