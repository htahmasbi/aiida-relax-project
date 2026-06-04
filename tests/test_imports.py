from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Requires aiida to be installed")


class TestProjectImport:
    def test_import_workflows(self):
        from aiida_relax_project.workflows import (
            DynamicRelaxWorkChain,
            DynamicSinglePointWorkChain,
            DynamicVolumeScanWorkChain,
        )

        assert DynamicSinglePointWorkChain is not None
        assert DynamicRelaxWorkChain is not None
        assert DynamicVolumeScanWorkChain is not None

    def test_import_transformations(self):
        from aiida_relax_project.transformations import (
            make_supercell_3x3,
            rotate_xy_to_xz,
        )

        assert make_supercell_3x3 is not None
        assert rotate_xy_to_xz is not None

    def test_import_datasets(self):
        from aiida_relax_project.datasets import (
            fetch_mc2d_structures,
            optimade_entry_to_pymatgen,
        )

        assert fetch_mc2d_structures is not None
        assert optimade_entry_to_pymatgen is not None
