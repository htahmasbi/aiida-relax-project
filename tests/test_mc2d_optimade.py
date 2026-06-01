from __future__ import annotations

import pytest

from aiida_relax_project.datasets.mc2d_optimade import optimade_entry_to_pymatgen


class TestOptimadeConversion:
    def test_convert_entry(self):
        entry = {
            "attributes": {
                "lattice_vectors": [
                    [2.46, 0.0, 0.0],
                    [0.0, 2.46, 0.0],
                    [0.0, 0.0, 10.0],
                ],
                "species_at_sites": ["C", "C"],
                "cartesian_site_positions": [[0.0, 0.0, 0.0], [1.23, 1.23, 5.0]],
            }
        }

        structure = optimade_entry_to_pymatgen(entry)

        assert len(structure) == 2
        assert structure.lattice.a == pytest.approx(2.46)

    def test_lattice_vectors(self):
        entry = {
            "attributes": {
                "lattice_vectors": [
                    [3.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 5.0],
                ],
                "species_at_sites": ["Si"],
                "cartesian_site_positions": [[0.5, 0.5, 0.5]],
            }
        }

        structure = optimade_entry_to_pymatgen(entry)

        assert structure.lattice.a == pytest.approx(3.0)
        assert structure.lattice.b == pytest.approx(4.0)
        assert structure.lattice.c == pytest.approx(5.0)
