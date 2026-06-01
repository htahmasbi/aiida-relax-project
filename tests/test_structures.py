from __future__ import annotations

from pymatgen.core import Lattice, Structure

from aiida_relax_project.transformations.structures import (
    make_supercell_3x3,
    rotate_xy_to_xz,
)


class TestRotateXyToXz:
    def test_rotate_preserves_atoms(self):
        lattice = Lattice([[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 3.0]])
        structure = Structure(
            lattice=lattice,
            species=["Si", "Ge"],
            coords=[[1.0, 2.0, 1.5], [0.5, 1.0, 0.75]],
            coords_are_cartesian=True,
        )

        rotated = rotate_xy_to_xz(structure, vacuum=15.0)

        assert len(rotated) == 2
        assert [s.symbol for s in rotated.species] == ["Si", "Ge"]

    def test_vacuum_direction_is_y(self):
        lattice = Lattice([[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 3.0]])
        structure = Structure(
            lattice=lattice,
            species=["C"],
            coords=[[1.0, 5.0, 1.5]],
            coords_are_cartesian=True,
        )

        rotated = rotate_xy_to_xz(structure, vacuum=20.0)

        b_vector = rotated.lattice.matrix[1]
        assert abs(b_vector[0]) < 1e-10
        assert abs(b_vector[1] - 20.0) < 1e-10
        assert abs(b_vector[2]) < 1e-10

    def test_custom_vacuum(self):
        lattice = Lattice([[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 3.0]])
        structure = Structure(
            lattice=lattice,
            species=["C"],
            coords=[[0.0, 0.0, 0.0]],
            coords_are_cartesian=True,
        )

        rotated = rotate_xy_to_xz(structure, vacuum=30.0)

        assert abs(rotated.lattice.matrix[1][1] - 30.0) < 1e-10


class TestMakeSupercell3x3:
    def test_supercell_size(self):
        lattice = Lattice([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        structure = Structure(
            lattice=lattice,
            species=["H"],
            coords=[[0.0, 0.0, 0.0]],
            coords_are_cartesian=True,
        )

        supercell = make_supercell_3x3(structure)

        assert len(supercell) == 9

    def test_original_unchanged(self):
        lattice = Lattice([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        structure = Structure(
            lattice=lattice,
            species=["H"],
            coords=[[0.0, 0.0, 0.0]],
            coords_are_cartesian=True,
        )
        original_len = len(structure)

        make_supercell_3x3(structure)

        assert len(structure) == original_len
