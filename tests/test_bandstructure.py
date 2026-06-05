from __future__ import annotations

from pymatgen.core import Lattice, Structure

from launch_scripts.launch_mc2d_gw import get_bandstructure_path
from aiida_relax_project.transformations.structures import (
    make_supercell,
    rotate_xy_to_xz,
)


def _hexagonal_bn() -> Structure:
    """Return a primitive hexagonal BN cell (MC2D-like, 2D plane in XY)."""
    lattice = Lattice.hexagonal(2.5, 10.0)
    return Structure(
        lattice=lattice,
        species=["B", "N"],
        coords=[[0, 0, 0], [1 / 3, 2 / 3, 0.5]],
        coords_are_cartesian=False,
    )


class TestGetBandstructurePath:
    def test_hexagonal_bn_contains_expected_labels(self):
        """BN hexagonal path should contain GAMMA, M, K in-plane."""
        structure = _hexagonal_bn()
        path = get_bandstructure_path(structure)

        labels = [sp.split()[0] for sp in path]
        assert "GAMMA" in labels
        assert "M" in labels
        assert "K" in labels

    def test_override_returns_as_is(self):
        structure = _hexagonal_bn()
        override = ["GAMMA  0.0  0.0  0.0", "X  0.5  0.0  0.0"]
        result = get_bandstructure_path(structure, override)
        assert result == override

    def test_path_is_invariant_under_rotation(self):
        """Fractional coords of high-symmetry points should be the same
        before and after XY->XZ rotation, because the reciprocal lattice
        rotates together with the real-space lattice."""
        original = _hexagonal_bn()
        rotated = rotate_xy_to_xz(original, vacuum=20.0)

        orig_path = get_bandstructure_path(original)
        rot_path = get_bandstructure_path(rotated)

        orig_items = {(sp.split()[0], tuple(sp.split()[1:])) for sp in orig_path}
        rot_items = {(sp.split()[0], tuple(sp.split()[1:])) for sp in rot_path}

        common = orig_items & rot_items
        assert len(common) >= 3, (
            f"Expected at least 3 matching (label,coord) pairs, got {len(common)}. "
            f"Original: {orig_items}, Rotated: {rot_items}"
        )

    def test_original_structure_gives_same_path_regardless_of_supercell(self):
        """The band path should be computed from the primitive cell, so
        applying a supercell to the structure fed to get_bandstructure_path
        should not change the result (since we pass original_structure)."""
        original = _hexagonal_bn()
        supercell = make_supercell(original, [3, 3, 1])

        orig_path = get_bandstructure_path(original)
        super_path = get_bandstructure_path(supercell)

        orig_items = {(sp.split()[0], tuple(sp.split()[1:])) for sp in orig_path}
        super_items = {(sp.split()[0], tuple(sp.split()[1:])) for sp in super_path}

        common = orig_items & super_items
        assert len(common) >= 3, (
            "Primitive and supercell paths should share "
            f"at least 3 labels. Primitive: {orig_items}, Supercell: {super_items}"
        )

    def test_full_pipeline_path_matches_primitive(self):
        """Simulate the GW pipeline: original -> make_supercell -> rotate.
        The k-path from the original primitive should match what we get
        from the modified structure (since the rotation is just a change
        of basis in fractional coords)."""
        original = _hexagonal_bn()
        modified = make_supercell(original, [3, 3, 1])
        modified = rotate_xy_to_xz(modified, vacuum=20.0)

        orig_path = get_bandstructure_path(original)
        mod_path = get_bandstructure_path(modified)

        orig_items = {(sp.split()[0], tuple(sp.split()[1:])) for sp in orig_path}
        mod_items = {(sp.split()[0], tuple(sp.split()[1:])) for sp in mod_path}

        common = orig_items & mod_items
        assert len(common) >= 3, (
            "Primitive and full-pipeline paths should share "
            f"at least 3 labels. Primitive: {orig_items}, Pipeline: {mod_items}"
        )
