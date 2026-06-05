from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from launch_scripts.launch_mc2d_gw import get_bandstructure_path


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
    def test_hexagonal_bn_inplane_path(self):
        """BN hexagonal path should map to rotated BZ coords.

        Original K at (1/3, 1/3, 0)  -> K at (1/3, 0, 1/3)  after swapping.
        Original M at (0.5, 0, 0)    -> M at (0.5, 0, 0)    unchanged.
        """
        structure = _hexagonal_bn()
        path = get_bandstructure_path(structure)

        labels = {sp.split()[0] for sp in path}
        assert "GAMMA" in labels
        assert "M" in labels
        assert "K" in labels

        # Check K has been mapped: (1/3, 1/3, 0) -> (1/3, 0, 1/3)
        k_lines = [sp for sp in path if sp.startswith("K")]
        assert len(k_lines) >= 1
        parts = k_lines[0].split()
        assert float(parts[1]) == pytest.approx(1 / 3, abs=1e-6)
        assert float(parts[2]) == pytest.approx(0.0, abs=1e-6)
        assert float(parts[3]) == pytest.approx(1 / 3, abs=1e-6)

    def test_override_returns_as_is(self):
        structure = _hexagonal_bn()
        override = ["GAMMA  0.0  0.0  0.0", "X  0.5  0.0  0.0"]
        result = get_bandstructure_path(structure, override)
        assert result == override

    def test_no_out_of_plane_segments(self):
        """Only in-plane (k_vac=0) segments should be included."""
        structure = _hexagonal_bn()
        path = get_bandstructure_path(structure)

        for sp in path:
            parts = sp.split()
            # y component (index 2 in split) should be 0 (vacuum direction)
            assert float(parts[2]) == pytest.approx(0.0, abs=1e-10), (
                f"Expected y=0 for in-plane path, got {sp}"
            )

    def test_path_starts_and_ends_at_gamma(self):
        """The band path should start and end at GAMMA."""
        structure = _hexagonal_bn()
        path = get_bandstructure_path(structure)

        assert path[0].startswith("GAMMA")
        assert path[-1].startswith("GAMMA")
