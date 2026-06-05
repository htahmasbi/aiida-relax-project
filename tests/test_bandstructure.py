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

    def test_rectangular_path_detected(self):
        """Rectangular in-plane lattice gives Γ→X→S→Y→Γ."""
        # Orthorhombic cell: a=6, b=4, all angles 90°
        lattice = Lattice.orthorhombic(6.0, 20.0, 4.0)
        structure = Structure(lattice, ["H"], [[0, 0, 0]], coords_are_cartesian=False)
        path = get_bandstructure_path(structure)

        labels = [sp.split()[0] for sp in path]
        assert labels == ["GAMMA", "X", "S", "Y", "GAMMA"]

        # Coordinate swap: (k_a1, k_a2, 0) → (k_a1, 0, k_a2)
        # X = (0.5, 0, 0) → (0.5, 0, 0)
        x_parts = [sp for sp in path if sp.startswith("X")][0].split()
        assert float(x_parts[1]) == 0.5  # x
        assert float(x_parts[2]) == 0.0  # y (vacuum)
        assert float(x_parts[3]) == 0.0  # z

        # S = (0.5, 0.5, 0) → (0.5, 0, 0.5)
        s_parts = [sp for sp in path if sp.startswith("S")][0].split()
        assert float(s_parts[1]) == 0.5
        assert float(s_parts[2]) == 0.0
        assert float(s_parts[3]) == 0.5

    def test_square_path_detected(self):
        """Square in-plane lattice gives Γ→X→M→Γ."""
        # a=b=5 in-plane, vacuum=20 in 3rd axis
        lattice = Lattice.orthorhombic(5.0, 5.0, 20.0)
        structure = Structure(lattice, ["H"], [[0, 0, 0]], coords_are_cartesian=False)
        path = get_bandstructure_path(structure)

        labels = [sp.split()[0] for sp in path]
        assert labels == ["GAMMA", "X", "M", "GAMMA"]

    def test_inse_rectangular_path(self):
        """In₂Se₃-like rectangular cell gives Γ→X→S→Y→Γ instead of
        pymatgen's monoclinic 3D path."""
        # Original (pre-rotation) InSe: a=7.941, b=5.731, vacuum=20 in z.
        # Atomic positions do not matter — only the in-plane lattice vectors
        # are used for 2D Bravais lattice detection.
        lattice = Lattice.orthorhombic(7.94104, 5.73095, 20.0)
        structure = Structure(lattice, ["H"], [[0, 0, 0]], coords_are_cartesian=False)
        path = get_bandstructure_path(structure)
        labels = [sp.split()[0] for sp in path]
        assert labels == ["GAMMA", "X", "S", "Y", "GAMMA"]

        s_parts = [sp for sp in path if sp.startswith("S")][0].split()
        assert float(s_parts[1]) == 0.5
        assert float(s_parts[2]) == 0.0
        assert float(s_parts[3]) == 0.5
