from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from launch_scripts.launch_mc2d_gw import (
    _classify_2d_inplane_lattice,
    get_bandstructure_path,
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
        lattice = Lattice.orthorhombic(7.94104, 5.73095, 20.0)
        structure = Structure(lattice, ["H"], [[0, 0, 0]], coords_are_cartesian=False)
        path = get_bandstructure_path(structure)
        labels = [sp.split()[0] for sp in path]
        assert labels == ["GAMMA", "X", "S", "Y", "GAMMA"]

        s_parts = [sp for sp in path if sp.startswith("S")][0].split()
        assert float(s_parts[1]) == 0.5
        assert float(s_parts[2]) == 0.0
        assert float(s_parts[3]) == 0.5


class TestClassify2DInplane:
    """2D Bravais lattice classification from in-plane vectors."""

    def test_hexagonal_120(self):
        s = Structure(Lattice.hexagonal(2.5, 10.0), ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "hexagonal"

    def test_hexagonal_60(self):
        s = Structure(Lattice.from_parameters(3.0, 3.0, 15.0, 90, 90, 60),
                       ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "hexagonal"

    def test_square(self):
        s = Structure(Lattice.orthorhombic(3.0, 3.0, 15.0), ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "square"

    def test_rectangular(self):
        s = Structure(Lattice.orthorhombic(3.0, 5.0, 15.0), ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "rectangular"

    def test_oblique(self):
        s = Structure(Lattice.from_parameters(3.0, 5.0, 15.0, 90, 90, 75),
                       ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "oblique"

    def test_monoclinic_sg3_like(self):
        """SG 3 (P2) 2D material — oblique in-plane."""
        s = Structure(Lattice.from_parameters(3.5, 5.2, 18.0, 90, 90, 105),
                       ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "oblique"

    def test_triclinic_like(self):
        s = Structure(Lattice.from_parameters(4.0, 6.0, 18.0, 90, 90, 70),
                       ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "oblique"

    def test_near_hexagonal(self):
        s = Structure(Lattice.from_parameters(3.0, 3.0, 15.0, 90, 90, 118),
                       ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "hexagonal"

    def test_near_rectangular(self):
        s = Structure(Lattice.from_parameters(3.0, 5.0, 15.0, 90, 90, 85),
                       ["H"], [[0, 0, 0]])
        assert _classify_2d_inplane_lattice(s) == "rectangular"


class TestBravaisEndToEnd:
    """End-to-end path generation for every 2D Bravais lattice type."""

    def _check(self, path):
        assert len(path) >= 3
        assert path[0].startswith("GAMMA")
        assert path[-1].startswith("GAMMA")
        for sp in path:
            p = sp.split()
            assert len(p) == 4
            assert float(p[2]) == pytest.approx(0.0, abs=1e-10)

    def test_hexagonal(self):
        self._check(get_bandstructure_path(
            Structure(Lattice.hexagonal(2.5, 10.0), ["H"], [[0, 0, 0]])))

    def test_square(self):
        self._check(get_bandstructure_path(
            Structure(Lattice.orthorhombic(3.0, 3.0, 15.0), ["H"], [[0, 0, 0]])))

    def test_rectangular(self):
        self._check(get_bandstructure_path(
            Structure(Lattice.orthorhombic(3.0, 5.0, 15.0), ["H"], [[0, 0, 0]])))

    def test_oblique(self):
        self._check(get_bandstructure_path(
            Structure(Lattice.from_parameters(3.0, 5.0, 15.0, 90, 90, 75),
                       ["H"], [[0, 0, 0]])))

    def test_monoclinic_sg3(self):
        """SG 3 (P2) 2D material produces a valid path."""
        self._check(get_bandstructure_path(
            Structure(Lattice.from_parameters(3.5, 5.2, 18.0, 90, 90, 105),
                       ["H"], [[0, 0, 0]])))
