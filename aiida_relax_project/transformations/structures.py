from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure


def center_slab_in_cell(structure: Structure) -> Structure:
    """Center the 2D slab in the cell along the vacuum direction (c-axis).

    Uses circular statistics to handle atoms that straddle the periodic
    boundary (fractional z near 0 or 1).
    """
    structure = structure.copy()
    z_frac = np.array(structure.frac_coords)[:, 2]
    mean_angle = np.arctan2(
        np.mean(np.sin(2 * np.pi * z_frac)),
        np.mean(np.cos(2 * np.pi * z_frac)),
    )
    mean_z = mean_angle / (2 * np.pi)
    shift_z = 0.5 - mean_z
    structure.translate_sites(
        list(range(len(structure))), [0, 0, shift_z],
    )
    return structure


def rotate_xy_to_xz(structure: Structure, vacuum: float = 20.0) -> Structure:
    """
    Rotate a 2D structure from the xy plane to the xz plane.

    Mapping:
        x_new = x_old
        y_new = z_old
        z_new = y_old

    The new vacuum direction is y.  Atoms are automatically centered
    in the new vacuum gap.
    """
    old_lattice = structure.lattice.matrix

    a_old = old_lattice[0]
    b_old = old_lattice[1]

    a_new = [a_old[0], 0.0, a_old[1]]
    b_new = [0.0, vacuum, 0.0]
    c_new = [b_old[0], 0.0, b_old[1]]

    new_lattice = Lattice([a_new, b_new, c_new])

    species = []
    coords = []

    for site in structure:
        x, y, z = site.coords
        species.append(site.species)
        coords.append([x, z, y])

    result = Structure(
        lattice=new_lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=True,
    )

    # Center slab in the new vacuum along y
    y_frac = np.array(result.frac_coords)[:, 1]
    mean_angle = np.arctan2(
        np.mean(np.sin(2 * np.pi * y_frac)),
        np.mean(np.cos(2 * np.pi * y_frac)),
    )
    mean_y = mean_angle / (2 * np.pi)
    shift_y = 0.5 - mean_y
    result.translate_sites(
        list(range(len(result))), [0, shift_y, 0],
    )
    return result


def make_supercell_3x3(structure: Structure) -> Structure:
    """Return a 3x3x1 supercell."""
    structure = structure.copy()
    structure.make_supercell([3, 3, 1])
    return structure
