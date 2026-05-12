from __future__ import annotations

from pymatgen.core import Lattice, Structure


def rotate_xy_to_xz(structure: Structure, vacuum: float = 20.0) -> Structure:
    """
    Rotate a 2D structure from the xy plane to the xz plane.

    Mapping:
        x_new = x_old
        y_new = z_old
        z_new = y_old

    The new vacuum direction is y.
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

    return Structure(
        lattice=new_lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=True,
    )


def make_supercell_3x3(structure: Structure) -> Structure:
    """Return a 3x3x1 supercell."""
    structure = structure.copy()
    structure.make_supercell([3, 3, 1])
    return structure
