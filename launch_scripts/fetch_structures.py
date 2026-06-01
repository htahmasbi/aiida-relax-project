"""Structure fetcher for OPTIMADE and AiiDA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from aiida import load_profile, orm
from aiida.plugins import DataFactory
from pymatgen.core import Structure

from aiida_relax_project.config import (
    CalculatorConfig,
    TransformationsConfig,
)
from aiida_relax_project.datasets.mc2d_optimade import fetch_mc2d_structures
from aiida_relax_project.transformations.structures import (
    make_supercell_3x3,
    rotate_xy_to_xz,
)


StructureData = DataFactory("core.structure")


def apply_transformations(
    structure: Structure, config: TransformationsConfig
) -> Structure:
    """Apply configured transformations to a structure."""
    if config.rotate_xy_to_xz:
        structure = rotate_xy_to_xz(structure, vacuum=config.vacuum)

    if config.make_supercell is not None:
        structure = structure.copy()
        structure.make_supercell(config.make_supercell)

    return structure


def fetch_from_optimade(
    config: CalculatorConfig, modifier: Optional[Callable] = None
) -> list[dict[str, Any]]:
    """
    Fetch structures from OPTIMADE API.

    Args:
        config: Calculator configuration.
        modifier: Optional transformation function.

    Returns:
        List of structure dictionaries.
    """
    return fetch_mc2d_structures(
        optimade_filter=config.optimade.filter,
        page_limit=config.optimade.page_limit,
        max_structures=config.optimade.max_structures,
        modifier=modifier,
    )


def fetch_from_group(group_label: str) -> list[StructureData]:
    """
    Fetch structures from AiiDA group.

    Args:
        group_label: Label of the AiiDA group.

    Returns:
        List of StructureData nodes.
    """
    group = orm.Group.collection.get(label=group_label)[0]
    return [node for node in group.nodes if isinstance(node, orm.StructureData)]


def create_inline_structure(config: CalculatorConfig) -> list[StructureData]:
    """
    Create StructureData from inline configuration.

    Args:
        config: Calculator configuration with inline structure.

    Returns:
        List containing single StructureData.
    """
    if not config.structure:
        return []

    struct_config = config.structure

    lattice = struct_config.lattice
    species = []
    coords = []

    for atom in struct_config.atoms:
        x, y, z = atom[0], atom[1], atom[2]
        element = atom[3]
        species.append(element)
        coords.append([x, y, z])

    from pymatgen.core import Lattice as PGLattice

    pymatgen_structure = Structure(
        lattice=PGLattice(lattice),
        species=species,
        coords=coords,
        coords_are_cartesian=not struct_config.coords_are_fractional,
    )

    aiida_structure = StructureData(pymatgen=pymatgen_structure)
    aiida_structure.label = "inline_structure"
    aiida_structure.store()

    return [aiida_structure]


def fetch_structures(config: CalculatorConfig) -> list[StructureData]:
    """
    Fetch structures based on configuration source.

    Args:
        config: Calculator configuration.

    Returns:
        List of StructureData nodes.
    """
    load_profile()

    structures: list[StructureData] = []

    if config.optimade.filter or config.optimade.max_structures:

        def transformation_modifier(structure: Structure) -> Structure:
            return apply_transformations(structure, config.transformations)

        data = fetch_from_optimade(config, modifier=transformation_modifier)

        for item in data:
            pymatgen_structure = item["structure"]

            aiida_structure = StructureData(pymatgen=pymatgen_structure)
            aiida_structure.label = f"{item['formula']} {item['id']}"
            aiida_structure.store()

            if config.output.group_label:
                group, _ = orm.Group.collection.get_or_create(
                    label=config.output.group_label
                )
                group.add_nodes(aiida_structure)

            structures.append(aiida_structure)

            if config.output.save_cif:
                output_dir = Path("structures")
                output_dir.mkdir(exist_ok=True)
                cif_path = output_dir / f"{item['id']}_{item['formula']}.cif"
                pymatgen_structure.to(filename=str(cif_path))

    elif config.structure_group:
        structures = fetch_from_group(config.structure_group)

        for structure in structures:
            transformed = structure.get_pymatgen_structure()
            transformed = apply_transformations(transformed, config.transformations)
            new_structure = StructureData(pymatgen=transformed)
            new_structure.label = f"{structure.label}_transformed"
            new_structure.store()
            structures[structures.index(structure)] = new_structure

    elif config.structure:
        structures = create_inline_structure(config)

        for structure in structures:
            transformed = structure.get_pymatgen_structure()
            transformed = apply_transformations(transformed, config.transformations)
            new_structure = StructureData(pymatgen=transformed)
            new_structure.label = f"{structure.label}_transformed"
            new_structure.store()
            structures[structures.index(structure)] = new_structure

    return structures