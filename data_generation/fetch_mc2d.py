from pathlib import Path

from aiida import load_profile, orm
from aiida.plugins import DataFactory

from aiida_relax_project.datasets.mc2d_optimade import fetch_mc2d_structures
from aiida_relax_project.transformations.structures import (
    make_supercell_3x3,
    rotate_xy_to_xz,
)


StructureData = DataFactory("core.structure")


def modifier(structure):
    structure = rotate_xy_to_xz(structure, vacuum=20.0)
    structure = make_supercell_3x3(structure)
    return structure


def main():
    load_profile()

    group, _ = orm.Group.collection.get_or_create(label="mc2d_bn_3x3_xz")

    data = fetch_mc2d_structures(
        optimade_filter='elements HAS ALL "B","N" AND nelements=2',
        max_structures=10,
        modifier=modifier,
    )

    output_dir = Path("mc2d_structures")
    output_dir.mkdir(exist_ok=True)

    for item in data:
        structure = item["structure"]

        # Save CIF for checking
        cif_path = output_dir / f"{item['id']}_{item['formula']}_xz_3x3.cif"
        structure.to(filename=str(cif_path))

        # Store in AiiDA
        aiida_structure = StructureData(pymatgen=structure)
        aiida_structure.label = f"MC2D {item['formula']} {item['id']} xz 3x3"
        aiida_structure.description = "Downloaded from MC2D OPTIMADE and modified with pymatgen."
        aiida_structure.store()

        group.add_nodes(aiida_structure)

        print(f"Stored {aiida_structure.label} as StructureData<{aiida_structure.pk}>")
        print(f"Wrote {cif_path}")


if __name__ == "__main__":
    main()
