from ase.build import bulk
from aiida import orm, load_profile


load_profile()


def create_silicon_set(group_label="si_test_set"):
    """Create a small group of Si structures with different lattice constants."""

    group, _ = orm.Group.collection.get_or_create(label=group_label)

    lattice_constants = [5.3, 5.4, 5.5]

    for a in lattice_constants:
        atoms = bulk("Si", "diamond", a=a)

        structure = orm.StructureData(ase=atoms)
        structure.label = f"Si_a_{a}"
        structure.store()

        group.add_nodes(structure)

        print(f"Stored structure with a={a} in group '{group_label}'")


if __name__ == "__main__":
    create_silicon_set()    
