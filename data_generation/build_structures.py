from ase.build import bulk
from aiida import orm, load_profile

load_profile() # Connect to your local AiiDA database

def create_silicon_set(group_label='si_test_set'):
    # 1. Create or get the group
    group, _ = orm.Group.get_or_create(label=group_label)
    
    # 2. Lattice constants to test (ASE part)
    lattice_constants = [5.3, 5.4, 5.5]
    
    for a in lattice_constants:
        atoms = bulk('Si', 'diamond', a=a)
        
        # 3. Convert ASE -> AiiDA
        structure = orm.StructureData(ase=atoms)
        structure.label = f"Si_a_{a}"
        structure.store()
        
        # 4. Add to group
        group.add_nodes(structure)
        print(f"Stored structure with a={a} in group '{group_label}'")

if __name__ == "__main__":
    create_silicon_set()
