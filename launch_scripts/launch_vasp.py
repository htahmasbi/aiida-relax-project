from ase.build import bulk
from aiida import orm, engine
from workflows.vasp_simple import MyVaspWorkChain

def launch():
    # 1. Setup Structure (via ASE)
    atoms = bulk('Si', 'diamond', a=5.43)
    structure = orm.StructureData(ase=atoms)

    # 2. Setup VASP Inputs (Your INCAR files)
    parameters = orm.Dict(dict={
        'ENCUT': 400,
        'ISMEAR': 0,
        'SIGMA': 0.05,
        'EDIFF': 1e-6,
    })

    # 3. Setup K-Points
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([4, 4, 4])

    # 4. Define your Code and Potentials
    # Replace these strings with your actual labels from 'verdi code list'
    code = orm.load_code('vasp@my_cluster')
    pot_family = orm.Str('PBE_Family') 
    pot_mapping = orm.Dict(dict={'Si': 'Si'})

    # 5. Build input dictionary for the WorkChain
    builder = MyVaspWorkChain.get_builder()
    builder.structure = structure
    builder.code = code
    builder.parameters = parameters
    builder.kpoints = kpoints
    builder.potential_family = pot_family
    builder.potential_mapping = pot_mapping

    # 6. Submit the WorkChain to the AiiDA Daemon
    node = engine.submit(builder)
    print(f"Submitted WorkChain with PK: {node.pk}")

if __name__ == '__main__':
    launch()
