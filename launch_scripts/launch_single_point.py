from aiida import orm, load_profile
from aiida.engine import submit
from aiida.plugins import DataFactory

from aiida_relax_project.workflows.single_point import VaspSinglePointWorkChain

StructureData = DataFactory("core.structure")


def main():
    load_profile()
    code = orm.load_code("vasp@localhost")  # change this to your configured code label

    structure = StructureData()
    structure.set_cell(
        [
            [5.43, 0.00, 0.00],
            [0.00, 5.43, 0.00],
            [0.00, 0.00, 5.43],
        ]
    )
    structure.append_atom(position=(0.00, 0.00, 0.00), symbols="Si")
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si")
    structure.store()

    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([4, 4, 4])

    parameters = orm.Dict(
        dict={
            "incar": {
                "ENCUT": 300,
                "PREC": "Normal",
                "EDIFF": 1e-5,
                "ISMEAR": 0,
                "SIGMA": 0.05,
            }
        }
    )

    inputs = {
        "code": code,
        "structure": structure,
        "parameters": parameters,
        "kpoints": kpoints,
        "potential_family": orm.Str("PBE.54"),
        "potential_mapping": orm.Dict(dict={"Si": "Si"}),
        "metadata_options": orm.Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 8,
                },
                "max_wallclock_seconds": 1800,
                "withmpi": True,
            }
        ),
    }

    node = submit(VaspSinglePointWorkChain, **inputs)
    print(f"Submitted VaspSinglePointWorkChain<{node.pk}>")


if __name__ == "__main__":
    main()
