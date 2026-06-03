from __future__ import annotations

from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import CalculationFactory, DataFactory
from aiida.orm import Dict, KpointsData
from pymatgen.core import Structure

from aiida_relax_project.core.config import get_config
from aiida_relax_project.datasets.mc2d_optimade import fetch_mc2d_structures
from aiida_relax_project.transformations.structures import (
    rotate_xy_to_xz,
    make_supercell_3x3,
)


Cp2kCalculation = CalculationFactory("cp2k")
StructureData = DataFactory("core.structure")


def modifier(structure: Structure) -> Structure:
    """Modify each MC2D pymatgen Structure before CP2K."""
    structure = rotate_xy_to_xz(structure, vacuum=20.0)
    structure = make_supercell_3x3(structure)
    return structure


def make_cp2k_parameters():
    """Minimal CP2K input dictionary for a PBE single-point test."""
    return Dict(
        {
            "GLOBAL": {
                "RUN_TYPE": "ENERGY",
                "PRINT_LEVEL": "MEDIUM",
            },
            "FORCE_EVAL": {
                "METHOD": "Quickstep",
                "DFT": {
                    "BASIS_SET_FILE_NAME": [
                        "/home/tahmas41/work/GW_2D/BASIS_AUG_MOLOPT/BASIS_GTH_MOLOPT_AUG_for_excited_states",
                        "/home/tahmas41/work/GW_2D/BASIS_AUG_MOLOPT/BASIS_GTH_MOLOPT_AUG_for_excited_states_RI",
                    ],
                    "POTENTIAL_FILE_NAME": "/home/tahmas41/work/GW_2D/cp2k/data/POTENTIAL_UZH",
                    "CHARGE": 0,
                    "MULTIPLICITY": 1,
                    "SCF": {
                        "SCF_GUESS": "ATOMIC",
                        "EPS_SCF": 1.0e-6,
                        "MAX_SCF": 200,
                        "DIAGONALIZATION": {
                            "ALGORITHM": "STANDARD",
                        },
                        "MIXING": {
                            "METHOD": "BROYDEN_MIXING",
                            "ALPHA": 0.2,
                            "BETA": 1.5,
                            "NBROYDEN": 8,
                        },
                    },
                    "XC": {
                        "XC_FUNCTIONAL": {
                            "_": "PBE",
                        },
                    },
                },
                "SUBSYS": {
                    "CELL": {
                        "PERIODIC": "XYZ",
                    },
                    "KIND": [
                        {
                            "_": "B",
                            "BASIS_SET ORB": "aug-SZV-MOLOPT-GTH-tier-1",
                            "BASIS_SET RI_AUX": "RI_aug-SZV-MOLOPT-GTH-tier-1_N_RI_009_s_p_d_f_g_h_i_3_2_0_0_0_0_0_error_1.1e-06",
                            "POTENTIAL": "GTH-PBE-q3",
                        },
                        {
                            "_": "N",
                            "BASIS_SET ORB": "aug-SZV-MOLOPT-GTH-tier-1",
                            "BASIS_SET RI_AUX": "RI_aug-SZV-MOLOPT-GTH-tier-1_N_RI_025_s_p_d_f_g_h_i_6_3_2_0_0_0_0_error_2.9e-06",
                            "POTENTIAL": "GTH-PBE-q5",
                        },
                    ],
                },
            },
        }
    )


def make_kpoints():
    """K-point mesh for the 3x3 rotated xz-plane supercell."""
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([4, 1, 4])
    return kpoints


def main():
    load_profile()

    config = get_config()
    code_label = f"cp2k@{config.code_label}"
    code = orm.load_code(code_label)

    group, _ = orm.Group.collection.get_or_create("mc2d_bn_cp2k_test_5")

    data = fetch_mc2d_structures(
        optimade_filter='elements HAS ALL "B","N" AND nelements=2',
        max_structures=5,
        modifier=modifier,
    )

    for item in data:
        pymatgen_structure = item["structure"]

        structure = StructureData(pymatgen=pymatgen_structure)
        structure.label = f"MC2D {item['formula']} {item['id']} xz 3x3"
        structure.description = "MC2D OPTIMADE structure modified with pymatgen."
        structure.store()

        group.add_nodes(structure)

        builder = Cp2kCalculation.get_builder()
        builder.code = code
        builder.structure = structure
        builder.parameters = make_cp2k_parameters()
        builder.kpoints = make_kpoints()

        builder.metadata.label = f"CP2K {item['formula']} {item['id']}"
        builder.metadata.description = (
            "CP2K single-point calculation for MC2D test set."
        )

        builder.metadata.options.resources = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 16,
        }
        builder.metadata.options.max_wallclock_seconds = 3600
        builder.metadata.options.withmpi = True

        node = submit(builder)

        print(
            f"Submitted {item['id']} {item['formula']} "
            f"as CalcJob<{node.pk}> for StructureData<{structure.pk}>"
        )


if __name__ == "__main__":
    main()
