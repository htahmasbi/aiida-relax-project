"""Launch GW + bandstructure calculations for MC2D BN structures with CP2K.

Usage:
    python launch_mc2d_gw.py
    python launch_mc2d_gw.py --max-structures 10
    python launch_mc2d_gw.py --group my_gw_group --scf-guess RESTART

The GW calculation uses a single CP2K run that performs both the SCF and the
GW correction / bandstructure in one job.  For a restart from a previous
wavefunction, pass ``--scf-guess RESTART``; the first run defaults to ATOMIC.

Configuration is read from ``config.toml`` (section ``[gw]``).  All paths
(basis set files, potential file) must be accessible on the cluster.
"""

from __future__ import annotations

import argparse

from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import CalculationFactory, DataFactory
from aiida.orm import Dict, KpointsData

from aiida_relax_project.core.config import get_config
from aiida_relax_project.datasets.mc2d_optimade import fetch_mc2d_structures
from aiida_relax_project.transformations.structures import (
    rotate_xy_to_xz,
    make_supercell_3x3,
)

Cp2kCalculation = CalculationFactory("cp2k")
StructureData = DataFactory("core.structure")


def modifier(structure):
    """Rotate 2D structure to xz-plane and build a 3×3 supercell."""
    structure = rotate_xy_to_xz(structure, vacuum=20.0)
    structure = make_supercell_3x3(structure)
    return structure


def make_gw_parameters(gw, scf_guess="ATOMIC"):
    """Build the full CP2K input dict for a GW + bandstructure run."""
    mesh_str = " ".join(str(k) for k in gw.kpoints_mesh)

    return Dict({
        "GLOBAL": {
            "RUN_TYPE": "ENERGY",
            "PRINT_LEVEL": "MEDIUM",
        },
        "FORCE_EVAL": {
            "METHOD": "Quickstep",
            "DFT": {
                "BASIS_SET_FILE_NAME": [
                    gw.basis_set_file,
                    gw.ri_basis_set_file,
                ],
                "POTENTIAL_FILE_NAME": gw.potential_file,
                "SORT_BASIS": "EXP",
                "CHARGE": 0,
                "MULTIPLICITY": 1,
                "MGRID": {
                    "CUTOFF": gw.cutoff,
                    "REL_CUTOFF": gw.rel_cutoff,
                },
                "QS": {
                    "METHOD": "GPW",
                    "EPS_DEFAULT": gw.eps_default,
                    "EPS_PGF_ORB": gw.eps_pgf_orb,
                },
                "SCF": {
                    "SCF_GUESS": scf_guess,
                    "EPS_SCF": gw.eps_scf,
                    "MAX_SCF": gw.max_scf,
                    "MIXING": {
                        "METHOD": "BROYDEN_MIXING",
                        "ALPHA": gw.mixing_alpha,
                        "BETA": gw.mixing_beta,
                        "NBROYDEN": gw.mixing_nbroyden,
                    },
                    "PRINT": {
                        "RESTART": {
                            "_": "ON",
                        },
                    },
                },
                "XC": {
                    "XC_FUNCTIONAL": {"_": "PBE"},
                },
                "KPOINTS": {
                    "SCHEME": f"MONKHORST-PACK {mesh_str}",
                    "PARALLEL_GROUP_SIZE": -1,
                },
                "POISSON": {
                    "PERIODIC": gw.periodic,
                    "POISSON_SOLVER": gw.poisson_solver,
                },
            },
            "PROPERTIES": {
                "BANDSTRUCTURE": {
                    "DOS": {},
                    "GW": {
                        "NUM_TIME_FREQ_POINTS": gw.num_time_freq,
                        "MEMORY_PER_PROC": gw.memory_per_proc,
                        "EPS_FILTER": gw.eps_filter,
                        "CUTOFF_RADIUS_RI": gw.cutoff_radius_ri,
                        "REGULARIZATION_RI": gw.regularization_ri,
                        "KPOINTS_W": list(gw.kpoints_w),
                    },
                    "BANDSTRUCTURE_PATH": {
                        "NPOINTS": gw.bs_npoints,
                        "UNITS": "B_VECTOR",
                        "SPECIAL_POINT": list(gw.special_points),
                    },
                },
            },
            "SUBSYS": {
                "CELL": {
                    "PERIODIC": gw.periodic,
                },
                "KIND": [
                    {
                        "_": "B",
                        "BASIS_SET ORB": gw.orb_basis,
                        "BASIS_SET RI_AUX": gw.ri_basis_B,
                        "POTENTIAL": gw.potential_B,
                    },
                    {
                        "_": "N",
                        "BASIS_SET ORB": gw.orb_basis,
                        "BASIS_SET RI_AUX": gw.ri_basis_N,
                        "POTENTIAL": gw.potential_N,
                    },
                ],
            },
        },
    })


def main():
    parser = argparse.ArgumentParser(
        description="Launch CP2K GW + bandstructure for MC2D BN structures."
    )
    parser.add_argument(
        "--max-structures", type=int, default=5,
        help="Maximum number of structures to process (default: 5)",
    )
    parser.add_argument(
        "--group", type=str, default="mc2d_bn_gw",
        help="AiiDA group label for the structures (default: mc2d_bn_gw)",
    )
    parser.add_argument(
        "--scf-guess", type=str, default="ATOMIC",
        choices=["ATOMIC", "RESTART"],
        help="SCF initial guess (default: ATOMIC; use RESTART to reuse a prior wavefunction)",
    )
    parser.add_argument(
        "--code-label", type=str,
        help="Computer@code label (default: from config.toml)",
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="Show GW configuration and exit",
    )
    args = parser.parse_args()

    config = get_config()

    if args.show_config:
        print("GW configuration:")
        for field, value in config.gw.model_dump().items():
            print(f"  {field}: {value}")
        return

    load_profile()

    code_label = args.code_label or config.code_label
    code = orm.load_code(f"cp2k@{code_label}")

    group, _ = orm.Group.collection.get_or_create(args.group)

    data = fetch_mc2d_structures(
        optimade_filter='elements HAS ALL "B","N" AND nelements=2',
        max_structures=args.max_structures,
        modifier=modifier,
    )

    gw = config.gw
    parameters = make_gw_parameters(gw, scf_guess=args.scf_guess)

    for item in data:
        pymatgen_structure = item["structure"]

        structure = StructureData(pymatgen=pymatgen_structure)
        structure.label = f"MC2D GW {item['formula']} {item['id']} xz 3x3"
        structure.description = "MC2D structure for CP2K GW + bandstructure."
        structure.store()

        group.add_nodes(structure)

        builder = Cp2kCalculation.get_builder()
        builder.code = code
        builder.structure = structure
        builder.parameters = parameters

        builder.metadata.label = f"GW {item['formula']} {item['id']}"
        builder.metadata.description = (
            "CP2K GW + bandstructure for MC2D test set."
        )

        builder.metadata.options.resources = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 16,
        }
        builder.metadata.options.max_wallclock_seconds = 86400
        builder.metadata.options.withmpi = True

        node = submit(builder)

        print(
            f"Submitted {item['id']} {item['formula']} "
            f"as CalcJob<{node.pk}> for StructureData<{structure.pk}>"
        )


if __name__ == "__main__":
    main()
