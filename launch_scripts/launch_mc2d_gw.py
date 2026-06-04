"""Launch GW + bandstructure calculations for MC2D structures with CP2K.

Usage:
    python launch_mc2d_gw.py
    python launch_mc2d_gw.py --max-structures 10
    python launch_mc2d_gw.py --filter 'elements HAS ALL "B","N" AND nelements=2'
    python launch_mc2d_gw.py --group my_gw_group --scf-guess RESTART
    python launch_mc2d_gw.py --filter 'chemical_formula_reduced="BN"'

The GW calculation uses a single CP2K run that performs both the SCF and the
GW correction / bandstructure in one job.  For a restart from a previous
wavefunction, pass ``--scf-guess RESTART``; the first run defaults to ATOMIC.

Configuration is read from ``config.toml`` (section ``[gw]``).  All paths
(basis set files, potential file) must be accessible on the cluster.
"""

from __future__ import annotations

import argparse

from aiida_relax_project.core.config import get_config


def make_modifier(vacuum: float = 20.0):
    """Return a modifier that builds a 3×3 supercell then rotates to xz-plane."""
    from aiida_relax_project.transformations.structures import (
        center_slab_in_cell,
        make_supercell_3x3,
        rotate_xy_to_xz,
    )
    def modifier(structure):
        structure = center_slab_in_cell(structure)
        structure = make_supercell_3x3(structure)
        structure = rotate_xy_to_xz(structure, vacuum=vacuum)
        return structure
    return modifier


def make_gw_parameters(gw, structure, scf_guess="ATOMIC"):
    """Build the full CP2K input dict for a GW + bandstructure run.

    KIND sections are generated dynamically from the unique elements
    present in *structure* (a pymatgen Structure).  Per-element settings
    (RI basis + potential) are looked up from *gw.element_settings*;
    if *gw.auto_resolve* is True and a setting is missing, it is resolved
    automatically from the configured files.
    """
    mesh_str = " ".join(str(k) for k in gw.kpoints_mesh)
    elements = sorted({site.species_string for site in structure})

    # --- resolve per-element settings ---------------------------------------
    settings = dict(gw.element_settings)  # copy — never mutate the original
    missing = [el for el in elements if el not in settings]

    if missing:
        if gw.auto_resolve:
            resolved = gw.resolve_elements(set(missing))
            settings.update(resolved)
            descs = [
                f"{el}(ri={cfg.ri_basis[:15]}..., pot={cfg.potential})"
                for el, cfg in sorted(resolved.items())
            ]
            print("  Auto-resolved for: " + ", ".join(descs))
        else:
            raise ValueError(
                f"No element_settings configured for {missing}. "
                f"Either:\n"
                f"  1. Add entries to [gw.element_settings] in config.toml\n"
                f"  2. Set auto_resolve = true in [gw] to read from files"
            )

    kinds = []
    for el in elements:
        cfg = settings[el]
        kinds.append({
            "_": el,
            "BASIS_SET ORB": gw.orb_basis,
            "BASIS_SET RI_AUX": cfg.ri_basis,
            "POTENTIAL": cfg.potential,
        }        )

    return {
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
                },
                "XC": {
                    "XC_FUNCTIONAL": {"_": "PBE"},
                },
                "KPOINTS": {
                    "SCHEME": f"MONKHORST-PACK {mesh_str}",
                    "PARALLEL_GROUP_SIZE": -1,
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
                        "KPOINTS_W": " ".join(str(k) for k in gw.kpoints_w),
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
                "KIND": kinds,
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch CP2K GW + bandstructure for MC2D BN structures."
    )
    parser.add_argument(
        "--max-structures", type=int, default=5,
        help="Maximum number of structures to process (default: 5)",
    )
    parser.add_argument(
        "--filter", type=str,
        default='elements HAS ALL "B","N" AND nelements=2',
        help="OPTIMADE filter string (default: binary BN)",
    )
    parser.add_argument(
        "--elements", type=str, nargs="*",
        help="Only process structures whose elements are a subset of this set "
             "(e.g. --elements B C N). Affects both the OPTIMADE query and "
             "client-side post-filtering.",
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
        "--num-nodes", type=int, default=1,
        help="Number of compute nodes (default: 1)",
    )
    parser.add_argument(
        "--tasks-per-node", type=int, default=16,
        help="MPI tasks per node (default: 16)",
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="Show GW configuration and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config()
    gw = config.gw

    if args.show_config:
        print("GW configuration:")
        for field, value in config.gw.model_dump().items():
            print(f"  {field}: {value}")
        return

    from aiida import load_profile, orm
    from aiida.engine import submit
    from aiida.orm import Dict
    from aiida.plugins import CalculationFactory, DataFactory

    from aiida_relax_project.datasets.mc2d_optimade import fetch_mc2d_structures

    Cp2kCalculation = CalculationFactory("cp2k")
    StructureData = DataFactory("core.structure")

    load_profile()

    code_label = args.code_label or config.code_label
    code = orm.load_code(f"cp2k@{code_label}")

    group, _ = orm.Group.collection.get_or_create(args.group)

    # Build OPTIMADE filter — use a broader HAS ANY query when
    # --elements is given, then post-filter client-side.
    if args.elements:
        quoted = '","'.join(args.elements)
        optimade_filter = (
            f'elements HAS ANY "{quoted}" AND nelements=2'
        )
    else:
        optimade_filter = args.filter

    # When client-side filtering is active, fetch extra structures so
    # enough pass the post-filter; cap to *max_structures* afterwards.
    fetch_limit = (
        args.max_structures * 10 if args.elements else args.max_structures
    )
    modifier = make_modifier(gw.vacuum)
    data = fetch_mc2d_structures(
        optimade_filter=optimade_filter,
        max_structures=fetch_limit,
        modifier=modifier,
    )

    # Client-side post-filter to keep only structures whose elements
    # are a subset of the requested set.
    if args.elements:
        allowed = set(args.elements)
        before = len(data)
        data = [
            item for item in data
            if set(item["structure"].symbol_set).issubset(allowed)
        ]
        skipped = before - len(data)
        if skipped:
            print(f"  Skipped {skipped} structure(s) with elements outside {{{', '.join(sorted(allowed))}}}")
        # Trim back to the user-requested count
        data = data[:args.max_structures]

    for item in data:
        pymatgen_structure = item["structure"]

        parameters = Dict(make_gw_parameters(gw, pymatgen_structure, scf_guess=args.scf_guess))

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
            "num_machines": args.num_nodes,
            "num_mpiprocs_per_machine": args.tasks_per_node,
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
