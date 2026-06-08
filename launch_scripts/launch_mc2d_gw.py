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


def make_modifier(vacuum: float = 20.0, supercell: list[int] | None = None):
    """Return a modifier that builds a supercell then rotates to xz-plane.

    Args:
        vacuum: Vacuum gap (A) along y after rotation.
        supercell: Scaling factors e.g. ``[3, 3, 1]``.  Defaults to ``[3, 3, 1]``.
    """
    if supercell is None:
        supercell = [3, 3, 1]
    from aiida_relax_project.transformations.structures import (
        center_slab_in_cell,
        make_supercell,
        rotate_xy_to_xz,
    )
    def modifier(structure):
        structure = center_slab_in_cell(structure)
        structure = make_supercell(structure, supercell)
        structure = rotate_xy_to_xz(structure, vacuum=vacuum)
        return structure
    return modifier


def _classify_2d_inplane_lattice(structure, angle_tol=15.0, ratio_tol=0.15):
    """Classify the 2D Bravais lattice from the in-plane lattice vectors.

    Examines the first two lattice vectors of *structure* and returns one of:
    ``'hexagonal'``, ``'square'``, ``'rectangular'``, or ``'oblique'``.

    Parameters
    ----------
    structure
        Pymatgen ``Structure`` (original, pre-rotation — vacuum along z).
    angle_tol
        Tolerance (degrees) for γ ≈ 120° or γ ≈ 90°.
    ratio_tol
        Tolerance for ``min(a,b) / max(a,b) ≈ 1``.

    Returns
    -------
    str
        2D Bravais lattice type.
    """
    import math

    import numpy as np

    matrix = structure.lattice.matrix
    a_vec = matrix[0]
    b_vec = matrix[1]

    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))

    cos_gamma = np.dot(a_vec, b_vec) / (a_len * b_len)
    cos_gamma = max(-1.0, min(1.0, cos_gamma))
    gamma = math.degrees(math.acos(cos_gamma))

    ratio = min(a_len, b_len) / max(a_len, b_len)

    if (abs(gamma - 120) < angle_tol or abs(gamma - 60) < angle_tol) and (1.0 - ratio) < ratio_tol:
        return "hexagonal"

    if abs(gamma - 90) < angle_tol and (1.0 - ratio) < ratio_tol:
        return "square"

    if abs(gamma - 90) < angle_tol:
        return "rectangular"

    return "oblique"


_2D_BRAVAIS_PATHS: dict[str, list[tuple[str, float, float, float]]] = {
    "hexagonal": [
        ("GAMMA", 0.0, 0.0, 0.0),
        ("M", 0.5, 0.0, 0.0),
        ("K", 1 / 3, 1 / 3, 0.0),
        ("GAMMA", 0.0, 0.0, 0.0),
    ],
    "square": [
        ("GAMMA", 0.0, 0.0, 0.0),
        ("X", 0.5, 0.0, 0.0),
        ("M", 0.5, 0.5, 0.0),
        ("GAMMA", 0.0, 0.0, 0.0),
    ],
    "rectangular": [
        ("GAMMA", 0.0, 0.0, 0.0),
        ("X", 0.5, 0.0, 0.0),
        ("S", 0.5, 0.5, 0.0),
        ("Y", 0.0, 0.5, 0.0),
        ("GAMMA", 0.0, 0.0, 0.0),
    ],
    "oblique": [
        ("GAMMA", 0.0, 0.0, 0.0),
        ("X", 0.5, 0.0, 0.0),
        ("M", 0.5, 0.5, 0.0),
        ("Y", 0.0, 0.5, 0.0),
        ("GAMMA", 0.0, 0.0, 0.0),
    ],
}


def _inplane_path_from_crystal_system(crystal_system):
    """Map a 3D crystal system to a standard 2D in-plane k-path.

    Points are in **original** fractional coordinates
    ``(label, k_a1, k_a2, k_vac=0)``.

    .. deprecated::
        Prefer :func:`_classify_2d_inplane_lattice` which determines the
        2D Bravais lattice directly from the in-plane lattice vectors.
    """
    paths = {
        "hexagonal": [
            ("GAMMA", 0.0, 0.0, 0.0),
            ("M", 0.5, 0.0, 0.0),
            ("K", 1 / 3, 1 / 3, 0.0),
            ("GAMMA", 0.0, 0.0, 0.0),
        ],
        "trigonal": [
            ("GAMMA", 0.0, 0.0, 0.0),
            ("M", 0.5, 0.0, 0.0),
            ("K", 1 / 3, 1 / 3, 0.0),
            ("GAMMA", 0.0, 0.0, 0.0),
        ],
        "orthorhombic": [
            ("GAMMA", 0.0, 0.0, 0.0),
            ("X", 0.5, 0.0, 0.0),
            ("S", 0.5, 0.5, 0.0),
            ("Y", 0.0, 0.5, 0.0),
            ("GAMMA", 0.0, 0.0, 0.0),
        ],
        "tetragonal": [
            ("GAMMA", 0.0, 0.0, 0.0),
            ("X", 0.5, 0.0, 0.0),
            ("M", 0.5, 0.5, 0.0),
            ("GAMMA", 0.0, 0.0, 0.0),
        ],
        "cubic": [
            ("GAMMA", 0.0, 0.0, 0.0),
            ("X", 0.5, 0.0, 0.0),
            ("M", 0.5, 0.5, 0.0),
            ("GAMMA", 0.0, 0.0, 0.0),
        ],
    }
    return paths.get(crystal_system)


def _reconstruct_inplane_path(kpath):
    """Extract a continuous in-plane 2D path from pymatgen's 3D path.

    1. Collect all points where k_vac (= k[2] for OPTIMADE) ≈ 0.
    2. Sort them by polar angle in the (k_a1, k_a2) plane.
    3. Build a round-trip: Γ → edge points in angular order → Γ.

    Returns list of ``(label, k_a1, k_a2, 0)`` tuples, or ``[]`` if
    fewer than 2 unique in-plane edge points exist.
    """
    import numpy as np

    kpoints = kpath.kpath["kpoints"]
    label_map = {"\\Gamma": "GAMMA", "Gamma": "GAMMA"}

    # Collect unique in-plane points (k_vac ≈ 0).
    inplane: dict[tuple[float, float], tuple[str, float, float]] = {}
    for label, k in kpoints.items():
        if abs(k[2]) > 1e-10:
            continue
        # Map label and round coords for dedup
        key = (round(k[0], 10), round(k[1], 10))
        mapped = label_map.get(label, label)
        if key not in inplane:
            inplane[key] = (mapped, k[0], k[1])

    items = list(inplane.values())
    # Separate Γ (origin) from edge points
    gamma = None
    edges: list[tuple[str, float, float]] = []
    for label, k0, k1 in items:
        if abs(k0) < 1e-10 and abs(k1) < 1e-10:
            if gamma is None:
                gamma = (label, 0.0, 0.0)
        else:
            edges.append((label, k0, k1))

    if gamma is None:
        return []

    # Sort edge points by polar angle around Γ
    edges.sort(key=lambda x: (np.arctan2(x[2], x[1]), x[1], x[2]))

    # Build the path: Γ → edge points → Γ
    result: list[tuple[str, float, float, float]] = [
        (gamma[0], gamma[1], gamma[2], 0.0)
    ]
    seen = set()
    for label, k0, k1 in edges:
        key = (round(k0, 10), round(k1, 10))
        if key not in seen:
            seen.add(key)
            result.append((label, k0, k1, 0.0))
    result.append((gamma[0], gamma[1], gamma[2], 0.0))

    return result


def get_bandstructure_path(original_structure, special_points_override=None):
    """Generate CP2K ``SPECIAL_POINT`` lines for a rotated 2D structure.

    The k-path is determined from the **2D Bravais lattice** of the
    in-plane lattice vectors, so it works for *any* 2D material
    regardless of its 3D space group classification (monoclinic,
    triclinic, etc.).

    ``rotate_xy_to_xz`` maps:
        original b1 → rotated b1  (x, in-plane)
        original b2 → rotated b3  (z, in-plane)
        original b3 → rotated b2  (y, vacuum)

    so k-points are mapped as *(k0, k1, k2*) → *(k0, k2, k1)*.

    Args:
        original_structure: Pymatgen ``Structure`` **before** modifier.
        special_points_override: Optional manual list of CP2K
            ``SPECIAL_POINT`` lines.

    Returns:
        List of ``"LABEL  x  y  z"`` strings.
    """
    if special_points_override is not None:
        return special_points_override

    # --- Step 1: 2D Bravais lattice from in-plane vectors ----------------
    bravais = _classify_2d_inplane_lattice(original_structure)
    std_path = _2D_BRAVAIS_PATHS.get(bravais)

    if std_path is not None:
        return [
            f"{label}  {k0:f}  {k2:f}  {k1:f}"
            for label, k0, k1, k2 in std_path
        ]

    # --- Step 2: fallback to 3D crystal system (backward compat) --------
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(original_structure, symprec=0.01)
    crystal_system = sga.get_crystal_system().lower()
    std_path = _inplane_path_from_crystal_system(crystal_system)

    if std_path is not None:
        return [
            f"{label}  {k0:f}  {k2:f}  {k1:f}"
            for label, k0, k1, k2 in std_path
        ]

    # --- Step 3: pymatgen path → reconstructed 2D in-plane path ---------
    from pymatgen.symmetry.bandstructure import HighSymmKpath
    kpath = HighSymmKpath(original_structure)
    inplane = _reconstruct_inplane_path(kpath)

    if inplane:
        return [
            f"{label}  {k0:f}  {k2:f}  {k1:f}"
            for label, k0, k1, k2 in inplane
        ]

    # --- Step 4: last resort — take the full 3D path as-is -------------
    kpoints = kpath.kpath["kpoints"]
    label_map = {"\\Gamma": "GAMMA", "Gamma": "GAMMA"}
    target = []
    for segment in kpath.kpath["path"]:
        for label in segment:
            if label not in target:
                target.append(label)

    return [
        f"{label_map.get(label, label)}  {kpoints[label][0]:f}  {kpoints[label][2]:f}  {kpoints[label][1]:f}"
        for label in target
    ]


def make_gw_parameters(gw, structure, scf_guess="ATOMIC", original_structure=None):
    """Build the full CP2K input dict for a GW + bandstructure run.

    KIND sections are generated dynamically from the unique elements
    present in *structure* (a pymatgen Structure).  Per-element settings
    (RI basis + potential) are looked up from *gw.element_settings*;
    if *gw.auto_resolve* is True and a setting is missing, it is resolved
    automatically from the configured files.
    """
    mesh_str = " ".join(str(k) for k in gw.get_kpoints_mesh(structure))
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
                        "KPOINTS_W": " ".join(str(k) for k in gw.get_kpoints_w(structure)),
                    },
                    "BANDSTRUCTURE_PATH": {
                        "NPOINTS": gw.bs_npoints,
                        "UNITS": "B_VECTOR",
                        "SPECIAL_POINT": get_bandstructure_path(
                            original_structure, gw.special_points
                        ),
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
        "--max-atoms", type=int, default=None,
        help="Only process structures with at most this many atoms (nsites)",
    )
    parser.add_argument(
        "--min-atoms", type=int, default=None,
        help="Only process structures with at least this many atoms (nsites)",
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
    modifier = make_modifier(gw.vacuum, gw.supercell)
    data = fetch_mc2d_structures(
        optimade_filter=optimade_filter,
        max_structures=fetch_limit,
        modifier=modifier,
        max_atoms=args.max_atoms,
        min_atoms=args.min_atoms,
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

        original_structure = item.get("original_structure")
        parameters = Dict(
            make_gw_parameters(
                gw, pymatgen_structure,
                scf_guess=args.scf_guess,
                original_structure=original_structure,
            )
        )

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
