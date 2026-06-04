"""Unified launch script supporting both VASP and CP2K engines.

Usage:
    # Single-point with VASP (uses config.toml defaults)
    python launch_unified.py --mode single-point

    # Single-point with CP2K
    python launch_unified.py --engine cp2k --mode single-point

    # Relaxation with VASP
    python launch_unified.py --mode relax --relax-type volume

    # Volume scan with CP2K
    python launch_unified.py --engine cp2k --mode volume-scan --group my_structures

    # Using generic parameters (auto-converted to engine format)
    python launch_unified.py --engine cp2k --mode single-point --generic-params encut=500,max_steps=100

Configuration (in order of priority):
    1. CLI arguments
    2. Environment variables (ENGINE, CODE_LABEL, etc.)
    3. config.toml file
    4. Defaults

Environment variables:
    ENGINE: 'vasp' or 'cp2k' (default: vasp)
    CODE_LABEL: Computer@code label (default: localhost)
    AIIDA_RELAX_CONFIG: Path to config file
"""

import argparse

from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import DataFactory

from aiida_relax_project.core.config import get_config
from aiida_relax_project.datasets.mc2d_optimade import fetch_mc2d_structures
from aiida_relax_project.transformations.structures import (
    make_supercell_3x3,
    rotate_xy_to_xz,
)
from aiida_relax_project.workflows.relaxation import DynamicRelaxWorkChain
from aiida_relax_project.workflows.single_point import DynamicSinglePointWorkChain
from aiida_relax_project.workflows.volume_scan import DynamicVolumeScanWorkChain

StructureData = DataFactory("core.structure")


def parse_generic_params(param_string: str | None) -> dict:
    """Parse generic parameters from comma-separated key=value pairs."""
    if not param_string:
        return {}

    params = {}
    for pair in param_string.split(","):
        key, value = pair.split("=")
        key = key.strip()
        value = value.strip()

        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
        elif value.isdigit():
            params[key] = int(value)
        elif "." in value and value.replace(".", "").isdigit():
            params[key] = float(value)
        else:
            params[key] = value

    return params


def create_example_structure(element: str = "Si") -> StructureData:
    """Create a simple example structure."""
    structure = StructureData()
    structure.set_cell([
        [5.43, 0.00, 0.00],
        [0.00, 5.43, 0.00],
        [0.00, 0.00, 5.43],
    ])
    structure.append_atom(position=(0.00, 0.00, 0.00), symbols=element)
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols=element)
    structure.label = f"Example {element} structure"
    structure.store()
    return structure


def create_kpoints(mesh: list) -> orm.KpointsData:
    """Create k-points data."""
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh(mesh)
    return kpoints


def launch_single_point(
    engine: str,
    code_label: str,
    structure: StructureData,
    parameters: dict,
    use_generic: bool = False,
    kpoints_mesh: list | None = None,
    metadata_options: dict | None = None,
):
    """Launch a single-point calculation."""
    code = orm.load_code(f"{engine}@{code_label}")

    params = orm.Dict(dict=parameters)

    kpoints = None
    if kpoints_mesh:
        kpoints = create_kpoints(kpoints_mesh)

    inputs = {
        "engine": orm.Str(engine),
        "code": code,
        "structure": structure,
        "parameters": params,
        "use_generic_params": orm.Bool(use_generic),
    }

    if kpoints is not None:
        inputs["kpoints"] = kpoints

    if engine == "vasp":
        inputs["potential_family"] = orm.Str("PBE.54")
        inputs["potential_mapping"] = orm.Dict(dict={"Si": "Si"})

    if metadata_options:
        inputs["metadata_options"] = orm.Dict(dict=metadata_options)

    node = submit(DynamicSinglePointWorkChain, **inputs)
    print(f"Submitted DynamicSinglePointWorkChain<{node.pk}> for {engine.upper()}")
    return node


def launch_relaxation(
    engine: str,
    code_label: str,
    structure: StructureData,
    parameters: dict,
    relax_type: str = "volume",
    use_generic: bool = False,
    kpoints_mesh: list | None = None,
    metadata_options: dict | None = None,
):
    """Launch a relaxation calculation."""
    code = orm.load_code(f"{engine}@{code_label}")

    params = orm.Dict(dict=parameters)

    kpoints = None
    if kpoints_mesh:
        kpoints = create_kpoints(kpoints_mesh)

    inputs = {
        "engine": orm.Str(engine),
        "code": code,
        "structure": structure,
        "parameters": params,
        "relaxation_type": orm.Str(relax_type),
        "use_generic_params": orm.Bool(use_generic),
    }

    if kpoints is not None:
        inputs["kpoints"] = kpoints

    if engine == "vasp":
        inputs["potential_family"] = orm.Str("PBE.54")
        inputs["potential_mapping"] = orm.Dict(dict={"Si": "Si"})

    if metadata_options:
        inputs["metadata_options"] = orm.Dict(dict=metadata_options)

    node = submit(DynamicRelaxWorkChain, **inputs)
    print(f"Submitted DynamicRelaxWorkChain<{node.pk}> for {engine.upper()} (relax_type={relax_type})")
    return node


def launch_volume_scan(
    engine: str,
    code_label: str,
    group_label: str,
    parameters: dict,
    use_generic: bool = False,
    kpoints_mesh: list | None = None,
    metadata_options: dict | None = None,
    continue_on_failure: bool = False,
    max_structures: int | None = None,
):
    """Launch a volume scan for structures in a group."""
    code = orm.load_code(f"{engine}@{code_label}")

    try:
        group = orm.Group.collection.get(label=group_label)
    except Exception:
        print(f"Group '{group_label}' not found. Creating it...")
        group, _ = orm.Group.collection.get_or_create(group_label)

        structures = fetch_mc2d_structures(
            optimade_filter='elements HAS ALL "B","N" AND nelements=2',
            max_structures=max_structures or 5,
            modifier=lambda s: rotate_xy_to_xz(make_supercell_3x3(s), vacuum=20.0),
        )

        for item in structures:
            pymatgen_structure = item["structure"]
            structure = StructureData(pymatgen=pymatgen_structure)
            structure.label = f"MC2D {item['formula']} {item['id']}"
            structure.store()
            group.add_nodes(structure)
            print(f"Added structure {item['id']} to group")

    params = orm.Dict(dict=parameters)

    kpoints = None
    if kpoints_mesh:
        kpoints = create_kpoints(kpoints_mesh)

    inputs = {
        "engine": orm.Str(engine),
        "structure_group": group,
        "code": code,
        "parameters": params,
        "use_generic_params": orm.Bool(use_generic),
        "continue_on_failure": orm.Bool(continue_on_failure),
    }

    if kpoints is not None:
        inputs["kpoints"] = kpoints

    if engine == "vasp":
        inputs["potential_family"] = orm.Str("PBE.54")
        inputs["potential_mapping"] = orm.Dict(dict={"B": "B", "N": "N"})

    if metadata_options:
        inputs["metadata_options"] = orm.Dict(dict=metadata_options)

    node = submit(DynamicVolumeScanWorkChain, **inputs)
    print(f"Submitted DynamicVolumeScanWorkChain<{node.pk}> for {engine.upper()}")
    return node


def main():
    parser = argparse.ArgumentParser(
        description="Unified launch script for VASP and CP2K calculations."
    )

    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        choices=["vasp", "cp2k"],
        help="Calculation engine (default: from config.toml or ENGINE env var)",
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        required=True,
        choices=["single-point", "relax", "volume-scan"],
        help="Calculation mode",
    )

    parser.add_argument(
        "--structure-element",
        type=str,
        help="Element for example structure (default: Si)",
    )

    parser.add_argument(
        "--group",
        type=str,
        help="Group label for volume scan",
    )

    parser.add_argument(
        "--relax-type",
        type=str,
        choices=["volume", "shape", "positions", "cell"],
        help="Relaxation type (for relax mode)",
    )

    parser.add_argument(
        "--generic-params",
        type=str,
        help="Generic parameters as key=value,key=value (e.g., encut=500,max_steps=100)",
    )

    parser.add_argument(
        "--kpoints",
        type=str,
        help="K-point mesh as kx,ky,kz (e.g., 4,4,4)",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        help="Metadata options as key=value,key=value",
    )

    parser.add_argument(
        "--code-label",
        type=str,
        help="Computer@code label (default: from config.toml or CODE_LABEL env var)",
    )

    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue processing other structures if one fails (volume-scan only)",
    )

    parser.add_argument(
        "--max-structures",
        type=int,
        help="Maximum number of structures to fetch for volume scan",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show loaded configuration and exit",
    )

    args = parser.parse_args()

    config = get_config()

    if args.show_config:
        print("Loaded configuration:")
        print(f"  engine: {config.engine}")
        print(f"  code_label: {config.code_label}")
        print(f"  vasp.kpoints_mesh: {config.vasp.kpoints_mesh}")
        print(f"  cp2k.kpoints_mesh: {config.cp2k.kpoints_mesh}")
        print(f"  vasp.potential_family: {config.vasp.potential_family}")
        print(f"  relax.relaxation_type: {config.relax.relaxation_type}")
        print(f"  volume_scan.max_structures: {config.volume_scan.max_structures}")
        return

    load_profile()

    engine = args.engine or config.engine

    if args.kpoints:
        kpoints_mesh = [int(x) for x in args.kpoints.split(",")]
    else:
        kpoints_mesh = config.get_kpoints_mesh()

    if args.metadata:
        metadata_options = parse_generic_params(args.metadata)
    else:
        metadata_options = config.metadata_options.to_dict()

    parameters = parse_generic_params(args.generic_params) if args.generic_params else {}

    use_generic = bool(parameters)

    if args.mode == "single-point":
        structure = create_example_structure(args.structure_element or "Si")
        launch_single_point(
            engine=engine,
            code_label=args.code_label or config.code_label,
            structure=structure,
            parameters=parameters,
            use_generic=use_generic,
            kpoints_mesh=kpoints_mesh,
            metadata_options=metadata_options,
        )

    elif args.mode == "relax":
        structure = create_example_structure(args.structure_element or "Si")
        launch_relaxation(
            engine=engine,
            code_label=args.code_label or config.code_label,
            structure=structure,
            parameters=parameters,
            relax_type=args.relax_type or config.relax.relaxation_type,
            use_generic=use_generic,
            kpoints_mesh=kpoints_mesh,
            metadata_options=metadata_options,
        )

    elif args.mode == "volume-scan":
        launch_volume_scan(
            engine=engine,
            code_label=args.code_label or config.code_label,
            group_label=args.group or config.volume_scan.default_group,
            parameters=parameters,
            use_generic=use_generic,
            kpoints_mesh=kpoints_mesh,
            metadata_options=metadata_options,
            continue_on_failure=args.continue_on_failure or config.volume_scan.continue_on_failure,
            max_structures=args.max_structures or config.volume_scan.max_structures,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
