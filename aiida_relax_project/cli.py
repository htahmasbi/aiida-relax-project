"""Professional CLI for aiida-relax-project using Typer."""

from __future__ import annotations

import sys
from typing import Optional, Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from aiida_relax_project.core.config import get_config, load_config, reset_config
from aiida_relax_project.core.enums import CalculationMode, RelaxType, RESOURCE_PRESETS
from aiida_relax_project.core.engine import EngineFactory
from aiida_relax_project.core.logging import setup_logging, get_logger
from aiida_relax_project.core.builders import (
    SinglePointBuilder,
    RelaxationBuilder,
    VolumeScanBuilder,
    create_example_structure,
    fetch_structures_from_optimade,
)

app = typer.Typer(
    name="aiida-relax",
    help="AiiDA workflows for VASP and CP2K calculations",
    add_completion=False,
)

console = Console()
logger = get_logger("cli")


@app.command()
def run(
    mode: Annotated[
        CalculationMode,
        typer.Option("--mode", "-m", help="Calculation mode")
    ] = CalculationMode.SINGLE_POINT,
    engine: Annotated[
        Optional[str],
        typer.Option("--engine", "-e", help="Calculation engine (vasp or cp2k)")
    ] = None,
    code_label: Annotated[
        Optional[str],
        typer.Option("--code", help="Computer@code label")
    ] = None,
    structure_element: Annotated[
        Optional[str],
        typer.Option("--element", help="Element for example structure")
    ] = None,
    group_label: Annotated[
        Optional[str],
        typer.Option("--group", help="Group label for volume scan")
    ] = None,
    relax_type: Annotated[
        Optional[RelaxType],
        typer.Option("--relax-type", help="Type of relaxation")
    ] = None,
    generic_params: Annotated[
        Optional[str],
        typer.Option(
            "--params",
            help="Parameters as key=value,key=value (e.g., encut=500,max_steps=100)"
        )
    ] = None,
    kpoints: Annotated[
        Optional[str],
        typer.Option("--kpoints", help="K-point mesh as kx,ky,kz")
    ] = None,
    continue_on_failure: Annotated[
        bool,
        typer.Option("--continue-on-failure", help="Continue on calculation failure")
    ] = False,
    max_structures: Annotated[
        Optional[int],
        typer.Option("--max-structures", help="Max structures for volume scan")
    ] = None,
    use_generic: Annotated[
        bool,
        typer.Option("--generic", help="Use generic parameter translation")
    ] = True,
) -> None:
    """Run a calculation workflow."""
    from aiida import load_profile, orm
    from aiida.engine import submit

    config = get_config()
    engine = engine or config.engine
    code_label = code_label or config.code_label

    try:
        load_profile()
        logger.info(f"Running {mode} with engine={engine}")
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to load AiiDA profile: {e}")
        raise typer.Exit(1)

    try:
        code = orm.load_code(f"{engine}@{code_label}")
    except Exception as exc:
        console.print(f"[red]Error:[/red] Code '{engine}@{code_label}' not found: {exc}")
        console.print("Run 'verdi code list' to see available codes.")
        raise typer.Exit(1)

    params_dict = _parse_params(generic_params)
    parameters = orm.Dict(dict=params_dict)

    kpoints_mesh = None
    if kpoints:
        kpoints_mesh = [int(x) for x in kpoints.split(",")]

    metadata = config.metadata_options.to_dict()

    if mode == CalculationMode.SINGLE_POINT:
        structure = create_example_structure(structure_element or "Si")

        if kpoints_mesh:
            adapter = EngineFactory.create(engine)
            kpoints_data = adapter.build_kpoints(kpoints_mesh)
        else:
            kpoints_data = None

        from aiida_relax_project.workflows.single_point import DynamicSinglePointWorkChain

        builder = SinglePointBuilder(config)
        inputs = builder.build_inputs(
            structure=structure,
            code=code,
            parameters=parameters,
            engine=engine,
            kpoints=kpoints_data,
            use_generic_params=use_generic,
            metadata_options=metadata,
        )

        node = submit(DynamicSinglePointWorkChain, **inputs)
        console.print(f"[green]Submitted[/green] DynamicSinglePointWorkChain<{node.pk}>")

    elif mode == CalculationMode.RELAX:
        structure = create_example_structure(structure_element or "Si")

        if kpoints_mesh:
            adapter = EngineFactory.create(engine)
            kpoints_data = adapter.build_kpoints(kpoints_mesh)
        else:
            kpoints_data = None

        from aiida_relax_project.workflows.relaxation import DynamicRelaxWorkChain

        builder = RelaxationBuilder(config)
        inputs = builder.build_inputs(
            structure=structure,
            code=code,
            parameters=parameters,
            engine=engine,
            relaxation_type=relax_type,
            kpoints=kpoints_data,
            use_generic_params=use_generic,
            metadata_options=metadata,
        )

        node = submit(DynamicRelaxWorkChain, **inputs)
        console.print(f"[green]Submitted[/green] DynamicRelaxWorkChain<{node.pk}>")

    elif mode == CalculationMode.VOLUME_SCAN:
        group_label = group_label or config.volume_scan.default_group
        max_structures = max_structures or config.volume_scan.max_structures

        group = fetch_structures_from_optimade(
            group_label=group_label,
            max_structures=max_structures,
            elements=["B", "N"] if engine == "cp2k" else None,
        )

        from aiida_relax_project.workflows.volume_scan import DynamicVolumeScanWorkChain

        builder = VolumeScanBuilder(config)
        inputs = builder.build_inputs(
            structure_group=group,
            code=code,
            parameters=parameters,
            engine=engine,
            use_generic_params=use_generic,
            continue_on_failure=continue_on_failure,
            metadata_options=metadata,
        )

        node = submit(DynamicVolumeScanWorkChain, **inputs)
        console.print(f"[green]Submitted[/green] DynamicVolumeScanWorkChain<{node.pk}>")


@app.command()
def config_show(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON")
    ] = False,
) -> None:
    """Show current configuration."""
    import json

    config = get_config()

    if as_json:
        console.print_json(data=config.to_dict())
    else:
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("engine", config.engine)
        table.add_row("code_label", config.code_label)
        table.add_row("resource_preset", config.resource_preset)
        table.add_row("", "")

        table.add_row("vasp.potential_family", config.vasp.potential_family)
        table.add_row("vasp.kpoints_mesh", str(config.vasp.kpoints_mesh))
        table.add_row("vasp.default_encut", str(config.vasp.default_encut))
        table.add_row("", "")

        table.add_row("cp2k.kpoints_mesh", str(config.cp2k.kpoints_mesh))
        table.add_row("cp2k.default_cutoff", str(config.cp2k.default_cutoff))
        table.add_row("", "")

        table.add_row("relax.relaxation_type", config.relax.relaxation_type)
        table.add_row("relax.default_max_steps", str(config.relax.default_max_steps))
        table.add_row("", "")

        table.add_row("volume_scan.max_structures", str(config.volume_scan.max_structures))
        table.add_row("volume_scan.default_group", config.volume_scan.default_group)

        console.print(table)


@app.command()
def engines() -> None:
    """List supported calculation engines."""
    table = Table(title="Supported Engines")
    table.add_column("Engine", style="cyan")
    table.add_column("Status", style="green")

    for engine_name in EngineFactory.supported_engines():
        try:
            adapter = EngineFactory.create(engine_name)
            table.add_row(engine_name, "[green]Available[/green]")
        except Exception as e:
            table.add_row(engine_name, f"[red]Error: {e}[/red]")

    console.print(table)


@app.command()
def validate(
    structure_file: Annotated[
        str,
        typer.Argument(help="Structure file (POSCAR, cif, etc.)")
    ] = None,
    engine: Annotated[
        Optional[str],
        typer.Option("--engine", "-e", help="Engine to validate against")
    ] = None,
) -> None:
    """Validate a structure file."""
    from aiida.plugins import DataFactory

    config = get_config()
    engine = engine or config.engine

    if structure_file:
        try:
            StructureData = DataFactory("core.structure")
            structure = StructureData.get_or_create(structure_file)[0]
        except (OSError, ValueError) as e:
            console.print(f"[red]Error:[/red] Failed to load structure: {e}")
            raise typer.Exit(1) from None
    else:
        structure = create_example_structure("Si")
        console.print("[yellow]No structure file provided, using example Si structure[/yellow]")

    adapter = EngineFactory.create(engine)

    try:
        adapter.validate_structure(structure)
        console.print(f"[green]Structure is valid for {engine}[/green]")

        symbols = [site.kind_name for site in structure.sites]
        unique_elements = list(dict.fromkeys(symbols))
        console.print(f"Elements: {', '.join(unique_elements)}")
        console.print(f"Number of atoms: {len(structure.sites)}")

    except (OSError, ValueError) as e:
        console.print(f"[red]Validation failed:[/red] {e}")
        raise typer.Exit(1) from None


def _parse_params(param_string: Optional[str]) -> dict:
    """Parse parameter string into dictionary."""
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


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output")
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress output")
    ] = False,
) -> None:
    """AiiDA workflows for VASP and CP2K calculations."""
    import logging

    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    setup_logging()


if __name__ == "__main__":
    app()