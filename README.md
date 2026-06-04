![CI Workflow test](https://github.com/htahmasbi/aiida-relax-project/actions/workflows/ci.yml/badge.svg)
# aiida-relax-project

Unified AiiDA workflows for VASP and CP2K calculations with a single CLI.

## Features

- **Engine-agnostic** — same workflow runs VASP or CP2K; generic parameters auto-translate
- **Single-point, relaxation, and volume-scan** workflows built-in
- **OPTIMADE integration** — fetch structures from materials databases (e.g. MC2D)
- **Pydantic-validated config** via `config.toml`, env vars, or CLI flags
- **Structure transformations** — rotation, vacuum padding, supercell expansion
- **CLI-first design** — `aiida-relax` command with Typer

## Project Structure

```text
aiida-relax-project/
├── LICENSE
├── README.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── config.toml                           # Main configuration file
├── aiida_relax_project/
│   ├── __init__.py                       # Package version & public API
│   ├── cli.py                            # aiida-relax CLI (Typer)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                     # Pydantic config (ProjectConfig)
│   │   ├── engine.py                     # Engine adapters (VaspAdapter, Cp2kAdapter)
│   │   ├── builders.py                   # Workflow input builders
│   │   ├── enums.py                      # EngineType, RunType, RelaxType, etc.
│   │   ├── exceptions.py                 # Custom exceptions
│   │   ├── logging.py                    # Logging setup
│   │   └── protocols.py                  # Type protocols
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── single_point.py               # Single-point WorkChains
│   │   ├── relaxation.py                 # Relaxation WorkChains
│   │   └── volume_scan.py                # Volume scan WorkChains
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── mc2d_optimade.py              # OPTIMADE structure fetcher
│   └── transformations/
│       ├── __init__.py
│       └── structures.py                 # Rotate, supercell, vacuum
├── launch_scripts/
│   ├── launch_single_point.py            # Standalone VASP launch script
│   ├── launch_unified.py                 # Unified argparse launcher
│   └── launch_mc2d_cp2k.py              # MC2D CP2K batch launcher
├── examples/
│   └── setup_cluster.sh                  # Cluster setup template
├── data_generation/
│   ├── build_structures.py
│   └── fetch_mc2d.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_config.py
    ├── test_core.py
    ├── test_engines.py
    ├── test_imports.py
    ├── test_mc2d_optimade.py
    └── test_structures.py
```

## Setup

```bash
conda create -n aiida-env python=3.11
conda activate aiida-env
pip install aiida-core aiida-vasp aiida-cp2k ase
pip install -e .
verdi quicksetup
verdi daemon start
```

## Quick Start

```bash
# Single-point energy with VASP
aiida-relax run --mode single-point --engine vasp --code vasp@localhost

# Single-point with CP2K and generic params
aiida-relax run --mode single-point --engine cp2k --code cp2k@localhost \
  --params encut=500,max_steps=100

# Relaxation
aiida-relax run --mode relax --engine vasp --relax-type volume

# Show current configuration
aiida-relax config-show

# List supported engines
aiida-relax engines
```

## Cluster Deployment

1. Edit `examples/setup_cluster.sh` with your cluster details
2. Run it to configure the AiiDA computer and codes
3. Update `config.toml` or use environment variables:

```bash
export CODE_LABEL=my_cluster
export ENGINE=vasp
aiida-relax run --mode single-point
```

## Configuration

Settings are loaded in order: CLI flags > environment variables > `config.toml` > defaults.

```bash
ENGINE=vasp
CODE_LABEL=my_cluster
VASP_POTENTIAL_FAMILY=PBE.54
DEFAULT_ENCUT=500
```

## Running Tests

```bash
pytest -v
```
