# aiida-relax-project

Learning project for building AiiDA workflows for VASP calculations.

## Goal

1. Run one VASP calculation through AiiDA.
2. Wrap it in a custom WorkChain.
3. Run many structures in a volume scan.
4. Collect energies.
5. Replace low-level VaspCalculation calls with aiida-vasp RelaxWorkChain.

## Project Structure

```text
iaiida-relax-project/
├── pyproject.toml
├── aiida_relax_project/
│   ├── __init__.py
│   └── workflows/
│       ├── __init__.py
│       ├── single_point.py
│       └── volume_scan.py
├── launch_scripts/
│   ├── launch_single_point.py
│   └── launch_volume_scan.py
└── data_generation/
    └── build_structures.py
```

## Setup

```bash
conda create -n aiida-vasp python=3.11
conda activate aiida-vasp
pip install aiida-core aiida-vasp ase
pip install -e .
verdi quicksetup
verdi daemon start

