# aiida-relax-project

Learning project for building AiiDA workflows for VASP and CP2K calculations.

## Goal

1. Run one VASP/CP2K calculation through AiiDA.
2. Wrap it in a custom WorkChain.
3. Run many structures in a volume scan.
4. Collect energies.
5. Replace low-level VaspCalculation calls with aiida-vasp/cp2k RelaxWorkChain.

## Project Structure

```text
aiida-relax-project/
├── LICENSE
├── README.md
├── pyproject.toml
├── aiida_relax_project/
│   ├── __init__.py
│   ├── datasets
│   ├── transformations
│   └── workflows/
│       ├── __init__.py
│       ├── single_point.py
│       └── volume_scan.py
├── examples/
│   └── setup_cluster.sh
├── launch_scripts/
│   └── launch_single_point.py
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
```
