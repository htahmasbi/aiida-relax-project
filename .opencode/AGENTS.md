# aiida-relax-project Session Notes

## Last Session (2026-06-02) — Cluster Testing Complete

### Setup
- AiiDA profile: `hossein` (AiiDA 2.8.0)
- Computer: `rosi5` → `rosi5.fz-rossendorf.de` (transport=`core.ssh_async`, scheduler=`core.slurm`, 64 MPI procs)
- Code: `cp2k@rosi5` with module loads (`ml purge; ml use ...; ml cp2k/master`)
- Work dir: `/bigdata/casus/fwuk/tahmas41/work/aiida-runs`
- Config defaults: `engine=cp2k`, `code_label=rosi5`, `kpoints=[24,1,24]`

### Fixed
- `examples/setup_cluster.sh` — full rewrite for ROSI5 with correct AiiDA 2.8.0 flags
- `config.toml` — defaults for CP2K@rosi5
- `launch_scripts/launch_mc2d_cp2k.py` — reads code label from config (not hardcoded)
- `aiida_relax_project/core/config.py` — unwrap `[default]` TOML section; fix f-string syntax error
- Transport reconfigured with `--no-use-login-shell` to fix scheduler parser

### How to run
```bash
git pull origin main
# Single point test
aiida-relax run --mode single-point
# MC2D batch
python launch_scripts/launch_mc2d_cp2k.py
```

### Known Issues
- RabbitMQ 3.10.8 is unsupported — upgrade advised if workflows crash
- `cp2k-output-tools` version conflict with `regex` prevents upgrade on this VM

### Next Steps
- Check calculation results: `verdi process list`
- If parser still fails, check raw output: `verdi calcjob outputcat <pk> | tail -50`
- For GW/bandstructure, add `[cp2k.raw_parameters]` sections in config.toml

### How to resume
The user starts the terok container, then the AI agent reads this file.
