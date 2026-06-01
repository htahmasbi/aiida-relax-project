#!/bin/bash
# -----------------------------------------------------------------------------
# Cluster Setup Script for aiida-relax-project
# -----------------------------------------------------------------------------
# This script sets up an AiiDA computer + code for a cluster (e.g. SLURM/PBS).
# Adjust all variables below for your specific cluster environment.
# Run: bash setup_cluster.sh
# -----------------------------------------------------------------------------

set -euo pipefail

# ============================== CONFIGURE THESE ==============================

CLUSTER_NAME="my_cluster"           # AiiDA computer name
CLUSTER_HOST="login.cluster.edu"    # SSH hostname
CLUSTER_DESCRIPTION="My HPC cluster"
SCHEDULER="slurm"                   # One of: slurm, pbspro, sge, torque
TRANSPORT="ssh"                     # One of: ssh, local
SSH_USER="myuser"                   # Your SSH username on the cluster
SSH_KEY="~/.ssh/id_rsa"            # SSH key for passwordless login
WORKDIR="/scratch/${SSH_USER}/aiida_work"  # Working directory on the cluster
MPI_PROCS_PER_MACHINE=32           # Number of MPI processes per node
NUM_MACHINES=1                      # Number of nodes per job

# VASP code
VASP_CODE_LABEL="vasp"
VASP_EXEC="/path/to/vasp_std"       # VASP binary path on cluster

# CP2K code
CP2K_CODE_LABEL="cp2k"
CP2K_EXEC="/path/to/cp2k.popt"      # CP2K binary path on cluster

# =============================================================================

echo "=== Setting up AiiDA computer: ${CLUSTER_NAME} ==="

# Remove if re-running
verdi computer delete "${CLUSTER_NAME}" 2>/dev/null || true

# --- Setup the computer ---
verdi computer setup --non-interactive \
    --label "${CLUSTER_NAME}" \
    --hostname "${CLUSTER_HOST}" \
    --description "${CLUSTER_DESCRIPTION}" \
    --transport "${TRANSPORT}" \
    --scheduler "${SCHEDULER}" \
    --work-dir "${WORKDIR}" \
    --mpirun "mpirun" \
    --mpiprocs-per-machine "${MPI_PROCS_PER_MACHINE}" \
    --num-machines "${NUM_MACHINES}"

# --- Configure SSH transport ---
verdi computer configure ssh "${CLUSTER_NAME}" --non-interactive \
    --username "${SSH_USER}" \
    --key-file "${SSH_KEY}" \
    --allow-agent \
    --look-for-keys

# --- Test the connection ---
echo "Testing connection to ${CLUSTER_NAME}..."
verdi computer test "${CLUSTER_NAME}"

# =============================================================================
# Setup VASP code
# =============================================================================
echo "=== Setting up VASP code: ${VASP_CODE_LABEL}@${CLUSTER_NAME} ==="

verdi code delete "${VASP_CODE_LABEL}@${CLUSTER_NAME}" 2>/dev/null || true

verdi code setup --non-interactive \
    --label "${VASP_CODE_LABEL}" \
    --computer "${CLUSTER_NAME}" \
    --remote-computer-exec "${VASP_EXEC}" \
    --input-plugin "vasp.vasp"

# =============================================================================
# Setup CP2K code
# =============================================================================
echo "=== Setting up CP2K code: ${CP2K_CODE_LABEL}@${CLUSTER_NAME} ==="

verdi code delete "${CP2K_CODE_LABEL}@${CLUSTER_NAME}" 2>/dev/null || true

verdi code setup --non-interactive \
    --label "${CP2K_CODE_LABEL}" \
    --computer "${CLUSTER_NAME}" \
    --remote-computer-exec "${CP2K_EXEC}" \
    --input-plugin "cp2k"

# =============================================================================
# Update config.toml
# =============================================================================
cat >> ../config.toml << EOF

# Override for cluster: use ${CLUSTER_NAME} as the default code label
# (only needed if not using environment variables)
# code_label = "${CLUSTER_NAME}"
EOF

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run calculations with:"
echo "  aiida-relax run --engine vasp --code ${VASP_CODE_LABEL}@${CLUSTER_NAME}"
echo "  aiida-relax run --engine cp2k --code ${CP2K_CODE_LABEL}@${CLUSTER_NAME}"
echo ""
echo "Or set environment variables:"
echo "  export CODE_LABEL=${CLUSTER_NAME}"
echo "  export ENGINE=vasp"
