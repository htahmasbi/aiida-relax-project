#!/bin/bash
# -----------------------------------------------------------------------------
# Profile Setup Script for aiida-relax-project — PostgreSQL + AiiDA profile
# -----------------------------------------------------------------------------
# Run this BEFORE setup_cluster.sh — it creates the AiiDA profile and database.
#
# Usage:
#   bash setup_profile.sh
#
# Prerequisites:
#   - PostgreSQL installed (sudo apt install postgresql)
#   - PostgreSQL running  (sudo service postgresql start)
#   - verdi in PATH
# -----------------------------------------------------------------------------

set -euo pipefail

# ============================== CONFIGURATION ================================

# --- Profile settings ---
PROFILE_NAME="aiida_cp2k"               # Name for your AiiDA profile
EMAIL="h.tahmasb@gmail.com"             # Your email
FIRST_NAME="Hossein"                    # Your first name
LAST_NAME="Tahmasbi"                    # Your last name
INSTITUTION="Casus"                     # Your institution

# --- PostgreSQL settings ---
DB_BACKEND="core.psql_dos"              # PostgreSQL backend
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="aiida_db_${PROFILE_NAME}"      # Database name
DB_USER="Hossein"                       # Database user
DB_PASS="aiida_secret"                  # Database password

# =============================================================================
# NOTE: Edit the variables above before running.
# The script does the rest automatically.
# =============================================================================

echo "=== 1. Checking prerequisites ==="
if ! command -v verdi &>/dev/null; then
    echo "ERROR: verdi not found. Install aiida-core first."
    exit 1
fi

if ! command -v psql &>/dev/null; then
    echo "WARNING: psql not found."
    echo "  Install PostgreSQL: sudo apt install postgresql"
    echo "  Then start it:      sudo service postgresql start"
    exit 1
fi

echo "  verdi: OK"
echo "  psql:  OK"

echo ""
echo "=== 2. Creating PostgreSQL role and database ==="
# This runs as the postgres system user (requires sudo).
# If the role already exists, it's skipped (no error).
# If the database already exists, it's skipped.

sudo -u postgres psql -tc \
    "SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}'" \
    | grep -q 1 \
    && echo "  Role '${DB_USER}' already exists — skipping create" \
    || {
        sudo -u postgres createuser "${DB_USER}"
        echo "  Created role '${DB_USER}'"
    }

sudo -u postgres psql -tc \
    "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" \
    | grep -q 1 \
    && echo "  Database '${DB_NAME}' already exists — skipping create" \
    || {
        sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
        echo "  Created database '${DB_NAME}'"
    }

# Set the password for the role
sudo -u postgres psql -c \
    "ALTER ROLE ${DB_USER} WITH PASSWORD '${DB_PASS}'" \
    &>/dev/null
echo "  Password set for role '${DB_USER}'"

echo ""
echo "=== 3. Checking for existing profile: ${PROFILE_NAME} ==="

if verdi profile show "${PROFILE_NAME}" &>/dev/null; then
    echo "  Profile '${PROFILE_NAME}' already exists."
    echo "  Delete it first if you want to recreate:"
    echo "    verdi profile delete ${PROFILE_NAME}"
    echo "  Then re-run this script."
    exit 1
fi

echo "  Profile '${PROFILE_NAME}' does not exist — proceeding with setup."

echo ""
echo "=== 4. Setting up AiiDA profile: ${PROFILE_NAME} ==="

verdi setup --non-interactive \
    --profile-name "${PROFILE_NAME}" \
    --email "${EMAIL}" \
    --first-name "${FIRST_NAME}" \
    --last-name "${LAST_NAME}" \
    --institution "${INSTITUTION}" \
    --db-backend "${DB_BACKEND}" \
    --db-host "${DB_HOST}" \
    --db-port "${DB_PORT}" \
    --db-name "${DB_NAME}" \
    --db-user "${DB_USER}" \
    --db-pass "${DB_PASS}" \
    --set-as-default

# If verdi setup fails, the script stops here (set -e).
# Your existing profiles are untouched — we never deleted anything.

echo ""
echo "=== 5. Verifying profile ==="
verdi profile list
verdi profile show "${PROFILE_NAME}"

echo ""
echo "============================================"
echo "  PROFILE SETUP COMPLETE!"
echo "============================================"
echo ""
echo "Now you can run setup_cluster.sh to set up computers and codes,"
echo "or start working directly:"
echo ""
echo "  verdi status"
echo "  bash examples/setup_cluster.sh"
echo ""
