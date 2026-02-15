#!/usr/bin/env bash
# Run once on the cluster (login or compute) to create healthbench/.venv so Slurm jobs work.
# Usage: from project root, bash healthbench/scripts/setup_healthbench_venv.sh
# Or:    cd healthbench && bash scripts/setup_healthbench_venv.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEALTHBENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${HEALTHBENCH_DIR}/.." && pwd)"

# Load Python 3.12 (same as Slurm)
module load python/3.12.11-fasrc02 2>/dev/null || true

cd "${PROJECT_ROOT}"

if [[ -d "${HEALTHBENCH_DIR}/.venv" ]]; then
  echo "healthbench/.venv already exists. Activating and ensuring blobfile is installed."
  source "${HEALTHBENCH_DIR}/.venv/bin/activate"
  pip install blobfile litellm tqdm --quiet
  echo "Done. Venv ready."
  exit 0
fi

if [[ -d "${HEALTHBENCH_DIR}/.venv.bak" ]]; then
  echo "Copying healthbench/.venv.bak to healthbench/.venv ..."
  cp -r "${HEALTHBENCH_DIR}/.venv.bak" "${HEALTHBENCH_DIR}/.venv"
  source "${HEALTHBENCH_DIR}/.venv/bin/activate"
  echo "Installing blobfile (and litellm/tqdm if missing) ..."
  pip install blobfile litellm tqdm --quiet
  echo "Done. Venv ready at healthbench/.venv"
  exit 0
fi

echo "Creating new healthbench/.venv with Python 3.12 ..."
python3 -m venv "${HEALTHBENCH_DIR}/.venv"
source "${HEALTHBENCH_DIR}/.venv/bin/activate"
pip install --upgrade pip --quiet
pip install blobfile litellm tqdm --quiet
echo "Done. Venv ready at healthbench/.venv"
