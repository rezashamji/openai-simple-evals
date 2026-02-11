#!/bin/bash
#SBATCH --job-name=ark-healthbench-grade
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ark-healthbench-grade-%j.out
#SBATCH --error=logs/ark-healthbench-grade-%j.err
#SBATCH --gres=gpu:1

# Phase 2 ONLY: Grade precomputed responses with HealthBench rubrics.
# Expects Phase 1 to have already created responses_ark_healthbench.jsonl.
#
# Tip: This script runs grading in a "fully sequential" mode by exporting `debug=1`
# (HealthBench uses this env var to disable thread pools) and by setting --n-threads 1.

set -e
set -u

module load python/3.12.11-fasrc02

HEALTHBENCH_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/healthbench"
PROJECT_ROOT="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji"
RESPONSES_JSONL="${HEALTHBENCH_DIR}/responses_ark_healthbench.jsonl"

# Phase 2: grading threads passed to HealthBenchEval (set 1 to minimize rate limit errors)
N_THREADS=1

cd "${PROJECT_ROOT}"
mkdir -p "${HEALTHBENCH_DIR}/logs"

# Activate the HealthBench venv (created under the loaded module Python).
if [[ -d "${HEALTHBENCH_DIR}/.venv" ]]; then
  source "${HEALTHBENCH_DIR}/.venv/bin/activate"
else
  echo "ERROR: Missing venv at ${HEALTHBENCH_DIR}/.venv. Create it and install deps." >&2
  exit 2
fi

# Load Azure env vars if present (keys/endpoints/versions). Do not commit real keys.
set -a
if [[ -f "${HEALTHBENCH_DIR}/.env" ]]; then
  source "${HEALTHBENCH_DIR}/.env"
fi
set +a

# HealthBench uses `debug` to disable its internal ThreadPool usage.
export debug=1

# Optional: verbose LiteLLM logs (noisy, but helpful for diagnosing 429s/timeouts).
# export LITELLM_LOG=DEBUG

if [[ ! -f "${RESPONSES_JSONL}" ]]; then
  echo "ERROR: Responses file not found: ${RESPONSES_JSONL}" >&2
  echo "Run Phase 1 first (healthbench/scripts/run_ark_on_healthbench.py) to generate it." >&2
  exit 1
fi

CHECKPOINT="${HEALTHBENCH_DIR}/responses_ark_healthbench_grading_checkpoint.jsonl"

echo "Phase 2 ONLY: Grade responses (debug=1, n_threads=${N_THREADS}, checkpoint=${CHECKPOINT})"
python -m healthbench.scripts.grade_ark_healthbench_responses \
  --responses-jsonl "${RESPONSES_JSONL}" \
  --output-dir "${HEALTHBENCH_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --n-threads "${N_THREADS}"

echo "Done. Report: ${HEALTHBENCH_DIR}/responses_ark_healthbench_report.html"
echo "Done. Metrics: ${HEALTHBENCH_DIR}/responses_ark_healthbench_metrics.json"
