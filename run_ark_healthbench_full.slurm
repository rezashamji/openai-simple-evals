#!/bin/bash
#SBATCH --job-name=ark-healthbench-full
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/ark-healthbench-full-%j.out
#SBATCH --error=logs/ark-healthbench-full-%j.err
#SBATCH --gres=gpu:1

# Full HealthBench on ARK. Pass graph/model here so the run is explicit (future runs can use different KG/model).
# Paper A.2: Prime KG, GPT-4.1, n=3 agents, Tmax=20. kempner_h100: 96 CPUs/node, 3-day max.

set -e
set -u

# Cluster: load Python (required for venv and subprocess).
module load python/3.12.11-fasrc02

HEALTHBENCH_DIR="/n/netscratch/mzitnik_lab/Lab/rshamji/healthbench"
ARK_DIR="/n/netscratch/mzitnik_lab/Lab/rshamji/ark"
PROJECT_ROOT="/n/netscratch/mzitnik_lab/Lab/rshamji"
INPUT_JSONL="${HEALTHBENCH_DIR}/2025-05-07-06-14-12_oss_eval.jsonl"
OUTPUT_JSONL="${HEALTHBENCH_DIR}/responses_ark_healthbench.jsonl"

# Phase 1: parallel questions (more workers = faster; tune to API limits)
N_WORKERS=16
# Phase 2: grading threads (HealthBench default 120; higher can speed if API allows)
N_THREADS=200

# ARK run config (passed explicitly so you can change for other runs)
GRAPH_NAME=prime
ARK_MODEL=azure/gpt-4.1
ARK_AGENTS=3
ARK_MAX_STEPS=20

# Optional: limit examples for a test run (e.g. LIMIT=100). Leave empty for full dataset.
LIMIT=""

cd "${PROJECT_ROOT}"
mkdir -p "${HEALTHBENCH_DIR}/logs"

# Prefer a local venv created under the loaded module Python.
# NOTE: `healthbench-venv/` in this repo is Python 3.6 and is incompatible with these scripts.
if [[ -d "${HEALTHBENCH_DIR}/.venv" ]]; then
  source "${HEALTHBENCH_DIR}/.venv/bin/activate"
elif [[ -d "${HEALTHBENCH_DIR}/healthbench-venv" ]]; then
  echo "ERROR: Found healthbench/healthbench-venv/ (Python 3.6). Create a Python 3.12+ venv at ${HEALTHBENCH_DIR}/.venv and install deps." >&2
  exit 2
fi

set -a
if [[ -f "${HEALTHBENCH_DIR}/.env" ]]; then
  source "${HEALTHBENCH_DIR}/.env"
fi
set +a

echo "Phase 1: HealthBench → ARK (graph=${GRAPH_NAME}, model=${ARK_MODEL}, agents=${ARK_AGENTS}, max_steps=${ARK_MAX_STEPS}) → response jsonl"
python -m healthbench.scripts.run_ark_on_healthbench \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --ark-dir "${ARK_DIR}" \
  --graph-name "${GRAPH_NAME}" \
  --ark-model "${ARK_MODEL}" \
  --ark-agents "${ARK_AGENTS}" \
  --ark-max-steps "${ARK_MAX_STEPS}" \
  --nl-model "${ARK_MODEL}" \
  --n-workers "${N_WORKERS}" \
  $([[ -n "${LIMIT}" ]] && echo "--limit ${LIMIT}")

echo "Phase 2: Grade responses (n_threads=${N_THREADS})"
python -m healthbench.scripts.grade_ark_healthbench_responses \
  --responses-jsonl "${OUTPUT_JSONL}" \
  --output-dir "${HEALTHBENCH_DIR}" \
  --n-threads "${N_THREADS}"

echo "Done. Report: ${HEALTHBENCH_DIR}/responses_ark_healthbench_report.html"
