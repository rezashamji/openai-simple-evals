#!/usr/bin/env bash
# Quick test that the ARK–HealthBench pipeline works before submitting the full Slurm run.
# Runs Phase 1 with --limit 2, then Phase 2 on that output. Needs .env (Azure) and healthbench/.venv.
# Usage: from project root, bash healthbench/scripts/run_small_ark_healthbench_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEALTHBENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${HEALTHBENCH_DIR}/.." && pwd)"
ARK_DIR="${PROJECT_ROOT}/ark"
INPUT_JSONL="${HEALTHBENCH_DIR}/2025-05-07-06-14-12_oss_eval.jsonl"
OUTPUT_JSONL="${HEALTHBENCH_DIR}/responses_ark_healthbench_small_test.jsonl"

module load python/3.12.11-fasrc02 2>/dev/null || true

if [[ ! -d "${HEALTHBENCH_DIR}/.venv" ]]; then
  echo "ERROR: healthbench/.venv not found. Run healthbench/scripts/setup_healthbench_venv.sh first." >&2
  exit 2
fi
source "${HEALTHBENCH_DIR}/.venv/bin/activate"

if [[ -f "${HEALTHBENCH_DIR}/.env" ]]; then
  set -a
  source "${HEALTHBENCH_DIR}/.env"
  set +a
fi

cd "${PROJECT_ROOT}"

# Remove any previous test output so Phase 1 (fresh) really runs ARK; then we test resume.
rm -f "${OUTPUT_JSONL}"

echo "Phase 1 (fresh): Running ARK on 2 HealthBench examples ..."
python -m healthbench.scripts.run_ark_on_healthbench \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --ark-dir "${ARK_DIR}" \
  --limit 2

echo "Phase 1 (resume): Re-running with same output; should reuse 2 by prompt_id, compute 0 ..."
RESUME_OUTPUT=$(python -m healthbench.scripts.run_ark_on_healthbench \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --ark-dir "${ARK_DIR}" \
  --limit 2 2>&1)
echo "${RESUME_OUTPUT}"
if ! echo "${RESUME_OUTPUT}" | grep -q "Resuming: reusing 2 results by prompt_id, computing 0"; then
  echo "ERROR: Resume did not report reusing 2 and computing 0. Check Phase 1 resume logic." >&2
  exit 2
fi

echo "Phase 2: Grading the 2 responses ..."
python -m healthbench.scripts.grade_ark_healthbench_responses \
  --responses-jsonl "${OUTPUT_JSONL}" \
  --output-dir "${HEALTHBENCH_DIR}" \
  --n-threads 2

echo "Small test passed (fresh + resume + grade). Full Slurm run should work."
