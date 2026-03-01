#!/bin/bash
#SBATCH --job-name=ark-healthbench-optimus-full
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/ark-healthbench-optimus-full-%j.out
#SBATCH --error=logs/ark-healthbench-optimus-full-%j.err
#SBATCH --gres=gpu:1

# Full HealthBench Level 3: Real ARK + OptmusKG Evaluation + Grading + Baseline Comparison
#
# CRITICAL INSIGHT: This script runs TWO separate Phase 1 evaluations with DIFFERENT response_text:
# 1. ARK + OptmusKG on HealthBench (generates response via ARK node retrieval + LLM reasoning)
# 2. Pure GPT-4.1 baseline on HealthBench (no KG, no ARK; direct LLM answers)
#
# Then we grade both separately, compare at criterion level (Step 8b), and generate
# binary labels (Step 9) showing whether KG helped improve answers per question.
#
# CHECKPOINTING: All phases support resuming from checkpoints if interrupted:
#   Phase 1A: Resume by prompt_id (load existing output, skip completed examples)
#   Phase 1B: Resume by prompt_id (load existing output, skip completed examples)
#   Phase 2: Per-question checkpoint files (grading_checkpoint.jsonl written after each grade)
#
# Pipeline:
#   Phase 1A: ARK+OptmusKG → phase1_ark_kg_grounded.jsonl (resume-capable)
#   Phase 1B: Pure GPT-4.1 → phase1_no_kg.jsonl (DIFFERENT response_text, resume-capable)
#   Phase 2: Grade both → phase2_kg_output/ + phase2_no_kg_output/ (checkpoint files)
#   Step 8b: Criterion-level comparison
#   Step 9: Binary labels (USEKG vs DONTUSEKG)

set -e
set -u

# Enable unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

module load python/3.12.11-fasrc02

SIMPLE_EVALS_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals"
ARK_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/ark"
PROJECT_ROOT="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji"
INPUT_JSONL="${SIMPLE_EVALS_DIR}/2025-05-07-06-14-12_oss_eval.jsonl"
OUTPUT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/kg_evaluation_results/$(date +%Y%m%d_%H%M%S)_optimus"
VENV_PYTHON="${SIMPLE_EVALS_DIR}/.venv/bin/python"  # Single consolidated venv with all dependencies

# Configuration
N_WORKERS=4              # Phase 1 parallel workers (ARK)
N_THREADS=10             # Phase 2 grading threads
GRAPH_NAME=optimus       # OptmusKG (192,682 nodes, 21.8M edges)
ARK_MODEL=azure/gpt-4.1
ARK_AGENTS=3
ARK_MAX_STEPS=20
BASELINE_MODEL=azure/gpt-4.1  # For pure GPT baseline (no KG) - must match Azure setup
LIMIT=""                    # Empty = full dataset (~5000 questions)

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs"

# Activate venv
if [[ -d "${SIMPLE_EVALS_DIR}/.venv" ]]; then
  source "${SIMPLE_EVALS_DIR}/.venv/bin/activate"
else
  echo "ERROR: Missing venv at ${SIMPLE_EVALS_DIR}/.venv" >&2
  exit 2
fi

# Load Azure credentials
set -a
if [[ -f "${SIMPLE_EVALS_DIR}/.env" ]]; then
  source "${SIMPLE_EVALS_DIR}/.env"
fi
set +a

echo "=========================================="
echo "LEVEL 3: Full HealthBench KG Evaluation"
echo "Graph: OptmusKG (192,682 nodes, 21.8M edges)"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Phase 1A (ARK+OptmusKG): graph=${GRAPH_NAME}, model=${ARK_MODEL}, agents=${ARK_AGENTS}, max_steps=${ARK_MAX_STEPS}"
echo "Phase 1B (Pure GPT): model=${BASELINE_MODEL}"
echo ""

# ============================================================================
# PHASE 1A: ARK with OptmusKG on HealthBench
# ============================================================================
PHASE1_KG="${OUTPUT_DIR}/phase1_ark_kg_grounded.jsonl"

echo "[Phase 1A] Running ARK + OptmusKG..."
"${VENV_PYTHON}" -m simple_evals.scripts.run_ark_on_healthbench \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${PHASE1_KG}" \
  --ark-dir "${ARK_DIR}" \
  --graph-name "${GRAPH_NAME}" \
  --ark-model "${ARK_MODEL}" \
  --ark-agents "${ARK_AGENTS}" \
  --ark-max-steps "${ARK_MAX_STEPS}" \
  --nl-model "${ARK_MODEL}" \
  --python "${ARK_DIR}/.venv/bin/python" \
  --n-workers "${N_WORKERS}" \
  $([[ -n "${LIMIT}" ]] && echo "--limit ${LIMIT}")

echo "[Phase 1A] Complete. Output: ${PHASE1_KG}"
echo ""

# ============================================================================
# PHASE 1B: Pure GPT-4.1 Baseline (no KG, no ARK)
# ============================================================================
PHASE1_NO_KG="${OUTPUT_DIR}/phase1_no_kg.jsonl"
echo "[Phase 1B] Running pure GPT-4.1 baseline (no KG, no ARK)..."
"${VENV_PYTHON}" "${SIMPLE_EVALS_DIR}/scripts/run_baseline_on_healthbench.py" \
  --examples-jsonl "${INPUT_JSONL}" \
  --output-path "${PHASE1_NO_KG}" \
  --model "${BASELINE_MODEL}" \
  --n-workers "${N_WORKERS}" \
  $([[ -n "${LIMIT}" ]] && echo "--limit ${LIMIT}")

echo "[Phase 1B] Complete. Output: ${PHASE1_NO_KG}"
echo ""

# ============================================================================
# PHASE 2A: Grade ARK+OptmusKG responses
# ============================================================================
echo "[Phase 2A] Grading ARK+OptmusKG responses (with KG metadata)..."
"${VENV_PYTHON}" -m simple_evals.scripts.grade_ark_healthbench_responses \
  --responses-jsonl "${PHASE1_KG}" \
  --output-dir "${OUTPUT_DIR}/phase2_kg_output/"

echo "[Phase 2A] Complete. Output: ${OUTPUT_DIR}/phase2_kg_output/"
echo ""

# ============================================================================
# PHASE 2B: Grade pure GPT baseline (no KG metadata, for fair comparison)
# ============================================================================
echo "[Phase 2B] Grading pure GPT baseline (without KG metadata)..."
"${VENV_PYTHON}" -m simple_evals.scripts.grade_ark_healthbench_responses \
  --responses-jsonl "${PHASE1_NO_KG}" \
  --output-dir "${OUTPUT_DIR}/phase2_no_kg_output/"

echo "[Phase 2B] Complete. Output: ${OUTPUT_DIR}/phase2_no_kg_output/"
echo ""

# ============================================================================
# STEP 8B: Criterion-level comparison (OptmusKG vs no-KG)
# ============================================================================
echo "[Step 8b] Running criterion-level comparison..."
"${VENV_PYTHON}" simple-evals/scripts/compare_kg_runs_criterion_level.py \
  --kg-graded "${OUTPUT_DIR}/phase2_kg_output/phase1_ark_kg_grounded_grading_checkpoint.jsonl" \
  --no-kg-graded "${OUTPUT_DIR}/phase2_no_kg_output/phase1_no_kg_grading_checkpoint.jsonl" \
  --output "${OUTPUT_DIR}/step8b_criterion_level.jsonl"

echo "[Step 8b] Complete. Output: ${OUTPUT_DIR}/step8b_criterion_level.jsonl"
echo ""

# ============================================================================
# STEP 9: Binary labels (USEKG vs DONTUSEKG per question)
# ============================================================================
echo "[Step 9] Generating binary labels..."
"${VENV_PYTHON}" simple-evals/scripts/build_kg_binary_labels.py \
  --criterion-level "${OUTPUT_DIR}/step8b_criterion_level.jsonl" \
  --output "${OUTPUT_DIR}/step9_labels.jsonl"

echo "[Step 9] Complete. Output: ${OUTPUT_DIR}/step9_labels.jsonl"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "✅ LEVEL 3 VALIDATION COMPLETE"
echo "=========================================="
echo "Results directory: ${OUTPUT_DIR}"
echo ""
echo "Outputs:"
echo "  Phase 1A (ARK+OptmusKG):  ${PHASE1_KG}"
echo "  Phase 1B (Pure GPT):  ${PHASE1_NO_KG}"
echo "  Phase 2A (KG graded): ${OUTPUT_DIR}/phase2_kg_output/"
echo "  Phase 2B (No-KG graded): ${OUTPUT_DIR}/phase2_no_kg_output/"
echo "  Step 8b (criterion comparison): ${OUTPUT_DIR}/step8b_criterion_level.jsonl"
echo "  Step 9 (binary labels): ${OUTPUT_DIR}/step9_labels.jsonl"
echo ""
echo "Analysis:"
echo "  Analyze ${OUTPUT_DIR}/step9_labels.jsonl to see KG benefit distribution:"
echo "  - USEKG: Questions where ARK+OptmusKG scored better than pure GPT"
echo "  - DONTUSEKG: Questions where pure GPT scored better or equal"
echo ""
python << 'EOF'
import json
import sys
from pathlib import Path

labels_file = Path(sys.argv[1])
if labels_file.exists():
    with open(labels_file) as f:
        labels = [json.loads(line) for line in f if line.strip()]

    usekg = sum(1 for l in labels if l.get('use_kg_label') == 'USEKG')
    dontusekg = sum(1 for l in labels if l.get('use_kg_label') == 'DONTUSEKG')
    total = len(labels)

    print(f"KG Benefit Summary:")
    print(f"  USEKG: {usekg} ({100*usekg/total:.1f}%)")
    print(f"  DONTUSEKG: {dontusekg} ({100*dontusekg/total:.1f}%)")
    print(f"  Total: {total} questions")
EOF
"${OUTPUT_DIR}/step9_labels.jsonl"
