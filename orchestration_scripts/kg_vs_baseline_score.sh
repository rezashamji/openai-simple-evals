#!/bin/bash
################################################################################
# kg_vs_baseline_score.sh
# KG VS BASELINE SCORING PIPELINE (NO NODE VALIDATION)
# Phase 1A (KG) + Phase 1B (Baseline) + Phase 2A + 2B
# Final output: per-question USEKG / DONTUSEKG labels based on score delta
################################################################################

#SBATCH --job-name=kg-vs-baseline
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/arkplus_evals/kg_vs_baseline_slurm_%j.out
#SBATCH --error=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/arkplus_evals/kg_vs_baseline_slurm_%j.err
#SBATCH --gres=gpu:1

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

################################################################################
# CONFIGURATION - EDIT THESE FOR DIFFERENT RUNS
################################################################################
LLM_MODEL="azure/gpt-5.4"          # Change this
KG_NAME="optimus"                  # Change this (optimus, prime, etc.)
SEARCH_MODE="embedding"            # Change this (hybrid, embedding, bm25)
LIMIT=5000                            # Change this
REASONING_EFFORT="medium"          # Change this (none, low, medium, high, xhigh)

################################################################################
# FIXED PATHS
################################################################################
PROJECT_ROOT="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji"
BASE_OUTPUT_DIR="${PROJECT_ROOT}/simple-evals/results/arkplus_evals"
INPUT_JSONL="${PROJECT_ROOT}/simple-evals/2025-05-07-06-14-12_oss_eval.jsonl"
ARK_DIR="${PROJECT_ROOT}/ark"
VENV_PYTHON="${PROJECT_ROOT}/simple-evals/.venv/bin/python"
ARK_PYTHON="${PROJECT_ROOT}/ark/.venv/bin/python"

# Generate output folder
MODEL_FOLDER=$(echo "$LLM_MODEL" | sed 's/.*\///; s/\./-/g; s/-//')
JOB_ID=${SLURM_JOB_ID:-"local"}
OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODEL_FOLDER}_${KG_NAME}_${SEARCH_MODE}_kg_vs_baseline_${JOB_ID}"

################################################################################
# ENVIRONMENT SETUP
################################################################################

# Force LiteLLM to handle Azure 429s automatically
export LITELLM_RETRY_COUNT=10
export LITELLM_RETRY_DELAY=8

# Help PyTorch manage memory fragmentation for the large embeddings matrix
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Ensure workers only see the one GPU allocated by Slurm
export CUDA_VISIBLE_DEVICES=0

# Export REASONING_EFFORT so all Python subprocesses pick it up
export REASONING_EFFORT

# Load Azure credentials
if [ ! -f "${PROJECT_ROOT}/.env" ]; then
    echo "ERROR: .env file not found at ${PROJECT_ROOT}/.env"
    exit 1
fi

set -a
source "${PROJECT_ROOT}/.env"
set +a

if [ -z "$AZURE_API_KEY" ] || [ -z "$AZURE_API_BASE" ]; then
    echo "ERROR: Azure credentials not set in .env"
    exit 1
fi

export VENV_PYTHON ARK_PYTHON PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

################################################################################
# VALIDATION
################################################################################

if [ ! -f "$INPUT_JSONL" ]; then
    echo -e "${RED}✗ Input file not found: $INPUT_JSONL${NC}"
    exit 1
fi

if [ ! -d "$ARK_DIR" ]; then
    echo -e "${RED}✗ ARK directory not found: $ARK_DIR${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Change to project root to avoid venv .pth shadowing types.py at simple-evals root
cd "$PROJECT_ROOT"

# Setup audit logging
AUDIT_FILE="${OUTPUT_DIR}/run_audit.jsonl"
SLURM_JOB_ID=${SLURM_JOB_ID:-"manual"}

log_audit() {
    local phase=$1 event=$2 status=$3 extra=$4
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"job_id\": \"$SLURM_JOB_ID\", \"phase\": \"$phase\", \"event\": \"$event\", \"status\": \"$status\", \"timestamp\": \"$timestamp\"$extra}" >> "$AUDIT_FILE"
}

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}KG VS BASELINE SCORING PIPELINE${NC}"
echo -e "${BLUE}Model: ${LLM_MODEL} | Graph: ${KG_NAME} | Search: ${SEARCH_MODE} | Limit: ${LIMIT}${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

log_audit "pipeline" "start" "in_progress" ", \"model\": \"$LLM_MODEL\", \"graph\": \"$KG_NAME\", \"search_mode\": \"$SEARCH_MODE\", \"limit\": $LIMIT, \"output_dir\": \"$OUTPUT_DIR\""

################################################################################
# PHASE 1A: KG-GROUNDED RUN
################################################################################

echo -e "${BLUE}PHASE 1A: KG-GROUNDED RESPONSES${NC}"
echo ""

PHASE1A_OUTPUT="${OUTPUT_DIR}/phase1a_kg_responses.jsonl"

if [ -f "$PHASE1A_OUTPUT" ]; then
    PHASE1A_LINES=$(wc -l < "$PHASE1A_OUTPUT")
    echo -e "${GREEN}✓ Phase 1A already complete: ${PHASE1A_LINES} questions (skipping)${NC}"
    log_audit "1a" "skipped" "success" ", \"reason\": \"output file exists\", \"lines\": $PHASE1A_LINES"
else
    log_audit "1a" "start" "in_progress" ""

    $VENV_PYTHON -m simple_evals.scripts.run_ark_on_healthbench \
        --input-jsonl "$INPUT_JSONL" \
        --output-jsonl "$PHASE1A_OUTPUT" \
        --ark-dir "$ARK_DIR" \
        --graph-name "$KG_NAME" \
        --search-mode "$SEARCH_MODE" \
        --ark-model "$LLM_MODEL" \
        --ark-agents 2 \
        --ark-max-steps 10 \
        --n-workers 2 \
        --limit "$LIMIT" \
        --run-tag "kg_grounded" \
        --log-interval 100 \
        2>&1 | tee "${OUTPUT_DIR}/phase1a_kg.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Phase 1A FAILED${NC}"
        PHASE1A_LINES=$(wc -l "$PHASE1A_OUTPUT" 2>/dev/null | awk '{print $1}' || echo 0)
        log_audit "1a" "end" "failed" ", \"lines_written\": $PHASE1A_LINES"
        exit 1
    fi

    PHASE1A_LINES=$(wc -l < "$PHASE1A_OUTPUT")
    echo -e "${GREEN}✓ Phase 1A complete: ${PHASE1A_LINES} questions${NC}"
    log_audit "1a" "end" "success" ", \"lines_written\": $PHASE1A_LINES"
fi
echo ""

# Let Azure token quota recover between Phase 1A and 1B
sleep 60

################################################################################
# PHASE 1B: BASELINE RUN
################################################################################

echo -e "${BLUE}PHASE 1B: BASELINE RESPONSES (NO KG)${NC}"
echo ""

PHASE1B_OUTPUT="${OUTPUT_DIR}/phase1b_baseline_responses.jsonl"

if [ -f "$PHASE1B_OUTPUT" ]; then
    PHASE1B_LINES=$(wc -l < "$PHASE1B_OUTPUT")
    echo -e "${GREEN}✓ Phase 1B already complete: ${PHASE1B_LINES} questions (skipping)${NC}"
    log_audit "1b" "skipped" "success" ", \"reason\": \"output file exists\", \"lines\": $PHASE1B_LINES"
else
    log_audit "1b" "start" "in_progress" ""

    $VENV_PYTHON -m simple_evals.scripts.run_baseline_on_healthbench \
        --examples-jsonl "$INPUT_JSONL" \
        --output-path "$PHASE1B_OUTPUT" \
        --model "$LLM_MODEL" \
        --limit "$LIMIT" \
        2>&1 | tee "${OUTPUT_DIR}/phase1b_baseline.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Phase 1B FAILED${NC}"
        PHASE1B_LINES=$(wc -l "$PHASE1B_OUTPUT" 2>/dev/null | awk '{print $1}' || echo 0)
        log_audit "1b" "end" "failed" ", \"lines_written\": $PHASE1B_LINES"
        exit 1
    fi

    PHASE1B_LINES=$(wc -l < "$PHASE1B_OUTPUT")
    echo -e "${GREEN}✓ Phase 1B complete: ${PHASE1B_LINES} questions${NC}"
    log_audit "1b" "end" "success" ", \"lines_written\": $PHASE1B_LINES"
fi
echo ""

################################################################################
# PHASE 2A: GRADE KG-GROUNDED RESPONSES
################################################################################

echo -e "${BLUE}PHASE 2A: GRADE KG-GROUNDED RESPONSES${NC}"
echo ""

PHASE2A_CHECKPOINT="${OUTPUT_DIR}/phase1a_kg_responses_grading_checkpoint.jsonl"
PHASE2A_EXISTING=$(wc -l "$PHASE2A_CHECKPOINT" 2>/dev/null | awk '{print $1}'); PHASE2A_EXISTING=${PHASE2A_EXISTING:-0}

if [ "$PHASE2A_EXISTING" -ge "$LIMIT" ]; then
    echo -e "${GREEN}✓ Phase 2A already complete: ${PHASE2A_EXISTING} lines (skipping)${NC}"
    log_audit "2a" "skipped" "success" ", \"reason\": \"checkpoint complete\", \"lines\": $PHASE2A_EXISTING"
else
    log_audit "2a" "start" "in_progress" ", \"checkpoint_lines_at_start\": $PHASE2A_EXISTING, \"expected_lines\": $LIMIT"

    $VENV_PYTHON -m simple_evals.scripts.grade_ark_healthbench_responses \
        --responses-jsonl "$PHASE1A_OUTPUT" \
        --output-dir "$OUTPUT_DIR" \
        --examples "$LIMIT" \
        --n-workers-phase2a 2 \
        --skip-on-error \
        --log-interval 100 \
        2>&1 | tee "${OUTPUT_DIR}/phase2a_kg.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Phase 2A FAILED${NC}"
        CHECKPOINT_LINES=$(wc -l "$PHASE2A_CHECKPOINT" 2>/dev/null | awk '{print $1}' || echo 0)
        log_audit "2a" "end" "failed" ", \"checkpoint_lines\": $CHECKPOINT_LINES, \"expected_lines\": $LIMIT"
        exit 1
    fi

    CHECKPOINT_LINES=$(wc -l "$PHASE2A_CHECKPOINT" 2>/dev/null | awk '{print $1}' || echo 0)
    echo -e "${GREEN}✓ Phase 2A complete${NC}"
    log_audit "2a" "end" "success" ", \"graded_lines\": $CHECKPOINT_LINES"
fi
echo ""

################################################################################
# PHASE 2B: GRADE BASELINE RESPONSES
################################################################################

echo -e "${BLUE}PHASE 2B: GRADE BASELINE RESPONSES${NC}"
echo ""

PHASE2B_CHECKPOINT="${OUTPUT_DIR}/phase1b_baseline_responses_grading_checkpoint.jsonl"
PHASE2B_EXISTING=$(wc -l "$PHASE2B_CHECKPOINT" 2>/dev/null | awk '{print $1}'); PHASE2B_EXISTING=${PHASE2B_EXISTING:-0}

if [ "$PHASE2B_EXISTING" -ge "$LIMIT" ]; then
    echo -e "${GREEN}✓ Phase 2B already complete: ${PHASE2B_EXISTING} lines (skipping)${NC}"
    log_audit "2b" "skipped" "success" ", \"reason\": \"checkpoint complete\", \"lines\": $PHASE2B_EXISTING"
else
    log_audit "2b" "start" "in_progress" ", \"checkpoint_lines_at_start\": $PHASE2B_EXISTING, \"expected_lines\": $LIMIT"

    $VENV_PYTHON -m simple_evals.scripts.grade_ark_healthbench_responses \
        --responses-jsonl "$PHASE1B_OUTPUT" \
        --output-dir "$OUTPUT_DIR" \
        --examples "$LIMIT" \
        --n-workers-phase2a 2 \
        --skip-on-error \
        --log-interval 100 \
        2>&1 | tee "${OUTPUT_DIR}/phase2b_baseline.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Phase 2B FAILED${NC}"
        CHECKPOINT_LINES=$(wc -l "$PHASE2B_CHECKPOINT" 2>/dev/null | awk '{print $1}' || echo 0)
        log_audit "2b" "end" "failed" ", \"checkpoint_lines\": $CHECKPOINT_LINES, \"expected_lines\": $LIMIT"
        exit 1
    fi

    CHECKPOINT_LINES=$(wc -l "$PHASE2B_CHECKPOINT" 2>/dev/null | awk '{print $1}' || echo 0)
    echo -e "${GREEN}✓ Phase 2B complete${NC}"
    log_audit "2b" "end" "success" ", \"graded_lines\": $CHECKPOINT_LINES"
fi
echo ""

################################################################################
# FINAL: COMPUTE PER-QUESTION USEKG / DONTUSEKG LABELS
################################################################################

echo -e "${BLUE}FINAL: COMPUTING PER-QUESTION KG VS BASELINE LABELS${NC}"
echo ""

LABELS_OUTPUT="${OUTPUT_DIR}/kg_routing_labels.jsonl"

$VENV_PYTHON - <<EOF
import json, sys

kg_file = "$PHASE2A_CHECKPOINT"
baseline_file = "$PHASE2B_CHECKPOINT"
output_file = "$LABELS_OUTPUT"

def load_by_id(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("example_level_metadata", {}).get("prompt_id") or rec.get("index")
            if qid is not None:
                data[qid] = rec
    return data

kg_data = load_by_id(kg_file)
baseline_data = load_by_id(baseline_file)

matched = set(kg_data.keys()) & set(baseline_data.keys())
print(f"Matched {len(matched)} questions across KG and baseline checkpoints", file=sys.stderr)

use_kg = 0
dont_use_kg = 0
results = []

for qid in sorted(matched, key=lambda x: str(x)):
    kg_score = kg_data[qid].get("score")
    baseline_score = baseline_data[qid].get("score")
    if kg_score is None or baseline_score is None:
        continue
    delta = kg_score - baseline_score
    label = "USEKG" if delta > 0 else "DONTUSEKG"
    if label == "USEKG":
        use_kg += 1
    else:
        dont_use_kg += 1
    results.append({
        "question_id": qid,
        "kg_score": kg_score,
        "baseline_score": baseline_score,
        "delta": round(delta, 6),
        "label": label,
    })

with open(output_file, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

avg_kg = sum(r["kg_score"] for r in results) / len(results) if results else 0
avg_baseline = sum(r["baseline_score"] for r in results) / len(results) if results else 0
avg_delta = sum(r["delta"] for r in results) / len(results) if results else 0

print(f"")
print(f"{'─'*60}")
print(f"  KG score (avg):       {avg_kg:.4f}")
print(f"  Baseline score (avg): {avg_baseline:.4f}")
print(f"  Avg delta:            {avg_delta:+.4f}")
print(f"  USEKG:                {use_kg} questions")
print(f"  DONTUSEKG:            {dont_use_kg} questions")
print(f"{'─'*60}")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Label generation FAILED${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Labels written to: ${LABELS_OUTPUT}${NC}"
log_audit "labels" "end" "success" ", \"output\": \"$LABELS_OUTPUT\""
echo ""

################################################################################
# SUCCESS
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ PIPELINE COMPLETE${NC}"
echo -e "${BLUE}Output: ${OUTPUT_DIR}${NC}"
echo -e "${BLUE}Labels: ${LABELS_OUTPUT}${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit 0
