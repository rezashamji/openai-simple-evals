#!/bin/bash
set -e

################################################################################
# llm_healthbench.sh
# BASELINE-ONLY PIPELINE (NO KG)
# Pipeline: Phase 1B (Baseline) + Phase 2B + Part 5 + 6
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

################################################################################
# CONFIGURATION - EDIT THESE OR PASS AS COMMAND-LINE ARGUMENTS
################################################################################

# If command-line args provided, use them; otherwise use defaults below
if [ $# -ge 4 ]; then
    # Command-line mode (for wrapper scripts calling this directly)
    OUTPUT_DIR="$1"
    LLM_MODEL="$2"
    LIMIT="$3"
    INPUT_JSONL="$4"
else
    # Editable config mode (for direct sbatch submission)
    LLM_MODEL="azure/gpt-5.4"                                           # Change this
    LIMIT=5000                                                          # Change this
    REASONING_EFFORT="medium"                                           # Change this (none, low, medium, high, xhigh)
    INPUT_JSONL="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/2025-05-07-06-14-12_oss_eval.jsonl"
    BASE_OUTPUT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/arkplus_evals"

    # Auto-generate folder name: MODEL_baseline_standalone (for standalone baseline runs)
    MODEL_FOLDER=$(echo "$LLM_MODEL" | sed 's/.*\///; s/\./-/g; s/-//')
    OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODEL_FOLDER}_baseline_standalone"
fi

# Defaults
VENV_PYTHON="${VENV_PYTHON:-python}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Export REASONING_EFFORT so all Python subprocesses (baseline model, grader, judge) pick it up
export REASONING_EFFORT="${REASONING_EFFORT:-medium}"

################################################################################
# VALIDATION
################################################################################

if [ ! -f "$INPUT_JSONL" ]; then
    echo -e "${RED}✗ Input file not found: $INPUT_JSONL${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Change to project root to avoid venv .pth shadowing types.py at simple-evals root
PROJECT_ROOT=$(dirname "$VENV_PYTHON")/../../..
cd "$PROJECT_ROOT"

# Setup audit logging
AUDIT_FILE="${OUTPUT_DIR}/run_audit.jsonl"
SLURM_JOB_ID=${SLURM_JOB_ID:-"manual"}

# Helper function to log to audit file
log_audit() {
    local phase=$1
    local event=$2
    local status=$3
    local extra=$4

    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"job_id\": \"$SLURM_JOB_ID\", \"phase\": \"$phase\", \"event\": \"$event\", \"status\": \"$status\", \"timestamp\": \"$timestamp\"$extra}" >> "$AUDIT_FILE"
}

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}BASELINE-ONLY PIPELINE${NC}"
echo -e "${BLUE}Model: ${LLM_MODEL} | Limit: ${LIMIT}${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

################################################################################
# PHASE 1B: BASELINE RUN
################################################################################

echo -e "${BLUE}PHASE 1B: BASELINE RESPONSES (NO KG)${NC}"
echo ""

PHASE1B_OUTPUT="${OUTPUT_DIR}/phase1b_baseline_responses.jsonl"

# Skip Phase 1B if already completed
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
        --n-workers 16 \
        --limit "$LIMIT" \
        2>&1 | tee "${OUTPUT_DIR}/phase1b_baseline.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Phase 1B FAILED${NC}"
        PHASE1B_LINES=$(wc -l < "$PHASE1B_OUTPUT" 2>/dev/null || echo 0)
        log_audit "1b" "end" "failed" ", \"lines_written\": $PHASE1B_LINES"
        exit 1
    fi

    PHASE1B_LINES=$(wc -l < "$PHASE1B_OUTPUT")
    echo -e "${GREEN}✓ Phase 1B complete: ${PHASE1B_LINES} questions${NC}"
    log_audit "1b" "end" "success" ", \"lines_written\": $PHASE1B_LINES"
fi
echo ""

################################################################################
# PHASE 2B: GRADE BASELINE RESPONSES
################################################################################

echo -e "${BLUE}PHASE 2B: GRADE BASELINE RESPONSES${NC}"
echo ""

log_audit "2b" "start" "in_progress" ""

$VENV_PYTHON -m simple_evals.scripts.grade_ark_healthbench_responses \
    --responses-jsonl "$PHASE1B_OUTPUT" \
    --output-dir "$OUTPUT_DIR" \
    --examples "$LIMIT" \
    --n-workers-phase2a 4 \
    --skip-on-error \
    --log-interval 100 \
    2>&1 | tee "${OUTPUT_DIR}/phase2b_baseline.log"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Phase 2B FAILED${NC}"
    CHECKPOINT_LINES=$(wc -l < "${OUTPUT_DIR}/phase1b_baseline_responses_grading_checkpoint.jsonl" 2>/dev/null || echo 0)
    log_audit "2b" "end" "failed" ", \"checkpoint_lines\": $CHECKPOINT_LINES, \"expected_lines\": $LIMIT"
    exit 1
fi

CHECKPOINT_LINES=$(wc -l < "${OUTPUT_DIR}/phase1b_baseline_responses_grading_checkpoint.jsonl" 2>/dev/null || echo 0)
echo -e "${GREEN}✓ Phase 2B complete${NC}"
log_audit "2b" "end" "success" ", \"graded_lines\": $CHECKPOINT_LINES"
echo ""

################################################################################
# SUCCESS
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ BASELINE PIPELINE COMPLETE (Phase 1B + 2B)${NC}"
echo -e "${BLUE}Output: ${OUTPUT_DIR}${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit 0
