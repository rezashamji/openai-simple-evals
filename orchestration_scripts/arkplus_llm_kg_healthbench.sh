#!/bin/bash
set -e

################################################################################
# arkplus_llm_kg_healthbench.sh
# KG-GROUNDED + BASELINE PIPELINE
# Full pipeline: Phase 1A (KG) + Phase 1B (Baseline) + Phase 2A + 2B + Part 5 + 6
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
if [ $# -ge 6 ]; then
    # Command-line mode (for wrapper scripts calling this directly)
    OUTPUT_DIR="$1"
    LLM_MODEL="$2"
    GRAPH_NAME="$3"
    SEARCH_MODE="$4"
    LIMIT="$5"
    INPUT_JSONL="$6"
else
    # Editable config mode (for direct sbatch submission)
    LLM_MODEL="azure/gpt-5.4"                                           # Change this
    GRAPH_NAME="optimus"                                                # Change this (optimus, prime, etc.)
    SEARCH_MODE="hybrid"                                                # Change this (hybrid, embedding, bm25)
    LIMIT=5000                                                          # Change this
    REASONING_EFFORT="medium"                                           # Change this (none, low, medium, high, xhigh)
    INPUT_JSONL="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/2025-05-07-06-14-12_oss_eval.jsonl"
    BASE_OUTPUT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/arkplus_evals"

    # Auto-generate folder name: MODEL_KG_SEARCH_kg (for standalone KG runs)
    MODEL_FOLDER=$(echo "$LLM_MODEL" | sed 's/.*\///; s/\./-/g; s/-//')
    OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODEL_FOLDER}_${GRAPH_NAME}_${SEARCH_MODE}_kg_standalone"
fi

# Defaults
ARK_DIR="${ARK_DIR:-/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/ark}"
VENV_PYTHON="${VENV_PYTHON:-python}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Export REASONING_EFFORT so all Python subprocesses (ARK agents, nodes_to_nl, grader, judge) pick it up
export REASONING_EFFORT="${REASONING_EFFORT:-medium}"

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
echo -e "${BLUE}ARKPLUS FULL PIPELINE${NC}"
echo -e "${BLUE}Model: ${LLM_MODEL} | Graph: ${GRAPH_NAME} | Search: ${SEARCH_MODE} | Limit: ${LIMIT}${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

log_audit "pipeline" "start" "in_progress" ", \"model\": \"$LLM_MODEL\", \"graph\": \"$GRAPH_NAME\", \"search_mode\": \"$SEARCH_MODE\", \"limit\": $LIMIT, \"output_dir\": \"$OUTPUT_DIR\""

################################################################################
# PHASE 1A: KG-GROUNDED RUN
################################################################################

echo -e "${BLUE}PHASE 1A: KG-GROUNDED RESPONSES${NC}"
echo ""

PHASE1A_OUTPUT="${OUTPUT_DIR}/phase1a_kg_responses.jsonl"

# Skip Phase 1A if already completed
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
        --graph-name "$GRAPH_NAME" \
        --search-mode "$SEARCH_MODE" \
        --ark-model "$LLM_MODEL" \
        --ark-agents 3 \
        --ark-max-steps 20 \
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
PHASE2A_EXISTING=$(wc -l "$PHASE2A_CHECKPOINT" 2>/dev/null | awk '{print $1}' || echo 0)

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
PHASE2B_EXISTING=$(wc -l "$PHASE2B_CHECKPOINT" 2>/dev/null | awk '{print $1}' || echo 0)

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
# PART 5: AGGREGATE TO QUESTION LEVEL
################################################################################

echo -e "${BLUE}PART 5: QUESTION-LEVEL AGGREGATION${NC}"
echo ""

PART5_OUTPUT="${OUTPUT_DIR}/step_5_output.jsonl"

if [ -f "$PART5_OUTPUT" ]; then
    PART5_LINES=$(wc -l < "$PART5_OUTPUT")
    echo -e "${GREEN}✓ Part 5 already complete: ${PART5_LINES} lines (skipping)${NC}"
    log_audit "5" "skipped" "success" ", \"reason\": \"output file exists\", \"lines\": $PART5_LINES"
else
    log_audit "5" "start" "in_progress" ""

    $VENV_PYTHON -m simple_evals.scripts.build_part5_question_metadata \
        --grading-output "${OUTPUT_DIR}/phase1a_kg_responses_grading_checkpoint.jsonl" \
        --output "$PART5_OUTPUT" \
        --limit "$LIMIT" \
        2>&1 | tee "${OUTPUT_DIR}/part5_build.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Part 5 FAILED${NC}"
        PART5_LINES=$(wc -l "$PART5_OUTPUT" 2>/dev/null | awk '{print $1}' || echo 0)
        log_audit "5" "end" "failed" ", \"lines_written\": $PART5_LINES"
        exit 1
    fi

    PART5_LINES=$(wc -l < "$PART5_OUTPUT")
    echo -e "${GREEN}✓ Part 5 complete${NC}"
    log_audit "5" "end" "success" ", \"lines_written\": $PART5_LINES"
fi
echo ""

################################################################################
# PART 6: MERGE ALL PHASES
################################################################################

echo -e "${BLUE}PART 6: MERGE ALL PHASES (FINAL OUTPUT)${NC}"
echo ""

PART6_OUTPUT="${OUTPUT_DIR}/part6_complete_metadata.jsonl"

if [ -f "$PART6_OUTPUT" ]; then
    PART6_LINES=$(wc -l < "$PART6_OUTPUT")
    echo -e "${GREEN}✓ Part 6 already complete: ${PART6_LINES} lines (skipping)${NC}"
    log_audit "6" "skipped" "success" ", \"reason\": \"output file exists\", \"lines\": $PART6_LINES"
else
    log_audit "6" "start" "in_progress" ""

    $VENV_PYTHON -m simple_evals.scripts.build_part6_complete_metadata \
        --phase1-kg "${OUTPUT_DIR}/phase1a_kg_responses.jsonl" \
        --phase1-baseline "${OUTPUT_DIR}/phase1b_baseline_responses.jsonl" \
        --phase2a-grading "${OUTPUT_DIR}/phase1a_kg_responses_grading_checkpoint.jsonl" \
        --phase2b-grading "${OUTPUT_DIR}/phase1b_baseline_responses_grading_checkpoint.jsonl" \
        --phase5-analysis "${OUTPUT_DIR}/step_5_output.jsonl" \
        --output "$PART6_OUTPUT" \
        2>&1 | tee "${OUTPUT_DIR}/part6_build.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Part 6 FAILED${NC}"
        PART6_LINES=$(wc -l "$PART6_OUTPUT" 2>/dev/null | awk '{print $1}' || echo 0)
        log_audit "6" "end" "failed" ", \"lines_written\": $PART6_LINES"
        exit 1
    fi

    PART6_LINES=$(wc -l < "$PART6_OUTPUT")
    echo -e "${GREEN}✓ Part 6 complete${NC}"
    log_audit "6" "end" "success" ", \"lines_written\": $PART6_LINES"
fi
echo ""

################################################################################
# SUCCESS
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ FULL PIPELINE COMPLETE${NC}"
echo -e "${BLUE}Output: ${OUTPUT_DIR}${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit 0
