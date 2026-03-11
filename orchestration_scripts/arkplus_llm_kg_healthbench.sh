#!/bin/bash
set -e

################################################################################
# arkplus_llm_kg_healthbench.sh
# Full pipeline: Phase 1A (KG) + Phase 1B (Baseline) + Phase 2A + 2B + Part 5 + 6
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

################################################################################
# ARGUMENT VALIDATION
################################################################################

if [ $# -lt 6 ]; then
    echo -e "${RED}Usage:${NC}"
    echo "  bash arkplus_llm_kg_healthbench.sh <OUTPUT_DIR> <LLM_MODEL> <GRAPH_NAME> <SEARCH_MODE> <LIMIT> <INPUT_JSONL>"
    echo ""
    echo "Example:"
    echo "  bash arkplus_llm_kg_healthbench.sh \\"
    echo "    /results/5000q_sweeps_mar10/run_001_gpt41_optimus_hybrid \\"
    echo "    'azure/gpt-4.1' \\"
    echo "    optimus \\"
    echo "    hybrid \\"
    echo "    5000 \\"
    echo "    /data/2025-05-07-06-14-12_oss_eval.jsonl"
    exit 1
fi

OUTPUT_DIR="$1"
LLM_MODEL="$2"
GRAPH_NAME="$3"
SEARCH_MODE="$4"
LIMIT="$5"
INPUT_JSONL="$6"

# Defaults
ARK_DIR="${ARK_DIR:-/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/ark}"
VENV_PYTHON="${VENV_PYTHON:-python}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}ARKPLUS FULL PIPELINE${NC}"
echo -e "${BLUE}Model: ${LLM_MODEL} | Graph: ${GRAPH_NAME} | Search: ${SEARCH_MODE} | Limit: ${LIMIT}${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

################################################################################
# PHASE 1A: KG-GROUNDED RUN
################################################################################

echo -e "${BLUE}PHASE 1A: KG-GROUNDED RESPONSES${NC}"
echo ""

PHASE1A_OUTPUT="${OUTPUT_DIR}/phase1a_kg_responses.jsonl"

$VENV_PYTHON -m simple_evals.scripts.run_ark_on_healthbench \
    --input-jsonl "$INPUT_JSONL" \
    --output-jsonl "$PHASE1A_OUTPUT" \
    --ark-dir "$ARK_DIR" \
    --graph-name "$GRAPH_NAME" \
    --search-mode "$SEARCH_MODE" \
    --ark-model "$LLM_MODEL" \
    --ark-agents 3 \
    --ark-max-steps 20 \
    --limit "$LIMIT" \
    --run-tag "kg_grounded" \
    --log-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/phase1a_kg.log"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Phase 1A FAILED${NC}"
    exit 1
fi

PHASE1A_LINES=$(wc -l < "$PHASE1A_OUTPUT")
echo -e "${GREEN}✓ Phase 1A complete: ${PHASE1A_LINES} questions${NC}"
echo ""

################################################################################
# PHASE 1B: BASELINE RUN
################################################################################

echo -e "${BLUE}PHASE 1B: BASELINE RESPONSES (NO KG)${NC}"
echo ""

PHASE1B_OUTPUT="${OUTPUT_DIR}/phase1b_baseline_responses.jsonl"

$VENV_PYTHON -m simple_evals.scripts.run_baseline_on_healthbench \
    --input-jsonl "$INPUT_JSONL" \
    --output-jsonl "$PHASE1B_OUTPUT" \
    --llm-model "$LLM_MODEL" \
    --limit "$LIMIT" \
    --run-tag "baseline" \
    --log-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/phase1b_baseline.log"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Phase 1B FAILED${NC}"
    exit 1
fi

PHASE1B_LINES=$(wc -l < "$PHASE1B_OUTPUT")
echo -e "${GREEN}✓ Phase 1B complete: ${PHASE1B_LINES} questions${NC}"
echo ""

################################################################################
# PHASE 2A: GRADE KG-GROUNDED RESPONSES
################################################################################

echo -e "${BLUE}PHASE 2A: GRADE KG-GROUNDED RESPONSES${NC}"
echo ""

$VENV_PYTHON -m simple_evals.scripts.grade_ark_healthbench_responses \
    --responses-jsonl "$PHASE1A_OUTPUT" \
    --output-dir "$OUTPUT_DIR" \
    --examples "$LIMIT" \
    --n-workers-phase2a 4 \
    --skip-on-error \
    --log-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/phase2a_kg.log"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Phase 2A FAILED${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Phase 2A complete${NC}"
echo ""

################################################################################
# PHASE 2B: GRADE BASELINE RESPONSES
################################################################################

echo -e "${BLUE}PHASE 2B: GRADE BASELINE RESPONSES${NC}"
echo ""

$VENV_PYTHON -m simple_evals.scripts.grade_ark_healthbench_responses \
    --responses-jsonl "$PHASE1B_OUTPUT" \
    --output-dir "$OUTPUT_DIR" \
    --examples "$LIMIT" \
    --n-workers-phase2a 4 \
    --skip-on-error \
    --log-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/phase2b_baseline.log"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Phase 2B FAILED${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Phase 2B complete${NC}"
echo ""

################################################################################
# PART 5: AGGREGATE TO QUESTION LEVEL
################################################################################

echo -e "${BLUE}PART 5: QUESTION-LEVEL AGGREGATION${NC}"
echo ""

$VENV_PYTHON -m simple_evals.scripts.build_part5_question_metadata \
    --output-dir "$OUTPUT_DIR" \
    --limit "$LIMIT" \
    2>&1 | tee "${OUTPUT_DIR}/part5_build.log"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Part 5 FAILED${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Part 5 complete${NC}"
echo ""

################################################################################
# PART 6: MERGE ALL PHASES
################################################################################

echo -e "${BLUE}PART 6: MERGE ALL PHASES (FINAL OUTPUT)${NC}"
echo ""

$VENV_PYTHON -m simple_evals.scripts.build_part6_complete_metadata \
    --output-dir "$OUTPUT_DIR" \
    --limit "$LIMIT" \
    2>&1 | tee "${OUTPUT_DIR}/part6_build.log"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Part 6 FAILED${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Part 6 complete${NC}"
echo ""

################################################################################
# SUCCESS
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ FULL PIPELINE COMPLETE${NC}"
echo -e "${BLUE}Output: ${OUTPUT_DIR}${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit 0
