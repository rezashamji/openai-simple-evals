#!/bin/bash
set -e

################################################################################
# orchestrate_5000q_sweeps.sh
# Master orchestrator for all 7 evaluation runs
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

################################################################################
# CONFIGURATION
################################################################################

BASE_OUTPUT_DIR="${1:-/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/5000q_sweeps_mar10}"
DRY_RUN="${2:-false}"

INPUT_JSONL="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/2025-05-07-06-14-12_oss_eval.jsonl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT=5000
LLM_MODEL="azure/gpt-4.1"

################################################################################
# VALIDATION
################################################################################

if [ ! -f "$INPUT_JSONL" ]; then
    echo -e "${RED}✗ Input file not found: $INPUT_JSONL${NC}"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/arkplus_llm_kg_healthbench.sh" ]; then
    echo -e "${RED}✗ Script not found: ${SCRIPT_DIR}/arkplus_llm_kg_healthbench.sh${NC}"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/llm_healthbench.sh" ]; then
    echo -e "${RED}✗ Script not found: ${SCRIPT_DIR}/llm_healthbench.sh${NC}"
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"

################################################################################
# HEADER
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}ORCHESTRATING 7 × 5000-QUESTION EVALUATION RUNS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Output Base: ${BASE_OUTPUT_DIR}"
echo "  Input File:  ${INPUT_JSONL}"
echo "  Limit:       ${LIMIT} questions per run"
echo "  Model:       ${LLM_MODEL}"
if [ "$DRY_RUN" = "true" ]; then
    echo -e "${YELLOW}  Mode:        DRY RUN (no execution)${NC}"
fi
echo ""

################################################################################
# RUN DEFINITIONS
################################################################################

declare -a RUNS=(
    "run_001_gpt41_optimus_hybrid|arkplus|optimus|hybrid"
    "run_002_gpt41_optimus_hybrid_baseline|baseline|||"
    "run_003_gpt41_optimus_embedding|arkplus|optimus|embedding"
    "run_004_gpt41_optimus_embedding_baseline|baseline|||"
    "run_005_gpt41_optimus_bm25|arkplus|optimus|bm25"
    "run_006_gpt41_optimus_bm25_baseline|baseline|||"
    "run_007_gpt41_prime_hybrid|arkplus|prime|hybrid"
)

TOTAL_RUNS=${#RUNS[@]}
PASSED=0
FAILED=0
FAILED_RUNS=()

################################################################################
# EXECUTE RUNS
################################################################################

for i in "${!RUNS[@]}"; do
    RUN_NUM=$((i + 1))
    IFS='|' read -r RUN_NAME RUN_TYPE GRAPH SEARCH_MODE <<< "${RUNS[$i]}"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}RUN ${RUN_NUM}/${TOTAL_RUNS}: ${RUN_NAME}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}DRY RUN:${NC} Would execute:"
        if [ "$RUN_TYPE" = "arkplus" ]; then
            echo "  bash ${SCRIPT_DIR}/arkplus_llm_kg_healthbench.sh \\"
            echo "    '${OUTPUT_DIR}' \\"
            echo "    '${LLM_MODEL}' \\"
            echo "    '${GRAPH}' \\"
            echo "    '${SEARCH_MODE}' \\"
            echo "    ${LIMIT} \\"
            echo "    '${INPUT_JSONL}'"
        else
            echo "  bash ${SCRIPT_DIR}/llm_healthbench.sh \\"
            echo "    '${OUTPUT_DIR}' \\"
            echo "    '${LLM_MODEL}' \\"
            echo "    ${LIMIT} \\"
            echo "    '${INPUT_JSONL}'"
        fi
        echo ""
        PASSED=$((PASSED + 1))
        continue
    fi

    # Execute appropriate script
    if [ "$RUN_TYPE" = "arkplus" ]; then
        bash "${SCRIPT_DIR}/arkplus_llm_kg_healthbench.sh" \
            "$OUTPUT_DIR" \
            "$LLM_MODEL" \
            "$GRAPH" \
            "$SEARCH_MODE" \
            "$LIMIT" \
            "$INPUT_JSONL"
    else
        bash "${SCRIPT_DIR}/llm_healthbench.sh" \
            "$OUTPUT_DIR" \
            "$LLM_MODEL" \
            "$LIMIT" \
            "$INPUT_JSONL"
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ RUN ${RUN_NUM} PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ RUN ${RUN_NUM} FAILED${NC}"
        FAILED=$((FAILED + 1))
        FAILED_RUNS+=("$RUN_NUM: $RUN_NAME")
    fi
    echo ""
done

################################################################################
# SUMMARY
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}ORCHESTRATION COMPLETE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Passed: ${PASSED}/${TOTAL_RUNS}"
echo "Failed: ${FAILED}/${TOTAL_RUNS}"
echo ""

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo -e "${RED}Failed runs:${NC}"
    for run in "${FAILED_RUNS[@]}"; do
        echo "  - $run"
    done
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ All runs completed successfully!${NC}"
    echo ""
    echo "Results available in: ${BASE_OUTPUT_DIR}"
    exit 0
fi
