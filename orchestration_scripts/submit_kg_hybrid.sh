#!/bin/bash
################################################################################
# submit_kg_hybrid.sh
# HYBRID SEARCH: KG-grounded + Baseline runs with configurable LLM and KG
################################################################################

#SBATCH --job-name=eval-kg-hybrid
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/arkplus_evals/hybrid_slurm_%j.out
#SBATCH --error=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/arkplus_evals/hybrid_slurm_%j.err
#SBATCH --gres=gpu:1

set -e

################################################################################
# CONFIGURATION - EDIT THESE FOR DIFFERENT RUNS
################################################################################
LLM_MODEL="azure/gpt-5.4"          # Change to any model (e.g., "azure/gpt-4.1", "azure/gpt-5.4")
KG_NAME="optimus"                  # Change to any KG (e.g., "optimus", "prime")
SEARCH_MODE="hybrid"               # Fixed for this batch file
LIMIT=5000

################################################################################
# FIXED PATHS
################################################################################
SCRIPT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/orchestration_scripts"
BASE_OUTPUT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/arkplus_evals"
INPUT_JSONL="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/2025-05-07-06-14-12_oss_eval.jsonl"
PROJECT_ROOT="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji"
VENV_PYTHON="${PROJECT_ROOT}/simple-evals/.venv/bin/python"
ARK_PYTHON="${PROJECT_ROOT}/ark/.venv/bin/python"

# Extract model name for folder (e.g., "azure/gpt-5.4" -> "gpt54")
MODEL_FOLDER=$(echo "$LLM_MODEL" | sed 's/.*\///; s/\./-/g; s/-//')

# Generate folder names: MODEL_KG_SEARCH_ROLE
RUN_KG_DIR="$BASE_OUTPUT_DIR/${MODEL_FOLDER}_${KG_NAME}_${SEARCH_MODE}_kg"

################################################################################
# ENVIRONMENT SETUP
################################################################################

# Force LiteLLM to handle the Azure 429s automatically
export LITELLM_RETRY_COUNT=10
export LITELLM_RETRY_DELAY=8

# Help PyTorch manage memory fragmentation for the large embeddings matrix
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Optional: Ensure workers only see the one GPU allocated by Slurm
export CUDA_VISIBLE_DEVICES=0

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

# Verify input file exists
if [ ! -f "$INPUT_JSONL" ]; then
    echo "ERROR: Input file not found at $INPUT_JSONL"
    exit 1
fi

export VENV_PYTHON ARK_PYTHON PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

mkdir -p "$BASE_OUTPUT_DIR"

################################################################################
# RUN BATCH
################################################################################

echo "=========================================="
echo "KG ${SEARCH_MODE^^} EVALUATION"
echo "LLM: $LLM_MODEL | KG: $KG_NAME | Search: $SEARCH_MODE"
echo "=========================================="
echo ""

# Run KG-grounded
echo "Running: ${MODEL_FOLDER}_${KG_NAME}_${SEARCH_MODE}_kg"
bash "$SCRIPT_DIR/arkplus_llm_kg_healthbench.sh" \
    "$RUN_KG_DIR" \
    "$LLM_MODEL" \
    "$KG_NAME" \
    "$SEARCH_MODE" \
    "$LIMIT" \
    "$INPUT_JSONL"

if [ $? -ne 0 ]; then
    echo "ERROR: KG run failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ ${SEARCH_MODE^^} EVALUATION COMPLETE"
echo "=========================================="
exit 0
