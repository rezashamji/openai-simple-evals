#!/bin/bash
#SBATCH --job-name=eval-batch1-optimus-hybrid
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/5000q_sweeps_mar10/batch1_slurm_%j.out
#SBATCH --error=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/5000q_sweeps_mar10/batch1_slurm_%j.err
#SBATCH --gres=gpu:1

set -e

SCRIPT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/orchestration_scripts"
BASE_OUTPUT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/5000q_sweeps_mar10"
INPUT_JSONL="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/2025-05-07-06-14-12_oss_eval.jsonl"
PROJECT_ROOT="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji"
VENV_PYTHON="${PROJECT_ROOT}/simple-evals/.venv/bin/python"
ARK_PYTHON="${PROJECT_ROOT}/ark/.venv/bin/python"

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

echo "=========================================="
echo "BATCH 1: OptimusKG Hybrid (Runs 1-2)"
echo "=========================================="
echo ""

# Run 1: KG + Hybrid
echo "RUN 1: arkplus optimus hybrid"
bash "$SCRIPT_DIR/arkplus_llm_kg_healthbench.sh" \
    "$BASE_OUTPUT_DIR/run_001_gpt41_optimus_hybrid" \
    "azure/gpt-4.1" \
    optimus \
    hybrid \
    5000 \
    "$INPUT_JSONL"

if [ $? -ne 0 ]; then
    echo "BATCH 1 FAILED at Run 1"
    exit 1
fi

echo ""
echo "RUN 2: baseline"
bash "$SCRIPT_DIR/llm_healthbench.sh" \
    "$BASE_OUTPUT_DIR/run_002_gpt41_optimus_hybrid_baseline" \
    "azure/gpt-4.1" \
    5000 \
    "$INPUT_JSONL"

if [ $? -ne 0 ]; then
    echo "BATCH 1 FAILED at Run 2"
    exit 1
fi

echo ""
echo "=========================================="
echo "BATCH 1 COMPLETE"
echo "=========================================="
exit 0
