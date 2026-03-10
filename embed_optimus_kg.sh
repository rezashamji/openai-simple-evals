#!/bin/bash
#SBATCH --job-name=embed-optimus-kg
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/embedding_logs/embed_%j.out
#SBATCH --error=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/embedding_logs/embed_%j.err
#SBATCH --gres=gpu:1

################################################################################
# EMBED OPTIMUS KG - One-time embedding of OptmusKG nodes using KaLM
#
# This script embeds all OptmusKG nodes with KaLM embeddings for use in
# embedding-based or hybrid (RRF fusion) search modes.
#
# Purpose:
# - Generate dense vector embeddings for all 192,682 OptmusKG nodes
# - Save embeddings to embeddings_kalm.npy (for embedding/hybrid search)
# - One-time operation (~5-10 minutes on GPU)
#
# Usage:
#   bash embed_optimus_kg.sh              # Embed OptmusKG with default settings
#   bash embed_optimus_kg.sh optimus      # Explicitly specify graph (default)
#
# Prerequisites:
# - ARK repo with OptmusKG graph data
# - simple-evals/.venv with sentence-transformers, torch, numpy, pandas installed
# - Azure GPU node (recommended; CPU will be very slow)
#
# Output:
# - embeddings_kalm.npy saved to: ark/benchmarks/stark/data/graphs/optimus/
# - Size: ~2.5GB (192,682 nodes × 3,840 dims × 4 bytes)
#
# After completion:
# - run_5q_optimus_test.sh can use --search-mode embedding or hybrid
# - Full SLURM jobs can use embedding/hybrid search across 5000 questions
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GRAPH_NAME=${1:-optimus}
PROJECT_ROOT="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji"
ARK_DIR="${PROJECT_ROOT}/ark"
GRAPH_PATH="${ARK_DIR}/benchmarks/stark/data/graphs/${GRAPH_NAME}"
NODES_PARQUET="${GRAPH_PATH}/nodes.parquet"
EMBEDDINGS_OUTPUT="${GRAPH_PATH}/embeddings_kalm.npy"

################################################################################
# STEP 0: VALIDATION
################################################################################

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}EMBED ${GRAPH_NAME^^} KG - KaLM Node Embeddings${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Check ARK repo exists
if [ ! -d "$ARK_DIR" ]; then
    echo -e "${RED}✗ ERROR: ARK repo not found at ${ARK_DIR}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ARK repo found at ${ARK_DIR}${NC}"

# Check graph exists
if [ ! -d "$GRAPH_PATH" ]; then
    echo -e "${RED}✗ ERROR: Graph '${GRAPH_NAME}' not found at ${GRAPH_PATH}${NC}"
    echo "  Available graphs:"
    ls -1 "${ARK_DIR}/benchmarks/stark/data/graphs/"
    exit 1
fi
echo -e "${GREEN}✓ Graph '${GRAPH_NAME}' found at ${GRAPH_PATH}${NC}"

# Check nodes.parquet exists
if [ ! -f "$NODES_PARQUET" ]; then
    echo -e "${RED}✗ ERROR: nodes.parquet not found at ${NODES_PARQUET}${NC}"
    exit 1
fi

# Get node count (approximate from file size since parquet is binary)
NODE_COUNT=$(python3 -c "import pandas; df = pandas.read_parquet('$NODES_PARQUET'); print(len(df))" 2>/dev/null || echo "?")
echo -e "${GREEN}✓ Nodes file found: ${NODE_COUNT} nodes${NC}"

# Check if embeddings already exist
if [ -f "$EMBEDDINGS_OUTPUT" ]; then
    EMBED_SIZE=$(ls -lh "$EMBEDDINGS_OUTPUT" | awk '{print $5}')
    echo -e "${YELLOW}⚠ Embeddings already exist at: ${EMBEDDINGS_OUTPUT} (${EMBED_SIZE})${NC}"
    read -p "  Overwrite existing embeddings? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✓ Using existing embeddings (skipping embedding step)${NC}"
        exit 0
    fi
fi

################################################################################
# STEP 1: CHECK VENV AND DEPENDENCIES
################################################################################

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 1: CHECK DEPENDENCIES${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

VENV_PYTHON="${PROJECT_ROOT}/simple-evals/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}✗ ERROR: Venv Python not found at ${VENV_PYTHON}${NC}"
    echo "  Create with: cd simple-evals && python -m venv .venv && .venv/bin/pip install sentence-transformers torch"
    exit 1
fi

# Check embedding dependencies
echo "Checking embedding dependencies..."
$VENV_PYTHON -c "
import sys
missing = []

try:
    from sentence_transformers import SentenceTransformer
    print('  ✓ sentence-transformers')
except ImportError:
    missing.append('sentence-transformers')
    print('  ✗ sentence-transformers')

try:
    import torch
    print('  ✓ torch')
except ImportError:
    missing.append('torch')
    print('  ✗ torch')

try:
    import numpy
    print('  ✓ numpy')
except ImportError:
    missing.append('numpy')
    print('  ✗ numpy')

try:
    import pandas
    print('  ✓ pandas')
except ImportError:
    missing.append('pandas')
    print('  ✗ pandas')

if missing:
    print(f'\nMissing: {missing}')
    print('Install with: pip install sentence-transformers torch pandas numpy')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ ERROR: Missing embedding dependencies${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All dependencies present${NC}"

################################################################################
# STEP 2: EMBED ALL NODES
################################################################################

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 2: EMBEDDING ALL NODES WITH KaLM${NC}"
echo -e "${BLUE}Model: tencent/KaLM-Embedding-Gemma3-12B-2511${NC}"
echo -e "${BLUE}Batch size: 256 | GPU enabled: auto-detect${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

echo ""
echo "Embedding process started..."
echo "  This will take 10-20 minutes depending on GPU availability"
echo "  Embeddings will be saved to: ${EMBEDDINGS_OUTPUT}"
echo ""

# Run embedding via embed_kg.py
cd "$PROJECT_ROOT"

$VENV_PYTHON -m simple_evals.embed_kg \
    --graph-path "$GRAPH_PATH" \
    --output-path "$EMBEDDINGS_OUTPUT" \
    --batch-size 256

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Embedding FAILED${NC}"
    exit 1
fi

################################################################################
# STEP 3: VERIFY EMBEDDINGS
################################################################################

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 3: VERIFY EMBEDDINGS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

if [ ! -f "$EMBEDDINGS_OUTPUT" ]; then
    echo -e "${RED}✗ ERROR: Embeddings file not created at ${EMBEDDINGS_OUTPUT}${NC}"
    exit 1
fi

# Check file size
EMBED_SIZE=$(ls -lh "$EMBEDDINGS_OUTPUT" | awk '{print $5}')
EMBED_SIZE_BYTES=$(stat -c%s "$EMBEDDINGS_OUTPUT")
EXPECTED_SIZE=$((192682 * 3840 * 4))  # nodes × dims × float32

echo -e "${GREEN}✓ Embeddings file exists${NC}"
echo "  Path: ${EMBEDDINGS_OUTPUT}"
echo "  Size: ${EMBED_SIZE} (${EMBED_SIZE_BYTES} bytes)"

# Quick validation
$VENV_PYTHON "$EMBEDDINGS_OUTPUT" << 'VALIDATE_EMBEDDINGS'
import numpy as np
import sys
from pathlib import Path

embeddings_path = Path(sys.argv[1])

try:
    embeddings = np.load(embeddings_path, allow_pickle=False)
    print(f"\n✓ Embeddings validated")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Min value: {embeddings.min():.6f}")
    print(f"  Max value: {embeddings.max():.6f}")
    print(f"  Mean value: {embeddings.mean():.6f}")

    # Check for NaN/inf
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    if nan_count == 0 and inf_count == 0:
        print(f"✓ No NaN or Inf values detected")
    else:
        print(f"⚠ Warning: {nan_count} NaN, {inf_count} Inf values detected")

except Exception as e:
    print(f"✗ ERROR: Failed to load embeddings: {e}")
    sys.exit(1)

VALIDATE_EMBEDDINGS

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Embedding validation failed${NC}"
    exit 1
fi

################################################################################
# COMPLETION
################################################################################

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ EMBEDDING COMPLETE - ${GRAPH_NAME^^} KG embeddings ready${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

echo ""
echo "Next steps:"
echo "  1. Run tests with embedding/hybrid search:"
echo "     bash simple-evals/run_5q_optimus_test.sh 5 ${GRAPH_NAME} embedding"
echo "     bash simple-evals/run_5q_optimus_test.sh 5 ${GRAPH_NAME} hybrid"
echo ""
echo "  2. Run full SLURM job with embedding/hybrid search:"
echo "     edit simple-evals/run_ark_healthbench_kg_full.slurm (set SEARCH_MODE=embedding or hybrid)"
echo "     sbatch simple-evals/run_ark_healthbench_kg_full.slurm"
echo ""
echo "  3. Embeddings location: ${EMBEDDINGS_OUTPUT}"
echo ""
