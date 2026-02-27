# KG Metadata Enrichment Evaluation Pipeline: Setup & Testing Guide

## Overview

This guide covers the complete KG metadata enrichment pipeline for HealthBench evaluation (9 steps):
- **Phase 1**: Generate responses (ARK → nodes → NL responses)
- **Phase 2**: Grade responses with KG enrichment (Azure LLM grading)
- **Steps 8b & 9**: Criterion-level comparison and binary label generation

## Prerequisites: Environment Setup

### 1. Python Virtual Environment

From the project root (`/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji`):

```bash
# Create/activate venv
cd simple-evals
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install litellm tqdm blobfile openai anthropic
```

**Key packages** needed:
- `litellm` ≥1.80 — Azure OpenAI integration
- `tqdm` — Progress bars
- `blobfile` — Cloud file handling (if using remote eval data)
- `openai` — OpenAI API client
- `anthropic` — Anthropic API client (optional)

### 2. Azure Credentials (`.env` file)

Create `.env` in the `simple-evals` directory **with the correct env var names**:

```bash
# Azure OpenAI credentials for LLM calls
AZURE_API_KEY=<your-key-here>
AZURE_API_BASE=https://azure-ai.hms.edu
AZURE_API_VERSION=2024-10-21
AZURE_DEPLOYMENT_NAME=gpt-4.1
```

**Important variable names:**
- LiteLLM looks for these exact names: `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`
- Alternative names also work: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`
- **Do NOT commit real keys** — add `.env` to `.gitignore`

**Load before running scripts:**
```bash
set -a && source simple-evals/.env && set +a
```

### 3. Python Module Setup

Because simple-evals uses hyphens (not valid in Python imports), create a symlink at the project root:

```bash
# From project root
ln -s simple-evals simple_evals
```

This allows `python -m simple_evals.scripts...` invocations to work properly.

---

## Test Validation Levels

### **Level 1: Mock Validation (TESTED ✓)**

Tests the **pipeline structure** with synthetic data (no real ARK nodes).

**What it validates:**
- ✅ Types.py shadowing issue fixed (imports work)
- ✅ Azure LLM integration functional (real API calls)
- ✅ KG metadata fields can be extracted
- ✅ Grading, criterion comparison, binary labels all work
- ✅ Full 9-step pipeline executes without error

**What it doesn't validate:**
- ❌ Real Phase 1 ARK node retrieval + contribution tracking
- ❌ Nodes-to-NL conversion with actual KG context
- ❌ True KG relevance on real questions

**When to use:**
- Rapid pipeline validation after code changes
- Testing infrastructure before full benchmark runs
- Verifying Azure credentials work

**Run:**
```bash
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji
source simple-evals/.venv/bin/activate
set -a && source simple-evals/.env && set +a

# Create mock data if not present
mkdir -p /tmp/kg_enrichment_mock_test
# ... generate or copy mock files ...

# Run grading pipeline
python -m simple_evals.scripts.grade_ark_healthbench_responses \
  --responses-jsonl /tmp/kg_enrichment_mock_test/mock_kg_grounded_phase1.jsonl \
  --output-dir /tmp/kg_enrichment_mock_test/

python -m simple_evals.scripts.grade_ark_healthbench_responses \
  --responses-jsonl /tmp/kg_enrichment_mock_test/mock_no_kg_phase1.jsonl \
  --output-dir /tmp/kg_enrichment_mock_test/

# Run criterion comparison (Step 8b)
python simple-evals/scripts/compare_kg_runs_criterion_level.py \
  --kg-graded /tmp/kg_enrichment_mock_test/mock_kg_grounded_phase1_grading_checkpoint.jsonl \
  --no-kg-graded /tmp/kg_enrichment_mock_test/mock_no_kg_phase1_grading_checkpoint.jsonl \
  --output /tmp/kg_enrichment_mock_test/step8b_criterion_level.jsonl

# Run binary label generation (Step 9)
python simple-evals/scripts/build_kg_binary_labels.py \
  --criterion-level /tmp/kg_enrichment_mock_test/step8b_criterion_level.jsonl \
  --output /tmp/kg_enrichment_mock_test/step9_labels.jsonl
```

**Expected output:**
- `*_grading_checkpoint.jsonl` (145+ KB) — graded responses with KG fields
- `step8b_criterion_level.jsonl` — criterion-level comparisons
- `step9_labels.jsonl` — binary labels (USEKG/DONTUSEKG)

---

### **Level 2: ARK Integration Test (NOT YET TESTED)**

Tests **Phase 1** (ARK node retrieval) → **Phase 2** (grading with real nodes).

**What it validates:**
- ✅ ARK subprocess integration works
- ✅ Node summaries retrieved correctly
- ✅ Nodes-to-NL LLM conversion with real KG context
- ✅ Node contributions tracked accurately
- ✅ Full pipeline with real KG metadata

**When to use:**
- Before committing Phase 1 changes
- Validating ARK integration after ARK updates
- Real HealthBench runs

**Prerequisites:**
- ARK repository with `benchmarks/stark/run_one_question.py`
- A KG graph (e.g., "prime") with nodes and edges
- Real HealthBench questions

**Run:**
```bash
# Phase 1: Get ARK nodes and generate responses
python -m simple_evals.scripts.run_ark_on_healthbench \
  --input-jsonl simple-evals/2025-05-07-06-14-12_oss_eval.jsonl \
  --output-jsonl /tmp/test_kg_responses.jsonl \
  --ark-dir /path/to/ark \
  --graph-name prime \
  --limit 5

# Phase 2: Grade with KG metadata
python -m simple_evals.scripts.grade_ark_healthbench_responses \
  --responses-jsonl /tmp/test_kg_responses.jsonl \
  --output-dir /tmp/test_kg_output/
```

---

### **Level 3: Full Benchmark Run (NOT YET TESTED)**

Tests the complete pipeline on all HealthBench questions.

**When to use:**
- Final validation before publication
- Benchmark reporting

---

## Files & Components

### Scripts

| File | Purpose | Status |
|------|---------|--------|
| `scripts/grade_ark_healthbench_responses.py` | Phase 2: Grade responses with KG enrichment | ✅ Fixed (relative imports) |
| `scripts/run_ark_on_healthbench.py` | Phase 1: Generate responses with ARK nodes | 🔄 Not tested with KG |
| `scripts/compare_kg_runs_criterion_level.py` | Step 8b: Criterion-level KG vs no-KG comparison | ✅ Tested |
| `scripts/build_kg_binary_labels.py` | Step 9: Generate binary labels | ✅ Tested |
| `healthbench_eval.py` | Rubric grading logic with KG relevance | ✅ Tested |
| `types.py` | Type definitions (OpenAI source) | ✅ Not modified |

### Key Files for KG Enrichment

- `healthbench_eval.py` lines 584-587: `kg_relevance_score` metric added
- `healthbench_eval.py` lines 308-400: KG reasoning and labeling logic
- `scripts/grade_ark_healthbench_responses.py` lines 30-46: Fixed to use relative imports

---

## Common Issues & Fixes

### Issue 1: `types.py` Shadow Import Error

```
ImportError: cannot import name 'MessageList' from types
```

**Cause:** Module-level `sys.path` hack trying to import `types` module (shadows stdlib).

**Fix:** ✅ Already fixed in commit 339c34a. Use relative imports:
```bash
python -m simple_evals.scripts.grade_ark_healthbench_responses ...
```

### Issue 2: Azure Credentials Not Found

```
APIConnectionError: Failed to establish a new connection
```

**Cause:** Missing or wrong `.env` variable names.

**Fix:**
```bash
# Verify correct variable names in .env
AZURE_API_KEY=...      # NOT AZURE_OPENAI_API_KEY
AZURE_API_BASE=...     # NOT AZURE_OPENAI_ENDPOINT
AZURE_API_VERSION=...

# Load before running
set -a && source simple-evals/.env && set +a
echo $AZURE_API_KEY  # Verify loaded
```

### Issue 3: Module Not Found When Running Script

```
ModuleNotFoundError: No module named 'simple_evals'
```

**Cause:** Missing symlink or wrong working directory.

**Fix:**
```bash
# Create symlink at project root
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji
ln -s simple-evals simple_evals

# Run from project root
python -m simple_evals.scripts.grade_ark_healthbench_responses ...
```

### Issue 4: LiteLLM Logging Errors

```
LiteLLM.LoggingError: Exception occurred while logging...
```

**Cause:** Non-blocking LiteLLM logging issue (doesn't stop execution).

**Fix:** Usually harmless; check if output files were created. If grading failed, check logs for actual API errors:
```bash
grep -i "error\|failed" /tmp/output/grade.log
```

---

## Next Steps

### To Validate Level 2 (ARK Integration):

1. **Ensure ARK repo is available** with `benchmarks/stark/run_one_question.py`
2. **Point Phase 1 at real KG** (graph_name, not mock nodes)
3. **Run Phase 1 → Phase 2 → 8b → 9** with small sample (--limit 5)
4. **Inspect node_contributions** in graded output to verify tracking works
5. **Verify kg_relevance_score variation** — real KG should show differences vs no-KG

### Documentation to Add:

- [ ] Add troubleshooting guide for Azure connection
- [ ] Document symlink requirement clearly
- [ ] Add example .env template
- [ ] Document node_contributions schema

---

## References

- **HealthBench paper**: https://arxiv.org/abs/2401.02999
- **Azure OpenAI setup**: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **LiteLLM docs**: https://docs.litellm.ai/docs/providers/azure
- **ARK repo**: (internal, contact team)
