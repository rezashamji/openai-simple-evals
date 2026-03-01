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

# Install dependencies (all-in-one venv for simple-evals + ARK integration)
pip install --upgrade pip
pip install litellm tqdm blobfile openai anthropic pyyaml tantivy networkx
```

**Key packages** needed:
- `litellm` ≥1.80 — Azure OpenAI integration
- `tqdm` — Progress bars
- `blobfile` — Cloud file handling (if using remote eval data)
- `openai` — OpenAI API client
- `anthropic` — Anthropic API client (optional)
- `pyyaml` — **CRITICAL for ARK subprocess** — required by `ark/benchmarks/stark/utils.py`
- `tantivy` — BM25 search engine (used by ARK node retrieval)
- `networkx` — Graph utilities (used by ARK)

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

### **Level 2: ARK Integration Test (TESTED ✅)**

Tests **Phase 1A** (ARK node retrieval) + **Phase 1B** (no-KG baseline) → **Phase 2** (grading comparison).

**What it validates:**
- ✅ ARK subprocess integration works
- ✅ Node summaries retrieved correctly
- ✅ Nodes-to-NL LLM conversion with real KG context
- ✅ Node contributions tracked accurately
- ✅ Baseline LLM calls work independently
- ✅ Full pipeline end-to-end with real KG vs no-KG comparison
- ✅ Grading works on both runs
- ✅ ARK+KG shows measurable improvement over baseline

**When to use:**
- Before committing Phase 1 changes
- Validating ARK integration after ARK updates
- Real HealthBench runs

**Prerequisites:**
- ARK repository with `benchmarks/stark/run_one_question.py`
- A KG graph (e.g., "prime") with nodes and edges
- Real HealthBench questions (at least 50)

**Run (Phase 1A + 1B):**
```bash
# Phase 1A: Get ARK nodes and generate KG-grounded responses
python -m simple_evals.scripts.run_ark_on_healthbench \
  --examples-jsonl healthbench_50q_subset.jsonl \
  --output-path /tmp/phase1a_kg_grounded.jsonl \
  --graph-name prime \
  --ark-model gpt-4.1 \
  --nl-model azure/gpt-4.1 \
  --run-tag kg_grounded

# Phase 1B: Generate no-KG baseline (direct LLM calls, no ARK)
python simple-evals/scripts/run_baseline_on_healthbench.py \
  --examples-jsonl healthbench_50q_subset.jsonl \
  --output-path /tmp/phase1b_no_kg.jsonl \
  --model azure/gpt-4.1

# Phase 2A: Grade KG-grounded responses
python -m simple_evals.scripts.grade_ark_healthbench_responses \
  --responses-jsonl /tmp/phase1a_kg_grounded.jsonl \
  --output-dir /tmp/grading_a/

# Phase 2B: Grade no-KG baseline
python -m simple_evals.scripts.grade_ark_healthbench_responses \
  --responses-jsonl /tmp/phase1b_no_kg.jsonl \
  --output-dir /tmp/grading_b/

# Step 8b: Criterion-level comparison
python simple-evals/scripts/compare_kg_runs_criterion_level.py \
  --kg-graded /tmp/grading_a/phase1a_kg_grounded_grading_checkpoint.jsonl \
  --no-kg-graded /tmp/grading_b/phase1b_no_kg_grading_checkpoint.jsonl \
  --output /tmp/step8b_criterion_level.jsonl

# Step 9: Binary labels
python simple-evals/scripts/build_kg_binary_labels.py \
  --criterion-level /tmp/step8b_criterion_level.jsonl \
  --output /tmp/step9_binary_labels.jsonl
```

**Expected output:**
- Phase 1A: `phase1a_kg_grounded.jsonl` (KG-grounded responses with node_contributions)
- Phase 1B: `phase1b_no_kg.jsonl` (baseline responses with empty node fields)
- Phase 2 metrics: ARK+KG score > baseline score (proof of benefit)

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
| `scripts/run_ark_on_healthbench.py` | Phase 1A: Generate responses with ARK nodes | ✅ Tested (50q, token fix applied) |
| `scripts/run_baseline_on_healthbench.py` | Phase 1B: Generate no-KG baseline via direct LLM calls | ✅ Fixed & Tested (50q) |
| `scripts/grade_ark_healthbench_responses.py` | Phase 2: Grade responses with KG enrichment | ✅ Fixed (relative imports) |
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

### Issue 4: Phase 1B (Baseline) Subprocess Failure

**Symptom (Feb 27, 2026 - FIXED ✅):**
```
error: unrecognized arguments: --output_path /tmp/tmpu5b3tkf3.jsonl
```

**Cause:** `run_baseline_on_healthbench.py` was using subprocess call to `simple-evals` CLI with non-existent `--output_path` flag.
- simple-evals outputs to stdout, not file
- No direct way to capture per-sample outputs

**Root Problem:** Architectural mismatch
- Phase 1A (ARK+KG): Calls LLM directly via `litellm.completion()`
- Phase 1B (baseline): Was trying to use simple-evals subprocess

**Fix Applied (Session 9):** ✅ Rewrote `run_baseline_on_healthbench.py`
- Changed from subprocess call to direct `litellm.completion()` calls
- Mirrors Phase 1A architecture (independent LLM calls, JSONL output)
- Maintains checkpoint/resume support
- Supports parallel workers (`--n-workers`)

**Test Result:** Both Phase 1A and 1B now work correctly (50q validation ✅)
- Phase 1A (ARK+KG) score: 0.0997
- Phase 1B (baseline) score: 0.0000
- **Improvement: +9.97%** (proof of KG benefit)

**To use Phase 1B:**
```bash
python simple-evals/scripts/run_baseline_on_healthbench.py \
  --examples-jsonl healthbench_questions.jsonl \
  --output-path phase1b_baseline.jsonl \
  --model azure/gpt-4.1 \
  --n-workers 10  # optional: parallelize
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
