# ARK + HealthBench: minimal bridge

We use **two oracles** and write **only the glue** between them.

## Oracles (we do not implement these)

1. **ARK** – Input: question(s), knowledge graph (e.g. prime), LLM (e.g. GPT-4.1). Output: ranked nodes (and summaries). We call it by subprocess to the ARK repo (`run_one_question.py`). All ARK internals (agents, aggregation, etc.) stay inside ARK.

2. **HealthBench eval** – Input: response jsonl (one NL answer per example). Output: metrics (grades each answer on rubrics). We do not implement grading; we run their script.

## Our code (four steps)

1. **Deconstruct HealthBench** – Read eval jsonl, extract questions (same order as HealthBench uses: `random.Random(0).sample`).
2. **Call ARK oracle** – For each question, subprocess `ark/benchmarks/stark/run_one_question.py` → get nodes JSON.
3. **Nodes → NL** – One LLM call per example to turn node summaries + question into natural language. Write response jsonl.
4. **Call HealthBench eval oracle** – Run `grade_ark_healthbench_responses.py` on the response jsonl → metrics.

## Files

- **run_ark_on_healthbench.py** – Phase 1: steps 1–3 (get Qs, ARK oracle, nodes→NL, write jsonl).
- **grade_ark_healthbench_responses.py** – Phase 2: step 4 (run HealthBench eval on response jsonl; no grading logic ours).
- **run_ark_healthbench_full.slurm** – One-job run: Phase 1 then Phase 2 (edit variables at top for paths, graph/model, N_WORKERS, N_THREADS).
- **tests/test_ark_healthbench.py** – Minimal tests (question extraction, Phase 2 with mock grader).

## Dependencies

Phase 1 and Phase 2 use **LiteLLM** with **Azure** (same as ARK). Install in the healthbench venv:

```bash
pip install litellm
```

## Environment (Azure, same as ARK)

Set Azure env vars so LiteLLM can call your model. Use this repo’s `.env` (or export in the shell). Do **not** commit real keys. **You must source `.env` before running** (e.g. `set -a && source .env && set +a`).

Any of these names work (scripts pass them through to LiteLLM):

- **Key:** `AZURE_API_KEY` or `AZURE_OPENAI_API_KEY`
- **Base URL:** `AZURE_API_BASE`, `AZURE_OPENAI_ENDPOINT`, or `AZURE_OPENAI_API_BASE` – e.g. `https://azure-ai.hms.edu`
- **Version:** `AZURE_API_VERSION` or `AZURE_OPENAI_API_VERSION` – e.g. `2024-10-21` (use what your endpoint supports)

Model used for nodes→NL and for the grader is **azure/gpt-4.1** (same as ARK). See `.env.example` in the repo root for a template.

## Run

From the project root (`/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji`) (venv activated, Azure env vars set, e.g. `set -a && source healthbench/.env && set +a`):

```bash
# Phase 1: HealthBench Qs → ARK → nodes→NL → response jsonl
python -m healthbench.scripts.run_ark_on_healthbench \
  --input-jsonl healthbench/2025-05-07-06-14-12_oss_eval.jsonl \
  --output-jsonl healthbench/responses.jsonl \
  --ark-dir /path/to/ark \
  --limit 5

# Optional: pass graph/model so the run is explicit (defaults: prime, azure/gpt-4.1, 3 agents, 20 steps)
# --graph-name prime --ark-model azure/gpt-4.1 --ark-agents 3 --ark-max-steps 20
# Optional: parallel questions (default 1) --n-workers 8

# Phase 2: response jsonl → HealthBench eval → metrics
python -m healthbench.scripts.grade_ark_healthbench_responses --responses-jsonl healthbench/responses.jsonl
# Optional: --n-threads 200 (default 120) --output-dir /path/to/dir
```

**Full run on Slurm:** Submit `run_ark_healthbench_full.slurm` from the repo root. The script loads `module load python/3.12.11-fasrc02`. Edit the variables at the top (paths, `GRAPH_NAME`, `ARK_MODEL`, `N_WORKERS`, `N_THREADS`, optional `LIMIT`).

ARK repo must contain `benchmarks/stark/run_one_question.py` (question in → nodes JSON out).
