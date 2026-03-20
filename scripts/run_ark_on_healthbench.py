"""
Phase 1: Our only logic. Four steps total; we use two oracles and do minimal glue.

  ORACLE 1 – ARK: question + KG (e.g. prime) + LLM → ranked nodes.
    We do not implement or import any ARK logic. We subprocess-call the ARK repo
    (run_one_question.py): question in → JSON with nodes+summaries out.

  OUR CODE:
    1. Deconstruct HealthBench dataset → get full conversation per example (same order as HealthBench eval).
    2. For each example: call ARK oracle with full conversation (or user-only via --ark-input user_only) → parse nodes JSON.
    3. nodes→NL: one LLM call with full HealthBench conversation + KG context; model continues (e.g. answer or follow-up). Write response jsonl.
    4. Write response jsonl (one line per example).

  We pass the full HealthBench conversation to the model and (by default) to ARK so the benchmark matches HealthBench: model sees full context and decides how to continue.

  ORACLE 2 – HealthBench eval: takes response jsonl, grades on rubrics, returns metrics.
    We do not implement grading. Phase 2 is grade_ark_healthbench_responses.py (calls their eval).

  ARK config: pass --graph-name, --ark-model, --ark-agents, --ark-max-steps when you run
  (e.g. from Slurm). Paper Appendix A.2 uses Prime KG, GPT-4.1, n=3 agents, Tmax=20.

  Resume: If --output-jsonl already exists, we load it and reuse results by prompt_id (no
  line-count assumption). We only run ARK+nodes_to_nl for examples whose prompt_id is not
  in the file, then write the full jsonl in current eval order so Phase 2 stays correct.
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Suppress LiteLLM verbose output (Provider List spam) in parent process
import litellm as _litellm_init
_litellm_init.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def tail_jsonl(filepath, n=1):
    """Read last n lines from a JSONL file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-n:] if line.strip()]
    except Exception:
        return []


def format_node_summary(node):
    """Format a single node for display."""
    return {
        'index': node.get('index'),
        'name': node.get('name', '')[:60],
        'summary': node.get('summary', '')[:100]
    }


def log_phase1_progress(output_jsonl, question_num, log_interval=10):
    """Log Phase 1A progress with sample nodes and contributions."""
    lines = tail_jsonl(output_jsonl, n=1)
    if not lines:
        return

    latest = lines[0]
    prompt_id = latest.get('prompt_id', 'N/A')[:8]
    has_nodes = len(latest.get('node_summaries', [])) > 0
    node_count = len(latest.get('node_summaries', []))
    contrib_count = len(latest.get('node_contributions', {}))
    response_len = len(latest.get('response_text', ''))

    print(f"\n{'='*80}")
    print(f"[Phase 1A Progress] {question_num} questions processed")
    print(f"{'='*80}")
    print(f"Latest question: {prompt_id}...")
    print(f"  Nodes found: {node_count}")
    print(f"  Nodes used (contributions): {contrib_count}")
    print(f"  Response length: {response_len} chars")

    if has_nodes and node_count > 0:
        nodes = latest.get('node_summaries', [])
        print(f"\n  Sample nodes (first 3 of {node_count}):")
        for node in nodes[:3]:
            formatted = format_node_summary(node)
            print(f"    - {formatted['name']}")
            print(f"      Summary: {formatted['summary']}")

    if contrib_count > 0:
        contribs = latest.get('node_contributions', {})
        print(f"\n  Sample node contributions (first 2 of {contrib_count}):")
        for node_id, contrib_data in list(contribs.items())[:2]:
            contributed = contrib_data.get('contributed', False)
            explanation = contrib_data.get('explanation', '')[:100]
            print(f"    - {node_id}: contributed={contributed}")
            print(f"      Explanation: {explanation}...")
    else:
        print(f"\n  No node contributions (empty response or no nodes)")

    sys.stdout.flush()

def get_question_from_prompt(prompt: list) -> str:
    """Extract user content from HealthBench prompt (list of message dicts). Same order as eval. Optional for ARK via --ark-input user_only."""
    parts = [m["content"] for m in prompt if m.get("role") == "user"]
    return "\n\n".join(parts).strip() if parts else ""


def normalize_node_indices(node_summaries: list, node_contributions: dict) -> tuple[list, dict]:
    """Normalize node indices to string format for consistency.

    node_summaries uses integer indices (e.g., 128100)
    node_contributions uses string keys (e.g., "node_128100")

    This function:
    1. Converts all node_summaries indices from int to string
    2. Ensures node_contributions keys match the string format
    3. Returns normalized (node_summaries, node_contributions) tuple

    This ensures downstream code has consistent index types.
    """
    # Normalize node_summaries: convert index to string
    normalized_summaries = []
    for node in node_summaries:
        normalized_node = node.copy()
        if "index" in node:
            normalized_node["index"] = str(node["index"])
        normalized_summaries.append(normalized_node)

    # Normalize node_contributions: ensure all keys are "node_<INDEX>" format
    normalized_contributions = {}
    for key, value in node_contributions.items():
        # Handle both formats: "node_128100" and "128100"
        if isinstance(key, str) and key.startswith("node_"):
            normalized_key = key
        else:
            normalized_key = f"node_{key}"
        normalized_contributions[normalized_key] = value

    return normalized_summaries, normalized_contributions


def format_prompt_as_conversation_string(prompt: list) -> str:
    """Format full HealthBench prompt (message list) as a single string for ARK: 'user: ...\\n\\nassistant: ...'."""
    return "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt).strip()


def load_id_to_result(path: Path) -> dict[str, dict]:
    """Load existing output jsonl into prompt_id -> result dict. Used for resume. Skips bad lines."""
    id_to_result: dict[str, dict] = {}
    if not path.exists():
        return id_to_result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                pid = d.get("prompt_id")
                if pid is not None:
                    id_to_result[pid] = d
            except json.JSONDecodeError:
                pass
    return id_to_result


def get_resume_to_compute(examples: list[dict], id_to_result: dict[str, dict]) -> list[int]:
    """Indices to compute when resuming: those whose prompt_id is not in id_to_result."""
    done_prompt_ids = set(id_to_result.keys())
    return [i for i, row in enumerate(examples) if row.get("prompt_id", str(i)) not in done_prompt_ids]


def build_results_in_order(
    examples: list[dict], id_to_result: dict[str, dict], new_results: dict[int, dict]
) -> list[dict]:
    """Merge cache and newly computed results in eval order for Phase 2 alignment."""
    results = []
    for i, row in enumerate(examples):
        pid = row.get("prompt_id", str(i))
        if pid in id_to_result:
            results.append(id_to_result[pid])
        else:
            results.append(new_results[i])
    return results


def nodes_to_nl(prompt: list, node_summaries: list, model_name: str = "azure/gpt-5.4") -> tuple[str, dict]:
    """Turn ARK's node summaries + full HealthBench conversation into response + track node contributions.

    Returns tuple of (response_text, node_contributions_dict) where:
    - response_text: clean natural language response (no markup)
    - node_contributions: dict mapping "node_<INDEX>" to {"explanation": str, "contributed": bool}

    The model generates JSON with two fields:
    1. final_answer: the clean response text
    2. node_contributions: which nodes it claims to have used and why

    Uses LiteLLM + AZURE_* env (same as ARK)."""
    from litellm import completion

    # Handle empty nodes case
    if not node_summaries:
        fallback = "I could not find relevant information in the knowledge base for this question."
        return (fallback, {})

    # Format nodes with their indices for the model to reference
    node_blocks = []
    for s in node_summaries:
        idx = s.get("index", "?")
        name = s.get("name", "")
        summary = s.get("summary", "")
        text = summary if summary else name
        node_blocks.append(f"Node {idx}: {text}")

    context = "\n\n".join(node_blocks)

    # New prompt: ask for JSON with final_answer + node_contributions
    kg_user_content = f"""You will answer a clinical health question using information from a knowledge graph (KG).

# Conversation (multi-turn)
[The conversation is provided above]

# Retrieved KG Nodes (ordered most to least relevant)
{context}

# Task
1. Generate a clear, accurate response to the latest user question.
2. After your response, reflect on which KG nodes you used and how.

# Output Format
Your response should be in this exact JSON format:

{{
  "final_answer": "Your clean natural language response here. Do not include any markup or citations.",
  "node_contributions": {{
    "node_<INDEX_A>": {{
      "explanation": "This node provided information about X. I incorporated it into the answer as [specific text/reasoning].",
      "contributed": true
    }},
    "node_<INDEX_B>": {{
      "explanation": "This node discussed Y, but Y was not relevant to the question / I already knew this from context / the user already said this.",
      "contributed": false
    }}
  }}
}}

Important:
- For EVERY retrieved node, include an entry in node_contributions (even if contributed=false).
- The "explanation" must be specific and grounded in the final_answer text.
- Do not include markdown, citations, or node references in final_answer.
- Return only valid JSON, no additional text."""

    # Build messages with actual conversation + KG context
    messages = list(prompt) + [{"role": "user", "content": kg_user_content}]

    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")
    api_base = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_BASE") or os.environ.get("AZURE_API_BASE")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("AZURE_API_VERSION")

    import litellm as _litellm
    from litellm.exceptions import RateLimitError
    import time as _time

    # Pick up reasoning_effort from env (set by orchestration script or CLI)
    reasoning_effort = os.environ.get("REASONING_EFFORT")

    max_retries = 3
    delay = 30
    last_exc = None
    for attempt in range(max_retries):
        try:
            extra_args = {}
            if reasoning_effort:
                extra_args["reasoning_effort"] = reasoning_effort
            response = completion(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=8192,
                timeout=120,
                num_retries=0,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                **extra_args,
            )
            choice = (response.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason", "unknown")
            content = choice.get("message", {}).get("content") or ""
            print(f"[nodes_to_nl] attempt {attempt+1}/{max_retries} finish_reason={finish_reason} content_len={len(content)}", file=sys.stderr)
            if finish_reason == "length":
                print(f"[nodes_to_nl WARNING] finish_reason=length — max_tokens budget exhausted (likely by reasoning tokens), content is truncated/empty", file=sys.stderr)
            parsed = json.loads(content.strip())

            response_text = parsed.get("final_answer", "").strip()
            node_contributions = parsed.get("node_contributions", {})

            return (response_text, node_contributions)
        except json.JSONDecodeError as e:
            last_exc = e
            # Log what the response actually contained so we can diagnose why content was empty/invalid
            try:
                raw_choices = response.get("choices") or []
                raw_content = raw_choices[0].get("message", {}).get("content") if raw_choices else None
                raw_finish = raw_choices[0].get("finish_reason") if raw_choices else None
                print(f"[nodes_to_nl ERROR] JSONDecodeError: finish_reason={raw_finish} raw_content={repr(str(raw_content)[:300])}", file=sys.stderr)
            except Exception:
                print(f"[nodes_to_nl ERROR] JSONDecodeError: could not inspect response", file=sys.stderr)
            if attempt < max_retries - 1:
                print(f"[nodes_to_nl ERROR] {type(e).__name__} (attempt {attempt+1}/{max_retries}), retrying in {delay}s...", file=sys.stderr)
        except (RateLimitError, _litellm.Timeout) as e:
            last_exc = e
            if attempt < max_retries - 1:
                print(f"[nodes_to_nl ERROR] {type(e).__name__} (attempt {attempt+1}/{max_retries}), retrying in {delay}s...", file=sys.stderr)
                _time.sleep(delay)
                delay *= 2
            else:
                print(f"[nodes_to_nl ERROR] {type(e).__name__} after {max_retries} attempts: {str(e)[:200]}", file=sys.stderr)
                return ("", {})
        except Exception as e:
            print(f"[nodes_to_nl ERROR] {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            if not api_key:
                print(f"  WARNING: api_key is None/empty", file=sys.stderr)
            if not api_base:
                print(f"  WARNING: api_base is None/empty", file=sys.stderr)
            if not api_version:
                print(f"  WARNING: api_version is None/empty", file=sys.stderr)
            return ("", {})


def main():
    parser = argparse.ArgumentParser(description="Phase 1: HealthBench Qs → ARK oracle → nodes→NL → response jsonl.")
    parser.add_argument("--input-jsonl", type=str, required=True, help="HealthBench eval jsonl.")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--ark-dir", type=str, required=True, help="Path to ARK repo (has benchmarks/stark/run_one_question.py).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples (same order as HealthBench eval).")
    parser.add_argument("--python", type=str, default=None, help="Python for ARK subprocess (default: ark/.venv/bin/python).")
    parser.add_argument("--nl-model", type=str, default="azure/gpt-5.4", help="Model for nodes→NL (default same as ARK).")
    parser.add_argument("--n-workers", type=int, default=16, help="Parallel workers for questions (default: 16, optimized from testing; 1=sequential). Speeds run without changing ARK method.")
    parser.add_argument("--graph-name", type=str, default="prime", help="ARK KG (e.g. prime). Pass when you run to make the run explicit.")
    parser.add_argument("--ark-model", type=str, default="azure/gpt-5.4", help="ARK backbone. Pass when you run (e.g. azure/gpt-5.4).")
    parser.add_argument("--ark-agents", type=int, default=3, help="ARK parallel agents (paper A.2: n=3).")
    parser.add_argument("--ark-max-steps", type=int, default=20, help="ARK max steps per trajectory (paper A.2: Tmax=20).")
    parser.add_argument("--reasoning-effort", type=str, default=None, choices=["none", "low", "medium", "high", "xhigh"], help="reasoning_effort for gpt-5.x models. Applied to ALL models in pipeline (ARK agents, nodes_to_nl). Falls back to REASONING_EFFORT env var.")
    parser.add_argument(
        "--ark-input",
        type=str,
        choices=["full_conversation", "user_only"],
        default="full_conversation",
        help="What to send to ARK: full_conversation (default, conversation-aware retrieval) or user_only (user turns only, e.g. for ablations).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="kg_grounded",
        help="Identifier for this run (e.g. 'kg_grounded', 'no_kg'). Saved to each output line.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log progress every N questions (default: 10, set to 0 to disable logging).",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["bm25", "embedding", "hybrid"],
        default="bm25",
        help="Search mode for ARK graph retrieval: bm25 (keyword), embedding (dense vectors), or hybrid (RRF fusion). Default: bm25.",
    )
    args = parser.parse_args()

    ark_dir = Path(args.ark_dir).resolve()
    stark_dir = ark_dir / "benchmarks" / "stark"
    run_one = stark_dir / "run_one_question.py"
    if not run_one.exists():
        print(f"Error: {run_one} not found.", file=sys.stderr)
        sys.exit(1)
    python_exe = args.python or str(ark_dir / ".venv" / "bin" / "python")
    if not Path(python_exe).exists():
        python_exe = sys.executable

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        examples = examples[:args.limit]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing output by prompt_id so we skip already-done examples (no line-count assumption).
    id_to_result = load_id_to_result(output_path)
    to_compute = get_resume_to_compute(examples, id_to_result)
    if id_to_result:
        print(f"Resuming: reusing {len(id_to_result)} results by prompt_id, computing {len(to_compute)}.", file=sys.stderr)

    def process_one(idx: int, row: dict) -> tuple[int, dict]:
        """Run ARK for one example, then nodes_to_nl. Returns (idx, output_dict) for ordered write."""
        prompt_id = row.get("prompt_id", str(idx))
        prompt = row.get("prompt", [])
        rubrics = row.get("rubrics", [])
        example_tags = row.get("example_tags", [])
        if args.ark_input == "user_only":
            question_for_ark = get_question_from_prompt(prompt) or "(No user message)"
        else:
            question_for_ark = format_prompt_as_conversation_string(prompt) or "(No messages)"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(question_for_ark)
            q_file = f.name
        # Paper A.2: Prime, GPT-4.1, 3 agents, Tmax=20 (passed explicitly)
        cmd = [
            python_exe, str(run_one),
            "--question_file", q_file,
            "--graph_name", args.graph_name,
            "--model_name", args.ark_model,
            "--max_steps", str(args.ark_max_steps),
            "--number_of_agents", str(args.ark_agents),
            "--search-mode", args.search_mode,
        ]
        # If using embedding/hybrid search, compute embeddings path
        if args.search_mode in ["embedding", "hybrid"]:
            graph_dir = ark_dir / "benchmarks" / "stark" / "data" / "graphs" / args.graph_name
            embeddings_path = graph_dir / "embeddings_kalm.npy"
            if embeddings_path.exists():
                cmd.extend(["--embeddings-path", str(embeddings_path)])
        # Pass reasoning_effort to ARK subprocess (CLI arg takes priority, then env var)
        resolved_effort = args.reasoning_effort or os.environ.get("REASONING_EFFORT")
        if resolved_effort:
            cmd.extend(["--reasoning-effort", resolved_effort])
        _RATE_LIMIT_PATTERNS = ("RateLimitReached", "RateLimitError", "429")
        _ark_delay = 30
        result = None
        try:
            for _attempt in range(3):
                result = subprocess.run(
                    cmd,
                    cwd=str(stark_dir),
                    capture_output=True,
                    text=True,
                    timeout=1200,
                    env={**os.environ, "LITELLM_LOG": "ERROR"},
                )
                if result.returncode == 0:
                    break
                is_rate_limit = any(p in result.stderr for p in _RATE_LIMIT_PATTERNS)
                if is_rate_limit and _attempt < 2:
                    print(f"[ARK_RATELIMIT] attempt {_attempt+1}/3, retrying in {_ark_delay}s...", file=sys.stderr)
                    import time as _t; _t.sleep(_ark_delay); _ark_delay *= 2
                else:
                    break
        finally:
            Path(q_file).unlink(missing_ok=True)
        # Initialize defaults for failure cases
        response_text = ""
        node_summaries = []
        node_contributions = {}

        if result.returncode != 0:
            if args.search_mode != "bm25":
                print(f"[ARK_ERROR] search_mode={args.search_mode} returncode={result.returncode}", file=sys.stderr)
                _NOISE = ("Provider List", "Give Feedback", "LiteLLM.Info", "Loading weights")
                filtered = [l for l in result.stderr.split('\n')
                            if l.strip() and not any(p in l for p in _NOISE)]
                if filtered:
                    print(f"[ARK_STDERR]\n" + "\n".join(filtered[:20]), file=sys.stderr)
            pass  # Defaults remain empty
        else:
            # Log unexpected stderr from subprocess, filtering known benign noise
            if result.stderr:
                stderr_lines = [l for l in result.stderr.split('\n')
                                if l.strip()
                                and 'Provider List' not in l
                                and 'Loading weights' not in l]
                if stderr_lines:
                    print(f"[ARK subprocess stderr]", file=sys.stderr)
                    for line in stderr_lines[:10]:
                        print(f"  {line}", file=sys.stderr)

            try:
                # Strip leading noise (e.g. LiteLLM "Provider List" prints) before JSON
                stdout_clean = result.stdout
                json_start = stdout_clean.find('{')
                if json_start > 0:
                    stdout_clean = stdout_clean[json_start:]
                ark_out = json.loads(stdout_clean.strip())
            except json.JSONDecodeError:
                print(f"[ARK_PARSE_ERROR] Failed to parse JSON from subprocess for search_mode={args.search_mode}", file=sys.stderr)
                pass  # Defaults remain empty
            else:
                if ark_out.get("error"):
                    pass  # Defaults remain empty
                else:
                    # Extract node_summaries and call nodes_to_nl() which returns tuple
                    node_summaries = ark_out.get("node_summaries", [])
                    response_text, node_contributions = nodes_to_nl(prompt, node_summaries, args.nl_model)
                    # Normalize node indices to string format for consistency (Part 0.5)
                    node_summaries, node_contributions = normalize_node_indices(node_summaries, node_contributions)

        return (idx, {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response_text": response_text,
            "rubrics": rubrics,
            "example_tags": example_tags,
            "node_summaries": node_summaries,
            "node_contributions": node_contributions,
            "run_tag": args.run_tag,
            "kg_name": args.graph_name,
            "llm_model": args.nl_model,
        })

    new_results: dict[int, dict] = {}
    n_workers = max(1, int(args.n_workers))
    log_interval = max(0, args.log_interval)  # 0 disables logging
    if to_compute:
        if n_workers == 1:
            for k, i in enumerate(to_compute):
                _, out = process_one(i, examples[i])
                new_results[i] = out
                if (k + 1) % 10 == 0 or (k + 1) == len(to_compute):
                    print(f"  {k + 1}/{len(to_compute)} computed.", file=sys.stderr)
                # Log progress if logging enabled and checkpoint written
                if log_interval > 0 and (k + 1) % log_interval == 0:
                    try:
                        log_phase1_progress(str(output_path), k + 1, log_interval)
                    except Exception:
                        pass  # Don't fail if logging fails
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_one, i, examples[i]): i for i in to_compute}
                for future in as_completed(futures):
                    i, out = future.result()
                    new_results[i] = out
                    done += 1
                    if done % 10 == 0 or done == len(to_compute):
                        print(f"  {done}/{len(to_compute)} computed.", file=sys.stderr)
                    # Log progress if logging enabled and checkpoint written
                    if log_interval > 0 and done % log_interval == 0:
                        try:
                            log_phase1_progress(str(output_path), done, log_interval)
                        except Exception:
                            pass  # Don't fail if logging fails
        for i in to_compute:
            if i not in new_results:
                raise RuntimeError(f"Missing result for example index {i}")

    # Build full results in eval order (cache + newly computed) so Phase 2 index alignment holds.
    results = build_results_in_order(examples, id_to_result, new_results)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for out in results:
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} lines to {output_path}.", file=sys.stderr)


if __name__ == "__main__":
    main()
