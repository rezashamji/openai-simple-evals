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
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def get_question_from_prompt(prompt: list) -> str:
    """Extract user content from HealthBench prompt (list of message dicts). Same order as eval. Optional for ARK via --ark-input user_only."""
    parts = [m["content"] for m in prompt if m.get("role") == "user"]
    return "\n\n".join(parts).strip() if parts else ""


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


def parse_json_to_dict(json_string: str) -> dict:
    """Parse JSON string, handling markdown code blocks. Return dict or empty dict on failure."""
    import json as json_lib
    json_string = json_string.strip()
    if json_string.startswith("```"):
        json_string = json_string.split("```")[1].strip()
        if json_string.startswith("json"):
            json_string = json_string[4:].strip()
    try:
        return json_lib.loads(json_string)
    except (json_lib.JSONDecodeError, ValueError):
        return {}


def nodes_to_nl(prompt: list, node_summaries: list, model_name: str = "azure/gpt-4.1") -> tuple[str, dict]:
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

    # Retry logic: retry if JSON is malformed or node_contributions is incomplete (per plan Step 2a)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model_name,
                messages=messages,
                max_tokens=2048,
                timeout=30,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
            )
            content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            parsed = parse_json_to_dict(content.strip())

            # Validate that we got both required fields
            if "final_answer" not in parsed or "node_contributions" not in parsed:
                if attempt < max_retries - 1:
                    print(f"Warning: JSON missing final_answer or node_contributions, retrying (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                    continue
                else:
                    # Fallback: return what we could parse
                    response_text = parsed.get("final_answer", "")
                    node_contributions = parsed.get("node_contributions", {})
                    return (response_text, node_contributions)

            response_text = parsed["final_answer"].strip()
            node_contributions = parsed["node_contributions"]

            # Validate node_contributions completeness using 60% threshold (per updated plan Step 2a)
            expected_node_count = len(node_summaries)
            actual_contrib_count = len(node_contributions)
            completion_ratio = actual_contrib_count / expected_node_count if expected_node_count > 0 else 0

            if completion_ratio < 0.6:
                if attempt < max_retries - 1:
                    print(f"Warning: node_contributions incomplete ({actual_contrib_count}/{expected_node_count} = {completion_ratio:.1%} < 60%), retrying (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                    continue
                else:
                    # Fallback: accept with poor tracking (will be reported in final stats)
                    print(f"Warning: Accepting incomplete node_contributions ({actual_contrib_count}/{expected_node_count} = {completion_ratio:.1%}) after max retries", file=sys.stderr)
                    return (response_text, node_contributions)

            # Success: valid JSON with both fields and ≥60% completion
            return (response_text, node_contributions)

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Warning: nodes_to_nl() call failed (attempt {attempt + 1}/{max_retries}): {e}", file=sys.stderr)
            else:
                print(f"Error: nodes_to_nl() failed after {max_retries} attempts: {e}", file=sys.stderr)
                return ("", {})

    # Should not reach here, but fallback just in case
    return ("", {})


def main():
    parser = argparse.ArgumentParser(description="Phase 1: HealthBench Qs → ARK oracle → nodes→NL → response jsonl.")
    parser.add_argument("--input-jsonl", type=str, required=True, help="HealthBench eval jsonl.")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--ark-dir", type=str, required=True, help="Path to ARK repo (has benchmarks/stark/run_one_question.py).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples (same order as HealthBench eval).")
    parser.add_argument("--python", type=str, default=None, help="Python for ARK subprocess (default: ark/.venv/bin/python).")
    parser.add_argument("--nl-model", type=str, default="azure/gpt-4.1", help="Model for nodes→NL (default same as ARK).")
    parser.add_argument("--n-workers", type=int, default=1, help="Parallel workers for questions (1=sequential). Speeds run without changing ARK method.")
    parser.add_argument("--graph-name", type=str, default="prime", help="ARK KG (e.g. prime). Pass when you run to make the run explicit.")
    parser.add_argument("--ark-model", type=str, default="azure/gpt-4.1", help="ARK backbone. Pass when you run (e.g. azure/gpt-4.1).")
    parser.add_argument("--ark-agents", type=int, default=3, help="ARK parallel agents (paper A.2: n=3).")
    parser.add_argument("--ark-max-steps", type=int, default=20, help="ARK max steps per trajectory (paper A.2: Tmax=20).")
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
        n = min(args.limit, len(examples))
        examples = random.Random(0).sample(examples, n)

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
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(stark_dir),
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ},
            )
        finally:
            Path(q_file).unlink(missing_ok=True)
        # Initialize defaults for failure cases
        response_text = ""
        node_summaries = []
        node_contributions = {}

        if result.returncode != 0:
            pass  # Defaults remain empty
        else:
            try:
                ark_out = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                pass  # Defaults remain empty
            else:
                if ark_out.get("error"):
                    pass  # Defaults remain empty
                else:
                    # Extract node_summaries and call nodes_to_nl() which returns tuple
                    node_summaries = ark_out.get("node_summaries", [])
                    response_text, node_contributions = nodes_to_nl(prompt, node_summaries, args.nl_model)

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
    if to_compute:
        if n_workers == 1:
            for k, i in enumerate(to_compute):
                _, out = process_one(i, examples[i])
                new_results[i] = out
                if (k + 1) % 10 == 0 or (k + 1) == len(to_compute):
                    print(f"  {k + 1}/{len(to_compute)} computed.", file=sys.stderr)
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
