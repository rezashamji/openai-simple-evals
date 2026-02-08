"""
Phase 1: Our only logic. Four steps total; we use two oracles and do minimal glue.

  ORACLE 1 – ARK: question + KG (e.g. prime) + LLM → ranked nodes.
    We do not implement or import any ARK logic. We subprocess-call the ARK repo
    (run_one_question.py): question in → JSON with nodes+summaries out.

  OUR CODE:
    1. Deconstruct HealthBench dataset → get questions (same order as HealthBench eval).
    2. For each question: call ARK oracle → parse nodes JSON.
    3. nodes→NL: one LLM call to turn node summaries + question into natural language.
    4. Write response jsonl (one line per example).

  ORACLE 2 – HealthBench eval: takes response jsonl, grades on rubrics, returns metrics.
    We do not implement grading. Phase 2 is grade_ark_healthbench_responses.py (calls their eval).

  ARK config: pass --graph-name, --ark-model, --ark-agents, --ark-max-steps when you run
  (e.g. from Slurm). Paper Appendix A.2 uses Prime KG, GPT-4.1, n=3 agents, Tmax=20.
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
    """Extract user content from HealthBench prompt (list of message dicts). Same order as eval."""
    parts = [m["content"] for m in prompt if m.get("role") == "user"]
    return "\n\n".join(parts).strip() if parts else ""


def nodes_to_nl(question: str, node_summaries: list, model_name: str = "azure/gpt-4.1") -> str:
    """Our only logic beyond glue: turn ARK's node summaries + question into one NL reply.
    Uses LiteLLM + AZURE_* env (same as ARK) so we use the same model/API."""
    from litellm import completion
    if not node_summaries:
        return "I could not find relevant information in the knowledge base for this question."
    blocks = [s.get("summary") or s.get("name") or f"(node {s.get('index', '')})" for s in node_summaries]
    context = "\n\n".join(blocks)
    prompt_text = f"""Question: {question}

The following is relevant information from the knowledge graph. Each paragraph (block of text separated by a blank line) is one item, ordered from most to least relevant to the question.

{context}

Using the relevant information above, write a complete, helpful reply to the question as a supportive assistant. Do not mention nodes, indices, or the graph."""
    messages = [{"role": "user", "content": prompt_text}]
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")
    api_base = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_BASE") or os.environ.get("AZURE_API_BASE")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("AZURE_API_VERSION")
    response = completion(
        model=model_name,
        messages=messages,
        max_tokens=1024,
        timeout=30,
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
    )
    content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return content.strip()


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

    def process_one(idx: int, row: dict) -> tuple[int, dict]:
        """Run ARK for one question, then nodes_to_nl. Returns (idx, output_dict) for ordered write."""
        prompt_id = row.get("prompt_id", str(idx))
        prompt = row.get("prompt", [])
        rubrics = row.get("rubrics", [])
        example_tags = row.get("example_tags", [])
        question = get_question_from_prompt(prompt) or "(No user message)"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(question)
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
        if result.returncode != 0:
            response_text = ""
        else:
            try:
                ark_out = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                response_text = ""
            else:
                if ark_out.get("error"):
                    response_text = ""
                else:
                    response_text = nodes_to_nl(question, ark_out.get("node_summaries", []), args.nl_model)
        return (idx, {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response_text": response_text,
            "rubrics": rubrics,
            "example_tags": example_tags,
        })

    n_workers = max(1, int(args.n_workers))
    if n_workers == 1:
        results = [None] * len(examples)
        for i, row in enumerate(examples):
            _, out = process_one(i, row)
            results[i] = out
            if (i + 1) % 10 == 0 or (i + 1) == len(examples):
                print(f"  {i + 1}/{len(examples)} done.", file=sys.stderr)
    else:
        results = [None] * len(examples)
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_one, i, row): i for i, row in enumerate(examples)}
            for future in as_completed(futures):
                i, out = future.result()
                results[i] = out
                done += 1
                if done % 10 == 0 or done == len(examples):
                    print(f"  {done}/{len(examples)} done.", file=sys.stderr)
        for i, out in enumerate(results):
            if out is None:
                raise RuntimeError(f"Missing result for example index {i}")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for out in results:
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} lines to {output_path}.", file=sys.stderr)


if __name__ == "__main__":
    main()
