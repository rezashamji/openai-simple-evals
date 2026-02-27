"""
Step 3: Baseline (No-KG) Runner using simple-evals

This script generates Phase 1-compatible JSONL output for the no-KG baseline run.
- Uses simple-evals to evaluate GPT-4.1 on HealthBench (no KG context)
- Reformats output to match Phase 1 schema for downstream comparison
- Output JSONL has identical 8 fields to run_ark_on_healthbench.py output

The key insight: simple-evals already handles GPT-4.1 calls, retries, and grading.
We just reformat its output to match Phase 1 schema so both runs are directly comparable.

Usage:
    python run_baseline_on_healthbench.py \\
        --examples-jsonl data/healthbench_examples.jsonl \\
        --output-path responses_no_kg.jsonl \\
        --model gpt-4-turbo
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_simple_evals_baseline(
    model: str = "gpt-4-turbo",
    n_threads: int = 120,
) -> dict:
    """
    Run simple-evals baseline (HealthBench, no KG).

    Returns: {prompt_id: {response_text, rubrics, example_tags, prompt, ...}}
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        temp_output = tmp.name

    try:
        # Run simple-evals with HealthBench eval
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "simple_evals.simple_evals",
                "--eval",
                "healthbench",
                "--model",
                model,
                "--n-threads",
                str(n_threads),
                "--output_path",
                temp_output,
            ],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max
        )

        if result.returncode != 0:
            print(
                f"simple-evals failed with return code {result.returncode}",
                file=sys.stderr,
            )
            print(f"stderr: {result.stderr}", file=sys.stderr)
            return {}

        # Parse simple-evals output JSONL
        id_to_result = {}
        with open(temp_output, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    prompt_id = obj.get("prompt_id")
                    if prompt_id:
                        id_to_result[prompt_id] = obj
                except json.JSONDecodeError as e:
                    print(
                        f"Failed to parse line in simple-evals output: {e}",
                        file=sys.stderr,
                    )
                    continue

        print(
            f"Loaded {len(id_to_result)} results from simple-evals",
            file=sys.stderr,
        )
        return id_to_result

    finally:
        Path(temp_output).unlink(missing_ok=True)


def reformat_to_phase1_schema(
    examples: list,
    simple_evals_results: dict,
) -> list:
    """
    Reformat simple-evals output to match Phase 1 JSONL schema.

    Phase 1 schema (8 fields, identical to run_ark_on_healthbench.py):
    {
        "prompt_id": str,
        "prompt": list[dict],  # conversation messages
        "response_text": str,
        "rubrics": list,       # rubric items with grades (from eval)
        "example_tags": list,  # original example tags
        "node_summaries": [],       # EMPTY for baseline (no KG)
        "node_contributions": {},   # EMPTY for baseline (no KG)
        "run_tag": "no_kg"
    }
    """
    phase1_results = []

    for example in examples:
        prompt_id = example.get("prompt_id")
        se_result = simple_evals_results.get(prompt_id, {})

        # Extract rubrics with grades from simple-evals output
        # simple-evals stores this in example_level_metadata.rubric_items
        rubric_items = []
        if "example_level_metadata" in se_result:
            metadata = se_result["example_level_metadata"]
            if "rubric_items" in metadata:
                rubric_items = metadata["rubric_items"]

        phase1_result = {
            "prompt_id": prompt_id,
            "prompt": example.get("prompt", []),
            "response_text": se_result.get("completion", [{"content": ""}])[0].get(
                "content", ""
            ),
            "rubrics": rubric_items,
            "example_tags": example.get("example_tags", []),
            "node_summaries": [],  # No KG for baseline
            "node_contributions": {},  # No KG for baseline
            "run_tag": "no_kg",
        }
        phase1_results.append(phase1_result)

    return phase1_results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline (no-KG) run using simple-evals, reformatted to Phase 1 schema."
    )
    parser.add_argument(
        "--examples-jsonl",
        type=str,
        required=True,
        help="Path to HealthBench examples JSONL (input).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output JSONL path (Phase 1 schema with run_tag='no_kg').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo",
        help="Model to use for baseline (default: gpt-4-turbo). Passed to simple-evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads for simple-evals (default: 120).",
    )
    args = parser.parse_args()

    # Load examples
    examples = []
    examples_path = Path(args.examples_jsonl).resolve()
    if not examples_path.exists():
        print(f"Error: {examples_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(examples_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Failed to parse example: {e}", file=sys.stderr)
                    continue

    print(f"Loaded {len(examples)} examples from {examples_path}", file=sys.stderr)

    # Run simple-evals baseline
    print("Running simple-evals baseline (no KG)...", file=sys.stderr)
    se_results = run_simple_evals_baseline(
        model=args.model,
        n_threads=args.n_threads,
    )

    if not se_results:
        print("Warning: simple-evals produced no results", file=sys.stderr)

    # Reformat to Phase 1 schema
    print("Reformatting to Phase 1 schema...", file=sys.stderr)
    phase1_results = reformat_to_phase1_schema(examples, se_results)

    # Write Phase 1 schema output
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in phase1_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(phase1_results)} lines to {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
