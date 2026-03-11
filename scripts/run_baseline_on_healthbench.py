"""
Step 3: Baseline (No-KG) Runner using direct litellm calls

This script generates Phase 1-compatible JSONL output for the no-KG baseline run.
- Calls GPT-4.1 directly via litellm (no KG context, no ARK)
- Generates response_text independently (different from Phase 1A responses)
- Output JSONL has identical 8 fields to run_ark_on_healthbench.py output
- SUPPORTS CHECKPOINTING: Can resume from partial output file

Pattern: Mirrors Phase 1A (run_ark_on_healthbench.py) but without ARK agent loop.
For each example: call GPT-4.1 directly → build Phase 1 schema dict → write JSONL

Resume behavior: If --output-path exists, loads it and only evaluates missing prompt_ids.

Usage:
    python run_baseline_on_healthbench.py \\
        --examples-jsonl data/healthbench_examples.jsonl \\
        --output-path responses_no_kg.jsonl \\
        --model gpt-4-turbo
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_completed_by_prompt_id(output_path: Path) -> dict[str, dict]:
    """Load existing output JSONL by prompt_id. Skip malformed lines."""
    completed = {}
    if not output_path.exists():
        return completed

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                prompt_id = obj.get("prompt_id")
                if prompt_id:
                    completed[prompt_id] = obj
            except json.JSONDecodeError as e:
                print(f"Skipping malformed checkpoint line: {e}", file=sys.stderr)
                continue

    return completed


def call_gpt_baseline(
    prompt: list,
    model_name: str = "azure/gpt-4.1",
    max_retries: int = 3,
) -> str:
    """
    Call GPT-4.1 directly via litellm for baseline (no KG, no ARK).

    Returns: response_text (clean string)
    Uses AZURE_* env vars (same as ARK nodes_to_nl).
    """
    from litellm import completion

    # Build user message: ask model to answer based on conversation only
    user_message = """You are a helpful clinical health assistant. Answer the latest question based on the conversation history provided. Do not use any external knowledge bases or graphs. Provide a clear, concise response."""

    # Build messages: use provided conversation + instruction
    messages = list(prompt) + [{"role": "user", "content": user_message}]

    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")
    api_base = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_BASE") or os.environ.get("AZURE_API_BASE")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("AZURE_API_VERSION")

    # Retry logic: retry on failure
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
            response_text = content.strip()

            if response_text:
                return response_text
            elif attempt < max_retries - 1:
                print(f"Warning: Empty response from GPT-4.1, retrying (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                continue
            else:
                return ""

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Warning: GPT-4.1 call failed (attempt {attempt + 1}/{max_retries}): {e}", file=sys.stderr)
            else:
                print(f"Error: GPT-4.1 call failed after {max_retries} attempts: {e}", file=sys.stderr)
                return ""

    return ""


def build_phase1_dict(
    prompt_id: str,
    prompt: list,
    response_text: str,
    example_tags: list,
) -> dict:
    """
    Build Phase 1 schema dict for baseline (no KG).

    Phase 1 schema (8 fields, identical to run_ark_on_healthbench.py):
    {
        "prompt_id": str,
        "prompt": list[dict],  # conversation messages
        "response_text": str,
        "rubrics": list,       # (empty for baseline, populated by Phase 2 grader)
        "example_tags": list,  # original example tags
        "node_summaries": [],       # EMPTY for baseline (no KG)
        "node_contributions": {},   # EMPTY for baseline (no KG)
        "run_tag": "no_kg"
    }
    """
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "response_text": response_text,
        "rubrics": [],  # Will be filled by Phase 2 grader
        "example_tags": example_tags,
        "node_summaries": [],  # No KG for baseline
        "node_contributions": {},  # No KG for baseline
        "run_tag": "no_kg",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Baseline (no-KG) runner: calls GPT-4.1 directly via litellm. Phase 1 schema output. Supports resuming from checkpoint."
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
        help="Output JSONL path (Phase 1 schema with run_tag='no_kg'). If exists, resumes from checkpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="azure/gpt-4.1",
        help="Model to use for baseline (default: azure/gpt-4.1, passed to litellm).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16, optimized from testing; 1=sequential). Set >1 for parallel evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N examples (for testing). Default: None (all examples).",
    )
    args = parser.parse_args()

    # Load examples
    examples = []
    examples_path = Path(args.examples_jsonl).resolve()
    if not examples_path.exists():
        print(f"Error: {examples_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(examples_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Failed to parse example {i}: {e}", file=sys.stderr)
                    continue

    print(f"Loaded {len(examples)} examples from {examples_path}", file=sys.stderr)

    # Apply limit BEFORE checkpoint filtering (so we only evaluate on limited set)
    if args.limit and len(examples) > args.limit:
        examples = examples[:args.limit]
        print(f"Limited to first {len(examples)} examples", file=sys.stderr)

    # Load checkpoint (resume if output already exists)
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_by_id = load_completed_by_prompt_id(output_path)
    if completed_by_id:
        print(f"Resuming: loaded {len(completed_by_id)} completed results from {output_path}", file=sys.stderr)

    # Identify which examples we still need to evaluate
    to_compute = [i for i, e in enumerate(examples) if e.get("prompt_id") not in completed_by_id]
    if not to_compute:
        print(f"All {len(examples)} examples already completed. Skipping evaluation.", file=sys.stderr)
    else:
        print(f"Running GPT-4.1 baseline on {len(to_compute)} remaining examples (no KG)...", file=sys.stderr)

        def process_one(idx: int) -> tuple[int, dict]:
            """Call GPT-4.1 for one example. Returns (idx, phase1_dict)."""
            example = examples[idx]
            prompt_id = example.get("prompt_id", str(idx))
            prompt = example.get("prompt", [])
            example_tags = example.get("example_tags", [])

            response_text = call_gpt_baseline(prompt, model_name=args.model)

            return (idx, build_phase1_dict(prompt_id, prompt, response_text, example_tags))

        new_results: dict[int, dict] = {}
        n_workers = max(1, int(args.n_workers))
        if n_workers == 1:
            # Sequential evaluation
            for k, i in enumerate(to_compute):
                _, out = process_one(i)
                new_results[i] = out
                if (k + 1) % 10 == 0 or (k + 1) == len(to_compute):
                    print(f"  {k + 1}/{len(to_compute)} completed.", file=sys.stderr)
        else:
            # Parallel evaluation
            done = 0
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_one, i): i for i in to_compute}
                for future in as_completed(futures):
                    i, out = future.result()
                    new_results[i] = out
                    done += 1
                    if done % 10 == 0 or done == len(to_compute):
                        print(f"  {done}/{len(to_compute)} completed.", file=sys.stderr)

        # Append new results to output file
        with open(output_path, "a", encoding="utf-8") as f:
            for result in new_results.values():
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Appended {len(new_results)} new results to {output_path}", file=sys.stderr)

    # Write final output in original example order
    print("Writing final output in original example order...", file=sys.stderr)
    final_completed = load_completed_by_prompt_id(output_path)

    final_results = []
    for i, example in enumerate(examples):
        prompt_id = example.get("prompt_id")
        if prompt_id in final_completed:
            final_results.append(final_completed[prompt_id])
        else:
            print(f"Warning: No result for example {i} (prompt_id={prompt_id})", file=sys.stderr)

    # Rewrite output file in correct order (idempotent)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in final_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final_results)} total lines to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
