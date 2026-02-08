"""
Phase 2: Run HealthBench eval oracle. Input: response jsonl (one NL answer per example).
Output: metrics (they grade on rubrics; we do not implement grading).

Usage (from healthbench repo root):
  python -m scripts.grade_ark_healthbench_responses \\
    --responses-jsonl ark_healthbench_responses.jsonl \\
    [--examples 5]  # optional: must match number of lines in jsonl if set
  # Or:
  python scripts/grade_ark_healthbench_responses.py \\
    --responses-jsonl ark_healthbench_responses.jsonl

Prints overall score and saves report HTML + full results JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import the upstream HealthBench implementation in this repo as a package.
# That code uses relative imports (e.g., `from . import common`), so we need the
# parent directory on sys.path and `healthbench/__init__.py` present.
repo_root = Path(__file__).resolve().parent.parent  # .../healthbench
repo_parent = repo_root.parent  # .../rshamji
if str(repo_parent) not in sys.path:
    sys.path.insert(0, str(repo_parent))

from healthbench import common, healthbench_eval  # type: ignore
from healthbench.healthbench_eval import HealthBenchEval  # type: ignore
from healthbench.sampler.precomputed_response_sampler import (  # type: ignore
    PrecomputedResponseSampler,
)
from healthbench.types import MessageList, SamplerResponse  # type: ignore

# Avoid importing OpenAI SDK just to get this constant.
OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."


class LiteLLMAzureGrader:
    """Grader that uses LiteLLM + AZURE_* env (same as ARK). Same interface as ChatCompletionSampler for HealthBenchEval."""

    def __init__(
        self,
        model: str = "azure/gpt-4.1",
        system_message: str | None = None,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.system_message = system_message
        self.max_tokens = max_tokens

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        from litellm import completion
        if self.system_message:
            message_list = [{"role": "system", "content": self.system_message}] + list(message_list)
        api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")
        api_base = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_BASE") or os.environ.get("AZURE_API_BASE")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("AZURE_API_VERSION")
        response = completion(
            model=self.model,
            messages=message_list,
            max_tokens=self.max_tokens,
            timeout=60,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return SamplerResponse(
            response_text=content,
            actual_queried_message_list=message_list,
            response_metadata={},
        )


def main():
    """
    Phase 2: Grade a response jsonl (from Phase 1) with HealthBench rubrics.
    Uses HealthBenchEval + PrecomputedResponseSampler; the grader model scores each
    rubric item per example, then we aggregate and write report + metrics.
    """
    parser = argparse.ArgumentParser(
        description="Grade precomputed responses (e.g. ARK) with HealthBench rubrics."
    )
    parser.add_argument(
        "--responses-jsonl",
        type=str,
        required=True,
        help="Path to response jsonl (same order as HealthBench examples).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Number of examples (default: number of lines in response jsonl). Must match.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Threads for grading (default 120, matches HealthBench for speed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write report and results (default: same dir as response jsonl).",
    )
    args = parser.parse_args()

    response_path = Path(args.responses_jsonl)
    if not response_path.exists():
        print(f"Error: {response_path} not found.", file=sys.stderr)
        sys.exit(1)

    # --- How many examples to grade: default = number of lines in the response jsonl ---
    with open(response_path, "r", encoding="utf-8") as f:
        num_responses = sum(1 for line in f if line.strip())
    num_examples = args.examples if args.examples is not None else num_responses
    if num_examples != num_responses:
        print(
            f"Warning: --examples={num_examples} but response jsonl has {num_responses} lines. Using {num_responses}.",
            file=sys.stderr,
        )
        num_examples = num_responses

    # --- Force HealthBenchEval to use the local eval jsonl (no network) ---
    local_eval_jsonl = repo_root / "2025-05-07-06-14-12_oss_eval.jsonl"
    if local_eval_jsonl.exists():
        # HealthBenchEval reads from module-level INPUT_PATH; blobfile can read local paths too.
        healthbench_eval.INPUT_PATH = str(local_eval_jsonl)
    else:
        print(
            f"Warning: local HealthBench eval jsonl not found at {local_eval_jsonl}; "
            "HealthBenchEval will fall back to its default INPUT_PATH.",
            file=sys.stderr,
        )

    # --- Grader: same model as ARK (LiteLLM + AZURE_* env), same interface as ChatCompletionSampler ---
    grader = LiteLLMAzureGrader(
        model="azure/gpt-4.1",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )

    # --- HealthBenchEval loads its own examples from the eval jsonl; we tell it how many to use ---
    # Order of examples is fixed by HealthBench (random.Random(0).sample), so Phase 1 must match.
    eval_obj = HealthBenchEval(
        grader_model=grader,
        num_examples=num_examples,
        n_repeats=1,
        n_threads=args.n_threads,
        subset_name=None,
    )

    # --- Sampler: no model call; we just return the next line's response_text for each example ---
    sampler = PrecomputedResponseSampler(response_jsonl_path=str(response_path))

    print("Running HealthBench grading on precomputed responses...", file=sys.stderr)
    result = eval_obj(sampler)

    # --- Write HTML report and metrics JSON (same format as normal HealthBench runs) ---
    out_dir = Path(args.output_dir) if args.output_dir else response_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = response_path.stem
    report_path = out_dir / f"{stem}_report.html"
    report_path.write_text(common.make_report(result), encoding="utf-8")
    print(f"Report saved to {report_path}", file=sys.stderr)

    metrics_path = out_dir / f"{stem}_metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics or {}, indent=2),
        encoding="utf-8",
    )
    print(f"Metrics saved to {metrics_path}", file=sys.stderr)

    print()
    print("Score:", result.score)
    # Print only top-level numeric metrics (skip tag-level and bootstrap stats for brevity)
    if result.metrics:
        for k, v in result.metrics.items():
            if ":" not in k and isinstance(v, (int, float)):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
