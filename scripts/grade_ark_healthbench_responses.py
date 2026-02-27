"""
Phase 2: Run HealthBench eval oracle. Input: response jsonl (one NL answer per example).
Output: metrics (they grade on rubrics; we do not implement grading).

Usage (from project root, i.e. parent of simple-evals/):
  python -m simple_evals.scripts.grade_ark_healthbench_responses \\
    --responses-jsonl ark_healthbench_responses.jsonl \\
    --output-dir output_dir/  \\
    [--examples 5]  # optional: must match number of lines in jsonl if set

Prints overall score and saves report HTML + full results JSON.
Supports --checkpoint for resumable grading after crashes.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Import from simple-evals package using relative imports
# Run as: python -m simple_evals.scripts.grade_ark_healthbench_responses
from .. import common  # type: ignore
from .. import healthbench_eval  # type: ignore
from ..healthbench_eval import (  # type: ignore
    HEALTHBENCH_HTML_JINJA,
    HealthBenchEval,
    _aggregate_get_clipped_mean,
    get_usage_dict,
)
from ..sampler.precomputed_response_sampler import (  # type: ignore
    PrecomputedResponseSampler,
)
from ..types import MessageList, SamplerResponse, SingleEvalResult  # type: ignore

# Avoid importing OpenAI SDK just to get this constant.
OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."


class ContentFilterSkipError(Exception):
    """Raised when grading hits a content policy violation; caller should skip this example."""


class LiteLLMAzureGrader:
    """Grader that uses LiteLLM + AZURE_* env (same as ARK). Same interface as ChatCompletionSampler for HealthBenchEval."""

    def __init__(
        self,
        model: str = "azure/gpt-4.1",
        system_message: str | None = None,
        max_tokens: int = 2048,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
    ):
        self.model = model
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        from litellm import RateLimitError, completion

        if self.system_message:
            message_list = [{"role": "system", "content": self.system_message}] + list(message_list)
        api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")
        api_base = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_BASE") or os.environ.get("AZURE_API_BASE")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("AZURE_API_VERSION")

        last_error = None
        for attempt in range(self.max_retries):
            try:
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
            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2**attempt)
                    logging.warning("Rate limit (429), retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, self.max_retries)
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                err_str = str(e).lower()
                if "content_filter" in err_str or "content management policy" in err_str or "responsibleaipolicyviolation" in err_str:
                    raise ContentFilterSkipError(str(e)) from e
                # 500 / server errors: retry with backoff (same as 429)
                if any(x in err_str for x in ("500", "internal server error", "server had an error", "502", "503")):
                    last_error = e
                    if attempt < self.max_retries - 1:
                        delay = self.retry_base_delay * (2**attempt)
                        logging.warning("Server error (5xx), retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, self.max_retries)
                        time.sleep(delay)
                    else:
                        raise
                    continue
                raise
        assert last_error is not None
        raise last_error


def _single_result_to_checkpoint_dict(index: int, r: SingleEvalResult) -> dict:
    """Serialize SingleEvalResult for JSONL checkpoint (omit None/non-serializable)."""
    return {
        "index": index,
        "score": r.score,
        "metrics": r.metrics,
        "html": r.html or "",
        "convo": r.convo or [],
        "example_level_metadata": r.example_level_metadata or {},
    }


def _checkpoint_dict_to_single_result(d: dict) -> SingleEvalResult:
    """Deserialize checkpoint line back to SingleEvalResult."""
    return SingleEvalResult(
        score=d.get("score"),
        metrics=d.get("metrics", {}),
        html=d.get("html") or None,
        convo=d.get("convo") or None,
        example_level_metadata=d.get("example_level_metadata") or None,
    )


def _load_checkpoint(checkpoint_path: Path) -> dict[int, SingleEvalResult]:
    """Load checkpoint file; return map index -> SingleEvalResult. Skip malformed lines."""
    completed: dict[int, SingleEvalResult] = {}
    if not checkpoint_path.exists():
        return completed
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                idx = d["index"]
                completed[idx] = _checkpoint_dict_to_single_result(d)
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning("Skipping malformed checkpoint line: %s", e)
    return completed


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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to grading checkpoint JSONL (default: {output_dir}/{stem}_grading_checkpoint.jsonl). "
        "Enables resumable grading after crashes.",
    )
    parser.add_argument(
        "--validate-node-contributions",
        action="store_true",
        default=False,
        help="Enable cross-checking of node contribution claims against raw node summaries (KG enrichment).",
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
    repo_root = Path(__file__).resolve().parent.parent  # .../simple-evals
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

    # --- Output dir and checkpoint path ---
    out_dir = Path(args.output_dir) if args.output_dir else response_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = response_path.stem
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / f"{stem}_grading_checkpoint.jsonl"

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

    # --- Load checkpoint if resuming ---
    completed: dict[int, SingleEvalResult] = _load_checkpoint(checkpoint_path)
    if completed:
        print(f"Resuming: loaded {len(completed)} graded examples from {checkpoint_path}", file=sys.stderr)

    # --- Grade loop with checkpoint: process each example, skip cached, handle content-filter ---
    examples = eval_obj.examples
    skipped_content_filter = 0
    print("Running HealthBench grading on precomputed responses...", file=sys.stderr)
    for i in tqdm(range(len(examples)), desc="Grading"):
        if i in completed:
            continue
        row = examples[i]
        prompt_id = row.get("prompt_id", f"index_{i}")
        prompt_messages = row["prompt"]
        logging.info("Grading index=%d prompt_id=%s", i, prompt_id)
        try:
            sampler_response = sampler.get_response_at_index(i, prompt_messages)
            response_text = sampler_response.response_text
            response_usage = sampler_response.response_metadata.get("usage", None)
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list

            # Extract KG-related fields from sampler response (contains full Phase 1 output)
            node_contributions = sampler_response.response_metadata.get("node_contributions", {})
            node_summaries = sampler_response.response_metadata.get("node_summaries", [])
            run_tag = sampler_response.response_metadata.get("run_tag", "no_kg")

            metrics, readable_explanation_str, rubric_items_with_grades, kg_reasoning_details = eval_obj.grade_sample(
                prompt=actual_queried_prompt_messages,
                response_text=response_text,
                rubric_items=row["rubrics"],
                example_tags=row["example_tags"],
                node_contributions=node_contributions,
                node_summaries=node_summaries,
                run_tag=run_tag,
                validate_node_contributions=args.validate_node_contributions,
            )
            score = metrics["overall_score"]

            html = common.jinja_env.from_string(
                HEALTHBENCH_HTML_JINJA.replace(
                    "{{ rubric_grades }}",
                    readable_explanation_str.replace("\n", "<br>"),
                )
            ).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=metrics["overall_score"],
                extracted_answer=response_text,
            )
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
            single_result = SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
                example_level_metadata={
                    "score": score,
                    "usage": get_usage_dict(response_usage),
                    "rubric_items": rubric_items_with_grades,
                    "prompt": actual_queried_prompt_messages,
                    "completion": [dict(content=response_text, role="assistant")],
                    "prompt_id": prompt_id,
                    "completion_id": hashlib.sha256(
                        (prompt_id + response_text).encode("utf-8")
                    ).hexdigest(),
                    "kg_reasoning_details": kg_reasoning_details,
                    "run_tag": run_tag,
                },
            )
            completed[i] = single_result
            # Append to checkpoint after each successful grade
            with open(checkpoint_path, "a", encoding="utf-8") as cf:
                cf.write(json.dumps(_single_result_to_checkpoint_dict(i, single_result)) + "\n")
        except ContentFilterSkipError as e:
            skipped_content_filter += 1
            logging.warning("Content filter skip index=%d prompt_id=%s: %s", i, prompt_id, e)
        except Exception as e:
            logging.error("Grading failed index=%d prompt_id=%s: %s", i, prompt_id, e)
            raise

    if skipped_content_filter:
        print(f"Skipped {skipped_content_filter} examples due to content filter.", file=sys.stderr)

    # --- Aggregate and write report (only successfully graded examples) ---
    results = [completed[i] for i in range(len(examples)) if i in completed]
    if not results:
        print("Error: No examples were graded successfully.", file=sys.stderr)
        sys.exit(1)
    result = _aggregate_get_clipped_mean(results)
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
