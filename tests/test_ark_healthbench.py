"""
Minimal tests for the ARK–HealthBench bridge.

We only test: (1) question extraction from HealthBench prompt, (2) Phase 2 runs and
produces a score when given a response jsonl (mock grader so no API).
Run from healthbench root: python tests/test_ark_healthbench.py
"""

import json
import sys
import tempfile
from pathlib import Path

if sys.version_info < (3, 8):
    print("SKIP: Python 3.8+ required", file=sys.stderr)
    sys.exit(0)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from healthbench.scripts.run_ark_on_healthbench import get_question_from_prompt


def test_get_question_from_prompt():
    """We extract user content from HealthBench prompt (same as eval order)."""
    assert get_question_from_prompt([{"role": "user", "content": "What is diabetes?"}]) == "What is diabetes?"
    assert get_question_from_prompt([
        {"role": "user", "content": "First."},
        {"role": "assistant", "content": "Reply."},
        {"role": "user", "content": "Follow-up."},
    ]) == "First.\n\nFollow-up."
    assert get_question_from_prompt([{"role": "assistant", "content": "x"}]) == ""


def test_phase2_grade_with_mock_grader():
    """Phase 2 (HealthBench eval oracle) runs on a response jsonl; we use mock grader so no API."""
    try:
        from healthbench import healthbench_eval
        from healthbench.healthbench_eval import HealthBenchEval
        from healthbench.sampler.precomputed_response_sampler import PrecomputedResponseSampler
        from healthbench.types import SamplerResponse
    except ImportError as e:
        print(f"SKIP Phase 2 test (missing deps, e.g. blobfile/numpy/pandas): {e}", file=sys.stderr)
        return

    class MockGrader:
        def __call__(self, message_list):
            return SamplerResponse(
                response_text='{"criteria_met": true, "explanation": "test"}',
                actual_queried_message_list=message_list,
                response_metadata={},
            )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({
            "prompt_id": "t",
            "prompt": [{"role": "user", "content": "Q"}],
            "response_text": "Answer.",
            "rubrics": [],
            "example_tags": [],
        }, ensure_ascii=False) + "\n")
        path = f.name
    try:
        # Force HealthBenchEval to use local data (no network/Azure credentials).
        local_eval_jsonl = REPO_ROOT / "2025-05-07-06-14-12_oss_eval.jsonl"
        if local_eval_jsonl.exists():
            healthbench_eval.INPUT_PATH = str(local_eval_jsonl)

        eval_obj = HealthBenchEval(grader_model=MockGrader(), num_examples=1, n_repeats=1, n_threads=1, subset_name=None)
        sampler = PrecomputedResponseSampler(response_jsonl_path=path)
        result = eval_obj(sampler)
        assert result.score is not None and 0 <= result.score <= 1
    finally:
        Path(path).unlink(missing_ok=True)


def run_all():
    test_get_question_from_prompt()
    test_phase2_grade_with_mock_grader()
    print("All tests passed.")



if __name__ == "__main__":
    run_all()
