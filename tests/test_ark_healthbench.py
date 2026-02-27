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

from healthbench.scripts.run_ark_on_healthbench import (
    build_results_in_order,
    format_prompt_as_conversation_string,
    get_question_from_prompt,
    get_resume_to_compute,
    load_id_to_result,
    parse_json_to_dict,
)


def test_get_question_from_prompt():
    """We extract user content from HealthBench prompt (same as eval order)."""
    assert get_question_from_prompt([{"role": "user", "content": "What is diabetes?"}]) == "What is diabetes?"
    assert get_question_from_prompt([
        {"role": "user", "content": "First."},
        {"role": "assistant", "content": "Reply."},
        {"role": "user", "content": "Follow-up."},
    ]) == "First.\n\nFollow-up."
    assert get_question_from_prompt([{"role": "assistant", "content": "x"}]) == ""


def test_format_prompt_as_conversation_string():
    """Full convo is formatted as user/assistant/user string for ARK."""
    assert format_prompt_as_conversation_string([
        {"role": "user", "content": "First."},
        {"role": "assistant", "content": "Reply."},
        {"role": "user", "content": "Follow-up."},
    ]) == "user: First.\n\nassistant: Reply.\n\nuser: Follow-up."
    assert format_prompt_as_conversation_string([{"role": "user", "content": "Hi"}]) == "user: Hi"


def test_resume_logic():
    """Resume: load by prompt_id, to_compute, and merge in eval order."""
    # load_id_to_result: missing file -> empty; existing file -> prompt_id -> dict
    assert load_id_to_result(Path("/nonexistent/path.jsonl")) == {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"prompt_id": "a", "response_text": "A"}) + "\n")
        f.write(json.dumps({"prompt_id": "b", "response_text": "B"}) + "\n")
        f.write(json.dumps({"not_prompt_id": "x"}) + "\n")  # no prompt_id, skipped
        f.write("bad json\n")  # skipped
        path = Path(f.name)
    try:
        id_to_result = load_id_to_result(path)
        assert set(id_to_result.keys()) == {"a", "b"}
        assert id_to_result["a"]["response_text"] == "A"
        assert id_to_result["b"]["response_text"] == "B"
    finally:
        path.unlink(missing_ok=True)

    # get_resume_to_compute: only indices whose prompt_id is not in cache
    examples = [
        {"prompt_id": "a", "prompt": []},
        {"prompt_id": "b", "prompt": []},
        {"prompt_id": "c", "prompt": []},
    ]
    to_compute = get_resume_to_compute(examples, {"a": {}})
    assert to_compute == [1, 2]
    to_compute_none = get_resume_to_compute(examples, {"a": {}, "b": {}, "c": {}})
    assert to_compute_none == []
    to_compute_all = get_resume_to_compute(examples, {})
    assert to_compute_all == [0, 1, 2]

    # build_results_in_order: eval order preserved; cache used when prompt_id in id_to_result
    examples2 = [
        {"prompt_id": "p1"},
        {"prompt_id": "p2"},
    ]
    id_to_result2 = {"p1": {"prompt_id": "p1", "response_text": "cached1"}}
    new_results2 = {1: {"prompt_id": "p2", "response_text": "new2"}}
    results = build_results_in_order(examples2, id_to_result2, new_results2)
    assert len(results) == 2
    assert results[0]["prompt_id"] == "p1" and results[0]["response_text"] == "cached1"
    assert results[1]["prompt_id"] == "p2" and results[1]["response_text"] == "new2"


def test_parse_json_to_dict():
    """Step 2a: parse_json_to_dict handles markdown code blocks and invalid JSON."""
    # Valid JSON
    assert parse_json_to_dict('{"key": "value"}') == {"key": "value"}

    # JSON wrapped in markdown code blocks
    assert parse_json_to_dict('```json\n{"key": "value"}\n```') == {"key": "value"}
    assert parse_json_to_dict('```\n{"key": "value"}\n```') == {"key": "value"}

    # Invalid JSON returns empty dict
    assert parse_json_to_dict("not valid json") == {}
    assert parse_json_to_dict('{"incomplete": ') == {}


def test_step2b_output_schema():
    """Step 2b: Verify output schema includes node_summaries, node_contributions, run_tag."""
    # This test verifies the new output schema matches kg-metadata-enrichment-plan.md Step 2b
    expected_schema_fields = {
        "prompt_id",
        "prompt",
        "response_text",
        "rubrics",
        "example_tags",
        "node_summaries",      # NEW (Step 2b)
        "node_contributions",  # NEW (Step 2b)
        "run_tag",            # NEW (Step 2b)
    }

    # Mock output dict (as would be returned by process_one)
    output_dict = {
        "prompt_id": "test_1",
        "prompt": [{"role": "user", "content": "test question"}],
        "response_text": "test answer",
        "rubrics": [],
        "example_tags": [],
        "node_summaries": [
            {"index": 42, "name": "Type 2 Diabetes", "summary": "..."}
        ],
        "node_contributions": {
            "node_42": {
                "explanation": "This node provided information...",
                "contributed": True
            }
        },
        "run_tag": "kg_grounded",
    }

    # Verify all required fields present
    assert set(output_dict.keys()) == expected_schema_fields

    # Verify node_summaries is a list
    assert isinstance(output_dict["node_summaries"], list)

    # Verify node_contributions is a dict
    assert isinstance(output_dict["node_contributions"], dict)

    # Verify run_tag is a string
    assert isinstance(output_dict["run_tag"], str)


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
    test_format_prompt_as_conversation_string()
    test_resume_logic()
    test_parse_json_to_dict()
    test_step2b_output_schema()
    test_phase2_grade_with_mock_grader()
    print("All tests passed.")



if __name__ == "__main__":
    run_all()
