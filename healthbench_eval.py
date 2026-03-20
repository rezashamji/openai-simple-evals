"""
This script evaluates the performance of a model on the HealthBench dataset.

To run HealthBench, HealthBench Consensus, or HealthBench Hard, use the simple-evals script:
- `python -m simple-evals.simple_evals --eval=healthbench --model=gpt-4.1`
- `python -m simple-evals.simple_evals --eval=healthbench_consensus --model=gpt-4.1`
- `python -m simple-evals.simple_evals --eval=healthbench_hard --model=gpt-4.1`

You can also evaluate physician ideal completions or reference completions against the HealthBench rubrics. To do so, run the following command:
- To evaluate physician ideal completions: `python -m simple-evals.healthbench_eval --run_mode=physician_completions`
- To evaluate reference model completions used by physicians: `python -m simple-evals.healthbench_eval --run_mode=physician_completion_references`
"""

import argparse
import copy
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import blobfile as bf
import numpy as np
import pandas as pd

from . import common
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from .types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult

INPUT_PATH = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
INPUT_PATH_HARD = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
INPUT_PATH_CONSENSUS = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

STAGE_1_ATTRIBUTION_TEMPLATE = """
You are validating whether a contributed knowledge graph (KG) node was actually mentioned in the grading explanation for a rubric criterion.

# Conversation
<<conversation>>

# Final Model Response
<<response>>

# Rubric Criterion
<<rubric_item>>

# Rubric Grader's Explanation (why criterion was met or not met)
<<rubric_explanation>>

# Node Contribution Claim (what this specific node claims it contributed)
<<node_contribution>>

# Task: STAGE 1 - NODE ATTRIBUTION VALIDATION

Question: Does the rubric grader's explanation DIRECTLY mention or reference what this node claims it contributed?

Think through:
1. What does this node claim it contributed? (from "Node Contribution Claim" above)
2. Read the grader's explanation carefully
3. Does the grader's explanation mention, reference, or require this node's contribution to justify its decision?

Return a JSON object with:
- "stage_1_passed": true if grader mentions/references this node's contribution, false if not
- "reasoning": Brief explanation
  - If true: "Grader mentions [concept] which node claims to contribute"
  - If false: "Grader's explanation does not mention [concept] from node"

Return just the JSON object in markdown format. Do not include any other text.
""".strip()

STAGE_2_RESPONSE_VALIDATION_TEMPLATE = """
You are validating whether a knowledge graph (KG) node's actual content appears in the model's response.

# Conversation
<<conversation>>

# Final Model Response
<<response>>

# Rubric Criterion
<<rubric_item>>

# Rubric Grader's Explanation
<<rubric_explanation>>

# Raw KG Node Summary (what this node actually contains)
<<node_summary>>

# Task: STAGE 2 - RESPONSE TEXT VALIDATION

Question: Does the exact node content appear in the response_text, specifically in the part that the grading_explanation references?

Think through:
1. What does the raw node summary actually say? (from "Raw KG Node Summary" above)
2. Read the grader's explanation - which part of the response does it reference?
3. Can you find exact phrases, concepts, or facts from the node summary in that specific part of the response that the grader references?

Return a JSON object with:
- "stage_2_passed": true if exact node content appears in the response part the grader references, false if not
- "reasoning": Brief explanation
  - If true: "Node content '[exact phrase]' found in response in the part grader references"
  - If false: "Node content does not appear in the part of response grader references"

Return just the JSON object in markdown format. Do not include any other text.
""".strip()

HEALTHBENCH_HTML_JINJA = (
    common.HTML_JINJA.replace(
        "<p>Correct Answer: {{ correct_answer }}</p>\n",
        "",
    )
    + "<p>Rubrics with grades: {{ rubric_grades }}</p>"
)


def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}


class RubricItem:
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )


def calculate_score(
    rubric_items: list[RubricItem], grading_response_list: list[dict]
) -> float | None:
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(
            rubric_items, grading_response_list, strict=True
        )
        if grading_response["criteria_met"]
    )
    overall_score = achieved_points / total_possible_points
    return overall_score


def calculate_kg_relevance_score(rubric_items: list[RubricItem], kg_labels: list[dict]) -> float | None:
    """Calculate KG relevance score from KG labels (helped/hurt/neutral).

    Score = (sum of points where kg_helped - sum of abs points where kg_hurt) / total_positive_points
    Returns None if no positive-point criteria exist or kg_labels is empty.
    """
    if not kg_labels:
        return None

    total_positive = sum(r.points for r in rubric_items if r.points > 0)
    if total_positive == 0:
        return None

    numerator = sum(
        r.points if kl.get("kg_label") == "kg_helped" else
        -abs(r.points) if kl.get("kg_label") == "kg_hurt" else 0
        for r, kl in zip(rubric_items, kg_labels)
    )
    return numerator / total_positive


def get_usage_dict(response_usage) -> dict[str, int | None]:
    if response_usage is None:
        return {
            "input_tokens": None,
            "input_cached_tokens": None,
            "output_tokens": None,
            "output_reasoning_tokens": None,
            "total_tokens": None,
        }

    try:
        return {
            "input_tokens": response_usage.input_tokens,
            "input_cached_tokens": response_usage.input_tokens_details.cached_tokens
            if hasattr(response_usage.input_tokens_details, "cached_tokens")
            else response_usage.input_tokens_details["cached_tokens"],
            "output_tokens": response_usage.output_tokens,
            "output_reasoning_tokens": response_usage.output_tokens_details.reasoning_tokens
            if hasattr(response_usage.output_tokens_details, "reasoning_tokens")
            else response_usage.output_tokens_details["reasoning_tokens"],
            "total_tokens": response_usage.total_tokens,
        }
    except AttributeError:
        return {
            "input_tokens": response_usage.prompt_tokens,
            "input_cached_tokens": response_usage.prompt_tokens_details.cached_tokens
            if hasattr(response_usage.prompt_tokens_details, "cached_tokens")
            else response_usage.prompt_tokens_details["cached_tokens"],
            "output_tokens": response_usage.completion_tokens,
            "output_reasoning_tokens": response_usage.completion_tokens_details.reasoning_tokens
            if hasattr(response_usage.completion_tokens_details, "reasoning_tokens")
            else response_usage.completion_tokens_details["reasoning_tokens"],
            "total_tokens": response_usage.total_tokens,
        }


PHYSICIAN_COMPLETION_MODES = {
    "Group 1": {
        "description": "No reference completions were provided to the physicians.",
        "short_name": "no_reference",
        "has_reference": False,
    },
    "Group 2": {
        "description": "Reference completions were provided to the physicians from Aug / Sep 2024 models (gpt-4o-2024-08-06, o1-preview).",
        "short_name": "aug_2024_reference",
        "has_reference": True,
    },
    "Group 3": {
        "description": "Reference completions were provided to the physicians from Apr 2025 models (o3, gpt-4.1).",
        "short_name": "apr_2025_reference",
        "has_reference": True,
    },
}


def _compute_clipped_stats(
    values: list,
    stat: str,
):
    """Computes the mean (clipped to [0, 1]), bootstrap std for that mean, and n_samples for final HealthBench scoring."""
    if stat == "mean":
        return np.clip(np.mean(values), 0, 1)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [
            _compute_clipped_stats(list(s), "mean") for s in bootstrap_samples
        ]
        return np.std(bootstrap_means)
    else:
        raise ValueError(f"Unknown {stat =}")


def _aggregate_get_clipped_mean(
    single_eval_results: list[SingleEvalResult],
) -> EvalResult:
    """
    Aggregate multiple SingleEvalResults into a single EvalResult for HealthBench.
    For each metric, returns the stats in _compute_clipped_stats.
    """
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        for stat in ["mean", "n_samples", "bootstrap_std"]:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_clipped_stats(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )


def validate_node_attribution(grader_model, conversation: str, response_text: str, rubric_item: RubricItem, grading_explanation: str, node_contribution: str) -> dict:
    """STAGE 1: Validate if node is mentioned in grading explanation.

    Args:
        grader_model: LLM to call for validation
        conversation: Full conversation string
        response_text: Model's response
        rubric_item: The rubric criterion
        grading_explanation: Grader's explanation for why criterion was met/not met
        node_contribution: What this node claims it contributed

    Returns:
        dict with keys: stage_1_passed (bool), reasoning (str)
    """
    prompt_text = STAGE_1_ATTRIBUTION_TEMPLATE \
        .replace("<<conversation>>", conversation) \
        .replace("<<response>>", response_text) \
        .replace("<<rubric_item>>", str(rubric_item)) \
        .replace("<<rubric_explanation>>", grading_explanation) \
        .replace("<<node_contribution>>", node_contribution)

    while True:
        result = grader_model([{"role": "user", "content": prompt_text}])
        parsed = parse_json_to_dict(result.response_text)
        if "stage_1_passed" in parsed and isinstance(parsed["stage_1_passed"], bool):
            break
        print("Stage 1 validation failed (invalid response), retrying...")

    return parsed


def validate_node_in_response(grader_model, conversation: str, response_text: str, rubric_item: RubricItem, grading_explanation: str, node_summary: str) -> dict:
    """STAGE 2: Validate if node content appears in response.

    Args:
        grader_model: LLM to call for validation
        conversation: Full conversation string
        response_text: Model's response
        rubric_item: The rubric criterion
        grading_explanation: Grader's explanation (which part of response is relevant)
        node_summary: Raw KG node summary

    Returns:
        dict with keys: stage_2_passed (bool), reasoning (str)
    """
    prompt_text = STAGE_2_RESPONSE_VALIDATION_TEMPLATE \
        .replace("<<conversation>>", conversation) \
        .replace("<<response>>", response_text) \
        .replace("<<rubric_item>>", str(rubric_item)) \
        .replace("<<rubric_explanation>>", grading_explanation) \
        .replace("<<node_summary>>", node_summary)

    while True:
        result = grader_model([{"role": "user", "content": prompt_text}])
        parsed = parse_json_to_dict(result.response_text)
        if "stage_2_passed" in parsed and isinstance(parsed["stage_2_passed"], bool):
            break
        print("Stage 2 validation failed (invalid response), retrying...")

    return parsed


def apply_semantic_impact(points: float, criteria_met: bool) -> str:
    """STAGE 3: Apply deterministic semantic logic to determine kg_label.

    IMPORTANT: This function is ONLY called when both Stage 1 and Stage 2 validation passed.
    (i.e., node was mentioned in grading explanation AND its content appears in response)

    Args:
        points: Criterion points (positive = desirable, negative = undesirable)
        criteria_met: Whether criterion was met (T/F)

    Returns:
        str: kg_label in ["kg_helped", "kg_hurt"]
    """
    if points > 0:  # Positive criterion (desirable)
        return "kg_helped" if criteria_met else "kg_hurt"
    else:  # points < 0, negative criterion (undesirable)
        return "kg_hurt" if criteria_met else "kg_helped"


def handle_validation_failure(stage_1_passed: bool, stage_2_reasoning: str = None) -> dict:
    """Handle cases where Stage 1 or Stage 2 validation failed.

    Args:
        stage_1_passed: Whether Stage 1 validation passed
        stage_2_reasoning: Stage 2 reasoning (if Stage 1 passed but Stage 2 failed)

    Returns:
        dict with keys: kg_label (str), validation_reasoning (str)
    """
    if not stage_1_passed:
        return {
            "kg_label": "kg_neutral",
            "validation_reasoning": "Node was not mentioned in grading explanation (Stage 1 failed) - node was not causal to grading decision"
        }
    else:  # stage_1_passed=true, but stage_2 failed
        return {
            "kg_label": "kg_attributed_but_not_in_response",
            "validation_reasoning": f"Node mentioned in grading explanation but content doesn't appear in response (Stage 2 failed). Stage 2 reasoning: {stage_2_reasoning}"
        }


class HealthBenchEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        num_examples: int | None = None,
        n_repeats: int = 1,
        # If set, evaluate human completions or reference completions instead of model completions.
        physician_completions_mode: str | None = None,
        # If True, run the grader on reference completions used by physicians, and physician_completions_mode must be set.
        run_reference_completions: bool = False,
        n_threads: int = 120,
        subset_name: Literal["hard", "consensus"] | None = None,
    ):
        if run_reference_completions:
            assert physician_completions_mode is not None, (
                "physician_completions_mode must be provided if run_reference_completions is True"
            )
            assert PHYSICIAN_COMPLETION_MODES[physician_completions_mode][
                "has_reference"
            ], (
                "physician_completions_mode must have reference completions if run_reference_completions is True"
            )

        if subset_name == "hard":
            input_path = INPUT_PATH_HARD
        elif subset_name == "consensus":
            input_path = INPUT_PATH_CONSENSUS
        elif subset_name is None:
            input_path = INPUT_PATH
        else:
            assert False, f"Invalid subset name: {subset_name}"
        with bf.BlobFile(input_path, "rb") as f:
            examples = [json.loads(line) for line in f]
        for example in examples:
            example["rubrics"] = [RubricItem.from_dict(d) for d in example["rubrics"]]

        rng = random.Random(0)

        # physician completions mode
        self.physician_completions_mode = physician_completions_mode
        if self.physician_completions_mode is not None:
            assert self.physician_completions_mode in PHYSICIAN_COMPLETION_MODES, (
                f"Invalid physician completions mode: {self.physician_completions_mode}; must be one of {PHYSICIAN_COMPLETION_MODES.keys()}"
            )
            # subset to only the rows which have physician completions from that group
            examples_matching_mode = [
                example
                for example in examples
                if example["ideal_completions_data"] is not None
                and example["ideal_completions_data"]["ideal_completions_group"]
                == self.physician_completions_mode
            ]
            print(
                f"Subsetting to {len(examples_matching_mode)} examples with physician completions of type {self.physician_completions_mode} ({PHYSICIAN_COMPLETION_MODES[self.physician_completions_mode]['description']})"
            )

            examples = []
            if run_reference_completions:
                for example in examples_matching_mode:
                    for completion in example["ideal_completions_data"][
                        "ideal_completions_ref_completions"
                    ]:
                        new_example = copy.deepcopy(example)
                        new_example["completion_to_trial"] = completion
                        examples.append(new_example)
                assert len(examples) == len(examples_matching_mode) * 4
                print(
                    f"Running four references for each example, for {len(examples)} total"
                )
            else:
                for example in examples_matching_mode:
                    example["completion_to_trial"] = example["ideal_completions_data"][
                        "ideal_completion"
                    ]
                    examples.append(example)
                assert len(examples) == len(examples_matching_mode)

            if len(examples) == 0:
                raise ValueError(
                    f"No examples found matching mode {self.physician_completions_mode}"
                )

        if num_examples is not None and num_examples < len(examples):
            examples = rng.sample(
                examples,
                num_examples,
            )

        self.examples = examples * n_repeats
        self.n_threads = n_threads
        self.grader_model = grader_model

    def grade_sample(
        self,
        prompt: list[dict[str, str]],
        response_text: str,
        example_tags: list[str],
        rubric_items: list[RubricItem],
        node_contributions: dict | None = None,
        node_summaries: list[dict] | None = None,
        run_tag: str = "no_kg",
        validate_node_contributions: bool = False,
    ) -> tuple[dict, str, list[dict], dict, dict]:
        # construct and grade the sample
        convo_with_response = prompt + [dict(content=response_text, role="assistant")]

        def grade_rubric_item(rubric_item: RubricItem) -> dict:
            convo_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response]
            )
            grader_prompt = GRADER_TEMPLATE.replace(
                "<<conversation>>", convo_str
            ).replace("<<rubric_item>>", str(rubric_item))
            messages: MessageList = [dict(content=grader_prompt, role="user")]
            while True:
                sampler_response = self.grader_model(messages)
                grading_response = sampler_response.response_text
                grading_response_dict = parse_json_to_dict(grading_response)
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    if label is True or label is False:
                        break
                print("Grading failed due to bad JSON output, retrying...")
            return grading_response_dict

        grading_response_list = common.map_with_progress(
            grade_rubric_item,
            rubric_items,
            pbar=False,
        )

        # compute the overall score
        overall_score = calculate_score(rubric_items, grading_response_list)
        assert overall_score is not None
        metrics = {
            "overall_score": overall_score,
        }

        # compute scores for example-level tags)
        example_tag_scores = {tag: overall_score for tag in example_tags}
        assert len(example_tag_scores) == len(example_tags)  # No duplicates.
        metrics.update(example_tag_scores)

        # compute scores for rubric-level tags
        rubric_tag_items_grades = defaultdict(list)
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            curr_item_tags = set()  # Ensure no duplicates in a rubric item.
            for tag in rubric_item.tags:
                rubric_tag_items_grades[tag].append((rubric_item, grading_response))
                assert tag not in curr_item_tags
                curr_item_tags.add(tag)

        rubric_tag_scores = {}
        for tag, items_grades in rubric_tag_items_grades.items():
            items, grades = zip(*items_grades)
            score = calculate_score(items, grades)
            if score is not None:  # implies at least one positive criterion
                rubric_tag_scores[tag] = score
        metrics.update(rubric_tag_scores)

        # construct the list of explanations and grades
        rubric_items_with_grades = []
        readable_explanation_list = []
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            explanation = grading_response.get("explanation", "No explanation provided")
            criteria_met = grading_response["criteria_met"]
            readable_explanation = (
                f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
            )
            readable_explanation_list.append(readable_explanation)
            rubric_items_with_grades.append(
                {
                    **rubric_item.to_dict(),
                    "criteria_met": criteria_met,
                    "explanation": explanation,
                }
            )

        readable_explanation_list.sort(
            key=lambda x: x.startswith("[False]"), reverse=True
        )
        readable_explanation_str = "\n\n".join(readable_explanation_list)
        readable_explanation_str = f"\n\n{readable_explanation_str}"

        # Build kg_labels by asking grader to map node contributions to criterion outcomes
        kg_reasoning_details = {}  # Store detailed reasoning for downstream cross-run analysis

        if run_tag == "kg_grounded" and node_contributions and node_summaries:
            # Import Issue 2.5 implementation (Parts 2-5)
            from . import kg_node_validation_part2

            # PART 2: Node-level preprocessing (validates final_contributed status)
            per_node_metadata = []
            for node_summary in node_summaries:
                node_id = str(node_summary.get("index", node_summary.get("name", "")))
                node_contrib = node_contributions.get(node_id, {})

                try:
                    node_part2 = kg_node_validation_part2.process_node_part2(
                        response_text=response_text,
                        node_index=node_id,
                        node_summary=node_summary.get("summary", ""),
                        initial_contributed=node_contrib.get("contributed", False),
                        initial_contribution_explanation=node_contrib.get("explanation", ""),
                        model=self.grader_model.model if hasattr(self.grader_model, "model") else "gpt-5.4",
                    )
                    # Add node_id to Part 2 output for downstream tracking
                    node_part2["node_id"] = node_id
                    per_node_metadata.append(node_part2)
                except Exception as e:
                    # Fallback: use initial_contributed if Part 2 fails
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Part 2 processing failed for node {node_id}: {e}")
                    per_node_metadata.append({
                        "node_id": node_id,
                        "final_contributed": node_contrib.get("contributed", False),
                        "final_node_contribution_explanation": node_contrib.get("explanation", ""),
                    })

            # PART 3: For each (node, criterion) pair, assign node-level label
            per_criterion_metadata = []
            for criterion_idx, (rubric_item, grading_response) in enumerate(
                zip(rubric_items, grading_response_list)
            ):
                criterion_data = {
                    "criterion_statement": rubric_item.criterion,
                    "points": rubric_item.points,
                    "criteria_met": grading_response["criteria_met"],
                    "grading_explanation": grading_response.get("explanation", ""),
                    "node_labels": [],
                }

                for node_part2 in per_node_metadata:
                    # Find corresponding node_summary
                    node_summary = next(
                        (n for n in node_summaries
                         if str(n.get("index", n.get("name", ""))) == node_part2["node_id"]),
                        {}
                    )

                    try:
                        node_label = kg_node_validation_part2.process_node_criterion_pair_part3(
                            node_index=node_part2["node_id"],
                            final_contributed=node_part2["final_contributed"],
                            final_node_contribution_explanation=node_part2["final_node_contribution_explanation"],
                            node_summary=node_summary.get("summary", ""),
                            criterion_statement=rubric_item.criterion,
                            criteria_met=grading_response["criteria_met"],
                            points=rubric_item.points,
                            grading_explanation=grading_response.get("explanation", ""),
                            model=self.grader_model.model if hasattr(self.grader_model, "model") else "gpt-5.4",
                        )
                        criterion_data["node_labels"].append({
                            "node_id": node_part2["node_id"],
                            **node_label
                        })
                    except Exception as e:
                        # Fallback if Part 3 fails
                        import logging
                        logger = logging.getLogger(__name__)
                        import traceback as _tb
                        logger.warning(f"Part 3 labeling failed for node {node_part2['node_id']}: {e}\n{_tb.format_exc()}")
                        criterion_data["node_labels"].append({
                            "node_id": node_part2["node_id"],
                            "final_node_label": "error_in_labeling",
                            "error": str(e)
                        })

                # Step 3.1.4b: Assign comparative causal weights to cited nodes
                try:
                    cited_nodes_for_causal = [
                        {
                            "node_index": nl.get("node_id", ""),
                            "node_summary": next(
                                (ns.get("summary", "") for ns in node_summaries
                                 if str(ns.get("index", "")) == str(nl.get("node_id", ""))),
                                ""
                            ),
                            "final_node_contribution_explanation": next(
                                (pm.get("final_node_contribution_explanation", "")
                                 for pm in per_node_metadata
                                 if str(pm.get("node_id", "")) == str(nl.get("node_id", ""))),
                                ""
                            ),
                            "node_used_as_justification_in_grading_explanation_reasoning": nl.get("node_used_as_justification_in_grading_explanation_reasoning", ""),
                            "node_direction_relative_to_criteria": nl.get("node_direction_relative_to_criteria", ""),
                            "node_direction_relative_to_criteria_reasoning": nl.get("node_direction_relative_to_criteria_reasoning", ""),
                            "final_node_label": nl.get("final_node_label", ""),
                        }
                        for nl in criterion_data["node_labels"]
                        if nl.get("node_used_as_justification_in_grading_explanation") is True
                    ]
                    if cited_nodes_for_causal:
                        _causal_model = self.grader_model.model if hasattr(self.grader_model, "model") else "gpt-5.4"
                        import logging as _logging
                        _logger = _logging.getLogger(__name__)
                        _logger.warning(f"[3.1.4b] calling with model={_causal_model!r}, grader_model type={type(self.grader_model)}, grader_model={self.grader_model!r}, num_cited={len(cited_nodes_for_causal)}")
                        causal_result = kg_node_validation_part2.step_3_1_4b_comparative_causal_weight(
                            cited_nodes=cited_nodes_for_causal,
                            criterion_statement=criterion_data.get("criterion_statement", ""),
                            criteria_met=criterion_data.get("criteria_met", False),
                            points=criterion_data.get("points", 0),
                            grading_explanation=criterion_data.get("grading_explanation", ""),
                            model=_causal_model,
                        )
                        # Write causal_weight back onto each cited node in node_labels
                        causal_by_index = {
                            str(cw["node_index"]): cw
                            for cw in causal_result.get("node_causal_weights", [])
                        }
                        for nl in criterion_data["node_labels"]:
                            node_id_str = str(nl.get("node_id", ""))
                            if node_id_str in causal_by_index:
                                nl["causal_weight"] = causal_by_index[node_id_str].get("causal_weight")
                                nl["causal_weight_reasoning"] = causal_by_index[node_id_str].get("reasoning", "")
                except Exception as e:
                    import logging, traceback
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Step 3.1.4b failed for criterion: {e}\n{traceback.format_exc()}")

                per_criterion_metadata.append(criterion_data)

            # PART 4: Aggregate per criterion (using actual Part 4 functions)
            for criterion_idx, criterion_data in enumerate(per_criterion_metadata):
                # Get node labels for this criterion (Part 3 outputs)
                node_criterion_labels = criterion_data["node_labels"]

                if not node_criterion_labels:
                    # No nodes for this criterion - mark as neutral
                    criterion_data["num_helped_nodes"] = 0
                    criterion_data["num_hurt_nodes"] = 0
                    criterion_data["num_neutral_nodes"] = 0
                    criterion_data["num_contradiction_nodes"] = 0
                    criterion_data["num_unclear_direction_nodes"] = 0
                    criterion_data["contradiction_ratio"] = 0.0
                    criterion_data["mixed_signals"] = False
                    criterion_data["kg_influence_label"] = "KG_NEUTRAL"
                    criterion_data["kg_label"] = "kg_neutral"
                    criterion_data["kg_reasoning"] = "No nodes available for this criterion"
                    continue

                # Step 4.1: Count node labels by type
                try:
                    import logging
                    logger = logging.getLogger(__name__)
                    part4_1_result = kg_node_validation_part2.step_4_1_count_labels(
                        node_criterion_labels
                    )
                    criterion_data.update(part4_1_result)
                except Exception as e:
                    logger.warning(f"Part 4.1 failed for criterion {criterion_idx}: {e}")
                    criterion_data["num_helped_nodes"] = 0
                    criterion_data["num_hurt_nodes"] = 0
                    criterion_data["num_neutral_nodes"] = len(node_criterion_labels)
                    criterion_data["num_contradiction_nodes"] = 0
                    criterion_data["num_unclear_direction_nodes"] = 0
                    criterion_data["contradiction_ratio"] = 0.0
                    criterion_data["mixed_signals"] = False

                # Step 4.5: High contradiction analysis (must run BEFORE 4.2)
                if criterion_data.get("contradiction_ratio", 0.0) >= 0.25:
                    try:
                        part4_5_result = kg_node_validation_part2.step_4_5_analyze_high_contradictions(
                            criterion_statement=criterion_data.get("criterion_statement", ""),
                            criteria_met=criterion_data.get("criteria_met", False),
                            grading_explanation=criterion_data.get("grading_explanation", ""),
                            response_text=response_text,
                            node_summaries=node_summaries,
                            node_contributions=node_contributions,
                            num_helped_nodes=criterion_data.get("num_helped_nodes", 0),
                            num_hurt_nodes=criterion_data.get("num_hurt_nodes", 0),
                            num_contradiction_nodes=criterion_data.get("num_contradiction_nodes", 0),
                            contradiction_ratio=criterion_data.get("contradiction_ratio", 0.0),
                            model=self.grader_model.model if hasattr(self.grader_model, "model") else "gpt-5.4",
                        )
                        criterion_data["high_contradiction_label_consistency"] = part4_5_result.get("high_contradiction_label_consistency")
                        criterion_data["high_contradiction_resolution_insight"] = part4_5_result.get("high_contradiction_resolution_insight")
                        criterion_data["high_contradiction_consistency_reasoning"] = part4_5_result.get("high_contradiction_consistency_reasoning")
                    except Exception as e:
                        import traceback as _tb
                        logger.warning(f"Part 4.5 failed for criterion {criterion_idx}: {e}\n{_tb.format_exc()}")

                # Step 4.7: Conflicting signals analysis (must run BEFORE 4.2)
                if criterion_data.get("mixed_signals", False):
                    try:
                        part4_7_result = kg_node_validation_part2.step_4_7_analyze_conflicting_signals(
                            criterion_statement=criterion_data.get("criterion_statement", ""),
                            criteria_met=criterion_data.get("criteria_met", False),
                            grading_explanation=criterion_data.get("grading_explanation", ""),
                            response_text=response_text,
                            node_summaries=node_summaries,
                            node_contributions=node_contributions,
                            num_helped_nodes=criterion_data.get("num_helped_nodes", 0),
                            num_hurt_nodes=criterion_data.get("num_hurt_nodes", 0),
                            model=self.grader_model.model if hasattr(self.grader_model, "model") else "gpt-5.4",
                        )
                        criterion_data["conflicting_signals_label_consistency"] = part4_7_result.get("conflicting_signals_label_consistency")
                        criterion_data["conflicting_signals_weighting"] = part4_7_result.get("conflicting_signals_weighting")
                        criterion_data["conflicting_signals_dominant_influence"] = part4_7_result.get("conflicting_signals_dominant_influence")
                        criterion_data["conflicting_signals_consistency_reasoning"] = part4_7_result.get("conflicting_signals_consistency_reasoning")
                    except Exception as e:
                        logger.warning(f"Part 4.7 failed for criterion {criterion_idx}: {e}")

                # Step 4.2: Assign KG influence label (runs AFTER 4.5 and 4.7 so their outputs are available)
                max_nodes_allowed = 20  # default; matches --max-nodes-per-question CLI default
                try:
                    part4_2_result = kg_node_validation_part2.step_4_2_assign_kg_influence_label(
                        num_helped_nodes=criterion_data.get("num_helped_nodes", 0),
                        num_hurt_nodes=criterion_data.get("num_hurt_nodes", 0),
                        num_neutral_nodes=criterion_data.get("num_neutral_nodes", 0),
                        num_neutral_not_contributed=criterion_data.get("num_neutral_not_contributed", 0),
                        num_neutral_in_response_not_cited=criterion_data.get("num_neutral_in_response_not_cited", 0),
                        num_neutral_in_response_direction_unclear=criterion_data.get("num_neutral_in_response_direction_unclear", 0),
                        num_contradiction_nodes=criterion_data.get("num_contradiction_nodes", 0),
                        num_push_met_but_criterion_not_met=criterion_data.get("num_push_met_but_criterion_not_met", 0),
                        num_push_not_met_but_criterion_met=criterion_data.get("num_push_not_met_but_criterion_met", 0),
                        total_nodes=criterion_data.get("total_nodes", 0),
                        max_nodes_allowed=max_nodes_allowed,
                        contradiction_ratio=criterion_data.get("contradiction_ratio", 0.0),
                        mixed_signals=criterion_data.get("mixed_signals", False),
                        num_primary_helped=criterion_data.get("num_primary_helped", 0),
                        num_supporting_helped=criterion_data.get("num_supporting_helped", 0),
                        num_incidental_helped=criterion_data.get("num_incidental_helped", 0),
                        num_primary_hurt=criterion_data.get("num_primary_hurt", 0),
                        num_supporting_hurt=criterion_data.get("num_supporting_hurt", 0),
                        num_incidental_hurt=criterion_data.get("num_incidental_hurt", 0),
                        high_contradiction_label_consistency=criterion_data.get("high_contradiction_label_consistency"),
                        conflicting_signals_label_consistency=criterion_data.get("conflicting_signals_label_consistency"),
                        conflicting_signals_dominant_influence=criterion_data.get("conflicting_signals_dominant_influence"),
                        criteria_met=criterion_data.get("criteria_met"),
                    )
                    criterion_data["kg_influence_label"] = part4_2_result.get("kg_influence_label", "KG_NEUTRAL")
                    criterion_data["assignment_reason"] = part4_2_result.get("assignment_reason", "")
                except Exception as e:
                    logger.warning(f"Part 4.2 failed for criterion {criterion_idx}: {e}")
                    criterion_data["kg_influence_label"] = "KG_NEUTRAL"
                    criterion_data["assignment_reason"] = f"Error in assignment: {str(e)}"

                # Derive 3-value kg_label from verbose kg_influence_label
                kg_inf_label = criterion_data.get("kg_influence_label", "")
                if kg_inf_label.startswith("KG_NODES_HELPED_AND_HURT"):
                    # Mixed signals — check dominant influence embedded in label
                    if "_HELPED_NODES_DROVE_OUTCOME_" in kg_inf_label:
                        criterion_data["kg_label"] = "kg_helped"
                    elif "_HURT_NODES_DROVE_OUTCOME_" in kg_inf_label:
                        criterion_data["kg_label"] = "kg_hurt"
                    else:
                        criterion_data["kg_label"] = "kg_neutral"
                elif kg_inf_label.startswith("KG_NODES_HELPED"):
                    criterion_data["kg_label"] = "kg_helped"
                elif kg_inf_label.startswith("KG_NODES_HURT"):
                    criterion_data["kg_label"] = "kg_hurt"
                else:
                    criterion_data["kg_label"] = "kg_neutral"

                criterion_data["kg_reasoning"] = (
                    f"{criterion_data.get('kg_influence_label', 'KG_NEUTRAL')}: "
                    f"Helped={criterion_data.get('num_helped_nodes', 0)}, "
                    f"Hurt={criterion_data.get('num_hurt_nodes', 0)}, "
                    f"Neutral={criterion_data.get('num_neutral_nodes', 0)}"
                )

            # Merge Part 2, 3, and 4 fields into rubric_items_with_grades
            for idx, criterion_data in enumerate(per_criterion_metadata):
                if idx < len(rubric_items_with_grades):
                    # Add Part 4 aggregation fields to grading output
                    part4_fields = {
                        "num_helped_nodes": criterion_data.get("num_helped_nodes"),
                        "num_hurt_nodes": criterion_data.get("num_hurt_nodes"),
                        "num_neutral_nodes": criterion_data.get("num_neutral_nodes"),
                        "num_contradiction_nodes": criterion_data.get("num_contradiction_nodes"),
                        "num_unclear_direction_nodes": criterion_data.get("num_unclear_direction_nodes"),
                        "total_nodes": criterion_data.get("total_nodes"),
                        "contradiction_ratio": criterion_data.get("contradiction_ratio"),
                        "mixed_signals": criterion_data.get("mixed_signals"),
                        # New step_4_1 sub-type counts
                        "num_neutral_not_contributed": criterion_data.get("num_neutral_not_contributed"),
                        "num_neutral_in_response_not_cited": criterion_data.get("num_neutral_in_response_not_cited"),
                        "num_neutral_in_response_direction_unclear": criterion_data.get("num_neutral_in_response_direction_unclear"),
                        "num_push_met_but_criterion_not_met": criterion_data.get("num_push_met_but_criterion_not_met"),
                        "num_push_not_met_but_criterion_met": criterion_data.get("num_push_not_met_but_criterion_met"),
                        "num_primary_helped": criterion_data.get("num_primary_helped"),
                        "num_supporting_helped": criterion_data.get("num_supporting_helped"),
                        "num_incidental_helped": criterion_data.get("num_incidental_helped"),
                        "num_primary_hurt": criterion_data.get("num_primary_hurt"),
                        "num_supporting_hurt": criterion_data.get("num_supporting_hurt"),
                        "num_incidental_hurt": criterion_data.get("num_incidental_hurt"),
                        # Part 4.2 fields
                        "kg_label": criterion_data.get("kg_label"),
                        "kg_influence_label": criterion_data.get("kg_influence_label"),
                        "assignment_reason": criterion_data.get("assignment_reason"),
                        # Part 4.5 fields (conditional)
                        "high_contradiction_label_consistency": criterion_data.get("high_contradiction_label_consistency"),
                        "high_contradiction_resolution_insight": criterion_data.get("high_contradiction_resolution_insight"),
                        "high_contradiction_consistency_reasoning": criterion_data.get("high_contradiction_consistency_reasoning"),
                        # Part 4.7 fields (conditional)
                        "conflicting_signals_label_consistency": criterion_data.get("conflicting_signals_label_consistency"),
                        "conflicting_signals_weighting": criterion_data.get("conflicting_signals_weighting"),
                        "conflicting_signals_dominant_influence": criterion_data.get("conflicting_signals_dominant_influence"),
                        "conflicting_signals_consistency_reasoning": criterion_data.get("conflicting_signals_consistency_reasoning"),
                        # Part 3: Include raw node_labels for complete auditability
                        "node_labels": criterion_data.get("node_labels", []),
                    }
                    rubric_items_with_grades[idx].update(part4_fields)

            # Build kg_reasoning_details from per_criterion_metadata
            kg_reasoning_details = {}
            for criterion_data in per_criterion_metadata:
                criterion_key = f"criterion_{criterion_data['criterion_statement'][:50]}"
                kg_reasoning_details[criterion_key] = {
                    "kg_label": criterion_data.get("kg_label"),
                    "kg_reasoning": criterion_data.get("kg_reasoning"),
                    "contributed_nodes": [
                        {"index": nl.get("node_id"), "label": nl.get("final_node_label")}
                        for nl in criterion_data.get("node_labels", [])
                        if nl.get("node_id") and nl.get("final_node_label") != "error_in_labeling"
                    ]
                }

            # Build kg_labels for metrics calculation (compatible with calculate_kg_relevance_score)
            kg_labels = []
            for criterion_data in per_criterion_metadata:
                kg_labels.append({
                    "kg_label": criterion_data.get("kg_label"),
                    "kg_reasoning": criterion_data.get("kg_reasoning", "")
                })
        else:
            # For no_kg runs: empty metadata and no labels
            kg_reasoning_details = {}
            kg_labels = [{"kg_label": None, "kg_reasoning": None}] * len(rubric_items)

        # Return per_node_metadata and per_criterion_metadata for complete auditability
        # These are intermediate Part 2-3 outputs needed for training predictors
        intermediate_metadata = {
            "per_node_metadata": per_node_metadata if 'per_node_metadata' in locals() else [],
            "per_criterion_metadata": per_criterion_metadata if 'per_criterion_metadata' in locals() else [],
        }

        return metrics, readable_explanation_str, rubric_items_with_grades, kg_reasoning_details, intermediate_metadata

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = row["prompt"]

            if self.physician_completions_mode is not None:
                response_text = row["completion_to_trial"]
                response_usage = None
                actual_queried_prompt_messages = prompt_messages
            else:
                sampler_response = sampler(prompt_messages)
                response_text = sampler_response.response_text
                response_dict = sampler_response.response_metadata
                actual_queried_prompt_messages = (
                    sampler_response.actual_queried_message_list
                )
                response_usage = response_dict.get("usage", None)

            metrics, readable_explanation_str, rubric_items_with_grades, kg_reasoning_details, intermediate_metadata = (
                self.grade_sample(
                    prompt=actual_queried_prompt_messages,
                    response_text=response_text,
                    rubric_items=row["rubrics"],
                    example_tags=row["example_tags"],
                    node_contributions=row.get("node_contributions"),
                    node_summaries=row.get("node_summaries"),
                    run_tag=row.get("run_tag", "no_kg"),
                    validate_node_contributions=False,
                )
            )

            # Extract KG labels from rubric_items_with_grades for kg_relevance_score calculation
            kg_labels = [item.get("kg_label") for item in rubric_items_with_grades] if rubric_items_with_grades else []
            kg_relevance_score = calculate_kg_relevance_score(row["rubrics"], [{"kg_label": label} for label in kg_labels])

            score = metrics["overall_score"]

            # Create HTML for each sample result
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

            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role="assistant")
            ]
            return SingleEvalResult(
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
                    "prompt_id": row["prompt_id"],
                    "completion_id": hashlib.sha256(
                        (row["prompt_id"] + response_text).encode("utf-8")
                    ).hexdigest(),
                    "run_tag": row.get("run_tag", "no_kg"),
                    "node_summaries": row.get("node_summaries", []),
                    "node_contributions": row.get("node_contributions", {}),
                    "kg_relevance_score": kg_relevance_score,
                    "kg_reasoning_details": kg_reasoning_details,
                },
            )

        results = common.map_with_progress(
            fn,
            self.examples,
            num_threads=self.n_threads,
            pbar=True,
        )
        final_metrics = _aggregate_get_clipped_mean(results)
        return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="HealthBenchEval specific run options, including e.g., running the eval on physician completions rows only."
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["physician_completions", "physician_completion_references"],
    )
    parser.add_argument("--examples", type=int, help="Number of examples to run")
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads to run",
    )
    parser.add_argument(
        "--max-nodes-per-question",
        type=int,
        default=20,
        help="Max KG nodes per question (used for KG coverage calculation in step 4.2)",
    )
    args = parser.parse_args()

    if args.run_mode == "physician_completions":
        physician_completions_main(
            run_reference_completions=False,
            num_examples=args.examples,
            n_threads=args.n_threads or 1,
        )
    elif args.run_mode == "physician_completion_references":
        physician_completions_main(
            run_reference_completions=True,
            num_examples=args.examples,
            n_threads=args.n_threads or 1,
        )

    else:
        raise ValueError(f"Invalid run mode: {args.run_mode}")


def physician_completions_main(
    run_reference_completions: bool = False,
    num_examples: int | None = None,
    n_threads: int = 120,
):
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")

    grading_sampler = ChatCompletionSampler(
        model="gpt-5.4",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )
    dummy_sampler = SamplerBase()

    merge_metrics = []
    for pc_mode in PHYSICIAN_COMPLETION_MODES.keys():
        if (
            run_reference_completions
            and not PHYSICIAN_COMPLETION_MODES[pc_mode]["has_reference"]
        ):
            continue

        # run
        eval = HealthBenchEval(
            grader_model=grading_sampler,
            physician_completions_mode=pc_mode,
            run_reference_completions=run_reference_completions,
            num_examples=num_examples,
            n_threads=n_threads,
        )
        result = eval(dummy_sampler)

        # report
        parsable_mode = PHYSICIAN_COMPLETION_MODES[pc_mode]["short_name"]
        if run_reference_completions:
            file_stem = f"healthbench_{parsable_mode}_referencecompletions_{date_str}"
        else:
            file_stem = f"healthbench_{parsable_mode}_humanbaseline_{date_str}"
        report_filename = Path(f"/tmp/{file_stem}.html")
        report_filename.write_text(common.make_report(result))
        print(f"Report saved to {report_filename}")

        # metrics
        assert result.metrics is not None
        metrics = result.metrics
        result_filename = Path(f"/tmp/{file_stem}.json")
        result_filename.write_text(json.dumps(metrics))
        print(f"Results saved to {result_filename}")

        full_result_dict = {
            "score": result.score,
            "metrics": result.metrics,
            "htmls": result.htmls,
            "convos": result.convos,
            "metadata": result.metadata,
        }
        full_result_filename = Path(f"/tmp/{file_stem}_allresults.json")
        full_result_filename.write_text(json.dumps(full_result_dict, indent=2))
        print(f"All results saved to {full_result_filename}")

        # metrics df
        merge_metrics.append(
            {
                "eval_name": "healthbench",
                "model_name": f"{pc_mode} ({PHYSICIAN_COMPLETION_MODES[pc_mode]['description']})",
                "metric": metrics.get("overall_score", None),
            }
        )

    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
