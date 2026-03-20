"""
Part 2: Node-level preprocessing for KG validation pipeline.

Implements Steps 2.1-2.4:
- Step 2.1: Content validation (check if node appears in response)
- Step 2.2: Conflict detection
- Step 2.3: Judge LLM for conflicts
- Step 2.4: Resolution of final_contributed values

All LLM calls use strict JSON output with validation.
"""

import json
import logging
import os
import time
from typing import Optional

import litellm
from litellm.exceptions import RateLimitError

litellm.drop_params = True  # Allow unsupported params (e.g., temperature) to be dropped for models like gpt-5

logger = logging.getLogger(__name__)

# ============================================================================
# LLM TEMPLATES
# ============================================================================

PHASE_A_CONTENT_VALIDATION_TEMPLATE = """
You are evaluating whether a knowledge graph (KG) node's content appears in a model response to a health question.

# Model Response
<<response_text>>

# Knowledge Graph Node
<<node_summary>>

# Task
Determine whether the content or information from this KG node is mentioned, cited, or referenced in the model response.

Look for:
- Exact phrases from the node
- Paraphrased content from the node
- Information that matches the node's key facts/entities

Return a JSON object with:
- "node_content_appears_in_response": boolean (true if node content is in response)
- "node_content_appears_in_response_reasoning": string (brief explanation of your finding)

Return ONLY valid JSON, no markdown code blocks or extra text.
""".strip()

PHASE_B_JUDGE_LLM_TEMPLATE = """
You are a judge evaluating a disagreement between two assessments of a knowledge graph (KG) node's contribution.

# Model Response
<<response_text>>

# Knowledge Graph Node
<<node_summary>>

# Initial Assessment (from the response generation model)
- Claimed node was contributed: <<initial_contributed>>
- Explanation: <<initial_node_contribution_explanation>>

# Conflict
An independent analysis found that the node content appears in response: <<node_content_appears_in_response>>

This creates a conflict: the initial model claimed <<initial_contributed_str>> but the node <<appearance_str>> in the response.

# Task
As a judge, determine which is more likely to be correct. Consider:
1. Does the KG node information actually match what appears in the response?
2. Was the initial model's judgment reasonable given the information available?
3. How confident are you in correcting the initial model?

Return a JSON object with:
- "judge_says_initial_correct": boolean (true = initial judgment was right, false = initial judgment was wrong)
- "judge_probability_initial_incorrect": float between 0.0-1.0 (probability that initial was wrong)
- "updated_contributed": boolean or null (if judge disagrees, what should the correct value be? else null)
- "updated_node_contribution_explanation": string or null (if judge disagrees, explain why. else null)
- "judge_reasoning": string (brief explanation of your judgment)

Rules:
- If judge_says_initial_correct=true, both updated fields MUST be null
- If judge_says_initial_correct=false, both updated fields MUST be non-null
- Probability should reflect your confidence in the correction (0.70+ threshold is used downstream)

Return ONLY valid JSON, no markdown code blocks or extra text.
""".strip()

PHASE_D2_CAUSAL_WEIGHT_TEMPLATE = """
You are analyzing which knowledge graph (KG) nodes were the primary drivers of a grading outcome for a medical criterion.

# Criterion
<<criterion_statement>>

# Criterion Outcome
Met: <<criteria_met>>
Points: <<points>>
Grader explanation: <<grading_explanation>>

# Cited KG Nodes
<<cited_nodes_list>>

For each node the following is provided:
- Node ID
- Node summary (the KG content)
- Part 2 contribution explanation (how the node appeared in the response)
- Justification reasoning (why the grader cited this node)
- Direction relative to criterion (PUSHED_TOWARD_MET / PUSHED_TOWARD_NOT_MET)
- Direction reasoning
- Final node label (HELPED / HURT / CONTRADICTION / etc.)

# Task
For each cited node, assign a causal weight based on how much it drove whether the criterion was met or not:
- PRIMARY: This node was the main driver of the criterion outcome. Without it, the outcome would likely have been different.
- SUPPORTING: This node contributed meaningfully but was not decisive on its own.
- INCIDENTAL: This node was cited in the explanation but had minimal causal impact on the outcome.

Return a JSON object:
{
  "node_causal_weights": [
    {"node_index": "<id>", "causal_weight": "PRIMARY|SUPPORTING|INCIDENTAL", "reasoning": "<one sentence>"}
  ]
}

Return ONLY valid JSON, no markdown.
""".strip()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_json_response(response: str, expected_keys: list[str]) -> dict:
    """
    Validate and parse JSON response from LLM.
    JSON is guaranteed valid by response_format constraint.

    Args:
        response: LLM response text (guaranteed valid JSON)
        expected_keys: List of required keys in JSON

    Returns:
        Parsed JSON dict

    Raises:
        ValueError: If required keys are missing
    """
    parsed = json.loads(response.strip())

    # Validate required keys
    missing = set(expected_keys) - set(parsed.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    return parsed


def call_llm_with_validation(
    template: str,
    replacements: dict[str, str],
    expected_keys: list[str],
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Call LLM with template substitution and strict JSON validation.

    Args:
        template: LLM prompt template with <<placeholders>>
        replacements: Dict mapping placeholder names to values
        expected_keys: List of required keys in JSON response
        model: LLM model to use

    Returns:
        Validated JSON response dict

    Raises:
        ValueError: If LLM response is invalid JSON or missing keys
    """
    # Build prompt by replacing placeholders
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(f"<<{key}>>", str(value))

    # Pick up reasoning_effort from env (set by orchestration script)
    reasoning_effort = os.environ.get("REASONING_EFFORT")

    logger.warning(f"[call_llm_with_validation] model={model}, reasoning_effort={reasoning_effort}, prompt_len={len(prompt)}, expected_keys={expected_keys}")
    logger.warning(f"[call_llm_with_validation] prompt preview: {prompt[:300]!r}")

    # Call LLM with response_format to guarantee valid JSON
    max_retries = 5
    delay = 15
    for attempt in range(max_retries):
        try:
            extra_args = {}
            if reasoning_effort:
                extra_args["reasoning_effort"] = reasoning_effort
            response = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2000,
                num_retries=0,
                **extra_args,
            )
            break
        except RateLimitError:
            logger.warning(f"[call_llm_with_validation] RateLimitError on attempt {attempt+1}/{max_retries}, sleeping {delay}s")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2

    response_text = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    usage = getattr(response, "usage", None)
    logger.warning(f"[call_llm_with_validation] finish_reason={finish_reason}, usage={usage}, response_text type={type(response_text)}, response_text={response_text!r}")

    # Validate required keys (JSON format guaranteed by response_format)
    return validate_json_response(response_text, expected_keys)


# ============================================================================
# STEP 2.1: NODE CONTENT VALIDATION
# ============================================================================


def step_2_1_validate_node_content(
    response_text: str,
    node_summary: str,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Step 2.1: Check if node content appears in response.

    Args:
        response_text: Model's response text
        node_summary: KG node summary (YAML-formatted)
        model: LLM model to use

    Returns:
        Dict with:
        - node_content_appears_in_response: bool
        - node_content_appears_in_response_reasoning: str
    """
    result = call_llm_with_validation(
        template=PHASE_A_CONTENT_VALIDATION_TEMPLATE,
        replacements={
            "response_text": response_text,
            "node_summary": node_summary,
        },
        expected_keys=[
            "node_content_appears_in_response",
            "node_content_appears_in_response_reasoning",
        ],
        model=model,
    )

    return {
        "node_content_appears_in_response": result["node_content_appears_in_response"],
        "node_content_appears_in_response_reasoning": result[
            "node_content_appears_in_response_reasoning"
        ],
    }


# ============================================================================
# STEP 2.2: CONFLICT DETECTION
# ============================================================================


def step_2_2_detect_conflict(
    node_content_appears: bool,
    initial_contributed: bool,
) -> bool:
    """
    Step 2.2: Detect conflict between claimed and actual contribution.

    Args:
        node_content_appears: Whether node content appears in response (from 2.1)
        initial_contributed: Whether initial model claimed contribution

    Returns:
        True if conflict exists, False otherwise
    """
    return node_content_appears != initial_contributed


# ============================================================================
# STEP 2.3: JUDGE LLM FOR CONFLICTS
# ============================================================================


def step_2_3_judge_conflict(
    response_text: str,
    node_summary: str,
    initial_contributed: bool,
    initial_contribution_explanation: str,
    node_content_appears: bool,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Step 2.3: Judge LLM for contribution conflicts.

    Only called when contribution_conflict == true.

    Args:
        response_text: Model's response text
        node_summary: KG node summary
        initial_contributed: Initial claimed contribution (T/F)
        initial_contribution_explanation: Initial explanation
        node_content_appears: Whether content actually appears in response
        model: LLM model to use

    Returns:
        Dict with:
        - judge_says_initial_correct: bool
        - judge_probability_initial_incorrect: float (0.0-1.0)
        - updated_contributed: bool or None
        - updated_node_contribution_explanation: str or None
        - judge_reasoning: str
    """
    # Format strings for template
    initial_contributed_str = "true" if initial_contributed else "false"
    appearance_str = "appears" if node_content_appears else "does not appear"

    result = call_llm_with_validation(
        template=PHASE_B_JUDGE_LLM_TEMPLATE,
        replacements={
            "response_text": response_text,
            "node_summary": node_summary,
            "initial_contributed": initial_contributed_str,
            "initial_node_contribution_explanation": initial_contribution_explanation,
            "node_content_appears_in_response": str(node_content_appears).lower(),
            "initial_contributed_str": initial_contributed_str,
            "appearance_str": appearance_str,
        },
        expected_keys=[
            "judge_says_initial_correct",
            "judge_probability_initial_incorrect",
            "updated_contributed",
            "updated_node_contribution_explanation",
            "judge_reasoning",
        ],
        model=model,
    )

    return {
        "judge_says_initial_correct": result["judge_says_initial_correct"],
        "judge_probability_initial_incorrect": result[
            "judge_probability_initial_incorrect"
        ],
        "updated_contributed": result["updated_contributed"],
        "updated_node_contribution_explanation": result[
            "updated_node_contribution_explanation"
        ],
        "judge_reasoning": result["judge_reasoning"],
    }


# ============================================================================
# STEP 2.4: RESOLUTION
# ============================================================================


def step_2_4_resolve_contribution(
    initial_contributed: bool,
    initial_contribution_explanation: str,
    contribution_conflict: bool,
    judge_result: Optional[dict] = None,
) -> dict:
    """
    Step 2.4: Resolve final_contributed and explanation for a node.

    Args:
        initial_contributed: Initial claimed contribution
        initial_contribution_explanation: Initial explanation
        contribution_conflict: Whether conflict was detected (2.2)
        judge_result: Judge LLM output (from 2.3), or None if no conflict

    Returns:
        Dict with:
        - final_contributed: bool
        - final_node_contribution_explanation: str
        - contribution_resolution_status: str (one of: "no_conflict_detected",
          "judge_ran_updated_applied", "judge_ran_initial_kept")
        - [optional judge fields if judge ran]
    """
    if not contribution_conflict:
        # No conflict: keep initial
        return {
            "final_contributed": initial_contributed,
            "final_node_contribution_explanation": initial_contribution_explanation,
            "contribution_resolution_status": "no_conflict_detected",
        }

    # Conflict exists: check judge result
    if judge_result is None:
        raise ValueError("judge_result required when contribution_conflict=true")

    # Threshold: 70%
    if (
        not judge_result["judge_says_initial_correct"]
        and judge_result["judge_probability_initial_incorrect"] >= 0.70
    ):
        # Judge strongly disagrees: apply update
        return {
            "final_contributed": judge_result["updated_contributed"],
            "final_node_contribution_explanation": judge_result[
                "updated_node_contribution_explanation"
            ],
            "contribution_resolution_status": "judge_ran_updated_applied",
            "judge_probability_initial_incorrect": judge_result[
                "judge_probability_initial_incorrect"
            ],
            "judge_reasoning": judge_result["judge_reasoning"],
            "updated_contributed": judge_result["updated_contributed"],
            "updated_node_contribution_explanation": judge_result[
                "updated_node_contribution_explanation"
            ],
        }
    else:
        # Judge agrees or below threshold: keep initial
        return {
            "final_contributed": initial_contributed,
            "final_node_contribution_explanation": initial_contribution_explanation,
            "contribution_resolution_status": "judge_ran_initial_kept",
            "judge_probability_initial_incorrect": judge_result[
                "judge_probability_initial_incorrect"
            ],
            "judge_reasoning": judge_result["judge_reasoning"],
            "updated_contributed": judge_result["updated_contributed"],
            "updated_node_contribution_explanation": judge_result[
                "updated_node_contribution_explanation"
            ],
        }


# ============================================================================
# MAIN PART 2 WRAPPER
# ============================================================================


def process_node_part2(
    response_text: str,
    node_index: str,
    node_summary: str,
    initial_contributed: bool,
    initial_contribution_explanation: str,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Process a single node through Part 2 (Steps 2.1-2.4).

    Args:
        response_text: Model's response text
        node_index: Node ID (for logging/tracking)
        node_summary: KG node summary
        initial_contributed: Initial claimed contribution
        initial_contribution_explanation: Initial explanation
        model: LLM model to use

    Returns:
        Dict with all Part 2 outputs (content check, conflict, judge, resolution)
    """
    logger.debug(f"Part 2 processing node {node_index}")

    # Step 2.1: Content validation
    content_check = step_2_1_validate_node_content(
        response_text=response_text,
        node_summary=node_summary,
        model=model,
    )
    node_content_appears = content_check["node_content_appears_in_response"]

    # Step 2.2: Conflict detection
    conflict = step_2_2_detect_conflict(
        node_content_appears=node_content_appears,
        initial_contributed=initial_contributed,
    )

    # Step 2.3: Judge if conflict exists
    judge_result = None
    if conflict:
        logger.info(f"Conflict detected for node {node_index}, calling judge")
        judge_result = step_2_3_judge_conflict(
            response_text=response_text,
            node_summary=node_summary,
            initial_contributed=initial_contributed,
            initial_contribution_explanation=initial_contribution_explanation,
            node_content_appears=node_content_appears,
            model=model,
        )

    # Step 2.4: Resolution
    resolution = step_2_4_resolve_contribution(
        initial_contributed=initial_contributed,
        initial_contribution_explanation=initial_contribution_explanation,
        contribution_conflict=conflict,
        judge_result=judge_result,
    )

    # Combine all results
    return {
        "node_content_appears_in_response": node_content_appears,
        "node_content_appears_in_response_reasoning": content_check[
            "node_content_appears_in_response_reasoning"
        ],
        "contribution_conflict": conflict,
        **resolution,
    }


# ============================================================================
# PART 3: MAIN NODE+CRITERION LABELING LOOP
# ============================================================================

PHASE_C_GRADING_JUSTIFICATION_TEMPLATE = """
You are checking whether a knowledge graph (KG) node was used as justification in the grading explanation for a specific criterion.

# Rubric Criterion
<<criterion_statement>>

# Grading Explanation (why this criterion was/was not met)
<<grading_explanation>>

# Knowledge Graph Node
<<node_summary>>

# Initial Node Contribution Explanation (how the model claimed to use this node)
<<node_contribution_explanation>>

# Task
Determine whether this specific KG node was mentioned, cited, or used as justification in the grading explanation for this criterion.

The grading explanation is the reasoning that determined whether the criterion was met or not. Check if the node (or its key information) was referenced in that reasoning.

Return a JSON object with:
- "node_used_as_justification_in_grading_explanation": boolean
- "node_used_as_justification_in_grading_explanation_reasoning": string (brief explanation)

Return ONLY valid JSON, no markdown code blocks or extra text.
""".strip()

PHASE_D_NODE_DIRECTION_TEMPLATE = """
You are analyzing the direction and impact of a knowledge graph (KG) node on a rubric criterion outcome.

# Rubric Criterion
<<criterion_statement>>

# Criterion Outcome
The criterion was met: <<criteria_met>>

# Grading Explanation (why this criterion was/was not met)
<<grading_explanation>>

# Knowledge Graph Node (used in grading)
<<node_summary>>

# Node Contribution Explanation
<<node_contribution_explanation>>

# Task
Analyze whether this KG node pushed the grading toward the criterion being met, or pushed toward it NOT being met.

Consider:
1. What information did the node provide?
2. Did that information support meeting the criterion (PUSHED_TOWARD_MET)?
3. Or did it support NOT meeting the criterion (PUSHED_TOWARD_NOT_MET)?
4. Or is the direction unclear from the available information?

Return a JSON object with:
- "node_direction_relative_to_criteria": one of ["PUSHED_TOWARD_MET", "PUSHED_TOWARD_NOT_MET", "UNCLEAR_DIRECTION"]
- "node_direction_relative_to_criteria_confidence": float between 0.0-1.0
- "node_direction_relative_to_criteria_reasoning": string (brief explanation)

Return ONLY valid JSON, no markdown code blocks or extra text.
""".strip()

# Label taxonomy
LABEL_NEUTRAL_NOT_IN_RESPONSE = "kg_node_neutral_to_grading_not_in_response"
LABEL_NEUTRAL_IN_RESPONSE = "kg_node_neutral_to_grading_but_in_response"
LABEL_HELPED_POSITIVE_POINTS = "kg_node_helped_led_to_awarding_positive_points"
LABEL_HURT_NO_POSITIVE = "kg_node_hurt_led_to_not_awarding_positive_points"
LABEL_HURT_NEGATIVE_POINTS = "kg_node_hurt_led_to_awarding_negative_points"
LABEL_HELPED_NEGATIVE_POINTS = "kg_node_helped_led_to_not_awarding_negative_points"
LABEL_CONTRADICTION_NOT_MET = "kg_node_push_not_met_but_criterion_met"
LABEL_CONTRADICTION_MET = "kg_node_push_met_but_criterion_not_met"


def step_3_1_1_check_contributed(final_contributed: bool) -> Optional[str]:
    """
    Step 3.1.1: If final_contributed == false, assign neutral label immediately.

    Args:
        final_contributed: Whether node was marked as contributed

    Returns:
        Label string if contributed=False, else None (continue to 3.1.2)
    """
    if not final_contributed:
        return LABEL_NEUTRAL_NOT_IN_RESPONSE
    return None


def step_3_1_2_check_grading_justification(
    criterion_statement: str,
    grading_explanation: str,
    node_summary: str,
    node_contribution_explanation: str,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Step 3.1.2: Check if node is used as justification in grading explanation.

    Args:
        criterion_statement: The rubric criterion
        grading_explanation: Explanation of why criterion was/wasn't met
        node_summary: KG node summary
        node_contribution_explanation: How node was claimed to contribute
        model: LLM model to use

    Returns:
        Dict with:
        - node_used_as_justification_in_grading_explanation: bool
        - node_used_as_justification_in_grading_explanation_reasoning: str
    """
    result = call_llm_with_validation(
        template=PHASE_C_GRADING_JUSTIFICATION_TEMPLATE,
        replacements={
            "criterion_statement": criterion_statement,
            "grading_explanation": grading_explanation,
            "node_summary": node_summary,
            "node_contribution_explanation": node_contribution_explanation,
        },
        expected_keys=[
            "node_used_as_justification_in_grading_explanation",
            "node_used_as_justification_in_grading_explanation_reasoning",
        ],
        model=model,
    )

    return {
        "node_used_as_justification_in_grading_explanation": result[
            "node_used_as_justification_in_grading_explanation"
        ],
        "node_used_as_justification_in_grading_explanation_reasoning": result[
            "node_used_as_justification_in_grading_explanation_reasoning"
        ],
    }


def step_3_1_3_check_not_in_justification(
    node_used_as_justification: bool,
) -> Optional[str]:
    """
    Step 3.1.3: If node NOT used in grading justification, assign neutral and stop.

    Args:
        node_used_as_justification: Whether node was used in grading explanation

    Returns:
        Label string if not used, else None (continue to 3.1.4)
    """
    if not node_used_as_justification:
        return LABEL_NEUTRAL_IN_RESPONSE
    return None


def step_3_1_4_analyze_node_direction(
    criterion_statement: str,
    criteria_met: bool,
    grading_explanation: str,
    node_summary: str,
    node_contribution_explanation: str,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Step 3.1.4: Analyze node direction relative to criterion outcome.

    Args:
        criterion_statement: The rubric criterion
        criteria_met: Whether criterion was met (true/false)
        grading_explanation: Explanation of why criterion was/wasn't met
        node_summary: KG node summary
        node_contribution_explanation: How node contributed
        model: LLM model to use

    Returns:
        Dict with:
        - node_direction_relative_to_criteria: one of ["PUSHED_TOWARD_MET", "PUSHED_TOWARD_NOT_MET", "UNCLEAR_DIRECTION"]
        - node_direction_relative_to_criteria_confidence: float
        - node_direction_relative_to_criteria_reasoning: str
    """
    result = call_llm_with_validation(
        template=PHASE_D_NODE_DIRECTION_TEMPLATE,
        replacements={
            "criterion_statement": criterion_statement,
            "criteria_met": str(criteria_met).lower(),
            "grading_explanation": grading_explanation,
            "node_summary": node_summary,
            "node_contribution_explanation": node_contribution_explanation,
        },
        expected_keys=[
            "node_direction_relative_to_criteria",
            "node_direction_relative_to_criteria_confidence",
            "node_direction_relative_to_criteria_reasoning",
        ],
        model=model,
    )

    return {
        "node_direction_relative_to_criteria": result["node_direction_relative_to_criteria"],
        "node_direction_relative_to_criteria_confidence": result[
            "node_direction_relative_to_criteria_confidence"
        ],
        "node_direction_relative_to_criteria_reasoning": result[
            "node_direction_relative_to_criteria_reasoning"
        ],
    }


def step_3_1_5_check_unclear_direction(node_direction: str) -> Optional[str]:
    """
    Step 3.1.5: If direction is UNCLEAR, assign neutral and stop.

    Args:
        node_direction: Direction value from step 3.1.4

    Returns:
        Label if UNCLEAR, else None (continue to 3.1.6)
    """
    if node_direction == "UNCLEAR_DIRECTION":
        return LABEL_NEUTRAL_IN_RESPONSE
    return None


def step_3_1_6_deterministic_labeling(
    points: int,
    criteria_met: bool,
    node_direction: str,
) -> str:
    """
    Step 3.1.6: Deterministic mapping to final helped/hurt label.

    Args:
        points: Criterion points (positive, negative, or 0)
        criteria_met: Whether criterion was met
        node_direction: Direction from step 3.1.4 ("PUSHED_TOWARD_MET" or "PUSHED_TOWARD_NOT_MET")

    Returns:
        Final node label (string)
    """
    # Handle zero points
    if points == 0:
        return LABEL_NEUTRAL_IN_RESPONSE

    # Positive points
    if points > 0:
        if criteria_met:
            if node_direction == "PUSHED_TOWARD_MET":
                return LABEL_HELPED_POSITIVE_POINTS
            else:  # PUSHED_TOWARD_NOT_MET
                return LABEL_CONTRADICTION_NOT_MET
        else:  # criteria_met == False
            if node_direction == "PUSHED_TOWARD_NOT_MET":
                return LABEL_HURT_NO_POSITIVE
            else:  # PUSHED_TOWARD_MET
                return LABEL_CONTRADICTION_MET

    # Negative points
    if points < 0:
        if criteria_met:
            if node_direction == "PUSHED_TOWARD_MET":
                return LABEL_HURT_NEGATIVE_POINTS
            else:  # PUSHED_TOWARD_NOT_MET
                return LABEL_CONTRADICTION_NOT_MET
        else:  # criteria_met == False
            if node_direction == "PUSHED_TOWARD_NOT_MET":
                return LABEL_HELPED_NEGATIVE_POINTS
            else:  # PUSHED_TOWARD_MET
                return LABEL_CONTRADICTION_MET

    # Should not reach here
    raise ValueError(f"Unexpected points value: {points}")


def process_node_criterion_pair_part3(
    node_index: str,
    final_contributed: bool,
    final_node_contribution_explanation: str,
    node_summary: str,
    criterion_statement: str,
    criteria_met: bool,
    points: int,
    grading_explanation: str,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Process a single (node, criterion) pair through Part 3 (Steps 3.1.1-3.1.6).

    Args:
        node_index: Node ID
        final_contributed: Whether node was marked as contributed (from Part 2)
        final_node_contribution_explanation: Node's contribution explanation
        node_summary: KG node summary
        criterion_statement: The rubric criterion
        criteria_met: Whether criterion was met
        points: Criterion points (can be negative)
        grading_explanation: Explanation of why criterion was/wasn't met
        model: LLM model to use

    Returns:
        Dict with all Part 3 outputs (node_used_as_justification, direction, final_label)
    """
    logger.debug(f"Part 3 processing (node={node_index}, criterion={criterion_statement[:50]}...)")

    # Step 3.1.1: Check if contributed
    label = step_3_1_1_check_contributed(final_contributed)
    if label:
        return {"final_node_label": label}

    # Step 3.1.2: Check grading justification
    justification_check = step_3_1_2_check_grading_justification(
        criterion_statement=criterion_statement,
        grading_explanation=grading_explanation,
        node_summary=node_summary,
        node_contribution_explanation=final_node_contribution_explanation,
        model=model,
    )
    node_used = justification_check["node_used_as_justification_in_grading_explanation"]

    # Step 3.1.3: Check if not in justification
    label = step_3_1_3_check_not_in_justification(node_used)
    if label:
        return {
            "node_used_as_justification_in_grading_explanation": node_used,
            "node_used_as_justification_in_grading_explanation_reasoning": justification_check[
                "node_used_as_justification_in_grading_explanation_reasoning"
            ],
            "final_node_label": label,
        }

    # Step 3.1.4: Analyze node direction
    direction_check = step_3_1_4_analyze_node_direction(
        criterion_statement=criterion_statement,
        criteria_met=criteria_met,
        grading_explanation=grading_explanation,
        node_summary=node_summary,
        node_contribution_explanation=final_node_contribution_explanation,
        model=model,
    )
    node_direction = direction_check["node_direction_relative_to_criteria"]

    # Step 3.1.5: Check if unclear direction
    label = step_3_1_5_check_unclear_direction(node_direction)
    if label:
        return {
            "node_used_as_justification_in_grading_explanation": node_used,
            "node_used_as_justification_in_grading_explanation_reasoning": justification_check[
                "node_used_as_justification_in_grading_explanation_reasoning"
            ],
            "node_direction_relative_to_criteria": node_direction,
            "node_direction_relative_to_criteria_confidence": direction_check[
                "node_direction_relative_to_criteria_confidence"
            ],
            "node_direction_relative_to_criteria_reasoning": direction_check[
                "node_direction_relative_to_criteria_reasoning"
            ],
            "final_node_label": label,
        }

    # Step 3.1.6: Deterministic labeling
    final_label = step_3_1_6_deterministic_labeling(
        points=points,
        criteria_met=criteria_met,
        node_direction=node_direction,
    )

    return {
        "node_used_as_justification_in_grading_explanation": node_used,
        "node_used_as_justification_in_grading_explanation_reasoning": justification_check[
            "node_used_as_justification_in_grading_explanation_reasoning"
        ],
        "node_direction_relative_to_criteria": node_direction,
        "node_direction_relative_to_criteria_confidence": direction_check[
            "node_direction_relative_to_criteria_confidence"
        ],
        "node_direction_relative_to_criteria_reasoning": direction_check[
            "node_direction_relative_to_criteria_reasoning"
        ],
        "final_node_label": final_label,
    }


def step_3_1_4b_comparative_causal_weight(
    cited_nodes: list[dict],
    criterion_statement: str,
    criteria_met: bool,
    points: int,
    grading_explanation: str,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Part 3.1.4b: Comparative causal weight LLM call.

    Takes all cited nodes for a criterion together, and assigns each a
    causal weight: PRIMARY, SUPPORTING, or INCIDENTAL.

    Only called for nodes where node_used_as_justification_in_grading_explanation=True.
    Uncited nodes are skipped — no causal_weight is assigned to them.

    Args:
        cited_nodes: List of node dicts (each has node_index, node_summary,
            final_node_contribution_explanation, node_used_as_justification_in_grading_explanation_reasoning,
            node_direction_relative_to_criteria, node_direction_relative_to_criteria_reasoning, final_node_label)
        criterion_statement: The rubric criterion text
        criteria_met: Whether criterion was met
        points: Points value for this criterion
        grading_explanation: The grader's explanation text
        model: LLM model to use

    Returns:
        Dict with "node_causal_weights": list of {node_index, causal_weight, reasoning}
    """
    if not cited_nodes:
        return {"node_causal_weights": []}

    logger.info(f"Part 3.1.4b: Assigning causal weights to {len(cited_nodes)} cited nodes")

    # Format cited nodes list
    cited_nodes_lines = []
    for node in cited_nodes:
        cited_nodes_lines.append(
            f"Node ID: {node.get('node_index', node.get('node_id', ''))}\n"
            f"Node summary: {node.get('node_summary', '')}\n"
            f"Part 2 contribution explanation: {node.get('final_node_contribution_explanation', '')}\n"
            f"Justification reasoning: {node.get('node_used_as_justification_in_grading_explanation_reasoning', '')}\n"
            f"Direction: {node.get('node_direction_relative_to_criteria', '')}\n"
            f"Direction reasoning: {node.get('node_direction_relative_to_criteria_reasoning', '')}\n"
            f"Final label: {node.get('final_node_label', '')}"
        )
    cited_nodes_str = "\n\n".join(cited_nodes_lines)

    result = call_llm_with_validation(
        template=PHASE_D2_CAUSAL_WEIGHT_TEMPLATE,
        replacements={
            "criterion_statement": criterion_statement,
            "criteria_met": "true" if criteria_met else "false",
            "points": str(points),
            "grading_explanation": grading_explanation,
            "cited_nodes_list": cited_nodes_str,
        },
        expected_keys=["node_causal_weights"],
        model=model,
    )

    return {"node_causal_weights": result.get("node_causal_weights", [])}


# ============================================================================
# PART 4.1: COUNT LABELS BY TYPE
# ============================================================================


def step_4_1_count_labels(node_criterion_labels: list[dict]) -> dict:
    """
    Part 4.1: Count node labels by type for a criterion.

    Args:
        node_criterion_labels: List of Part 3 outputs (node+criterion pair labels)

    Returns:
        Dict with counts of each label type, including:
        - 3-way neutral sub-types (not_contributed, in_response_not_cited, direction_unclear)
        - 2 push direction sub-types (push_met_but_criterion_not_met, push_not_met_but_criterion_met)
        - 6 causal weight counters (primary/supporting/incidental for helped and hurt)
    """
    logger.info(f"Part 4.1: Counting labels across {len(node_criterion_labels)} nodes")

    num_helped_nodes = 0
    num_hurt_nodes = 0
    num_neutral_nodes = 0
    num_contradiction_nodes = 0
    num_unclear_direction_nodes = 0

    # 3-way neutral sub-types
    num_neutral_not_contributed = 0       # step 3.1.1: final_contributed=False — node never appeared in response
    num_neutral_in_response_not_cited = 0 # step 3.1.3: in response but grader didn't cite it
    num_neutral_in_response_direction_unclear = 0  # step 3.1.5: cited but UNCLEAR_DIRECTION

    # Push direction sub-types
    num_push_met_but_criterion_not_met = 0   # pushed toward met, criterion failed
    num_push_not_met_but_criterion_met = 0   # pushed toward not-met, criterion passed

    # Causal weight counters (only for nodes that have causal_weight set)
    num_primary_helped = 0
    num_supporting_helped = 0
    num_incidental_helped = 0
    num_primary_hurt = 0
    num_supporting_hurt = 0
    num_incidental_hurt = 0

    for label_result in node_criterion_labels:
        final_label = label_result.get("final_node_label")
        causal_weight = label_result.get("causal_weight")  # only set on cited nodes

        # HELPED labels
        if final_label in [
            LABEL_HELPED_POSITIVE_POINTS,
            LABEL_HELPED_NEGATIVE_POINTS
        ]:
            num_helped_nodes += 1
            if causal_weight == "PRIMARY":
                num_primary_helped += 1
            elif causal_weight == "SUPPORTING":
                num_supporting_helped += 1
            elif causal_weight == "INCIDENTAL":
                num_incidental_helped += 1

        # HURT labels
        elif final_label in [
            LABEL_HURT_NO_POSITIVE,
            LABEL_HURT_NEGATIVE_POINTS
        ]:
            num_hurt_nodes += 1
            if causal_weight == "PRIMARY":
                num_primary_hurt += 1
            elif causal_weight == "SUPPORTING":
                num_supporting_hurt += 1
            elif causal_weight == "INCIDENTAL":
                num_incidental_hurt += 1

        # NEUTRAL labels — track 3-way sub-types
        elif final_label == LABEL_NEUTRAL_NOT_IN_RESPONSE:
            num_neutral_nodes += 1
            num_neutral_not_contributed += 1
        elif final_label == LABEL_NEUTRAL_IN_RESPONSE:
            num_neutral_nodes += 1
            # Distinguish: was it cited by grader (direction_unclear) or not cited at all?
            if label_result.get("node_direction_relative_to_criteria") == "UNCLEAR_DIRECTION":
                num_neutral_in_response_direction_unclear += 1
            else:
                num_neutral_in_response_not_cited += 1

        # Diagnostic (contradiction/push) labels — track direction sub-types
        elif final_label == LABEL_CONTRADICTION_NOT_MET:
            # pushed toward met but criterion not met (criteria_met=False)
            num_contradiction_nodes += 1
            num_push_met_but_criterion_not_met += 1
        elif final_label == LABEL_CONTRADICTION_MET:
            # pushed toward not-met but criterion passed (criteria_met=True)
            num_contradiction_nodes += 1
            num_push_not_met_but_criterion_met += 1

        # Track unclear direction (diagnostic)
        if label_result.get("node_direction_relative_to_criteria") == "UNCLEAR_DIRECTION":
            num_unclear_direction_nodes += 1

    # Calculate additional metrics for downstream steps
    total_nodes = (
        num_helped_nodes + num_hurt_nodes + num_neutral_nodes +
        num_contradiction_nodes + num_unclear_direction_nodes
    )

    contradiction_ratio = (
        num_contradiction_nodes / total_nodes if total_nodes > 0 else 0.0
    )

    mixed_signals = (num_helped_nodes > 0 and num_hurt_nodes > 0)

    return {
        "num_helped_nodes": num_helped_nodes,
        "num_hurt_nodes": num_hurt_nodes,
        "num_neutral_nodes": num_neutral_nodes,
        "num_contradiction_nodes": num_contradiction_nodes,
        "num_unclear_direction_nodes": num_unclear_direction_nodes,
        # 3-way neutral sub-types
        "num_neutral_not_contributed": num_neutral_not_contributed,
        "num_neutral_in_response_not_cited": num_neutral_in_response_not_cited,
        "num_neutral_in_response_direction_unclear": num_neutral_in_response_direction_unclear,
        # Push direction sub-types
        "num_push_met_but_criterion_not_met": num_push_met_but_criterion_not_met,
        "num_push_not_met_but_criterion_met": num_push_not_met_but_criterion_met,
        # Causal weight counters
        "num_primary_helped": num_primary_helped,
        "num_supporting_helped": num_supporting_helped,
        "num_incidental_helped": num_incidental_helped,
        "num_primary_hurt": num_primary_hurt,
        "num_supporting_hurt": num_supporting_hurt,
        "num_incidental_hurt": num_incidental_hurt,
        # Derived metrics
        "total_nodes": total_nodes,
        "contradiction_ratio": contradiction_ratio,
        "mixed_signals": mixed_signals,
    }


# ============================================================================
# PART 4.2: ASSIGN KG INFLUENCE LABEL
# ============================================================================

def step_4_2_assign_kg_influence_label(
    # Node counts from step_4_1
    num_helped_nodes: int,
    num_hurt_nodes: int,
    num_neutral_nodes: int,
    # 3-way neutral sub-type counts
    num_neutral_not_contributed: int,
    num_neutral_in_response_not_cited: int,
    num_neutral_in_response_direction_unclear: int,
    num_contradiction_nodes: int,
    num_push_met_but_criterion_not_met: int,
    num_push_not_met_but_criterion_met: int,
    total_nodes: int,
    max_nodes_allowed: int,
    contradiction_ratio: float,
    mixed_signals: bool,
    # Causal weight counts
    num_primary_helped: int = 0,
    num_supporting_helped: int = 0,
    num_incidental_helped: int = 0,
    num_primary_hurt: int = 0,
    num_supporting_hurt: int = 0,
    num_incidental_hurt: int = 0,
    # Step 4.5 outputs (only if contradiction_ratio >= 0.25)
    high_contradiction_label_consistency: Optional[str] = None,
    # Step 4.7 outputs (only if mixed_signals=True)
    conflicting_signals_label_consistency: Optional[str] = None,
    conflicting_signals_dominant_influence: Optional[str] = None,
    # Criterion outcome (needed for push direction tokens in Easy Case / Case A)
    criteria_met: Optional[bool] = None,
) -> dict:
    """
    Part 4.2: Assign verbose KG influence label using 6-case structure.

    Cases determined by mixed_signals x contradiction_ratio:
    - Easy Case: not mixed_signals and contradiction_ratio == 0
    - Case A: not mixed_signals and 0 < contradiction_ratio < 0.25
    - Case E: mixed_signals and contradiction_ratio == 0
    - Case B: mixed_signals and 0 < contradiction_ratio < 0.25
    - Case C: not mixed_signals and contradiction_ratio >= 0.25
    - Case D: mixed_signals and contradiction_ratio >= 0.25

    Returns:
        Dict with kg_influence_label (verbose string) and assignment_reason
    """
    logger.info(
        f"Part 4.2: Assigning KG influence label "
        f"(helped={num_helped_nodes}, hurt={num_hurt_nodes}, "
        f"contradiction_ratio={contradiction_ratio:.3f}, mixed_signals={mixed_signals})"
    )

    # ------------------------------------------------------------------
    # Helper: KG coverage token
    # ------------------------------------------------------------------
    def _coverage_token():
        if max_nodes_allowed <= 0:
            return "MEDIUM_KG_COVERAGE"
        ratio = total_nodes / max_nodes_allowed
        if ratio <= 0.15:
            return "LOW_KG_COVERAGE"
        elif ratio <= 0.45:
            return "MEDIUM_KG_COVERAGE"
        else:
            return "HIGH_KG_COVERAGE"

    # ------------------------------------------------------------------
    # Helper: Causal weight token for dominant side (helped or hurt)
    # ------------------------------------------------------------------
    def _causal_weight_token(side: str) -> Optional[str]:
        """side is 'helped' or 'hurt'"""
        if side == "helped":
            primary, supporting, incidental = num_primary_helped, num_supporting_helped, num_incidental_helped
            all_token = "ALL_HELPED_NODES_WERE_MAIN_DRIVER"
            mostly_main = "HELPED_NODES_MOSTLY_MAIN_DRIVER"
            mostly_support = "HELPED_NODES_MOSTLY_MEANINGFUL_NOT_DECISIVE"
            mostly_incidental = "HELPED_NODES_MOSTLY_MINOR_CAUSAL_IMPACT"
        else:
            primary, supporting, incidental = num_primary_hurt, num_supporting_hurt, num_incidental_hurt
            all_token = "ALL_HURT_NODES_WERE_MAIN_DRIVER"
            mostly_main = "HURT_NODES_MOSTLY_MAIN_DRIVER"
            mostly_support = "HURT_NODES_MOSTLY_MEANINGFUL_NOT_DECISIVE"
            mostly_incidental = "HURT_NODES_MOSTLY_MINOR_CAUSAL_IMPACT"

        total_w = primary + supporting + incidental
        if total_w == 0:
            return None  # no causal weights set (step_3_1_4b didn't run or no cited nodes)
        if primary == total_w:
            return all_token
        # majority bucket (tie-break: higher weight wins)
        if primary > total_w / 2:
            return mostly_main
        elif supporting > total_w / 2:
            return mostly_support
        elif incidental > total_w / 2:
            return mostly_incidental
        else:
            # tie — use highest-weight bucket
            if primary >= supporting and primary >= incidental:
                return mostly_main
            elif supporting >= incidental:
                return mostly_support
            else:
                return mostly_incidental

    # ------------------------------------------------------------------
    # Helper: 3-way neutral suffix tokens
    # ------------------------------------------------------------------
    def _neutral_tokens() -> str:
        nc = "ALL_NODES_APPEARED_IN_RESPONSE" if num_neutral_not_contributed == 0 else "SOME_NODES_NEVER_APPEARED_IN_RESPONSE"
        cited = "ALL_RESPONSE_NODES_CITED_BY_GRADER" if num_neutral_in_response_not_cited == 0 else "SOME_NODES_IN_RESPONSE_BUT_GRADER_DID_NOT_CITE"
        clear = "ALL_CITED_NODES_HAD_CLEAR_IMPACT" if num_neutral_in_response_direction_unclear == 0 else "SOME_NODES_GRADER_CITED_BUT_IMPACT_UNCLEAR"
        return f"__{nc}__{cited}__{clear}"

    # ------------------------------------------------------------------
    # Helper: push contradiction token (for Case B suffix)
    # ------------------------------------------------------------------
    def _push_token_suffix() -> str:
        if num_contradiction_nodes == 0:
            return "__NO_PUSH_CONTRADICTIONS"
        # Only one direction can be > 0 per criterion
        if num_push_met_but_criterion_not_met > 0:
            pct = num_push_met_but_criterion_not_met / total_nodes if total_nodes > 0 else 0
            magnitude = "MAJOR" if pct >= 0.25 else "MINOR"
            return f"__{magnitude}_PUSH_TOWARD_MET_BUT_CRITERION_FAILED"
        else:
            pct = num_push_not_met_but_criterion_met / total_nodes if total_nodes > 0 else 0
            magnitude = "MAJOR" if pct >= 0.25 else "MINOR"
            return f"__{magnitude}_PUSH_TOWARD_NOT_MET_BUT_CRITERION_PASSED"

    # ------------------------------------------------------------------
    # Helper: contradiction magnitude token (Cases C/D)
    # ------------------------------------------------------------------
    def _magnitude_token() -> str:
        if contradiction_ratio < 0.50:
            return "MODERATE"
        elif contradiction_ratio < 0.75:
            return "HIGH"
        else:
            return "EXTREME"

    # ------------------------------------------------------------------
    # Helper: push token embedded in base label (Cases A)
    # ------------------------------------------------------------------
    def _push_label_fragment() -> str:
        """Returns the push fragment for Case A base labels."""
        if num_push_met_but_criterion_not_met > 0:
            pct = num_push_met_but_criterion_not_met / total_nodes if total_nodes > 0 else 0
            magnitude = "MAJOR" if pct >= 0.25 else "MINOR"
            return f"{magnitude}_PUSH_TOWARD_MET_BUT_CRITERION_FAILED"
        else:
            pct = num_push_not_met_but_criterion_met / total_nodes if total_nodes > 0 else 0
            magnitude = "MAJOR" if pct >= 0.25 else "MINOR"
            return f"{magnitude}_PUSH_TOWARD_NOT_MET_BUT_CRITERION_PASSED"

    # ------------------------------------------------------------------
    # Helper: assemble full label from base + suffixes
    # ------------------------------------------------------------------
    def _build_label(base: str, side: Optional[str] = None, include_push_suffix: bool = False, include_magnitude: bool = False, magnitude: Optional[str] = None) -> str:
        label = base
        if include_magnitude and magnitude:
            label += f"_WITH_{magnitude}_CONTRADICTIONS"
        label += f"__{_coverage_token()}"
        if side:
            cw = _causal_weight_token(side)
            if cw:
                label += f"__{cw}"
        if include_push_suffix:
            label += _push_token_suffix()
        label += _neutral_tokens()
        return label

    coverage = _coverage_token()

    # ------------------------------------------------------------------
    # EASY CASE: not mixed_signals, contradiction_ratio == 0
    # ------------------------------------------------------------------
    if not mixed_signals and contradiction_ratio == 0:
        if total_nodes == 0:
            return {
                "kg_influence_label": "KG_NO_NODES_RETRIEVED_NO_SIGNAL",
                "assignment_reason": "No KG nodes retrieved",
            }
        if num_helped_nodes > 0 and num_hurt_nodes == 0:
            base = "KG_NODES_HELPED_NO_CONFLICTS_NO_CONTRADICTIONS"
            return {
                "kg_influence_label": _build_label(base, side="helped"),
                "assignment_reason": f"Easy case: helped={num_helped_nodes}, hurt=0, contradiction_ratio=0",
            }
        if num_hurt_nodes > 0 and num_helped_nodes == 0:
            base = "KG_NODES_HURT_NO_CONFLICTS_NO_CONTRADICTIONS"
            return {
                "kg_influence_label": _build_label(base, side="hurt"),
                "assignment_reason": f"Easy case: hurt={num_hurt_nodes}, helped=0, contradiction_ratio=0",
            }
        # helped=0, hurt=0 — neutral sub-cases
        if num_neutral_not_contributed > 0 and (num_neutral_in_response_not_cited + num_neutral_in_response_direction_unclear) == 0:
            base = "KG_NODES_ALL_ABSENT_FROM_RESPONSE_NO_EFFECT"
        elif num_neutral_not_contributed == 0:
            base = "KG_NODES_PRESENT_IN_RESPONSE_BUT_NO_DIRECTIONAL_EFFECT"
        else:
            base = "KG_NODES_PARTIALLY_PRESENT_IN_RESPONSE_NO_DIRECTIONAL_EFFECT"
        return {
            "kg_influence_label": _build_label(base),
            "assignment_reason": f"Easy case: no directional nodes (neutral only)",
        }

    # ------------------------------------------------------------------
    # CASE A: not mixed_signals, 0 < contradiction_ratio < 0.25
    # ------------------------------------------------------------------
    if not mixed_signals and 0 < contradiction_ratio < 0.25:
        push_frag = _push_label_fragment()
        if num_helped_nodes > 0 and num_hurt_nodes == 0:
            base = f"KG_NODES_HELPED_{push_frag}_WITH_MINOR_CONTRADICTIONS"
            return {
                "kg_influence_label": _build_label(base, side="helped"),
                "assignment_reason": f"Case A: helped={num_helped_nodes}, contradiction_ratio={contradiction_ratio:.3f}",
            }
        if num_hurt_nodes > 0 and num_helped_nodes == 0:
            base = f"KG_NODES_HURT_{push_frag}_WITH_MINOR_CONTRADICTIONS"
            return {
                "kg_influence_label": _build_label(base, side="hurt"),
                "assignment_reason": f"Case A: hurt={num_hurt_nodes}, contradiction_ratio={contradiction_ratio:.3f}",
            }
        # helped=0, hurt=0 — pure push nodes
        base = f"KG_NODES_{push_frag}_NO_NET_HELPED_OR_HURT_WITH_MINOR_CONTRADICTIONS"
        return {
            "kg_influence_label": _build_label(base),
            "assignment_reason": f"Case A: no helped/hurt, only push nodes, contradiction_ratio={contradiction_ratio:.3f}",
        }

    # ------------------------------------------------------------------
    # CASE E: mixed_signals, contradiction_ratio == 0
    # ------------------------------------------------------------------
    if mixed_signals and contradiction_ratio == 0:
        consistency = conflicting_signals_label_consistency or "UNCLEAR"
        dominant = conflicting_signals_dominant_influence or "unclear"
        if consistency == "INCONSISTENT":
            base = "KG_NODES_HELPED_AND_HURT__OUTCOME_DRIVER_UNCLEAR__GRADER_EXPLANATION_INTERNALLY_INCONSISTENT__NO_CONTRADICTIONS"
            return {
                "kg_influence_label": _build_label(base),
                "assignment_reason": f"Case E INCONSISTENT: mixed_signals, consistency={consistency}",
            }
        if consistency == "UNCLEAR":
            base = "KG_NODES_HELPED_AND_HURT__OUTCOME_DRIVER_UNCLEAR__GRADER_COULD_NOT_DETERMINE_WHICH_NODES_DROVE_OUTCOME__NO_CONTRADICTIONS"
            return {
                "kg_influence_label": _build_label(base),
                "assignment_reason": f"Case E UNCLEAR: mixed_signals, consistency={consistency}",
            }
        # CONSISTENT
        if dominant == "helped_nodes":
            count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_helped_nodes > num_hurt_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
            base = f"KG_NODES_HELPED_AND_HURT__HELPED_NODES_DROVE_OUTCOME__GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME__{count_token}__NO_CONTRADICTIONS"
            side = "helped"
        elif dominant == "hurt_nodes":
            count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_hurt_nodes > num_helped_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
            base = f"KG_NODES_HELPED_AND_HURT__HURT_NODES_DROVE_OUTCOME__GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME__{count_token}__NO_CONTRADICTIONS"
            side = "hurt"
        elif dominant == "mixed_with_nonkg_factors":
            if num_helped_nodes > num_hurt_nodes:
                count_token = "HELPED_COUNT_HIGHER"
            elif num_hurt_nodes > num_helped_nodes:
                count_token = "HURT_COUNT_HIGHER"
            else:
                count_token = "COUNTS_EQUAL"
            base = f"KG_NODES_HELPED_AND_HURT__NON_KG_FACTORS_DROVE_OUTCOME__GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME__{count_token}__NO_CONTRADICTIONS"
            side = None
        else:  # unclear
            if num_helped_nodes > num_hurt_nodes:
                count_token = "HELPED_COUNT_HIGHER"
            elif num_hurt_nodes > num_helped_nodes:
                count_token = "HURT_COUNT_HIGHER"
            else:
                count_token = "COUNTS_EQUAL"
            base = f"KG_NODES_HELPED_AND_HURT__OUTCOME_DRIVER_UNCLEAR__GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME__{count_token}__NO_CONTRADICTIONS"
            side = None
        return {
            "kg_influence_label": _build_label(base, side=side),
            "assignment_reason": f"Case E CONSISTENT: dominant={dominant}, helped={num_helped_nodes}, hurt={num_hurt_nodes}",
        }

    # ------------------------------------------------------------------
    # CASE B: mixed_signals, 0 < contradiction_ratio < 0.25
    # ------------------------------------------------------------------
    if mixed_signals and 0 < contradiction_ratio < 0.25:
        consistency = conflicting_signals_label_consistency or "UNCLEAR"
        dominant = conflicting_signals_dominant_influence or "unclear"
        if consistency == "INCONSISTENT":
            base = "KG_NODES_HELPED_AND_HURT__OUTCOME_DRIVER_UNCLEAR__GRADER_EXPLANATION_INTERNALLY_INCONSISTENT__WITH_MINOR_CONTRADICTIONS"
            return {
                "kg_influence_label": _build_label(base, include_push_suffix=True),
                "assignment_reason": f"Case B INCONSISTENT: mixed_signals, consistency={consistency}",
            }
        if consistency == "UNCLEAR":
            base = "KG_NODES_HELPED_AND_HURT__OUTCOME_DRIVER_UNCLEAR__GRADER_COULD_NOT_DETERMINE_WHICH_NODES_DROVE_OUTCOME__WITH_MINOR_CONTRADICTIONS"
            return {
                "kg_influence_label": _build_label(base, include_push_suffix=True),
                "assignment_reason": f"Case B UNCLEAR: mixed_signals, consistency={consistency}",
            }
        # CONSISTENT
        if dominant == "helped_nodes":
            count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_helped_nodes > num_hurt_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
            base = f"KG_NODES_HELPED_AND_HURT__HELPED_NODES_DROVE_OUTCOME__GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME__{count_token}__WITH_MINOR_CONTRADICTIONS"
            side = "helped"
        elif dominant == "hurt_nodes":
            count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_hurt_nodes > num_helped_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
            base = f"KG_NODES_HELPED_AND_HURT__HURT_NODES_DROVE_OUTCOME__GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME__{count_token}__WITH_MINOR_CONTRADICTIONS"
            side = "hurt"
        elif dominant == "mixed_with_nonkg_factors":
            if num_helped_nodes > num_hurt_nodes:
                count_token = "HELPED_COUNT_HIGHER"
            elif num_hurt_nodes > num_helped_nodes:
                count_token = "HURT_COUNT_HIGHER"
            else:
                count_token = "COUNTS_EQUAL"
            base = f"KG_NODES_HELPED_AND_HURT__NON_KG_FACTORS_DROVE_OUTCOME__GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME__{count_token}__WITH_MINOR_CONTRADICTIONS"
            side = None
        else:  # unclear
            if num_helped_nodes > num_hurt_nodes:
                count_token = "HELPED_COUNT_HIGHER"
            elif num_hurt_nodes > num_helped_nodes:
                count_token = "HURT_COUNT_HIGHER"
            else:
                count_token = "COUNTS_EQUAL"
            base = f"KG_NODES_HELPED_AND_HURT__OUTCOME_DRIVER_UNCLEAR__GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME__{count_token}__WITH_MINOR_CONTRADICTIONS"
            side = None
        return {
            "kg_influence_label": _build_label(base, side=side, include_push_suffix=True),
            "assignment_reason": f"Case B CONSISTENT: dominant={dominant}, helped={num_helped_nodes}, hurt={num_hurt_nodes}, contradiction_ratio={contradiction_ratio:.3f}",
        }

    # ------------------------------------------------------------------
    # CASE C: not mixed_signals, contradiction_ratio >= 0.25
    # ------------------------------------------------------------------
    if not mixed_signals and contradiction_ratio >= 0.25:
        consistency = high_contradiction_label_consistency or "UNCLEAR"
        magnitude = _magnitude_token()
        if num_helped_nodes > 0:
            side_prefix = "KG_NODES_HELPED"
            side = "helped"
        elif num_hurt_nodes > 0:
            side_prefix = "KG_NODES_HURT"
            side = "hurt"
        else:
            side_prefix = "KG_NO_NET_DIRECTION"
            side = None

        if consistency == "CONSISTENT":
            base = f"{side_prefix}__WITH_{magnitude}_CONTRADICTIONS__NON_KG_FACTORS_EXPLAIN_OUTCOME_DESPITE_NODES"
        elif consistency == "INCONSISTENT":
            base = f"{side_prefix}__BUT_{magnitude}_CONTRADICTIONS__GRADING_EXPLANATION_INCONSISTENT_WITH_NODES"
        else:
            base = f"{side_prefix}__BUT_{magnitude}_CONTRADICTIONS__GRADING_EXPLANATION_UNCLEAR"

        return {
            "kg_influence_label": _build_label(base, side=side),
            "assignment_reason": f"Case C: contradiction_ratio={contradiction_ratio:.3f} ({magnitude}), consistency={consistency}, helped={num_helped_nodes}, hurt={num_hurt_nodes}",
        }

    # ------------------------------------------------------------------
    # CASE D: mixed_signals, contradiction_ratio >= 0.25
    # ------------------------------------------------------------------
    if mixed_signals and contradiction_ratio >= 0.25:
        cons_4_5 = high_contradiction_label_consistency or "UNCLEAR"
        cons_4_7 = conflicting_signals_label_consistency or "UNCLEAR"
        dominant = conflicting_signals_dominant_influence or "unclear"
        magnitude = _magnitude_token()

        # Both CONSISTENT — trust both analyses
        if cons_4_5 == "CONSISTENT" and cons_4_7 == "CONSISTENT":
            combined = "OUTCOME_SENSIBLE_AND_GRADER_CLEARLY_EXPLAINS_DRIVER"
            if dominant == "helped_nodes":
                count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_helped_nodes > num_hurt_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__HELPED_NODES_DROVE_OUTCOME__{combined}__{count_token}"
                side = "helped"
            elif dominant == "hurt_nodes":
                count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_hurt_nodes > num_helped_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__HURT_NODES_DROVE_OUTCOME__{combined}__{count_token}"
                side = "hurt"
            elif dominant == "mixed_with_nonkg_factors":
                if num_helped_nodes > num_hurt_nodes:
                    count_token = "HELPED_COUNT_HIGHER"
                elif num_hurt_nodes > num_helped_nodes:
                    count_token = "HURT_COUNT_HIGHER"
                else:
                    count_token = "COUNTS_EQUAL"
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__NON_KG_FACTORS_DROVE_OUTCOME__{combined}__{count_token}"
                side = None
            else:
                if num_helped_nodes > num_hurt_nodes:
                    count_token = "HELPED_COUNT_HIGHER"
                elif num_hurt_nodes > num_helped_nodes:
                    count_token = "HURT_COUNT_HIGHER"
                else:
                    count_token = "COUNTS_EQUAL"
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_DRIVER_UNCLEAR__{combined}__{count_token}"
                side = None
            return {
                "kg_influence_label": _build_label(base, side=side),
                "assignment_reason": f"Case D both CONSISTENT: dominant={dominant}, magnitude={magnitude}",
            }

        # 4.5=CONSISTENT, 4.7=INCONSISTENT
        if cons_4_5 == "CONSISTENT" and cons_4_7 == "INCONSISTENT":
            base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_MAKES_SENSE_DESPITE_CONTRADICTION_NODES__OUTCOME_DRIVER_UNCLEAR__GRADER_EXPLANATION_INTERNALLY_INCONSISTENT"
            return {"kg_influence_label": _build_label(base), "assignment_reason": f"Case D 4.5=CONSISTENT/4.7=INCONSISTENT: magnitude={magnitude}"}

        # 4.5=CONSISTENT, 4.7=UNCLEAR
        if cons_4_5 == "CONSISTENT" and cons_4_7 == "UNCLEAR":
            base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_MAKES_SENSE_DESPITE_CONTRADICTION_NODES__OUTCOME_DRIVER_UNCLEAR__GRADER_COULD_NOT_DETERMINE_WHICH_NODES_DROVE_OUTCOME"
            return {"kg_influence_label": _build_label(base), "assignment_reason": f"Case D 4.5=CONSISTENT/4.7=UNCLEAR: magnitude={magnitude}"}

        # 4.5=INCONSISTENT, 4.7=CONSISTENT — trust dominant_influence
        if cons_4_5 == "INCONSISTENT" and cons_4_7 == "CONSISTENT":
            surprise = "OUTCOME_SURPRISING_GIVEN_CONTRADICTION_NODE_PATTERN"
            clearly = "GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME"
            if dominant == "helped_nodes":
                count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_helped_nodes > num_hurt_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__HELPED_NODES_DROVE_OUTCOME__{surprise}__{clearly}__{count_token}"
                side = "helped"
            elif dominant == "hurt_nodes":
                count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_hurt_nodes > num_helped_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__HURT_NODES_DROVE_OUTCOME__{surprise}__{clearly}__{count_token}"
                side = "hurt"
            elif dominant == "mixed_with_nonkg_factors":
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__NON_KG_FACTORS_DROVE_OUTCOME__{surprise}__{clearly}"
                side = None
            else:
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_DRIVER_UNCLEAR__{surprise}__{clearly}"
                side = None
            return {"kg_influence_label": _build_label(base, side=side), "assignment_reason": f"Case D 4.5=INCONSISTENT/4.7=CONSISTENT: dominant={dominant}, magnitude={magnitude}"}

        # 4.5=INCONSISTENT, 4.7=INCONSISTENT
        if cons_4_5 == "INCONSISTENT" and cons_4_7 == "INCONSISTENT":
            base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_SURPRISING_GIVEN_CONTRADICTION_NODE_PATTERN__OUTCOME_DRIVER_UNCLEAR__GRADER_EXPLANATION_INTERNALLY_INCONSISTENT"
            return {"kg_influence_label": _build_label(base), "assignment_reason": f"Case D both INCONSISTENT: magnitude={magnitude}"}

        # 4.5=INCONSISTENT, 4.7=UNCLEAR
        if cons_4_5 == "INCONSISTENT" and cons_4_7 == "UNCLEAR":
            base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_SURPRISING_GIVEN_CONTRADICTION_NODE_PATTERN__OUTCOME_DRIVER_UNCLEAR__GRADER_COULD_NOT_DETERMINE_WHICH_NODES_DROVE_OUTCOME"
            return {"kg_influence_label": _build_label(base), "assignment_reason": f"Case D 4.5=INCONSISTENT/4.7=UNCLEAR: magnitude={magnitude}"}

        # 4.5=UNCLEAR, 4.7=CONSISTENT — trust dominant_influence
        if cons_4_5 == "UNCLEAR" and cons_4_7 == "CONSISTENT":
            unclear_cons = "OUTCOME_CONSISTENCY_WITH_CONTRADICTION_NODES_UNCLEAR"
            clearly = "GRADER_CLEARLY_EXPLAINS_WHICH_NODES_DROVE_OUTCOME"
            if dominant == "helped_nodes":
                count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_helped_nodes > num_hurt_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__HELPED_NODES_DROVE_OUTCOME__{unclear_cons}__{clearly}__{count_token}"
                side = "helped"
            elif dominant == "hurt_nodes":
                count_token = "RAW_NODE_COUNTS_AGREE_WITH_LLM" if num_hurt_nodes > num_helped_nodes else "RAW_NODE_COUNTS_DISAGREE_WITH_LLM"
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__HURT_NODES_DROVE_OUTCOME__{unclear_cons}__{clearly}__{count_token}"
                side = "hurt"
            elif dominant == "mixed_with_nonkg_factors":
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__NON_KG_FACTORS_DROVE_OUTCOME__{unclear_cons}__{clearly}"
                side = None
            else:
                base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_DRIVER_UNCLEAR__{unclear_cons}__{clearly}"
                side = None
            return {"kg_influence_label": _build_label(base, side=side), "assignment_reason": f"Case D 4.5=UNCLEAR/4.7=CONSISTENT: dominant={dominant}, magnitude={magnitude}"}

        # 4.5=UNCLEAR, 4.7=INCONSISTENT
        if cons_4_5 == "UNCLEAR" and cons_4_7 == "INCONSISTENT":
            base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_CONSISTENCY_WITH_CONTRADICTION_NODES_UNCLEAR__OUTCOME_DRIVER_UNCLEAR__GRADER_EXPLANATION_INTERNALLY_INCONSISTENT"
            return {"kg_influence_label": _build_label(base), "assignment_reason": f"Case D 4.5=UNCLEAR/4.7=INCONSISTENT: magnitude={magnitude}"}

        # 4.5=UNCLEAR, 4.7=UNCLEAR
        base = f"KG_NODES_HELPED_AND_HURT__WITH_{magnitude}_CONTRADICTIONS__OUTCOME_CONSISTENCY_WITH_CONTRADICTION_NODES_UNCLEAR__OUTCOME_DRIVER_UNCLEAR__GRADER_COULD_NOT_DETERMINE_WHICH_NODES_DROVE_OUTCOME"
        return {"kg_influence_label": _build_label(base), "assignment_reason": f"Case D both UNCLEAR: magnitude={magnitude}"}

    # Fallback (should not reach here)
    logger.warning(
        f"Step 4.2 reached end without assignment (helped={num_helped_nodes}, hurt={num_hurt_nodes}, "
        f"contradiction_ratio={contradiction_ratio:.3f}, mixed_signals={mixed_signals})"
    )
    return {
        "kg_influence_label": f"KG_NODES_PRESENT_IN_RESPONSE_BUT_NO_DIRECTIONAL_EFFECT__{_coverage_token()}{_neutral_tokens()}",
        "assignment_reason": "Fallback: unable to classify with given inputs",
    }


# ============================================================================
# PART 4.5: HIGH CONTRADICTION RATIO ANALYSIS (conditional)
# ============================================================================

HIGH_CONTRADICTION_ANALYSIS_TEMPLATE = """
You are analyzing a criterion where we found conflicting KG node signals that seem to contradict the outcome.

# Criterion
<<criterion_statement>>

# Criterion Outcome
The criterion was met: <<criteria_met>>

# Grading Explanation
<<grading_explanation>>

# Our Node Analysis
- <<num_helped_nodes>> nodes appeared to help this criterion
- <<num_hurt_nodes>> nodes appeared to hurt this criterion
- <<num_contradiction_nodes>> nodes had contradictory signals (pushing opposite direction from outcome)
- Contradiction ratio: <<contradiction_ratio>>%

# Response Text
<<response_text>>

# Knowledge Graph Nodes and Their Contributions
<<node_contributions>>

# Task
1. Explain HOW it's possible that despite these conflicting signals, the criterion ended up <<criteria_met>>
2. Did non-KG parts of the response (independent reasoning/facts) override the KG signals?
3. Given your understanding of the conflicting nodes, does this outcome make semantic sense?

Provide your reasoning in free text. Focus on semantic consistency: do the grading explanation and node signals align logically?

Return a JSON object with:
- "high_contradiction_resolution_insight": string (free text explanation of how outcome was reached)
- "high_contradiction_label_consistency": one of ["CONSISTENT", "INCONSISTENT", "UNCLEAR"]
  - "CONSISTENT": The outcome makes semantic sense despite contradictory signals
  - "INCONSISTENT": The outcome contradicts the dominant KG signals
  - "UNCLEAR": Cannot determine semantic consistency from available information
- "high_contradiction_consistency_reasoning": string (brief explanation of consistency assessment)

Return ONLY valid JSON, no markdown code blocks or extra text.
""".strip()


def step_4_5_analyze_high_contradictions(
    criterion_statement: str,
    criteria_met: bool,
    grading_explanation: str,
    response_text: str,
    node_summaries: list[dict],
    node_contributions: dict,
    num_helped_nodes: int,
    num_hurt_nodes: int,
    num_contradiction_nodes: int,
    contradiction_ratio: float,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Part 4.5: Analyze high contradiction cases where diagnostic labels dominate.

    Trigger: Only run if contradiction_ratio >= 0.25

    Purpose: Explain HOW the outcome occurred despite contradictory node signals.
    The LLM judges semantic consistency: does the grading explanation align with node signals?

    Args:
        criterion_statement: The rubric criterion text
        criteria_met: Whether criterion was met (true/false)
        grading_explanation: LLM's grading explanation
        response_text: Model's original response text
        node_summaries: List of node summary dicts with index keys
        node_contributions: Dict mapping node indices to contribution explanations
        num_helped_nodes: Count of HELPED-labeled nodes
        num_hurt_nodes: Count of HURT-labeled nodes
        num_contradiction_nodes: Count of diagnostic contradiction labels
        contradiction_ratio: Ratio of contradictions to total nodes
        model: LLM model to use

    Returns:
        Dict with:
        - high_contradiction_resolution_insight: str
        - high_contradiction_label_consistency: str (CONSISTENT/INCONSISTENT/UNCLEAR)
        - high_contradiction_consistency_reasoning: str
    """
    logger.info(
        f"Part 4.5: Analyzing high contradictions "
        f"(contradiction_ratio={contradiction_ratio:.1%}, criterion_met={criteria_met})"
    )

    # Format node contributions for template
    node_contributions_str = "Node contributions by index:\n"
    for node_idx, explanation in node_contributions.items():
        node_contributions_str += f"- {node_idx}: {explanation}\n"

    # Call LLM with validation
    result = call_llm_with_validation(
        template=HIGH_CONTRADICTION_ANALYSIS_TEMPLATE,
        replacements={
            "criterion_statement": criterion_statement,
            "criteria_met": "true" if criteria_met else "false",
            "grading_explanation": grading_explanation,
            "response_text": response_text,
            "num_helped_nodes": str(num_helped_nodes),
            "num_hurt_nodes": str(num_hurt_nodes),
            "num_contradiction_nodes": str(num_contradiction_nodes),
            "contradiction_ratio": f"{contradiction_ratio*100:.1f}",
            "node_contributions": node_contributions_str,
        },
        expected_keys=[
            "high_contradiction_resolution_insight",
            "high_contradiction_label_consistency",
            "high_contradiction_consistency_reasoning",
        ],
        model=model,
    )

    return {
        "high_contradiction_resolution_insight": result[
            "high_contradiction_resolution_insight"
        ],
        "high_contradiction_label_consistency": result[
            "high_contradiction_label_consistency"
        ],
        "high_contradiction_consistency_reasoning": result[
            "high_contradiction_consistency_reasoning"
        ],
    }


# ============================================================================
# PART 4.7: CONFLICTING SIGNALS ANALYSIS (conditional)
# ============================================================================

CONFLICTING_SIGNALS_ANALYSIS_TEMPLATE = """
You are analyzing a criterion where KG nodes had conflicting influences on the grading decision.

# Criterion
<<criterion_statement>>

# Criterion Outcome
The criterion was met: <<criteria_met>>

# Grading Explanation
<<grading_explanation>>

# Our Node Analysis
- <<num_helped_nodes>> nodes appeared to help this criterion
- <<num_hurt_nodes>> nodes appeared to hurt this criterion

# Response Text
<<response_text>>

# Knowledge Graph Nodes and Their Contributions
<<node_contributions>>

# Task
1. Which nodes pushed TOWARD this criterion being met? How did they contribute?
2. Which nodes pushed AWAY FROM this criterion being met? How did they work against it?
3. How did the grader weigh these conflicting signals to reach the final decision: criterion_met = <<criteria_met>>?
4. Which signal won out (helped or hurt), and why?

Provide your reasoning in free text. Focus on understanding how the grader resolved the conflict.

Return a JSON object with:
- "conflicting_signals_weighting": string (free text explanation of how conflicting signals were weighed)
- "conflicting_signals_dominant_influence": one of ["helped_nodes", "hurt_nodes", "mixed_with_nonkg_factors", "unclear"]
  - "helped_nodes": Helped nodes dominated the grading decision
  - "hurt_nodes": Hurt nodes dominated the grading decision
  - "mixed_with_nonkg_factors": Both KG signals and non-KG reasoning contributed equally
  - "unclear": Cannot determine which signal dominated
- "conflicting_signals_label_consistency": one of ["CONSISTENT", "INCONSISTENT", "UNCLEAR"]
  - "CONSISTENT": The grading explanation clearly explains which signal won and why
  - "INCONSISTENT": The explanation contradicts the dominant signal
  - "UNCLEAR": Cannot determine semantic consistency from available information
- "conflicting_signals_consistency_reasoning": string (brief explanation of consistency assessment)

Return ONLY valid JSON, no markdown code blocks or extra text.
""".strip()


def step_4_7_analyze_conflicting_signals(
    criterion_statement: str,
    criteria_met: bool,
    grading_explanation: str,
    response_text: str,
    node_summaries: list[dict],
    node_contributions: dict,
    num_helped_nodes: int,
    num_hurt_nodes: int,
    model: str = "azure/gpt-5.4",
) -> dict:
    """
    Part 4.7: Analyze cases where both HELPED and HURT nodes are cited in grading.

    Trigger: Only run if num_helped_nodes > 0 AND num_hurt_nodes > 0

    Purpose: Explain HOW the grader weighed conflicting node signals to reach the final decision.
    The LLM judges semantic consistency: does the explanation clearly identify which signal won?

    Independent of Part 4.5: Both triggers can run independently. A criterion can have both
    high contradictions AND conflicting signals.

    Args:
        criterion_statement: The rubric criterion text
        criteria_met: Whether criterion was met (true/false)
        grading_explanation: LLM's grading explanation
        response_text: Model's original response text
        node_summaries: List of node summary dicts with index keys
        node_contributions: Dict mapping node indices to contribution explanations
        num_helped_nodes: Count of HELPED-labeled nodes
        num_hurt_nodes: Count of HURT-labeled nodes
        model: LLM model to use

    Returns:
        Dict with:
        - conflicting_signals_weighting: str
        - conflicting_signals_dominant_influence: str (helped_nodes/hurt_nodes/mixed_with_nonkg_factors/unclear)
        - conflicting_signals_label_consistency: str (CONSISTENT/INCONSISTENT/UNCLEAR)
        - conflicting_signals_consistency_reasoning: str
    """
    logger.info(
        f"Part 4.7: Analyzing conflicting signals "
        f"({num_helped_nodes} helped vs {num_hurt_nodes} hurt, criterion_met={criteria_met})"
    )

    # Format node contributions for template
    node_contributions_str = "Node contributions by index:\n"
    for node_idx, explanation in node_contributions.items():
        node_contributions_str += f"- {node_idx}: {explanation}\n"

    # Call LLM with validation
    result = call_llm_with_validation(
        template=CONFLICTING_SIGNALS_ANALYSIS_TEMPLATE,
        replacements={
            "criterion_statement": criterion_statement,
            "criteria_met": "true" if criteria_met else "false",
            "grading_explanation": grading_explanation,
            "response_text": response_text,
            "num_helped_nodes": str(num_helped_nodes),
            "num_hurt_nodes": str(num_hurt_nodes),
            "node_contributions": node_contributions_str,
        },
        expected_keys=[
            "conflicting_signals_weighting",
            "conflicting_signals_dominant_influence",
            "conflicting_signals_label_consistency",
            "conflicting_signals_consistency_reasoning",
        ],
        model=model,
    )

    return {
        "conflicting_signals_weighting": result["conflicting_signals_weighting"],
        "conflicting_signals_dominant_influence": result[
            "conflicting_signals_dominant_influence"
        ],
        "conflicting_signals_label_consistency": result[
            "conflicting_signals_label_consistency"
        ],
        "conflicting_signals_consistency_reasoning": result[
            "conflicting_signals_consistency_reasoning"
        ],
    }


# ============================================================================
# PART 5: QUESTION-LEVEL METADATA AGGREGATION
# ============================================================================


def step_5_1_aggregate_kg_influence_labels(per_criterion_metadata: list) -> dict:
    """
    Step 5.1: Aggregate KG influence labels per question using prefix-based grouping.

    Groups criteria by kg_label (3-value rollup: kg_helped/kg_hurt/kg_neutral)
    and sums points by prediction category.

    Args:
        per_criterion_metadata: List of criterion metadata dicts from Part 4.
            Each dict contains kg_label (kg_helped/kg_hurt/kg_neutral) and points.

    Returns:
        Dict with:
        - num_kg_helped_criteria: Count of kg_helped criteria
        - num_kg_hurt_criteria: Count of kg_hurt criteria
        - num_kg_uncertain_criteria: Count of kg_neutral criteria
        - kg_points_helped/hurt/uncertain: Sum of abs(points) per category
    """
    num_kg_helped_criteria = 0
    num_kg_hurt_criteria = 0
    num_kg_uncertain_criteria = 0
    kg_points_helped = 0.0
    kg_points_hurt = 0.0
    kg_points_uncertain = 0.0

    for criterion in per_criterion_metadata:
        kg_label = criterion.get("kg_label", "kg_neutral")
        points = criterion.get("points", 0)
        abs_points = abs(points)

        if kg_label == "kg_helped":
            num_kg_helped_criteria += 1
            kg_points_helped += abs_points
        elif kg_label == "kg_hurt":
            num_kg_hurt_criteria += 1
            kg_points_hurt += abs_points
        else:
            num_kg_uncertain_criteria += 1
            kg_points_uncertain += abs_points

    return {
        "num_kg_helped_criteria": num_kg_helped_criteria,
        "num_kg_hurt_criteria": num_kg_hurt_criteria,
        "num_kg_uncertain_criteria": num_kg_uncertain_criteria,
        "kg_points_helped": kg_points_helped,
        "kg_points_hurt": kg_points_hurt,
        "kg_points_uncertain": kg_points_uncertain,
    }


def step_5_3_preserve_per_criterion_metadata(
    per_criterion_metadata: list, criteria_data: list
) -> list:
    """
    Step 5.3: Preserve complete per-criterion metadata with Part 4 outputs.

    Merges Part 4 metadata with criterion details from rubric.

    Args:
        per_criterion_metadata: List of criterion metadata dicts from Part 4.
            Contains: kg_influence_label, confidence_level, node counts,
            contradiction_ratio, mixed_signals, and conditional fields from 4.5/4.7.
        criteria_data: List of criterion dicts from rubric.
            Contains: criterion_statement, points, criteria_met, etc.

    Returns:
        List of enriched criterion dicts with all Part 4 metadata plus
        criterion_statement and points.
    """
    # Create a lookup by criteria_met or criterion_statement for matching
    criteria_by_stmt = {c.get("criterion_statement"): c for c in criteria_data}

    enriched_metadata = []
    for criterion in per_criterion_metadata:
        enriched = criterion.copy()

        # Add criterion text and points from rubric if available
        stmt = criterion.get("criterion_statement")
        if stmt and stmt in criteria_by_stmt:
            rubric_criterion = criteria_by_stmt[stmt]
            enriched["points"] = rubric_criterion.get("points")

        enriched_metadata.append(enriched)

    return enriched_metadata


def step_5_4_question_level_summary_output(
    question_id: str,
    per_criterion_metadata: list,
    num_criteria: int,
    kg_influence_summary: dict,
) -> dict:
    """
    Step 5.4: Generate question-level summary output.

    Assembles all aggregated metadata into final question-level structure.

    Args:
        question_id: The question ID
        per_criterion_metadata: Complete per-criterion metadata from Step 5.3
        num_criteria: Total number of criteria for this question
        kg_influence_summary: Output from Step 5.1

    Returns:
        Dict with complete question-level summary.
        NOTE: NO final USEKG/DONTUSEKG label. That is reserved for Part 6.
    """
    return {
        "question_id": question_id,
        "num_criteria": num_criteria,
        "kg_influence_summary": kg_influence_summary,
        "per_criterion_metadata": per_criterion_metadata,
    }


def aggregate_part5_for_question(
    question_id: str,
    per_criterion_metadata: list,
    criteria_data: list,
) -> dict:
    """
    Orchestrator for Part 5 aggregation.

    Runs Steps 5.1, 5.3, 5.4 in sequence and returns complete question-level
    metadata aggregation.

    Args:
        question_id: Question ID
        per_criterion_metadata: List of criterion metadata from Part 4
        criteria_data: List of criterion details from rubric

    Returns:
        Complete question-level summary per Step 5.4 spec.
    """
    # Step 5.1: Aggregate kg_influence_labels
    kg_influence_summary = step_5_1_aggregate_kg_influence_labels(
        per_criterion_metadata
    )

    # Step 5.3: Preserve complete per-criterion metadata
    enriched_metadata = step_5_3_preserve_per_criterion_metadata(
        per_criterion_metadata, criteria_data
    )

    # Step 5.4: Assemble question-level summary
    summary = step_5_4_question_level_summary_output(
        question_id=question_id,
        per_criterion_metadata=enriched_metadata,
        num_criteria=len(criteria_data),
        kg_influence_summary=kg_influence_summary,
    )

    return summary
