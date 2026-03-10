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
from typing import Optional

import litellm

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
    model: str = "gpt-4-turbo",
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

    # Call LLM with response_format to guarantee valid JSON
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
        max_tokens=500,
    )

    response_text = response.choices[0].message.content
    logger.debug(f"LLM response: {response_text}")

    # Validate required keys (JSON format guaranteed by response_format)
    return validate_json_response(response_text, expected_keys)


# ============================================================================
# STEP 2.1: NODE CONTENT VALIDATION
# ============================================================================


def step_2_1_validate_node_content(
    response_text: str,
    node_summary: str,
    model: str = "gpt-4-turbo",
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
    model: str = "gpt-4-turbo",
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
    model: str = "gpt-4-turbo",
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
    logger.info(f"Part 2 processing node {node_index}")

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
    model: str = "gpt-4-turbo",
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
    model: str = "gpt-4-turbo",
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
    model: str = "gpt-4-turbo",
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
    logger.info(f"Part 3 processing (node={node_index}, criterion={criterion_statement[:50]}...)")

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


# ============================================================================
# PART 4.1: COUNT LABELS BY TYPE
# ============================================================================


def step_4_1_count_labels(node_criterion_labels: list[dict]) -> dict:
    """
    Part 4.1: Count node labels by type for a criterion.

    Args:
        node_criterion_labels: List of Part 3 outputs (node+criterion pair labels)

    Returns:
        Dict with counts of each label type
    """
    logger.info(f"Part 4.1: Counting labels across {len(node_criterion_labels)} nodes")

    num_helped_nodes = 0
    num_hurt_nodes = 0
    num_neutral_nodes = 0
    num_contradiction_nodes = 0
    num_unclear_direction_nodes = 0

    for label_result in node_criterion_labels:
        final_label = label_result.get("final_node_label")

        # HELPED labels
        if final_label in [
            LABEL_HELPED_POSITIVE_POINTS,
            LABEL_HELPED_NEGATIVE_POINTS
        ]:
            num_helped_nodes += 1

        # HURT labels
        elif final_label in [
            LABEL_HURT_NO_POSITIVE,
            LABEL_HURT_NEGATIVE_POINTS
        ]:
            num_hurt_nodes += 1

        # NEUTRAL labels
        elif final_label in [
            LABEL_NEUTRAL_NOT_IN_RESPONSE,
            LABEL_NEUTRAL_IN_RESPONSE
        ]:
            num_neutral_nodes += 1

        # Diagnostic (contradiction) labels
        elif final_label in [
            LABEL_CONTRADICTION_NOT_MET,
            LABEL_CONTRADICTION_MET
        ]:
            num_contradiction_nodes += 1

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
        "contradiction_ratio": contradiction_ratio,
        "mixed_signals": mixed_signals,
    }


# ============================================================================
# PART 4.2: ASSIGN KG INFLUENCE LABEL
# ============================================================================

# 7 possible KG influence labels
KG_INFLUENCE_HELPED = "KG_HELPED"
KG_INFLUENCE_HURT = "KG_HURT"
KG_INFLUENCE_NEUTRAL = "KG_NEUTRAL"
KG_INFLUENCE_HELPED_DESPITE_CONFLICTS = "KG_HELPED_DESPITE_CONFLICTS"
KG_INFLUENCE_HURT_DESPITE_CONFLICTS = "KG_HURT_DESPITE_CONFLICTS"
KG_INFLUENCE_UNCLEAR_MIXED_SIGNALS = "KG_UNCLEAR_MIXED_SIGNALS"
KG_INFLUENCE_OVERRIDDEN_BY_NON_KG = "KG_OVERRIDDEN_BY_NON_KG"
KG_INFLUENCE_UNEXPLAINED_CONTRADICTIONS = "KG_UNEXPLAINED_CONTRADICTIONS"


def step_4_2_assign_kg_influence_label(
    num_helped_nodes: int,
    num_hurt_nodes: int,
    num_neutral_nodes: int,
    num_contradiction_nodes: int,
    contradiction_ratio: float,
    mixed_signals: bool,
    conflicting_signals_label_consistency: Optional[str] = None,
    high_contradiction_label_consistency: Optional[str] = None,
) -> dict:
    """
    Part 4.2: Assign KG influence label based on node counts and consistency checks.

    Logic (in order):
    1. Simple cases: Only helped OR only hurt OR no impact
    2. Conflicting signals case: Both helped AND hurt nodes exist → use Step 4.7 consistency check
    3. High contradiction case: contradiction_ratio >= 0.25 → use Step 4.5 consistency check

    Args:
        num_helped_nodes: Count of HELPED-labeled nodes
        num_hurt_nodes: Count of HURT-labeled nodes
        num_neutral_nodes: Count of NEUTRAL-labeled nodes
        num_contradiction_nodes: Count of diagnostic contradiction labels
        contradiction_ratio: Ratio of contradiction nodes to total nodes
        mixed_signals: Whether both helped AND hurt nodes exist
        conflicting_signals_label_consistency: Output from Step 4.7 (CONSISTENT/INCONSISTENT/UNCLEAR)
        high_contradiction_label_consistency: Output from Step 4.5 (CONSISTENT/INCONSISTENT/UNCLEAR)

    Returns:
        Dict with:
        - kg_influence_label: One of 8 possible labels
        - assignment_reason: Brief explanation of why this label was assigned
    """
    logger.info(
        f"Part 4.2: Assigning KG influence label "
        f"(helped={num_helped_nodes}, hurt={num_hurt_nodes}, "
        f"contradiction_ratio={contradiction_ratio:.3f}, mixed_signals={mixed_signals})"
    )

    # ========================================================================
    # HIGH CONTRADICTION CASE (contradiction_ratio >= 0.25) - CHECK FIRST
    # Trigger Step 4.5 analysis first
    # ========================================================================

    if contradiction_ratio >= 0.25:
        # Step 4.5 should have already run; use its consistency result
        if high_contradiction_label_consistency is None:
            logger.warning(
                "contradiction_ratio >= 0.25 but high_contradiction_label_consistency is None. "
                "Step 4.5 should have been called first. Defaulting to UNCLEAR."
            )
            high_contradiction_label_consistency = "UNCLEAR"

        if high_contradiction_label_consistency == "CONSISTENT":
            return {
                "kg_influence_label": KG_INFLUENCE_OVERRIDDEN_BY_NON_KG,
                "assignment_reason": (
                    f"High contradiction ratio ({contradiction_ratio:.1%}). "
                    "Diagnostic labels dominated, but non-KG reasoning resolved outcome (Step 4.5 consistent)."
                ),
            }
        else:
            # INCONSISTENT or UNCLEAR
            return {
                "kg_influence_label": KG_INFLUENCE_UNEXPLAINED_CONTRADICTIONS,
                "assignment_reason": (
                    f"High contradiction ratio ({contradiction_ratio:.1%}). "
                    f"Too many contradictions to explain (Step 4.5 consistency: {high_contradiction_label_consistency})."
                ),
            }

    # ========================================================================
    # CONFLICTING SIGNALS CASE (both helped AND hurt nodes exist)
    # Trigger Step 4.7 analysis first
    # ========================================================================

    if mixed_signals and contradiction_ratio < 0.25:
        # Step 4.7 should have already run; use its consistency result
        if conflicting_signals_label_consistency is None:
            logger.warning(
                "mixed_signals=True but conflicting_signals_label_consistency is None. "
                "Step 4.7 should have been called first. Defaulting to UNCLEAR."
            )
            conflicting_signals_label_consistency = "UNCLEAR"

        if conflicting_signals_label_consistency == "CONSISTENT":
            # LLM explained which signal won out
            if num_helped_nodes > num_hurt_nodes:
                return {
                    "kg_influence_label": KG_INFLUENCE_HELPED_DESPITE_CONFLICTS,
                    "assignment_reason": (
                        f"Conflicting signals ({num_helped_nodes} helped vs {num_hurt_nodes} hurt), "
                        f"but helped nodes dominated (Step 4.7 consistent)"
                    ),
                }
            else:
                return {
                    "kg_influence_label": KG_INFLUENCE_HURT_DESPITE_CONFLICTS,
                    "assignment_reason": (
                        f"Conflicting signals ({num_helped_nodes} helped vs {num_hurt_nodes} hurt), "
                        f"but hurt nodes dominated (Step 4.7 consistent)"
                    ),
                }
        else:
            # INCONSISTENT or UNCLEAR: can't determine which signal dominated
            return {
                "kg_influence_label": KG_INFLUENCE_UNCLEAR_MIXED_SIGNALS,
                "assignment_reason": (
                    f"Conflicting signals ({num_helped_nodes} helped vs {num_hurt_nodes} hurt), "
                    f"but Step 4.7 consistency check was {conflicting_signals_label_consistency}"
                ),
            }

    # ========================================================================
    # SIMPLE CASES (no conflicting signals and no high contradictions)
    # ========================================================================

    if num_helped_nodes > 0 and num_hurt_nodes == 0:
        return {
            "kg_influence_label": KG_INFLUENCE_HELPED,
            "assignment_reason": f"Only helpful nodes ({num_helped_nodes}), no harmful nodes",
        }

    if num_hurt_nodes > 0 and num_helped_nodes == 0:
        return {
            "kg_influence_label": KG_INFLUENCE_HURT,
            "assignment_reason": f"Only harmful nodes ({num_hurt_nodes}), no helpful nodes",
        }

    if num_helped_nodes == 0 and num_hurt_nodes == 0:
        return {
            "kg_influence_label": KG_INFLUENCE_NEUTRAL,
            "assignment_reason": "No nodes had measurable impact (all neutral/diagnostic)",
        }

    # Fallback (should not reach here with valid input)
    logger.warning(
        "Step 4.2 reached end without assignment. This indicates unexpected input combination. "
        f"(helped={num_helped_nodes}, hurt={num_hurt_nodes}, contradiction_ratio={contradiction_ratio:.3f})"
    )
    return {
        "kg_influence_label": KG_INFLUENCE_NEUTRAL,
        "assignment_reason": "Fallback: unable to classify with given inputs",
    }


# ============================================================================
# PART 4.3: CALCULATE CONFIDENCE LEVEL
# ============================================================================


def step_4_3_calculate_confidence_level(
    contradiction_ratio: float,
    num_helped_nodes: int,
    num_hurt_nodes: int,
) -> dict:
    """
    Part 4.3: Calculate confidence level in the KG influence label.

    Confidence is based on node signal quality and consistency.

    Args:
        contradiction_ratio: Ratio of diagnostic contradiction labels to total nodes
        num_helped_nodes: Count of HELPED-labeled nodes
        num_hurt_nodes: Count of HURT-labeled nodes

    Returns:
        Dict with:
        - confidence_level: "HIGH" | "MEDIUM" | "LOW"
        - confidence_reasoning: Brief explanation
    """
    logger.info(
        f"Part 4.3: Calculating confidence level "
        f"(contradiction_ratio={contradiction_ratio:.3f}, helped={num_helped_nodes}, hurt={num_hurt_nodes})"
    )

    # HIGH contradiction ratio → LOW confidence
    if contradiction_ratio >= 0.25:
        return {
            "confidence_level": "LOW",
            "confidence_reasoning": (
                f"Too many contradictory signals ({contradiction_ratio:.1%} diagnostic labels). "
                "Outcome is uncertain."
            ),
        }

    # Mixed signals (both helped AND hurt) but not too many contradictions → MEDIUM confidence
    if num_helped_nodes > 0 and num_hurt_nodes > 0:
        return {
            "confidence_level": "MEDIUM",
            "confidence_reasoning": (
                f"Mixed signals ({num_helped_nodes} helped vs {num_hurt_nodes} hurt) "
                f"but contradiction ratio is low ({contradiction_ratio:.1%})."
            ),
        }

    # Clean signal (only helped OR only hurt OR no nodes) → HIGH confidence
    return {
        "confidence_level": "HIGH",
        "confidence_reasoning": "Clear directional signal from nodes or no nodes present.",
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
    model: str = "gpt-4-turbo",
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
    model: str = "gpt-4-turbo",
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
    Step 5.1: Aggregate KG influence labels per question.

    Groups criteria by kg_influence_label and sums points by prediction category.

    Args:
        per_criterion_metadata: List of criterion metadata dicts from Part 4.
            Each dict contains:
            - kg_influence_label: One of 7 labels
            - points: Integer points for this criterion
            - Other Part 4 fields

    Returns:
        Dict with:
        - num_kg_helped_criteria: Count of KG_HELPED + KG_HELPED_DESPITE_CONFLICTS
        - num_kg_hurt_criteria: Count of KG_HURT + KG_HURT_DESPITE_CONFLICTS
        - num_kg_uncertain_criteria: Count of KG_NEUTRAL + KG_UNCLEAR_MIXED_SIGNALS + KG_OVERRIDDEN_BY_NON_KG + KG_UNEXPLAINED_CONTRADICTIONS
        - kg_points_helped: Sum of abs(points) for helped criteria
        - kg_points_hurt: Sum of abs(points) for hurt criteria
        - kg_points_uncertain: Sum of abs(points) for uncertain criteria
    """
    helped_labels = {"KG_HELPED", "KG_HELPED_DESPITE_CONFLICTS"}
    hurt_labels = {"KG_HURT", "KG_HURT_DESPITE_CONFLICTS"}
    uncertain_labels = {
        "KG_NEUTRAL",
        "KG_UNCLEAR_MIXED_SIGNALS",
        "KG_OVERRIDDEN_BY_NON_KG",
        "KG_UNEXPLAINED_CONTRADICTIONS",
    }

    num_kg_helped_criteria = 0
    num_kg_hurt_criteria = 0
    num_kg_uncertain_criteria = 0
    kg_points_helped = 0.0
    kg_points_hurt = 0.0
    kg_points_uncertain = 0.0

    for criterion in per_criterion_metadata:
        label = criterion.get("kg_influence_label")
        points = criterion.get("points", 0)
        abs_points = abs(points)

        if label in helped_labels:
            num_kg_helped_criteria += 1
            kg_points_helped += abs_points
        elif label in hurt_labels:
            num_kg_hurt_criteria += 1
            kg_points_hurt += abs_points
        elif label in uncertain_labels:
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


def step_5_2_aggregate_confidence_level_distribution(
    per_criterion_metadata: list,
) -> dict:
    """
    Step 5.2: Aggregate confidence level distribution per question.

    Counts criteria by confidence level (HIGH/MEDIUM/LOW) separately for
    helped and hurt predictions.

    Args:
        per_criterion_metadata: List of criterion metadata dicts from Part 4.
            Each dict contains:
            - kg_influence_label: One of 7 labels
            - confidence_level: HIGH | MEDIUM | LOW

    Returns:
        Dict with two sub-dicts:
        - kg_helped_confidence_breakdown:
            - num_helped_high_confidence
            - num_helped_medium_confidence
            - num_helped_low_confidence
        - kg_hurt_confidence_breakdown:
            - num_hurt_high_confidence
            - num_hurt_medium_confidence
            - num_hurt_low_confidence
    """
    helped_labels = {"KG_HELPED", "KG_HELPED_DESPITE_CONFLICTS"}
    hurt_labels = {"KG_HURT", "KG_HURT_DESPITE_CONFLICTS"}

    helped_high = 0
    helped_medium = 0
    helped_low = 0
    hurt_high = 0
    hurt_medium = 0
    hurt_low = 0

    for criterion in per_criterion_metadata:
        label = criterion.get("kg_influence_label")
        confidence = criterion.get("confidence_level")

        if label in helped_labels:
            if confidence == "HIGH":
                helped_high += 1
            elif confidence == "MEDIUM":
                helped_medium += 1
            elif confidence == "LOW":
                helped_low += 1
        elif label in hurt_labels:
            if confidence == "HIGH":
                hurt_high += 1
            elif confidence == "MEDIUM":
                hurt_medium += 1
            elif confidence == "LOW":
                hurt_low += 1

    return {
        "kg_helped_confidence_breakdown": {
            "num_helped_high_confidence": helped_high,
            "num_helped_medium_confidence": helped_medium,
            "num_helped_low_confidence": helped_low,
        },
        "kg_hurt_confidence_breakdown": {
            "num_hurt_high_confidence": hurt_high,
            "num_hurt_medium_confidence": hurt_medium,
            "num_hurt_low_confidence": hurt_low,
        },
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
    kg_helped_confidence_breakdown: dict,
    kg_hurt_confidence_breakdown: dict,
) -> dict:
    """
    Step 5.4: Generate question-level summary output.

    Assembles all aggregated metadata into final question-level structure.

    Args:
        question_id: The question ID
        per_criterion_metadata: Complete per-criterion metadata from Step 5.3
        num_criteria: Total number of criteria for this question
        kg_influence_summary: Output from Step 5.1
        kg_helped_confidence_breakdown: From Step 5.2
        kg_hurt_confidence_breakdown: From Step 5.2

    Returns:
        Dict with complete question-level summary:
        - question_id
        - num_criteria
        - kg_influence_summary (counts + points)
        - kg_helped_confidence_breakdown
        - kg_hurt_confidence_breakdown
        - per_criterion_metadata (all criteria with full metadata)

        NOTE: NO final USEKG/DONTUSEKG label. That is reserved for Part 6.
    """
    return {
        "question_id": question_id,
        "num_criteria": num_criteria,
        "kg_influence_summary": kg_influence_summary,
        "kg_helped_confidence_breakdown": kg_helped_confidence_breakdown,
        "kg_hurt_confidence_breakdown": kg_hurt_confidence_breakdown,
        "per_criterion_metadata": per_criterion_metadata,
    }


def aggregate_part5_for_question(
    question_id: str,
    per_criterion_metadata: list,
    criteria_data: list,
) -> dict:
    """
    Orchestrator for Part 5 aggregation.

    Runs Steps 5.1-5.4 in sequence and returns complete question-level
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

    # Step 5.2: Aggregate confidence level distribution
    confidence_data = step_5_2_aggregate_confidence_level_distribution(
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
        kg_helped_confidence_breakdown=confidence_data[
            "kg_helped_confidence_breakdown"
        ],
        kg_hurt_confidence_breakdown=confidence_data["kg_hurt_confidence_breakdown"],
    )

    return summary
