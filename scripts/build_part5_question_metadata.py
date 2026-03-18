#!/usr/bin/env python3
"""
Part 5: Question-level Metadata Aggregation

Aggregates per-criterion metadata from Phase 2A grading (Part 4) into
question-level summaries with KG influence aggregation and confidence breakdown.

Input:  Phase 2A grading checkpoint JSONL
        (contains example_level_metadata.rubric_items with Part 4 per-criterion data)

Output: step_5_output.jsonl
        One JSON object per line with aggregated metadata per question:
        {
          "question_id": "...",
          "num_criteria": N,
          "kg_influence_summary": {
            "num_kg_helped_criteria": X,
            "num_kg_hurt_criteria": Y,
            "num_kg_uncertain_criteria": Z,
            ...
          },
          "kg_helped_confidence_breakdown": {
            "num_helped_high_confidence": A,
            "num_helped_medium_confidence": B,
            ...
          },
          "kg_hurt_confidence_breakdown": { ... },
          "per_criterion_metadata": [
            { criterion details from Part 4 + criterion_statement + points ... }
          ]
        }

Usage:
  python -m simple_evals.scripts.build_part5_question_metadata \\
    --grading-output phase1a_kg_responses_grading_checkpoint.jsonl \\
    --output step_5_output.jsonl \\
    [--limit N]  # optional: limit to first N questions
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import Part 5 aggregation function
try:
    from simple_evals.kg_node_validation_part2 import aggregate_part5_for_question
except ImportError:
    # Fallback: try relative import for local testing
    try:
        from kg_node_validation_part2 import aggregate_part5_for_question
    except ImportError as e:
        print(f"ERROR: Could not import aggregate_part5_for_question: {e}", file=sys.stderr)
        print("Make sure kg_node_validation_part2.py is in the PYTHONPATH", file=sys.stderr)
        sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {line_num}: {e}")
                        continue
        logger.info(f"Loaded {len(data)} questions from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return []


def extract_part4_metadata(grading_output: Dict[str, Any]) -> tuple[Optional[str], Optional[List[Dict]]]:
    """
    Extract Part 4 per-criterion metadata from Phase 2A grading output.

    Returns:
        (question_id, per_criterion_metadata) where per_criterion_metadata is a list
        of criterion dicts with all Part 4 fields (kg_influence_label, confidence, node counts, etc.)
    """
    question_id = grading_output.get('example_level_metadata', {}).get('prompt_id') or grading_output.get('index') or grading_output.get('question_id')

    # Extract rubric_items from example_level_metadata
    example_metadata = grading_output.get('example_level_metadata', {})
    rubric_items = example_metadata.get('rubric_items', [])

    if not rubric_items:
        logger.warning(f"No rubric_items found for question {question_id}")
        return question_id, []

    # Convert rubric items to per_criterion_metadata format
    # Each rubric_item contains Part 4 fields like kg_influence_label, confidence_level, etc.
    per_criterion_metadata = []
    for item in rubric_items:
        criterion_data = {
            'criterion_statement': item.get('criterion'),  # criterion text (from RubricItem.to_dict())
            'points': item.get('points'),

            # Part 4.1: Node counts (now populated by healthbench_eval.py Part 4 integration)
            'num_helped_nodes': item.get('num_helped_nodes'),
            'num_hurt_nodes': item.get('num_hurt_nodes'),
            'num_neutral_nodes': item.get('num_neutral_nodes'),
            'num_contradiction_nodes': item.get('num_contradiction_nodes'),
            'num_unclear_direction_nodes': item.get('num_unclear_direction_nodes'),
            'total_nodes': item.get('total_nodes'),
            'contradiction_ratio': item.get('contradiction_ratio'),
            'mixed_signals': item.get('mixed_signals'),

            # Part 4.2: KG influence label
            'kg_influence_label': item.get('kg_influence_label'),
            'assignment_reason': item.get('assignment_reason'),

            # Part 4.3: Confidence
            'confidence_level': item.get('confidence_level'),
            'confidence_reasoning': item.get('confidence_reasoning'),

            # Part 4.5: High contradiction analysis (conditional)
            'high_contradiction_label_consistency': item.get('high_contradiction_label_consistency'),
            'high_contradiction_resolution_insight': item.get('high_contradiction_resolution_insight'),
            'high_contradiction_consistency_reasoning': item.get('high_contradiction_consistency_reasoning'),

            # Part 4.7: Conflicting signals analysis (conditional)
            'conflicting_signals_label_consistency': item.get('conflicting_signals_label_consistency'),
            'conflicting_signals_weighting': item.get('conflicting_signals_weighting'),
            'conflicting_signals_dominant_influence': item.get('conflicting_signals_dominant_influence'),
            'conflicting_signals_consistency_reasoning': item.get('conflicting_signals_consistency_reasoning'),
        }
        per_criterion_metadata.append(criterion_data)

    return question_id, per_criterion_metadata


def build_part5_for_grading_output(grading_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build Part 5 metadata for all questions in grading output.

    Args:
        grading_outputs: List of Phase 2A grading checkpoint dicts

    Returns:
        List of Part 5 question-level metadata dicts
    """
    part5_results = []

    for idx, grading_output in enumerate(grading_outputs, 1):
        question_id, per_criterion_metadata = extract_part4_metadata(grading_output)

        if not question_id:
            logger.warning(f"Skipping question {idx}: no question_id found")
            continue

        if not per_criterion_metadata:
            logger.warning(f"Skipping question {question_id}: no per-criterion metadata found")
            continue

        # Extract rubric data (criterion_statement + points) from the metadata
        criteria_data = [
            {
                'criterion_statement': item.get('criterion_statement'),
                'points': item.get('points'),
            }
            for item in per_criterion_metadata
        ]

        # Call Part 5 aggregation function
        try:
            part5_output = aggregate_part5_for_question(
                question_id=question_id,
                per_criterion_metadata=per_criterion_metadata,
                criteria_data=criteria_data,
            )
            part5_results.append(part5_output)

            if idx % 100 == 0:
                logger.info(f"Processed {idx} questions")

        except Exception as e:
            logger.error(f"Error aggregating Part 5 for question {question_id}: {e}")
            continue

    return part5_results


def main():
    parser = argparse.ArgumentParser(
        description="Build Part 5 question-level metadata from Phase 2A grading output"
    )
    parser.add_argument(
        "--grading-output",
        required=True,
        help="Path to Phase 2A grading checkpoint JSONL"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output step_5_output.jsonl"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N questions (optional)"
    )

    args = parser.parse_args()

    # Load Phase 2A grading output
    logger.info(f"Loading Phase 2A grading output from {args.grading_output}")
    grading_outputs = load_jsonl(args.grading_output)

    if not grading_outputs:
        logger.error("No grading output loaded. Exiting.")
        sys.exit(1)

    # Apply limit if specified
    if args.limit:
        grading_outputs = grading_outputs[:args.limit]
        logger.info(f"Limited to first {len(grading_outputs)} questions")

    # Build Part 5 metadata
    logger.info(f"Building Part 5 metadata for {len(grading_outputs)} questions...")
    part5_results = build_part5_for_grading_output(grading_outputs)

    if not part5_results:
        logger.error("No Part 5 results generated. Exiting.")
        sys.exit(1)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {len(part5_results)} Part 5 questions to {args.output}")
    with open(output_path, 'w') as f:
        for result in part5_results:
            f.write(json.dumps(result) + '\n')

    logger.info(f"✓ Part 5 metadata generation complete")
    logger.info(f"  Questions processed: {len(part5_results)}")
    logger.info(f"  Output file: {output_path}")


if __name__ == '__main__':
    main()
