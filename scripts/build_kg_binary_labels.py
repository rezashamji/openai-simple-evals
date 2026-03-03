#!/usr/bin/env python3
"""
Step 9: Build Question-Level Binary Classification Labels

Aggregates criterion-level data from compare_kg_runs_criterion_level.py
to create question-level training data with binary USEKG vs DONTUSEKG labels.

Uses a PLACEHOLDER decision rule for initial data generation (will be refined later):
  if score_delta > 0: "USEKG"
  else: "DONTUSEKG"

This script reads the criterion-level comparison JSONL and produces a labeled dataset
suitable for training a classifier on when to use KG vs baseline.

Usage:
  python build_kg_binary_labels.py \
    --criterion-level criterion_level_comparison.jsonl \
    --output classifier_training_data_healthbench_optimus.jsonl
"""

import json
import argparse
import sys
from typing import Any, Optional
from collections import defaultdict


def count_kg_labels(criterion_data: list[dict[str, Any]]) -> dict[str, int]:
    """Count occurrences of each KG label in criterion data."""
    counts = defaultdict(int)
    for criterion in criterion_data:
        kg_label = criterion.get("kg_label")
        if kg_label:
            counts[kg_label] += 1
    return dict(counts)


def placeholder_decision_rule(score_delta: float) -> str:
    """
    PLACEHOLDER decision rule for binary classification.
    Will be replaced with more sophisticated logic after analysis.

    Args:
        score_delta: KG score minus non-KG score

    Returns:
        "USEKG" or "DONTUSEKG"
    """
    if score_delta > 0:
        return "USEKG"
    else:
        return "DONTUSEKG"


def process_criterion_level_file(
    input_file: str,
    output_file: str,
) -> None:
    """
    Read criterion-level comparison JSONL and produce question-level labels.

    Args:
        input_file: Path to criterion_level_comparison.jsonl
        output_file: Path to output classifier_training_data.jsonl
    """
    print(f"Reading criterion-level data from {input_file}...", file=sys.stderr)

    output_lines = []
    total_questions = 0
    usekg_count = 0
    dontusekg_count = 0

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR at line {line_num}: {e}", file=sys.stderr)
                continue

            total_questions += 1
            prompt_id = item.get("prompt_id")
            kg_score = item.get("kg_score", 0)
            nonkg_score = item.get("nonkg_score", 0)
            score_delta = item.get("score_delta", kg_score - nonkg_score)
            bucket_distribution = item.get("bucket_distribution", {})
            criterion_level_data = item.get("criterion_level_data", [])

            # Apply placeholder decision rule
            use_kg_label = placeholder_decision_rule(score_delta)
            if use_kg_label == "USEKG":
                usekg_count += 1
            else:
                dontusekg_count += 1

            # Count KG labels
            kg_label_counts = count_kg_labels(criterion_level_data)

            # Extract kg_name and llm_model from criterion-level data (passed through from Phase 1)
            kg_name = item.get("kg_name", "unknown")
            llm_model = item.get("llm_model", "unknown")

            # Build output item
            output_item = {
                "prompt_id": prompt_id,
                "benchmark": "healthbench",
                "kg_name": kg_name,
                "llm_model": llm_model,

                # Question-level scores
                "kg_score": kg_score,
                "nonkg_score": nonkg_score,
                "score_delta": score_delta,

                # Binary classification (using placeholder rule)
                "use_kg_label": use_kg_label,
                "classification_reasoning": (
                    f"KG-grounded run scored {kg_score:.2f} vs baseline {nonkg_score:.2f} "
                    f"(delta={score_delta:+.2f}). Placeholder rule: use KG if score_delta > 0."
                ),
                "classification_note": (
                    "PLACEHOLDER LOGIC: This binary label was generated using a simple "
                    "score-delta threshold for initial data exploration. The real decision rule "
                    "should be refined based on criterion-level analysis, bucket distributions, "
                    "and category-level patterns. This placeholder is sufficient for generating "
                    "the training dataset structure but should be replaced with a more nuanced "
                    "rule in production."
                ),

                # Distribution of criterion-level buckets (for interpretability)
                "bucket_distribution": bucket_distribution,

                # Aggregated KG label counts
                "kg_label_counts": {
                    "kg_helped": kg_label_counts.get("kg_helped", 0),
                    "kg_neutral": kg_label_counts.get("kg_neutral", 0),
                    "kg_hurt": kg_label_counts.get("kg_hurt", 0),
                },

                # All criterion-level data for future classifier
                "criterion_level_data": criterion_level_data,
            }

            output_lines.append(output_item)

    # Write output
    print(f"Writing {len(output_lines)} questions to {output_file}...", file=sys.stderr)
    with open(output_file, 'w') as f:
        for item in output_lines:
            f.write(json.dumps(item) + '\n')

    # Print summary
    print("\n=== Question-Level Summary ===", file=sys.stderr)
    print(f"Total questions processed: {total_questions}", file=sys.stderr)
    print(f"Labeled USEKG: {usekg_count} ({100*usekg_count/total_questions:.1f}%)", file=sys.stderr)
    print(f"Labeled DONTUSEKG: {dontusekg_count} ({100*dontusekg_count/total_questions:.1f}%)", file=sys.stderr)
    print(f"\nOutput written to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Build question-level binary classification labels from criterion-level data"
    )
    parser.add_argument(
        "--criterion-level",
        required=True,
        help="Path to criterion_level_comparison.jsonl from Step 8b",
    )
    parser.add_argument(
        "--output",
        default="classifier_training_data_healthbench_optimus.jsonl",
        help="Output file for question-level labeled data",
    )

    args = parser.parse_args()

    process_criterion_level_file(args.criterion_level, args.output)


if __name__ == "__main__":
    main()
