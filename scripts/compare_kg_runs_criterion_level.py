#!/usr/bin/env python3
"""
Step 8b: Cross-Run Criterion-Level Comparison Script

Compares KG-grounded vs. non-KG baseline runs at the criterion level.
For each criterion on each question, assigns to one of 6 buckets:
  - both_correct: Both runs met criterion
  - kg_necessary: KG met, baseline didn't, KG label="kg_helped"
  - kg_lucky: KG met, baseline didn't, KG label="kg_neutral"
  - kg_resilient: KG met, baseline didn't, KG label="kg_hurt"
  - kg_hurt: Baseline met, KG didn't
  - both_failed: Neither met criterion

Usage:
  python compare_kg_runs_criterion_level.py \
    --kg-graded responses_ark_healthbench_grading_checkpoint.jsonl \
    --no-kg-graded responses_baseline_healthbench_grading_checkpoint.jsonl \
    --output criterion_level_comparison.jsonl \
    [--enable-edge-case-analysis]
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict


def assign_criterion_bucket(
    kg_criteria_met: bool,
    nonkg_criteria_met: bool,
    kg_label: Optional[str],
) -> str:
    """
    Assign a criterion to one of 6 buckets based on outcomes and KG label.

    Args:
        kg_criteria_met: Whether KG-grounded run met this criterion
        nonkg_criteria_met: Whether non-KG baseline met this criterion
        kg_label: KG label from grader ("kg_helped", "kg_hurt", "kg_neutral")

    Returns:
        Bucket name: "both_correct", "kg_necessary", "kg_lucky", "kg_resilient", "kg_hurt", "both_failed"
    """
    if kg_criteria_met and nonkg_criteria_met:
        return "both_correct"
    elif kg_criteria_met and not nonkg_criteria_met:
        if kg_label == "kg_helped":
            return "kg_necessary"
        elif kg_label == "kg_neutral":
            return "kg_lucky"
        elif kg_label == "kg_hurt":
            return "kg_resilient"
        else:
            # Fallback for unexpected label
            return "kg_lucky"
    elif not kg_criteria_met and nonkg_criteria_met:
        return "kg_hurt"
    else:  # not kg_criteria_met and not nonkg_criteria_met
        return "both_failed"


def get_bucket_signal_value(bucket: str) -> str:
    """Return signal value for each bucket."""
    signal_map = {
        "both_correct": "neutral",
        "kg_necessary": "high",
        "kg_lucky": "suspicious",
        "kg_resilient": "edge_case",
        "kg_hurt": "high",
        "both_failed": "neutral",
    }
    return signal_map.get(bucket, "unknown")


def get_bucket_reason(bucket: str, kg_label: Optional[str] = None) -> str:
    """Return human-readable reason for bucket assignment."""
    reasons = {
        "both_correct": "Both runs met criterion; both are good",
        "kg_necessary": "KG-grounded met criterion, baseline didn't, KG label confirms help",
        "kg_lucky": "KG-grounded met criterion, baseline didn't, but KG label is neutral (suspicious)",
        "kg_resilient": "KG-grounded met criterion, baseline didn't, but KG label says hurt (edge case)",
        "kg_hurt": "Baseline met criterion, KG-grounded didn't (KG harmful)",
        "both_failed": "Neither run met criterion",
    }
    return reasons.get(bucket, "Unknown")


def load_jsonl(filepath: str) -> dict[str, Any]:
    """Load JSONL file and index by prompt_id."""
    data_by_prompt_id = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # prompt_id should be in the item or in convo[0] metadata
                prompt_id = item.get("prompt_id") or item.get("index")
                if prompt_id is not None:
                    data_by_prompt_id[prompt_id] = item
    return data_by_prompt_id


def extract_criteria_data(single_result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract criterion data from a single eval result.

    Returns list of dicts with keys:
      - criterion, points, criteria_met, kg_label, explanation, kg_explanation, kg_reasoning
    """
    example_metadata = single_result.get("example_level_metadata", {})
    rubric_items = example_metadata.get("rubric_items", [])

    criteria_list = []
    for item in rubric_items:
        criteria_list.append({
            "criterion": item.get("criterion", ""),
            "points": item.get("points", 0),
            "criteria_met": item.get("criteria_met", False),
            "kg_label": item.get("kg_label"),  # Only in kg_grounded run
            "kg_explanation": item.get("kg_explanation"),
            "kg_reasoning": item.get("kg_reasoning"),
            "explanation": item.get("explanation", ""),
        })

    return criteria_list


def compare_runs(
    kg_result: dict[str, Any],
    nonkg_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Compare two results (kg_grounded vs no_kg) at the criterion level.

    Returns list of criterion-level comparison dicts with bucket assignments.
    """
    kg_criteria = extract_criteria_data(kg_result)
    nonkg_criteria = extract_criteria_data(nonkg_result)

    # Assume same number of criteria in same order
    comparisons = []
    for idx, (kg_crit, nonkg_crit) in enumerate(zip(kg_criteria, nonkg_criteria)):
        kg_met = kg_crit["criteria_met"]
        nonkg_met = nonkg_crit["criteria_met"]
        kg_label = kg_crit.get("kg_label")

        bucket = assign_criterion_bucket(kg_met, nonkg_met, kg_label)

        comparison = {
            "criterion_idx": idx,
            "criterion_text": kg_crit["criterion"],
            "points": kg_crit["points"],

            # Outcomes
            "kg_criteria_met": kg_met,
            "nonkg_criteria_met": nonkg_met,
            "bucket": bucket,

            # KG-grounded run details
            "kg_explanation": kg_crit.get("explanation", ""),
            "kg_label": kg_label,
            "kg_reasoning": kg_crit.get("kg_reasoning"),

            # Non-KG run details
            "nonkg_explanation": nonkg_crit.get("explanation", ""),

            # Cross-run analysis
            "cross_run_analysis": {
                "bucket": bucket,
                "signal_value": get_bucket_signal_value(bucket),
                "reason": get_bucket_reason(bucket, kg_label),
            },
        }

        comparisons.append(comparison)

    return comparisons


def count_bucket_distribution(comparisons: list[dict[str, Any]]) -> dict[str, int]:
    """Count occurrences of each bucket in comparisons."""
    distribution = defaultdict(int)
    for comp in comparisons:
        bucket = comp["bucket"]
        distribution[bucket] += 1

    # Ensure all buckets are present
    all_buckets = ["both_correct", "kg_necessary", "kg_lucky", "kg_resilient", "kg_hurt", "both_failed"]
    for bucket in all_buckets:
        if bucket not in distribution:
            distribution[bucket] = 0

    return dict(distribution)


def main():
    parser = argparse.ArgumentParser(
        description="Compare KG-grounded vs no-KG runs at criterion level"
    )
    parser.add_argument(
        "--kg-graded",
        required=True,
        help="Path to KG-grounded graded responses (JSONL)",
    )
    parser.add_argument(
        "--no-kg-graded",
        required=True,
        help="Path to non-KG baseline graded responses (JSONL)",
    )
    parser.add_argument(
        "--output",
        default="criterion_level_comparison.jsonl",
        help="Output file for criterion-level comparisons (JSONL)",
    )
    parser.add_argument(
        "--enable-edge-case-analysis",
        action="store_true",
        help="Enable LLM-based analysis for kg_lucky edge cases (not implemented yet)",
    )

    args = parser.parse_args()

    # Load both runs
    print(f"Loading KG-grounded results from {args.kg_graded}...", file=sys.stderr)
    kg_data = load_jsonl(args.kg_graded)
    print(f"  Loaded {len(kg_data)} results", file=sys.stderr)

    print(f"Loading non-KG baseline results from {args.no_kg_graded}...", file=sys.stderr)
    nonkg_data = load_jsonl(args.no_kg_graded)
    print(f"  Loaded {len(nonkg_data)} results", file=sys.stderr)

    # Find common prompt_ids
    common_ids = set(kg_data.keys()) & set(nonkg_data.keys())
    print(f"Found {len(common_ids)} questions in both runs", file=sys.stderr)

    if len(common_ids) == 0:
        print("ERROR: No common prompt_ids found between runs!", file=sys.stderr)
        sys.exit(1)

    # Compare at criterion level
    output_lines = []
    total_buckets = defaultdict(int)

    for prompt_id in sorted(common_ids):
        kg_result = kg_data[prompt_id]
        nonkg_result = nonkg_data[prompt_id]

        comparisons = compare_runs(kg_result, nonkg_result)
        bucket_dist = count_bucket_distribution(comparisons)

        # Aggregate bucket counts
        for bucket, count in bucket_dist.items():
            total_buckets[bucket] += count

        # Output this question's criterion-level data
        output_item = {
            "prompt_id": prompt_id,
            "kg_score": kg_result.get("score"),
            "nonkg_score": nonkg_result.get("score"),
            "score_delta": (kg_result.get("score", 0) - nonkg_result.get("score", 0)),
            "bucket_distribution": bucket_dist,
            "criterion_level_data": comparisons,
        }

        output_lines.append(output_item)

    # Write output
    print(f"Writing {len(output_lines)} questions to {args.output}...", file=sys.stderr)
    with open(args.output, 'w') as f:
        for item in output_lines:
            f.write(json.dumps(item) + '\n')

    # Print summary statistics
    print("\n=== Criterion-Level Summary ===", file=sys.stderr)
    print(f"Total criteria analyzed: {sum(total_buckets.values())}", file=sys.stderr)
    print("Bucket distribution:", file=sys.stderr)
    for bucket in ["both_correct", "kg_necessary", "kg_lucky", "kg_resilient", "kg_hurt", "both_failed"]:
        count = total_buckets[bucket]
        pct = 100 * count / sum(total_buckets.values()) if sum(total_buckets.values()) > 0 else 0
        print(f"  {bucket}: {count} ({pct:.1f}%)", file=sys.stderr)

    print(f"\nOutput written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
