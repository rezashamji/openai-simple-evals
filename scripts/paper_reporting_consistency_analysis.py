#!/usr/bin/env python3
"""
Paper Reporting: Consistency Rate Analysis

Computes two key metrics for paper reporting:
1. High Contradiction Consistency Rate - % of high-contradiction criteria with consistent labels
2. Conflicting Signals Consistency Rate - % of conflicting-signal criteria with consistent labels

Input: Part 6 complete metadata JSONL (one line per question with all criteria)
Output: Formatted text ready for paper inclusion
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def load_jsonl(filepath: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_all_criteria(part6_metadata: List[Dict]) -> List[Dict]:
    """
    Extract all criteria from Part 6 metadata.

    Each question in Part 6 has a list of criteria under 'phase5_analysis.per_criterion_metadata'.
    Flatten all criteria from all questions into a single list.
    """
    all_criteria = []

    for question_data in part6_metadata:
        phase5 = question_data.get("phase5_analysis", {})
        criteria = phase5.get("per_criterion_metadata", [])

        # Add all criteria from this question
        all_criteria.extend(criteria)

    return all_criteria


def compute_high_contradiction_rate(all_criteria: List[Dict]) -> Tuple[int, int, float, float]:
    """
    Compute High Contradiction Consistency Rate.

    - Trigger: contradiction_ratio >= 0.25
    - Metric: % with high_contradiction_label_consistency == "CONSISTENT"

    Returns: (total_high_contradiction, consistent_count, consistency_rate, percentage_of_all)
    """
    criteria_with_high_contradiction = [
        c for c in all_criteria
        if c.get("contradiction_ratio", 0) >= 0.25
    ]

    consistent_high_contradiction = [
        c for c in criteria_with_high_contradiction
        if c.get("high_contradiction_label_consistency") == "CONSISTENT"
    ]

    total_criteria = len(all_criteria)
    count_hc = len(criteria_with_high_contradiction)
    count_consistent = len(consistent_high_contradiction)

    # Consistency rate: of the high-contradiction criteria, how many are consistent
    consistency_rate = (count_consistent / count_hc * 100) if count_hc > 0 else 0.0

    # Percentage of all criteria: how many of all criteria are high-contradiction
    percentage_of_all = (count_hc / total_criteria * 100) if total_criteria > 0 else 0.0

    return count_hc, count_consistent, consistency_rate, percentage_of_all


def compute_conflicting_signals_rate(all_criteria: List[Dict]) -> Tuple[int, int, float, float]:
    """
    Compute Conflicting Signals Consistency Rate.

    - Trigger: num_helped_nodes > 0 AND num_hurt_nodes > 0
    - Metric: % with conflicting_signals_label_consistency == "CONSISTENT"

    Returns: (total_conflicting, consistent_count, consistency_rate, percentage_of_all)
    """
    criteria_with_conflicting = [
        c for c in all_criteria
        if c.get("num_helped_nodes", 0) > 0 and c.get("num_hurt_nodes", 0) > 0
    ]

    consistent_conflicting = [
        c for c in criteria_with_conflicting
        if c.get("conflicting_signals_label_consistency") == "CONSISTENT"
    ]

    total_criteria = len(all_criteria)
    count_cf = len(criteria_with_conflicting)
    count_consistent = len(consistent_conflicting)

    # Consistency rate: of the conflicting-signals criteria, how many are consistent
    consistency_rate = (count_consistent / count_cf * 100) if count_cf > 0 else 0.0

    # Percentage of all criteria: how many of all criteria have conflicting signals
    percentage_of_all = (count_cf / total_criteria * 100) if total_criteria > 0 else 0.0

    return count_cf, count_consistent, consistency_rate, percentage_of_all


def format_paper_text(
    hc_total: int,
    hc_consistent: int,
    hc_rate: float,
    hc_percentage: float,
    cs_total: int,
    cs_consistent: int,
    cs_rate: float,
    cs_percentage: float,
    total_criteria: int
) -> str:
    """Format the consistency rates into paper-ready text."""

    text = f"""
PAPER REPORTING: Consistency Rate Analysis
============================================

Total Criteria Analyzed: {total_criteria}

HIGH CONTRADICTION CONSISTENCY RATE
-----------------------------------
Criteria with high contradiction (contradiction_ratio >= 0.25): {hc_total} ({hc_percentage:.1f}% of total)
Semantically consistent labels: {hc_consistent}/{hc_total}
Consistency Rate: {hc_rate:.1f}%

Paper Text:
High contradiction criteria ({hc_percentage:.1f}% of total) had {hc_rate:.1f}% semantic consistency
with the LLM's own reasoning about node influences, suggesting our labeling logic successfully
handles cases with many contradictory diagnostic signals.

CONFLICTING SIGNALS CONSISTENCY RATE
------------------------------------
Criteria with conflicting signals (both helped and hurt nodes): {cs_total} ({cs_percentage:.1f}% of total)
Semantically consistent labels: {cs_consistent}/{cs_total}
Consistency Rate: {cs_rate:.1f}%

Paper Text:
Conflicting signals criteria ({cs_percentage:.1f}% of total) had {cs_rate:.1f}% semantic consistency,
indicating the grader's explanations align with how we characterize cases where both HELPED and
HURT nodes are present.
"""
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Compute paper reporting consistency rates from Part 6 metadata"
    )
    parser.add_argument(
        "part6_metadata",
        help="Path to Part 6 complete metadata JSONL file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file for paper text (default: print to stdout)"
    )

    args = parser.parse_args()

    # Load Part 6 metadata
    print(f"Loading Part 6 metadata from {args.part6_metadata}...", file=sys.stderr)
    part6_data = load_jsonl(args.part6_metadata)
    print(f"Loaded {len(part6_data)} questions", file=sys.stderr)

    # Extract all criteria
    print("Extracting criteria from all questions...", file=sys.stderr)
    all_criteria = extract_all_criteria(part6_data)
    total_criteria = len(all_criteria)
    print(f"Found {total_criteria} total criteria", file=sys.stderr)

    # Compute high contradiction rate
    print("\nComputing High Contradiction Consistency Rate...", file=sys.stderr)
    hc_total, hc_consistent, hc_rate, hc_percentage = compute_high_contradiction_rate(all_criteria)
    print(f"  High contradiction criteria: {hc_total} ({hc_percentage:.1f}%)", file=sys.stderr)
    print(f"  Consistent: {hc_consistent}/{hc_total} ({hc_rate:.1f}%)", file=sys.stderr)

    # Compute conflicting signals rate
    print("\nComputing Conflicting Signals Consistency Rate...", file=sys.stderr)
    cs_total, cs_consistent, cs_rate, cs_percentage = compute_conflicting_signals_rate(all_criteria)
    print(f"  Conflicting signals criteria: {cs_total} ({cs_percentage:.1f}%)", file=sys.stderr)
    print(f"  Consistent: {cs_consistent}/{cs_total} ({cs_rate:.1f}%)", file=sys.stderr)

    # Format paper text
    paper_text = format_paper_text(
        hc_total, hc_consistent, hc_rate, hc_percentage,
        cs_total, cs_consistent, cs_rate, cs_percentage,
        total_criteria
    )

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(paper_text)
        print(f"\nPaper text saved to {args.output}", file=sys.stderr)
    else:
        print(paper_text)


if __name__ == "__main__":
    main()
