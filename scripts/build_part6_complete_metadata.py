#!/usr/bin/env python3
"""
Part 6: Build Complete Metadata Storage

Combines metadata from all phases (1-5) into comprehensive JSONL output.
Each line contains ALL metadata for one question from the entire pipeline.

Input Files:
1. phase1_kg.jsonl - Phase 1 KG output
2. phase1_no_kg.jsonl - Phase 1 Baseline output
3. *_grading_checkpoint.jsonl (KG) - Phase 2A output
4. *_grading_checkpoint.jsonl (Baseline) - Phase 2B output
5. step_5_output.jsonl - Phase 5 aggregation output

Output:
- part6_complete_metadata.jsonl - One JSON object per line with all metadata per question

Critical Preservation Rules:
- Preserve ALL Phase 4 metadata (node counts, influence labels, confidence, etc.)
- Include conditional fields (4.5/4.7) as null if not triggered (don't omit)
- Preserve grading explanations for both KG and Baseline runs
- Keep node_summaries and node_contributions from Phase 1
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


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
        return []
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {filepath}: {e}", file=sys.stderr)
        return []


def build_part6_metadata(
    phase1_kg: List[Dict],
    phase1_baseline: List[Dict],
    phase2a_grading: List[Dict],
    phase2b_grading: List[Dict],
    phase5_analysis: List[Dict],
) -> List[Dict]:
    """
    Combine all phase outputs into complete metadata per question.

    Args:
        phase1_kg: Phase 1 KG run output
        phase1_baseline: Phase 1 Baseline run output
        phase2a_grading: Phase 2A KG grading checkpoint
        phase2b_grading: Phase 2B Baseline grading checkpoint
        phase5_analysis: Phase 5 aggregated analysis

    Returns:
        List of complete metadata objects, one per question
    """
    results = []

    # Verify all inputs have same length
    num_questions = len(phase1_kg)
    if len(phase1_baseline) != num_questions:
        print(f"WARNING: phase1_baseline has {len(phase1_baseline)} questions, expected {num_questions}", file=sys.stderr)
    if len(phase2a_grading) != num_questions:
        print(f"WARNING: phase2a_grading has {len(phase2a_grading)} questions, expected {num_questions}", file=sys.stderr)
    if len(phase2b_grading) != num_questions:
        print(f"WARNING: phase2b_grading has {len(phase2b_grading)} questions, expected {num_questions}", file=sys.stderr)

    # Build phase5 lookup by question_id for flexible matching
    phase5_by_qid = {}
    for item in phase5_analysis:
        qid = item.get("question_id")
        if qid:
            phase5_by_qid[qid] = item

    # Process each question by index
    for idx in range(num_questions):
        # Get data from each phase
        p1_kg = phase1_kg[idx] if idx < len(phase1_kg) else {}
        p1_baseline = phase1_baseline[idx] if idx < len(phase1_baseline) else {}
        p2a = phase2a_grading[idx] if idx < len(phase2a_grading) else {}
        p2b = phase2b_grading[idx] if idx < len(phase2b_grading) else {}

        # Get Phase 5 data by matching question_id from Phase 2A
        question_id = p2a.get("example_level_metadata", {}).get("prompt_id", f"q_{idx}")
        p5 = phase5_by_qid.get(question_id, {})

        # Build question-level metadata object (OPTION B: merge grading into phase1)
        complete_metadata = {
            "question_id": question_id,
            "phase1": build_phase1_section(p1_kg, p1_baseline, p2a, p2b),
            "phase5_analysis": build_phase5_section(p2a, p2b, p5),
        }

        results.append(complete_metadata)

    return results


def build_phase1_section(p1_kg: Dict, p1_baseline: Dict, p2a: Dict = None, p2b: Dict = None) -> Dict:
    """
    Build Phase 1 section with both KG and Baseline runs.

    OPTION B: Merges Phase 2A/2B grading data into phase1 for unified data access.
    This makes phase1.kg_run and phase1.baseline_run self-contained with all grading data.
    """
    p2a_metadata = p2a.get("example_level_metadata", {}) if p2a else {}
    p2b_metadata = p2b.get("example_level_metadata", {}) if p2b else {}

    # Build KG run: Phase 1 response + nodes + Phase 2A grading data
    kg_run = {
        "response_text": p1_kg.get("response_text", ""),
        "node_summaries": p1_kg.get("node_summaries", []),
        "node_contributions": p1_kg.get("node_contributions", {}),
    }

    # Merge Phase 2A grading into kg_run for unified access
    if p2a:
        kg_run.update({
            "score": p2a.get("score"),
            "metrics": p2a_metadata.get("metrics", p2a.get("metrics", {})),
            "example_level_metadata": {
                "kg_reasoning_details": p2a_metadata.get("kg_reasoning_details", {}),
                "per_criterion_metadata": p2a_metadata.get("per_criterion_metadata", []),
                "per_node_metadata": p2a_metadata.get("per_node_metadata", []),
                "rubric_items": p2a_metadata.get("rubric_items", []),
                "score": p2a_metadata.get("score"),
                "usage": p2a_metadata.get("usage"),
                "run_tag": p2a_metadata.get("run_tag"),
            },
        })

    # Build baseline run: Phase 1 response + Phase 2B grading data
    baseline_run = {
        "response_text": p1_baseline.get("response_text", ""),
    }

    # Merge Phase 2B grading into baseline_run
    if p2b:
        baseline_run.update({
            "score": p2b.get("score"),
            "metrics": p2b_metadata.get("metrics", p2b.get("metrics", {})),
            "example_level_metadata": {
                "rubric_items": p2b_metadata.get("rubric_items", []),
                "score": p2b_metadata.get("score"),
                "usage": p2b_metadata.get("usage"),
                "run_tag": p2b_metadata.get("run_tag"),
            },
        })

    return {
        "prompt_id": p1_kg.get("prompt_id", ""),
        "prompt": p1_kg.get("prompt", []),
        "rubrics": p1_kg.get("rubrics", []),
        "example_tags": p1_kg.get("example_tags", []),
        "kg_run": kg_run,
        "baseline_run": baseline_run,
        "metadata": {
            "kg_name": p1_kg.get("kg_name"),
            "llm_model": p1_kg.get("llm_model"),
        },
    }


def build_phase2a_section(p2a: Dict) -> Dict:
    """Extract Phase 2A (KG Grading) metadata."""
    metadata = p2a.get("example_level_metadata", {})

    return {
        "overall_score": p2a.get("score"),
        "metrics": metadata.get("metrics", p2a.get("metrics", {})),
        "rubric_items_with_grades": metadata.get("rubric_items", []),
        "kg_reasoning_details": metadata.get("kg_reasoning_details", {}),
        # Complete audit trail: Parts 2-3 intermediate data for training
        "per_node_metadata": metadata.get("per_node_metadata", []),
        "per_criterion_metadata": metadata.get("per_criterion_metadata", []),
    }


def build_phase2b_section(p2b: Dict) -> Dict:
    """Extract Phase 2B (Baseline Grading) metadata."""
    metadata = p2b.get("example_level_metadata", {})

    return {
        "overall_score": p2b.get("score"),
        "metrics": metadata.get("metrics", p2b.get("metrics", {})),
        "rubric_items_with_grades": metadata.get("rubric_items", []),
    }


def build_phase5_section(p2a: Dict, p2b: Dict, p5: Dict) -> Dict:
    """
    Build Phase 5 section by merging Phase 5 output with grading data.

    For each criterion, adds:
    - kg_criteria_met and kg_explanation from Phase 2A
    - baseline_criteria_met and baseline_explanation from Phase 2B
    - all Part 4 analysis fields from Phase 5
    """
    # Extract grading data
    p2a_metadata = p2a.get("example_level_metadata", {})
    p2b_metadata = p2b.get("example_level_metadata", {})

    p2a_rubrics = {item.get("criterion"): item for item in p2a_metadata.get("rubric_items", [])}
    p2b_rubrics = {item.get("criterion"): item for item in p2b_metadata.get("rubric_items", [])}

    # Process per-criterion metadata from Phase 5
    per_criterion = []
    if "per_criterion_metadata" in p5:
        for criterion_data in p5["per_criterion_metadata"]:
            criterion_stmt = criterion_data.get("criterion_statement", "")

            # Find matching rubric items from Phase 2A and 2B
            p2a_item = p2a_rubrics.get(criterion_stmt, {})
            p2b_item = p2b_rubrics.get(criterion_stmt, {})

            # Build complete criterion entry with all fields
            complete_criterion = {
                # Basic info
                "criterion_statement": criterion_stmt,
                "points": criterion_data.get("points"),

                # From Phase 2A/2B grading
                "kg_criteria_met": p2a_item.get("criteria_met"),
                "kg_explanation": p2a_item.get("explanation", ""),
                "baseline_criteria_met": p2b_item.get("criteria_met"),
                "baseline_explanation": p2b_item.get("explanation", ""),

                # From Part 4.1: Node counts
                "num_helped_nodes": criterion_data.get("num_helped_nodes"),
                "num_hurt_nodes": criterion_data.get("num_hurt_nodes"),
                "num_neutral_nodes": criterion_data.get("num_neutral_nodes"),
                "num_contradiction_nodes": criterion_data.get("num_contradiction_nodes"),
                "num_unclear_direction_nodes": criterion_data.get("num_unclear_direction_nodes"),
                "total_nodes": criterion_data.get("total_nodes"),

                # From Part 4.1: Metrics
                "contradiction_ratio": criterion_data.get("contradiction_ratio"),
                "mixed_signals": criterion_data.get("mixed_signals"),

                # From new step_4_1 sub-type counts
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

                # From Part 4.2: KG Influence Label
                "kg_influence_label": criterion_data.get("kg_influence_label"),
                "assignment_reason": criterion_data.get("assignment_reason", ""),

                # From Part 4.5: High Contradiction Analysis (conditional)
                "high_contradiction_label_consistency": criterion_data.get("high_contradiction_label_consistency"),
                "high_contradiction_resolution_insight": criterion_data.get("high_contradiction_resolution_insight"),
                "high_contradiction_consistency_reasoning": criterion_data.get("high_contradiction_consistency_reasoning"),

                # From Part 4.7: Conflicting Signals Analysis (conditional)
                "conflicting_signals_label_consistency": criterion_data.get("conflicting_signals_label_consistency"),
                "conflicting_signals_weighting": criterion_data.get("conflicting_signals_weighting"),
                "conflicting_signals_dominant_influence": criterion_data.get("conflicting_signals_dominant_influence"),
                "conflicting_signals_consistency_reasoning": criterion_data.get("conflicting_signals_consistency_reasoning"),
            }

            per_criterion.append(complete_criterion)

    return {
        "num_criteria": p5.get("num_criteria", len(per_criterion)),
        "kg_relevance_score": p2a.get("example_level_metadata", {}).get("kg_relevance_score"),
        "kg_influence_summary": p5.get("kg_influence_summary", {}),
        "per_criterion_metadata": per_criterion,
    }


def validate_output(metadata: Dict) -> bool:
    """Validate that metadata has all required top-level sections."""
    # OPTION B: phase2a/2b grading data is now merged into phase1, so don't require separate sections
    required_sections = ["question_id", "phase1", "phase5_analysis"]
    for section in required_sections:
        if section not in metadata:
            print(f"WARNING: Missing section: {section}", file=sys.stderr)
            return False

    # Verify phase1 has kg_run with grading data
    phase1 = metadata.get("phase1", {})
    kg_run = phase1.get("kg_run", {})
    if "example_level_metadata" not in kg_run or "kg_reasoning_details" not in kg_run.get("example_level_metadata", {}):
        print(f"WARNING: phase1.kg_run missing grading data for {metadata.get('question_id')}", file=sys.stderr)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build Part 6 complete metadata from all pipeline phases"
    )
    parser.add_argument("--phase1-kg", required=True, help="Path to phase1_kg.jsonl")
    parser.add_argument("--phase1-baseline", required=True, help="Path to phase1_no_kg.jsonl")
    parser.add_argument("--phase2a-grading", required=True, help="Path to Phase 2A grading checkpoint")
    parser.add_argument("--phase2b-grading", required=True, help="Path to Phase 2B grading checkpoint")
    parser.add_argument("--phase5-analysis", required=True, help="Path to step_5_output.jsonl")
    parser.add_argument("--output", required=True, help="Output path for part6_complete_metadata.jsonl")

    args = parser.parse_args()

    print("Loading input files...", file=sys.stderr)
    phase1_kg = load_jsonl(args.phase1_kg)
    phase1_baseline = load_jsonl(args.phase1_baseline)
    phase2a_grading = sorted(load_jsonl(args.phase2a_grading), key=lambda x: x.get("index", 0))
    phase2b_grading = sorted(load_jsonl(args.phase2b_grading), key=lambda x: x.get("index", 0))
    phase5_analysis = load_jsonl(args.phase5_analysis)

    print(f"Loaded {len(phase1_kg)} questions from phase1_kg", file=sys.stderr)
    print(f"Loaded {len(phase1_baseline)} questions from phase1_baseline", file=sys.stderr)
    print(f"Loaded {len(phase2a_grading)} questions from phase2a_grading", file=sys.stderr)
    print(f"Loaded {len(phase2b_grading)} questions from phase2b_grading", file=sys.stderr)
    print(f"Loaded {len(phase5_analysis)} questions from phase5_analysis", file=sys.stderr)

    print("Building Part 6 metadata...", file=sys.stderr)
    complete_metadata = build_part6_metadata(
        phase1_kg,
        phase1_baseline,
        phase2a_grading,
        phase2b_grading,
        phase5_analysis,
    )

    print(f"Writing {len(complete_metadata)} questions to {args.output}", file=sys.stderr)
    with open(args.output, 'w') as f:
        for metadata in complete_metadata:
            if validate_output(metadata):
                f.write(json.dumps(metadata) + '\n')
            else:
                print(f"WARNING: Skipping invalid metadata for {metadata.get('question_id')}", file=sys.stderr)

    print(f"✓ Complete", file=sys.stderr)


if __name__ == "__main__":
    main()
