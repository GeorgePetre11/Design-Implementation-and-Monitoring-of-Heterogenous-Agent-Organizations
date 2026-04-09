#!/usr/bin/env python3
"""
CLI tool for evaluating consulting reports without running the server.

Usage:
  python evaluate_cli.py --question "Should we expand?" --report path/to/report.md --level 3
  python evaluate_cli.py --question "Should we expand?" --report-text "# Report\n..." --level 1

Results are saved to evaluator/results/<id>.json
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a consulting report using DeepSeek-R1:70b"
    )
    parser.add_argument(
        "--question", "-q", required=True,
        help="The original client business question.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--report", "-r",
        help="Path to a file containing the consulting report.",
    )
    group.add_argument(
        "--report-text",
        help="The report text passed directly as a string.",
    )
    parser.add_argument(
        "--level", "-l", type=int, required=True, choices=[1, 2, 3, 4],
        help="Which complexity level produced this report.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save the result JSON (default: results/<id>.json).",
    )

    args = parser.parse_args()

    # Load report
    if args.report:
        report_path = Path(args.report)
        if not report_path.exists():
            print(f"Error: file not found: {args.report}", file=sys.stderr)
            sys.exit(1)
        report = report_path.read_text(encoding="utf-8")
    else:
        report = args.report_text

    if not report.strip():
        print("Error: report is empty.", file=sys.stderr)
        sys.exit(1)

    # Run evaluation
    evaluation_id = str(uuid.uuid4())
    agent = Evaluator()

    print(f"Evaluator model: {agent.model}")
    print(f"Level: {args.level}")
    print(f"Question: {args.question}")
    print(f"Report length: {len(report)} chars")
    print("-" * 50)
    print("Running evaluation (this may take a while)...")
    print()

    t0 = time.time()
    scorecard = agent.run(args.question, report, args.level)
    elapsed = round(time.time() - t0, 2)

    result = {
        "evaluation_id": evaluation_id,
        "level": args.level,
        "model": agent.model,
        "question": args.question,
        "elapsed_seconds": elapsed,
        "scorecard": scorecard,
    }

    # Save result
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = Path(args.output) if args.output else results_dir / f"{evaluation_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print scorecard
    print("=" * 50)
    print("EVALUATION SCORECARD")
    print("=" * 50)
    print()

    criteria = [
        "completeness", "accuracy", "coherence",
        "structure", "actionability", "critical_depth",
    ]
    for c in criteria:
        entry = scorecard[c]
        label = c.replace("_", " ").upper()
        print(f"  {label:<18} {entry['score']}/10")
        print(f"    {entry['justification']}")
        print()

    print(f"  {'OVERALL':<18} {scorecard['overall_score']:.2f}/10")
    print()
    print(f"Summary: {scorecard['summary']}")
    print()
    print("Strengths:")
    for s in scorecard["strengths"]:
        print(f"  + {s}")
    print()
    print("Weaknesses:")
    for w in scorecard["weaknesses"]:
        print(f"  - {w}")
    print()
    print(f"Time: {elapsed}s | Saved to: {output_path}")


if __name__ == "__main__":
    main()
