"""
Run LLM-as-Judge evaluation using Claude Sonnet.

Usage:
    python scripts/run_judge.py                           # Evaluate all 4 methods
    python scripts/run_judge.py --dry-run                 # Only 5 examples each (~1 min)
    python scripts/run_judge.py --methods qlora dora      # Specific methods only
    python scripts/run_judge.py --report-only             # Just aggregate existing scores
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.judge_runner import JudgeRunner
from src.evaluation.score_aggregator import ScoreAggregator


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation (Claude Sonnet)")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        choices=["qlora", "dora", "ia3", "rag"],
        help="Methods to evaluate (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only evaluate 5 examples per method"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip evaluation, just aggregate existing scores"
    )
    parser.add_argument(
        "--rate-limit-delay", type=float, default=0.5,
        help="Seconds between API calls (default: 0.5)"
    )
    parser.add_argument(
        "--prediction-dir", type=str, default="evaluation",
        help="Directory containing prediction files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation/judge_scores",
        help="Directory to save judge scores"
    )
    args = parser.parse_args()

    if not args.report_only:
        runner = JudgeRunner(
            prediction_dir=args.prediction_dir,
            output_dir=args.output_dir,
            rate_limit_delay=args.rate_limit_delay,
        )
        runner.evaluate_all(methods=args.methods, dry_run=args.dry_run)

    # Generate report
    print("\n[REPORT] Generating aggregate report...")
    aggregator = ScoreAggregator(scores_dir=args.output_dir)
    aggregator.generate_full_report(output_path="evaluation/judge_report.json")


if __name__ == "__main__":
    main()
