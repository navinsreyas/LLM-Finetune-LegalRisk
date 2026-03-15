"""
Orchestrates LLM-as-Judge evaluation across all methods.

Handles loading predictions, calling the judge, rate limiting,
progress tracking, crash recovery (resume), and saving results.
"""

import json
import time
from pathlib import Path
from tqdm import tqdm

from src.evaluation.llm_judge import LLMJudge


class JudgeRunner:
    def __init__(
        self,
        prediction_dir: str = "evaluation",
        output_dir: str = "evaluation/judge_scores",
        rate_limit_delay: float = 0.5,
    ):
        """
        Args:
            prediction_dir: Directory containing *_predictions.jsonl files
            output_dir: Directory to save judge scores
            rate_limit_delay: Seconds between API calls.
                              Anthropic rate limits are generous for Sonnet.
                              0.5s is safe. Increase to 1.0 if you hit limits.
        """
        self.prediction_dir = Path(prediction_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.judge = LLMJudge()

    def load_predictions(self, method: str) -> list[dict]:
        """Load prediction file for a given method."""
        filepath = self.prediction_dir / f"{method}_predictions.jsonl"
        if not filepath.exists():
            raise FileNotFoundError(f"Prediction file not found: {filepath}")

        predictions = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                predictions.append(json.loads(line.strip()))

        return predictions

    def evaluate_method(
        self,
        method: str,
        dry_run: bool = False,
        resume: bool = True,
    ) -> list[dict]:
        """
        Run LLM judge on all predictions for a single method.

        Args:
            method: One of 'qlora', 'dora', 'ia3', 'rag'
            dry_run: If True, only evaluate first 5 predictions
            resume: If True, skip already-scored predictions (crash recovery)

        Returns:
            List of judge results (one per prediction)
        """
        predictions = self.load_predictions(method)

        if dry_run:
            predictions = predictions[:5]
            print(f"[DRY RUN] Evaluating only 5 predictions for {method}")

        # Resume support — load existing scores
        output_path = self.output_dir / f"{method}_judge_scores.jsonl"
        existing_results = []
        scored_indices = set()

        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        result = json.loads(line)
                        existing_results.append(result)
                        scored_indices.add(result["test_index"])
            print(f"[RESUME] Found {len(existing_results)} existing scores for {method}")

        results = list(existing_results)

        # Find remaining predictions
        remaining = [p for p in predictions if p["test_index"] not in scored_indices]
        if not remaining:
            print(f"[JUDGE] All {len(predictions)} predictions already scored for {method}")
            return results

        print(f"\n[JUDGE] Evaluating {method.upper()}: {len(remaining)} remaining "
              f"(of {len(predictions)} total)")

        # Estimate time
        est_minutes = len(remaining) * (self.rate_limit_delay + 1.5) / 60
        print(f"[JUDGE] Estimated time: ~{est_minutes:.0f} minutes")

        eval_failed_count = 0

        for pred in tqdm(remaining, desc=f"{method} judge"):
            test_index = pred["test_index"]
            clause_text = pred["input"]["clause_text"]
            clause_type = pred["input"]["clause_type"]
            ground_truth = pred["ground_truth"]
            prediction = pred["prediction"]

            # Call the judge
            judge_result = self.judge.evaluate_single(
                clause_text=clause_text,
                clause_type=clause_type,
                ground_truth=ground_truth,
                prediction=prediction,
            )

            # Package full result
            full_result = {
                "test_index": test_index,
                "method": method,
                "clause_type": clause_type,
                "ground_truth_risk_level": ground_truth.get("risk_level", "unknown"),
                "predicted_risk_level": (
                    prediction.get("risk_level", "unknown") if prediction else "FAILED"
                ),
                "json_parse_success": pred["json_parse_success"],
                "scores": judge_result["scores"],
                "overall": judge_result["overall"],
                "justification": judge_result.get("justification", ""),
                "eval_failed": judge_result.get("eval_failed", False),
                "latency_ms": pred.get("latency_ms", 0),
            }

            results.append(full_result)

            if judge_result.get("eval_failed"):
                eval_failed_count += 1

            # Write to file immediately (crash resilience)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(full_result, ensure_ascii=False) + "\n")

            # Rate limiting
            time.sleep(self.rate_limit_delay)

        # Summary
        scores = [r["overall"] for r in results if not r.get("eval_failed")]
        avg_score = sum(scores) / len(scores) if scores else 0

        print(f"\n[JUDGE] {method.upper()} evaluation complete:")
        print(f"  Total evaluated:  {len(results)}")
        print(f"  Eval failures:    {eval_failed_count}")
        print(f"  Average overall:  {avg_score:.2f}/10")

        return results

    def evaluate_all(
        self,
        methods: list[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Run LLM judge on all methods.

        Args:
            methods: List of methods to evaluate. Default: all 4
            dry_run: If True, only evaluate 5 examples per method

        Returns:
            dict mapping method name to list of results
        """
        if methods is None:
            methods = ["qlora", "dora", "ia3", "rag"]

        n_examples = 5 if dry_run else 107
        total_calls = len(methods) * n_examples
        est_cost = total_calls * 0.0075

        print("=" * 70)
        print("  LegalRisk-LLM Phase 3B: LLM-as-Judge Evaluation")
        print("  Judge Model: Claude Sonnet (claude-sonnet-4-20250514)")
        if dry_run:
            print("  [DRY RUN MODE: 5 examples per method]")
        print("=" * 70)
        print(f"  Methods:          {', '.join(methods)}")
        print(f"  Total calls:      ~{total_calls}")
        print(f"  Estimated cost:   ~${est_cost:.2f}")
        print("=" * 70)

        all_results = {}

        for method in methods:
            try:
                results = self.evaluate_method(method, dry_run=dry_run)
                all_results[method] = results
            except FileNotFoundError as e:
                print(f"[WARNING] Skipping {method}: {e}")
                continue

        # Print cost
        cost = self.judge.get_cost_estimate()
        print(f"\n{'='*70}")
        print(f"  Evaluation Complete!")
        print(f"  API Cost: ${cost['total_cost_usd']:.4f}")
        print(f"  Tokens: {cost['input_tokens']:,} in + {cost['output_tokens']:,} out")
        print(f"{'='*70}")

        return all_results
