"""
Aggregates and analyzes LLM judge scores across all methods.

Produces summary statistics, comparisons, and detailed breakdowns
for statistical analysis.
"""

import json
from pathlib import Path
from collections import defaultdict


class ScoreAggregator:
    def __init__(self, scores_dir: str = "evaluation/judge_scores"):
        self.scores_dir = Path(scores_dir)
        self.methods = ["qlora", "dora", "ia3", "rag"]
        self.dimensions = [
            "accuracy", "completeness", "legal_reasoning", "clarity", "actionability"
        ]
        self.all_scores = {}

        for method in self.methods:
            filepath = self.scores_dir / f"{method}_judge_scores.jsonl"
            if filepath.exists():
                scores = []
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            scores.append(json.loads(line))
                self.all_scores[method] = scores
                print(f"[AGG] Loaded {len(scores)} scores for {method}")
            else:
                print(f"[AGG] No scores found for {method}")

    def method_summary(self) -> dict:
        """Overall scores per method."""
        summary = {}
        for method, scores in self.all_scores.items():
            valid = [s for s in scores if not s.get("eval_failed")]
            if not valid:
                continue

            stats = {}
            for dim in self.dimensions:
                vals = [s["scores"][dim] for s in valid]
                stats[dim] = round(sum(vals) / len(vals), 2)

            overalls = [s["overall"] for s in valid]
            stats["overall"] = round(sum(overalls) / len(overalls), 2)
            stats["n_evaluated"] = len(valid)
            stats["n_failed"] = len(scores) - len(valid)

            summary[method] = stats

        return summary

    def category_breakdown(self) -> dict:
        """Scores broken down by clause type per method."""
        breakdown = {}
        for method, scores in self.all_scores.items():
            valid = [s for s in scores if not s.get("eval_failed")]
            by_cat = defaultdict(list)
            for s in valid:
                by_cat[s["clause_type"]].append(s)

            method_cats = {}
            for cat, cat_scores in by_cat.items():
                cat_stats = {}
                for dim in self.dimensions:
                    vals = [s["scores"][dim] for s in cat_scores]
                    cat_stats[dim] = round(sum(vals) / len(vals), 2)
                overalls = [s["overall"] for s in cat_scores]
                cat_stats["overall"] = round(sum(overalls) / len(overalls), 2)
                cat_stats["n"] = len(cat_scores)
                method_cats[cat] = cat_stats

            breakdown[method] = method_cats

        return breakdown

    def dimension_comparison(self) -> dict:
        """For each dimension, compare all methods. For radar charts."""
        summary = self.method_summary()
        comparison = {}
        for dim in self.dimensions + ["overall"]:
            comparison[dim] = {}
            for method in self.methods:
                if method in summary:
                    comparison[dim][method] = summary[method].get(dim, 0)
        return comparison

    def head_to_head(self) -> dict:
        """Pairwise win/loss/tie counts between all methods."""
        results = {}
        method_list = [m for m in self.methods if m in self.all_scores]

        for i, m1 in enumerate(method_list):
            for m2 in method_list[i + 1:]:
                s1_map = {
                    s["test_index"]: s
                    for s in self.all_scores[m1]
                    if not s.get("eval_failed")
                }
                s2_map = {
                    s["test_index"]: s
                    for s in self.all_scores[m2]
                    if not s.get("eval_failed")
                }

                common = set(s1_map.keys()) & set(s2_map.keys())
                m1_wins, m2_wins, ties = 0, 0, 0

                for idx in common:
                    diff = s1_map[idx]["overall"] - s2_map[idx]["overall"]
                    if abs(diff) < 0.01:
                        ties += 1
                    elif diff > 0:
                        m1_wins += 1
                    else:
                        m2_wins += 1

                results[f"{m1}_vs_{m2}"] = {
                    f"{m1}_wins": m1_wins,
                    f"{m2}_wins": m2_wins,
                    "ties": ties,
                    "total": len(common),
                }

        return results

    def worst_examples(self, method: str, n: int = 5) -> list[dict]:
        """Find N lowest-scoring examples for failure analysis."""
        if method not in self.all_scores:
            return []
        valid = [s for s in self.all_scores[method] if not s.get("eval_failed")]
        return sorted(valid, key=lambda x: x["overall"])[:n]

    def best_examples(self, method: str, n: int = 5) -> list[dict]:
        """Find N highest-scoring examples."""
        if method not in self.all_scores:
            return []
        valid = [s for s in self.all_scores[method] if not s.get("eval_failed")]
        return sorted(valid, key=lambda x: x["overall"], reverse=True)[:n]

    def generate_full_report(self, output_path: str = "evaluation/judge_report.json"):
        """Generate comprehensive JSON report with all analyses."""
        report = {
            "judge_model": "claude-sonnet-4-20250514",
            "method_summary": self.method_summary(),
            "category_breakdown": self.category_breakdown(),
            "dimension_comparison": self.dimension_comparison(),
            "head_to_head": self.head_to_head(),
        }

        for method in self.methods:
            if method in self.all_scores:
                report[f"{method}_worst_5"] = self.worst_examples(method, 5)
                report[f"{method}_best_5"] = self.best_examples(method, 5)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n[REPORT] Full judge report saved to {output_path}")

        # Print summary table
        summary = report["method_summary"]
        print(f"\n{'='*70}")
        print(f"  LLM JUDGE RESULTS (Claude Sonnet)")
        print(f"{'='*70}")

        dims = self.dimensions + ["overall"]
        header = f"{'Dimension':20s}"
        for m in self.methods:
            if m in summary:
                header += f" {m:>8s}"
        print(header)
        print("-" * (20 + 9 * len(summary)))

        for dim in dims:
            row = f"{dim:20s}"
            for m in self.methods:
                if m in summary:
                    val = summary[m].get(dim, 0)
                    row += f" {val:>8.2f}"
            print(row)

        print(f"\n  Head-to-Head Win Rates:")
        for matchup, result in report["head_to_head"].items():
            methods_pair = matchup.split("_vs_")
            print(f"    {matchup}: "
                  f"{methods_pair[0]} wins {result[f'{methods_pair[0]}_wins']}, "
                  f"{methods_pair[1]} wins {result[f'{methods_pair[1]}_wins']}, "
                  f"ties {result['ties']}")

        print(f"{'='*70}")

        return report
