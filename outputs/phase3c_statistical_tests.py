"""
Phase 3C: Statistical Analysis of LegalRisk-LLM Evaluation Results

Compares 4 methods (QLoRA, DoRA, IA³, RAG) on:
- LLM judge scores (5 dimensions + overall)
- JSON parse success rates
- Risk level prediction accuracy
- Latency profiles
- Per-clause-type performance

Statistical tests used:
- Shapiro-Wilk: Test normality of each method's scores
- Kruskal-Wallis H: Are any methods significantly different? (non-parametric ANOVA)
- Mann-Whitney U: Pairwise comparisons between methods
- Rank-biserial correlation: Effect size for pairwise comparisons
- Chi-square: Are JSON parse rates significantly different?
- Descriptive stats: Mean, std, median, quartiles

Why Kruskal-Wallis instead of ANOVA?
  - LLM judge scores (1-10 integers) are ordinal, not continuous
  - With only 107 examples per method, normality is not guaranteed
  - Non-parametric tests make no distribution assumptions
  - More appropriate for small samples with ordinal data

No API calls. Pure scipy/numpy/pandas computation.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    shapiro,
    kruskal,
    mannwhitneyu,
    chi2_contingency,
    rankdata,
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types that stdlib json can't serialize.

    Why: scipy/numpy return np.bool_, np.int64, np.float64 etc. — these are
    NOT Python builtins and json.dump() rejects them with TypeError.
    The fix: detect them and cast to native Python types before encoding.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =========================================================
# CONFIG: Update these paths to match your local files
# =========================================================
BASE_DIR = Path(__file__).parent.parent  # Project root

PREDICTIONS_DIR = BASE_DIR / "evaluation"
JUDGE_SCORES_DIR = BASE_DIR / "evaluation" / "judge_scores"
OUTPUT_DIR = BASE_DIR / "results"

METHODS = ["qlora", "dora", "ia3", "rag"]
DIMENSIONS = ["accuracy", "completeness", "legal_reasoning", "clarity", "actionability"]


# =========================================================
# DATA LOADING
# =========================================================

def load_predictions(method: str) -> list[dict]:
    """Load raw inference predictions for a method."""
    filepath = PREDICTIONS_DIR / f"{method}_predictions.jsonl"
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_judge_scores(method: str) -> list[dict]:
    """Load LLM judge scores for a method."""
    filepath = JUDGE_SCORES_DIR / f"{method}_judge_scores.jsonl"
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_data() -> tuple[dict, dict]:
    """Load all predictions and judge scores into dicts keyed by method."""
    all_preds = {}
    all_scores = {}

    for method in METHODS:
        try:
            all_preds[method] = load_predictions(method)
            print(f"  [LOAD] {method}: {len(all_preds[method])} predictions")
        except FileNotFoundError as e:
            print(f"  [WARN] Missing predictions for {method}: {e}")

        try:
            all_scores[method] = [s for s in load_judge_scores(method) if not s.get("eval_failed")]
            print(f"  [LOAD] {method}: {len(all_scores[method])} judge scores")
        except FileNotFoundError as e:
            print(f"  [WARN] Missing judge scores for {method}: {e}")

    return all_preds, all_scores


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def rank_biserial_correlation(x: list, y: list) -> float:
    """
    Effect size for Mann-Whitney U test.

    Interpretation:
      0.0 - 0.1: negligible
      0.1 - 0.3: small
      0.3 - 0.5: medium
      0.5+:      large

    Formula: r = 1 - (2U) / (n1 * n2)
    """
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    n1, n2 = len(x), len(y)
    return 1 - (2 * u_stat) / (n1 * n2)


def effect_size_label(r: float) -> str:
    """Human-readable label for effect size."""
    r = abs(r)
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


def descriptive_stats(values: list) -> dict:
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=1)), 4),
        "min": round(float(np.min(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(np.max(arr)), 4),
    }


# =========================================================
# SECTION 1: Descriptive Statistics
# =========================================================

def compute_descriptive_stats(all_scores: dict) -> dict:
    """
    Per-method, per-dimension descriptive statistics.

    This is the foundation — before running any inferential tests,
    understand the raw distributions. Are scores clustered around 6-8?
    Is one method consistently higher?
    """
    print("\n[STATS] Computing descriptive statistics...")
    result = {}

    for method in METHODS:
        if method not in all_scores:
            continue
        scores = all_scores[method]
        method_stats = {}

        # Overall scores
        overalls = [s["overall"] for s in scores]
        method_stats["overall"] = descriptive_stats(overalls)

        # Per-dimension scores
        for dim in DIMENSIONS:
            dim_scores = [s["scores"][dim] for s in scores]
            method_stats[dim] = descriptive_stats(dim_scores)

        result[method] = method_stats

    return result


# =========================================================
# SECTION 2: Normality Tests
# =========================================================

def test_normality(all_scores: dict) -> dict:
    """
    Shapiro-Wilk test for normality on overall scores per method.

    Why this matters: If scores are normally distributed, we CAN use
    ANOVA (parametric). If NOT, we must use Kruskal-Wallis (non-parametric).
    In practice, integer scores 1-10 with N=107 are rarely normal.
    """
    print("[STATS] Testing normality (Shapiro-Wilk)...")
    result = {}

    for method in METHODS:
        if method not in all_scores:
            continue
        overalls = [s["overall"] for s in all_scores[method]]

        stat, p_value = shapiro(overalls)
        is_normal = p_value > 0.05  # Fail to reject H0 = likely normal

        result[method] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "is_normal": is_normal,
            "interpretation": (
                "Normally distributed (p > 0.05)" if is_normal
                else "NOT normally distributed (p <= 0.05) — use non-parametric tests"
            ),
        }

    return result


# =========================================================
# SECTION 3: Kruskal-Wallis (Global Significance Test)
# =========================================================

def kruskal_wallis_test(all_scores: dict) -> dict:
    """
    Kruskal-Wallis H-test: Are any of the 4 methods significantly different?

    This is the non-parametric equivalent of one-way ANOVA.
    H0: All methods come from the same distribution (no difference).
    Ha: At least one method has different score distribution.

    If p < 0.05: At least one pair is different → run pairwise tests.
    If p >= 0.05: No statistically significant difference detected.
    """
    print("[STATS] Running Kruskal-Wallis H-test...")

    # Gather overall scores for each method
    method_scores = {}
    for method in METHODS:
        if method in all_scores:
            method_scores[method] = [s["overall"] for s in all_scores[method]]

    if len(method_scores) < 2:
        return {"error": "Need at least 2 methods"}

    # Run test across all methods
    stat, p_value = kruskal(*method_scores.values())

    # Eta-squared effect size for Kruskal-Wallis
    n_total = sum(len(v) for v in method_scores.values())
    k = len(method_scores)
    eta_squared = (stat - k + 1) / (n_total - k)

    # Per-dimension tests
    dim_results = {}
    for dim in DIMENSIONS:
        dim_scores_by_method = {}
        for method in METHODS:
            if method in all_scores:
                dim_scores_by_method[method] = [s["scores"][dim] for s in all_scores[method]]

        if len(dim_scores_by_method) >= 2:
            d_stat, d_p = kruskal(*dim_scores_by_method.values())
            dim_results[dim] = {
                "statistic": round(float(d_stat), 4),
                "p_value": round(float(d_p), 6),
                "significant": d_p < 0.05,
            }

    return {
        "overall": {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < 0.05,
            "eta_squared": round(float(eta_squared), 4),
            "interpretation": (
                "Significant difference exists between methods (p < 0.05)"
                if p_value < 0.05
                else "No significant difference detected (p >= 0.05)"
            ),
        },
        "per_dimension": dim_results,
    }


# =========================================================
# SECTION 4: Pairwise Mann-Whitney U Tests
# =========================================================

def pairwise_mann_whitney(all_scores: dict) -> dict:
    """
    Pairwise Mann-Whitney U tests with Bonferroni correction.

    After Kruskal-Wallis tells us "some methods differ," Mann-Whitney
    tells us WHICH pairs are different.

    Bonferroni correction: With 6 pairs (C(4,2)=6), multiply p-values by 6
    to control family-wise error rate. Adjusted threshold: 0.05/6 ≈ 0.0083.

    Also computes rank-biserial correlation as effect size.
    """
    print("[STATS] Running pairwise Mann-Whitney U tests...")

    results = {}
    method_scores = {}
    for method in METHODS:
        if method in all_scores:
            method_scores[method] = [s["overall"] for s in all_scores[method]]

    available_methods = list(method_scores.keys())
    n_pairs = len(available_methods) * (len(available_methods) - 1) // 2

    for i, m1 in enumerate(available_methods):
        for m2 in available_methods[i + 1:]:
            x = method_scores[m1]
            y = method_scores[m2]

            u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
            effect_r = rank_biserial_correlation(x, y)
            p_adjusted = min(p_value * n_pairs, 1.0)  # Bonferroni

            pair_key = f"{m1}_vs_{m2}"
            results[pair_key] = {
                "u_statistic": round(float(u_stat), 2),
                "p_value_raw": round(float(p_value), 6),
                "p_value_bonferroni": round(float(p_adjusted), 6),
                "significant_raw": p_value < 0.05,
                "significant_bonferroni": p_adjusted < 0.05,
                "effect_size_r": round(float(effect_r), 4),
                "effect_size_label": effect_size_label(effect_r),
                "winner": m1 if np.mean(x) > np.mean(y) else m2,
                "mean_diff": round(float(np.mean(x) - np.mean(y)), 4),
            }

    return results


# =========================================================
# SECTION 5: JSON Parse Success Rate Analysis
# =========================================================

def json_parse_analysis(all_preds: dict) -> dict:
    """
    Chi-square test: Are JSON parse success rates significantly different?

    Each method either succeeds or fails at producing valid JSON.
    This is a 2×4 contingency table:
      Methods: qlora, dora, ia3, rag
      Outcomes: success, failure

    H0: Parse success rate is the same across all methods.
    """
    print("[STATS] Analyzing JSON parse success rates...")

    rates = {}
    contingency = []

    for method in METHODS:
        if method not in all_preds:
            continue
        preds = all_preds[method]
        successes = sum(1 for p in preds if p.get("json_parse_success"))
        failures = len(preds) - successes

        rates[method] = {
            "n_total": len(preds),
            "n_success": successes,
            "n_failure": failures,
            "success_rate": round(successes / len(preds), 4) if preds else 0,
        }
        contingency.append([successes, failures])

    # Chi-square test (only if we have at least 2 methods)
    chi2_result = {}
    if len(contingency) >= 2:
        try:
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
            chi2_result = {
                "statistic": round(float(chi2_stat), 4),
                "p_value": round(float(p_value), 6),
                "dof": int(dof),
                "significant": p_value < 0.05,
                "interpretation": (
                    "Parse success rates are significantly different (p < 0.05)"
                    if p_value < 0.05
                    else "No significant difference in parse success rates (p >= 0.05)"
                ),
            }
        except Exception as e:
            chi2_result = {"error": str(e)}

    return {
        "per_method": rates,
        "chi_square_test": chi2_result,
    }


# =========================================================
# SECTION 6: Risk Level Accuracy
# =========================================================

def risk_level_accuracy(all_scores: dict) -> dict:
    """
    How often does each method predict the correct risk level?

    Risk levels: Low, Medium, High, Critical
    We treat this as a classification problem and compute:
    - Exact match accuracy
    - Off-by-one accuracy (one level away = partial credit)
    - Confusion breakdown
    """
    print("[STATS] Computing risk level accuracy...")

    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    results = {}

    for method in METHODS:
        if method not in all_scores:
            continue
        scores = all_scores[method]

        exact_matches = 0
        off_by_one = 0
        off_by_two = 0
        confusion = defaultdict(int)
        n = 0

        for s in scores:
            gt = s.get("ground_truth_risk_level", "").lower()
            pred = s.get("predicted_risk_level", "").lower()

            if not gt or not pred or pred == "failed":
                continue

            n += 1
            confusion[f"{gt}→{pred}"] += 1

            gt_rank = risk_order.get(gt, -1)
            pred_rank = risk_order.get(pred, -1)

            if gt_rank == -1 or pred_rank == -1:
                continue

            diff = abs(gt_rank - pred_rank)
            if diff == 0:
                exact_matches += 1
                off_by_one += 1  # 0-away is also ≤1 away
                off_by_two += 1
            elif diff == 1:
                off_by_one += 1
                off_by_two += 1
            elif diff == 2:
                off_by_two += 1

        results[method] = {
            "n_evaluated": n,
            "exact_match_rate": round(exact_matches / n, 4) if n > 0 else 0,
            "off_by_one_rate": round(off_by_one / n, 4) if n > 0 else 0,
            "off_by_two_rate": round(off_by_two / n, 4) if n > 0 else 0,
            "top_confusions": dict(sorted(confusion.items(), key=lambda x: -x[1])[:8]),
        }

    return results


# =========================================================
# SECTION 7: Latency Analysis
# =========================================================

def latency_analysis(all_preds: dict) -> dict:
    """
    Compare inference latency across methods.

    RAG has retrieval overhead. Fine-tuned models run fast.
    This quantifies the speed-quality tradeoff.
    """
    print("[STATS] Analyzing inference latency...")

    results = {}

    for method in METHODS:
        if method not in all_preds:
            continue
        latencies = [p.get("latency_ms", 0) / 1000.0 for p in all_preds[method]]  # → seconds
        latencies = [l for l in latencies if l > 0]  # Remove zeros

        if latencies:
            results[method] = {
                **descriptive_stats(latencies),
                "total_seconds": round(float(sum(latencies)), 1),
                "unit": "seconds",
            }

    # Kruskal-Wallis on latencies
    available = {m: [p.get("latency_ms", 0) for p in all_preds[m] if p.get("latency_ms")]
                 for m in METHODS if m in all_preds}
    if len(available) >= 2:
        stat, p_val = kruskal(*available.values())
        results["kruskal_wallis"] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": p_val < 0.05,
        }

    return results


# =========================================================
# SECTION 8: Per-Category Breakdown
# =========================================================

def per_category_stats(all_scores: dict) -> dict:
    """
    Performance breakdown by clause type (termination, liability, ip, etc.)

    Legal risk assessment is NOT uniform. A model might excel at
    termination clauses but struggle with IP clauses. This reveals
    whether one method has domain-specific strengths.
    """
    print("[STATS] Computing per-category statistics...")

    results = {}

    for method in METHODS:
        if method not in all_scores:
            continue
        scores = all_scores[method]
        by_cat = defaultdict(list)

        for s in scores:
            cat = s.get("clause_type", "unknown")
            by_cat[cat].append(s["overall"])

        cat_stats = {}
        for cat, cat_overalls in sorted(by_cat.items()):
            cat_stats[cat] = descriptive_stats(cat_overalls)

        results[method] = cat_stats

    # Cross-method comparison per category
    all_cats = set()
    for method in all_scores:
        scores = all_scores[method]
        all_cats.update(s.get("clause_type") for s in scores)

    cross_category = {}
    for cat in sorted(all_cats):
        cat_by_method = {}
        for method in METHODS:
            if method not in all_scores:
                continue
            cat_scores = [s["overall"] for s in all_scores[method] if s.get("clause_type") == cat]
            if cat_scores:
                cat_by_method[method] = {
                    "mean": round(float(np.mean(cat_scores)), 4),
                    "n": len(cat_scores),
                }
        cross_category[cat] = cat_by_method

    return {"per_method": results, "cross_method": cross_category}


# =========================================================
# SECTION 9: Rankings and Summary Table
# =========================================================

def build_rankings(all_scores: dict, desc_stats: dict) -> dict:
    """
    Rank all 4 methods across all dimensions.
    Rank 1 = best, 4 = worst.

    This is the "scoreboard" — the single view that tells you
    which method won Phase 3.
    """
    print("[STATS] Building rankings...")

    rankings = {}
    all_dims = ["overall"] + DIMENSIONS

    for dim in all_dims:
        method_means = {}
        for method in METHODS:
            if method in desc_stats:
                method_means[method] = desc_stats[method][dim]["mean"]

        # Rank: higher mean = better rank
        sorted_methods = sorted(method_means.items(), key=lambda x: -x[1])
        rankings[dim] = {
            method: {"rank": i + 1, "mean": mean}
            for i, (method, mean) in enumerate(sorted_methods)
        }

    # Count how many times each method ranked 1st
    first_place_counts = defaultdict(int)
    for dim in all_dims:
        for method, info in rankings[dim].items():
            if info["rank"] == 1:
                first_place_counts[method] += 1

    return {
        "by_dimension": rankings,
        "first_place_counts": dict(first_place_counts),
        "overall_winner": max(first_place_counts.items(), key=lambda x: x[1])[0] if first_place_counts else "unknown",
    }


# =========================================================
# MAIN: Run All Tests
# =========================================================

def main():
    print("=" * 70)
    print("  Phase 3C: Statistical Analysis")
    print("  LegalRisk-LLM — 4-Method Comparison")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("\n[LOAD] Loading predictions and judge scores...")
    all_preds, all_scores = load_all_data()

    available_methods = [m for m in METHODS if m in all_scores]
    total_scored = sum(len(all_scores[m]) for m in available_methods)
    print(f"\n[LOAD] Total scored examples: {total_scored}")
    print(f"[LOAD] Available methods: {available_methods}")

    # Run all analyses
    results = {
        "metadata": {
            "phase": "3C",
            "description": "Statistical comparison of 4 LLM training methods",
            "methods": available_methods,
            "n_examples_per_method": {m: len(all_scores[m]) for m in available_methods},
            "n_predictions_per_method": {m: len(all_preds.get(m, [])) for m in METHODS},
        }
    }

    results["descriptive_stats"] = compute_descriptive_stats(all_scores)
    results["normality_tests"] = test_normality(all_scores)
    results["kruskal_wallis"] = kruskal_wallis_test(all_scores)
    results["pairwise_mann_whitney"] = pairwise_mann_whitney(all_scores)
    results["json_parse_analysis"] = json_parse_analysis(all_preds)
    results["risk_level_accuracy"] = risk_level_accuracy(all_scores)
    results["latency_analysis"] = latency_analysis(all_preds)
    results["per_category_stats"] = per_category_stats(all_scores)
    results["rankings"] = build_rankings(all_scores, results["descriptive_stats"])

    # Save to results/
    output_path = OUTPUT_DIR / "phase3c_statistical_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # Print human-readable summary
    print_summary(results)

    print(f"\n[DONE] Full results saved to: {output_path}")
    return results


def print_summary(results: dict):
    """Print a human-readable summary of key findings."""
    print("\n" + "=" * 70)
    print("  STATISTICAL SUMMARY")
    print("=" * 70)

    # Rankings table
    print("\n  OVERALL SCORE RANKINGS:")
    rankings = results["rankings"]["by_dimension"].get("overall", {})
    for method, info in sorted(rankings.items(), key=lambda x: x[1]["rank"]):
        bar = "#" * int(info["mean"])
        print(f"    #{info['rank']} {method:8s}  {info['mean']:.3f}/10  {bar}")

    # Kruskal-Wallis
    kw = results["kruskal_wallis"]["overall"]
    print(f"\n  KRUSKAL-WALLIS (Global Significance):")
    print(f"    H={kw['statistic']:.3f}, p={kw['p_value']:.4f}")
    print(f"    → {kw['interpretation']}")

    # Pairwise
    print(f"\n  PAIRWISE COMPARISONS (Bonferroni-corrected):")
    for pair, info in results["pairwise_mann_whitney"].items():
        sig = "* SIGNIFICANT" if info["significant_bonferroni"] else ""
        print(f"    {pair:25s}  Δ={info['mean_diff']:+.3f}  "
              f"p={info['p_value_bonferroni']:.4f}  "
              f"effect={info['effect_size_label']}  {sig}")

    # JSON Parse rates
    print(f"\n  JSON PARSE SUCCESS RATES:")
    for method, info in results["json_parse_analysis"]["per_method"].items():
        bar = "#" * int(info["success_rate"] * 20)
        print(f"    {method:8s}  {info['success_rate']*100:.1f}%  {bar}")

    # Risk level accuracy
    print(f"\n  RISK LEVEL ACCURACY (exact match):")
    for method, info in results["risk_level_accuracy"].items():
        print(f"    {method:8s}  {info['exact_match_rate']*100:.1f}%  "
              f"(±1 step: {info['off_by_one_rate']*100:.1f}%)")

    # Per-dimension
    print(f"\n  PER-DIMENSION MEANS:")
    desc = results["descriptive_stats"]
    header = f"  {'Dimension':22s}"
    for m in results["metadata"]["methods"]:
        header += f" {m:>7s}"
    print(header)
    print("  " + "-" * (22 + 8 * len(results["metadata"]["methods"])))

    all_dims = ["overall"] + ["accuracy", "completeness", "legal_reasoning", "clarity", "actionability"]
    for dim in all_dims:
        row = f"  {dim:22s}"
        for m in results["metadata"]["methods"]:
            val = desc.get(m, {}).get(dim, {}).get("mean", 0)
            row += f" {val:>7.3f}"
        print(row)

    # Winner
    winner = results["rankings"]["overall_winner"]
    print(f"\n  OVERALL WINNER: {winner.upper()} "
          f"(ranked #1 in {results['rankings']['first_place_counts'].get(winner, 0)}/6 dimensions)")

    print("=" * 70)


if __name__ == "__main__":
    main()
