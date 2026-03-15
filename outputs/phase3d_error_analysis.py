"""
Phase 3D: Error Analysis of LegalRisk-LLM Evaluation Results

Deep-dive into failure modes across all 4 methods (QLoRA, DoRA, IA3, RAG):

1. Risk level confusion matrices — what gets misclassified as what?
2. Worst examples per method — the 10 lowest-scoring predictions with context
3. Dimension failure patterns — which scoring dimensions pull scores down?
4. Per-category failure rates — which clause types are hardest to handle?
5. JSON parse failure analysis — RAG had ~6% failures; what went wrong?
6. Cross-method failure overlap — do all methods fail on the same examples?
7. Risk level bias analysis — do methods over/under-predict severity?
8. Score correlations between methods — how aligned are their errors?

No API calls. Pure computation on cached predictions and judge scores.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that stdlib json can't serialize."""
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
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).parent.parent  # Project root

PREDICTIONS_DIR = BASE_DIR / "evaluation"
JUDGE_SCORES_DIR = BASE_DIR / "evaluation" / "judge_scores"
OUTPUT_DIR = BASE_DIR / "results"

METHODS = ["qlora", "dora", "ia3", "rag"]
RISK_LEVELS = ["low", "medium", "high", "critical"]
DIMENSIONS = ["accuracy", "completeness", "legal_reasoning", "clarity", "actionability"]

# Risk level ordinal mapping for computing over/under-prediction
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# "Failing" threshold — scores below this warrant investigation
FAIL_THRESHOLD = 5.0


# =========================================================
# DATA LOADING
# =========================================================

def load_predictions(method: str) -> list[dict]:
    filepath = PREDICTIONS_DIR / f"{method}_predictions.jsonl"
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_judge_scores(method: str) -> list[dict]:
    filepath = JUDGE_SCORES_DIR / f"{method}_judge_scores.jsonl"
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_data() -> tuple[dict, dict]:
    all_preds = {}
    all_scores = {}

    for method in METHODS:
        try:
            all_preds[method] = load_predictions(method)
            print(f"  [LOAD] {method}: {len(all_preds[method])} predictions")
        except FileNotFoundError as e:
            print(f"  [WARN] Missing predictions: {e}")

        try:
            raw_scores = load_judge_scores(method)
            all_scores[method] = raw_scores  # Keep all, including eval_failed
            valid = [s for s in raw_scores if not s.get("eval_failed")]
            print(f"  [LOAD] {method}: {len(valid)} valid judge scores "
                  f"({len(raw_scores) - len(valid)} eval_failed)")
        except FileNotFoundError as e:
            print(f"  [WARN] Missing judge scores: {e}")

    return all_preds, all_scores


def merge_by_index(preds: list[dict], scores: list[dict]) -> list[dict]:
    """
    Join predictions + judge scores by test_index.

    Why: Judge scores have quality ratings but no clause text.
    Predictions have clause text but no quality ratings.
    We need both together to understand WHY a prediction scored low.
    """
    pred_map = {p["test_index"]: p for p in preds}
    score_map = {s["test_index"]: s for s in scores if not s.get("eval_failed")}

    merged = []
    common_indices = sorted(set(pred_map.keys()) & set(score_map.keys()))

    for idx in common_indices:
        pred = pred_map[idx]
        score = score_map[idx]

        clause_text = pred.get("input", {}).get("clause_text", "")
        justification = score.get("justification", "")

        merged.append({
            "test_index": idx,
            "clause_type": score.get("clause_type") or pred.get("input", {}).get("clause_type", "unknown"),
            "clause_text_snippet": clause_text[:200] + "..." if len(clause_text) > 200 else clause_text,
            "ground_truth_risk_level": score.get("ground_truth_risk_level", "unknown"),
            "predicted_risk_level": score.get("predicted_risk_level", "unknown"),
            "json_parse_success": pred.get("json_parse_success", False),
            "latency_ms": pred.get("latency_ms", 0),
            "overall": score.get("overall"),
            "judge_scores": score.get("scores", {}),
            "justification_snippet": justification[:200] + "..." if len(justification) > 200 else justification,
        })

    return merged


# =========================================================
# SECTION 1: Risk Level Confusion Matrices
# =========================================================

def confusion_matrix_analysis(all_merged: dict) -> dict:
    """
    4x4 confusion matrix: ground_truth_risk_level vs predicted_risk_level.

    Reveals systematic misclassification patterns:
    - Does the model always predict "Medium" even for "Critical" clauses?
    - Does RAG over-predict "High" because similar training cases were High?

    Also computes per-class precision, recall, and F1.
    """
    print("\n[ERROR] Building risk level confusion matrices...")
    results = {}

    for method, records in all_merged.items():
        # Build raw matrix
        matrix = defaultdict(lambda: defaultdict(int))
        for r in records:
            gt = r.get("ground_truth_risk_level", "unknown").lower()
            pred = r.get("predicted_risk_level", "unknown").lower()
            matrix[gt][pred] += 1

        # Per-class Precision / Recall / F1
        per_class = {}
        for level in RISK_LEVELS:
            tp = matrix[level].get(level, 0)
            fp = sum(matrix[gt].get(level, 0) for gt in RISK_LEVELS if gt != level)
            fn = sum(matrix[level].get(p, 0) for p in RISK_LEVELS if p != level)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            per_class[level] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        # Macro-averaged F1
        macro_f1 = round(
            sum(per_class[l]["f1"] for l in RISK_LEVELS) / len(RISK_LEVELS), 4
        )

        # Exact match accuracy
        total = sum(sum(row.values()) for row in matrix.values())
        correct = sum(matrix[level].get(level, 0) for level in RISK_LEVELS)
        accuracy = round(correct / total, 4) if total > 0 else 0.0

        # Convert defaultdict to regular dict for JSON
        matrix_dict = {
            gt: {pred: count for pred, count in pred_counts.items()}
            for gt, pred_counts in matrix.items()
        }

        results[method] = {
            "matrix": matrix_dict,
            "per_class_metrics": per_class,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "n_evaluated": total,
        }

    return results


# =========================================================
# SECTION 2: Worst Examples Per Method
# =========================================================

def worst_examples_analysis(all_merged: dict, n: int = 10) -> dict:
    """
    The 10 lowest-scoring examples per method with full context.

    Why: Looking at failures is more instructive than celebrating successes.
    If QLoRA consistently scores low on liability clauses, that's a training
    data gap — not a model capacity issue.

    Includes clause text snippet so you can see WHAT the model saw.
    Includes judge justification so you can see WHY it scored low.
    """
    print("[ERROR] Finding worst examples per method...")
    results = {}

    for method, records in all_merged.items():
        # Sort by overall score ascending (worst first)
        sorted_records = sorted(records, key=lambda x: x.get("overall", 10))
        worst = sorted_records[:n]

        results[method] = [
            {
                "test_index": r["test_index"],
                "clause_type": r["clause_type"],
                "clause_text_snippet": r["clause_text_snippet"],
                "ground_truth_risk_level": r["ground_truth_risk_level"],
                "predicted_risk_level": r["predicted_risk_level"],
                "json_parse_success": r["json_parse_success"],
                "overall": r["overall"],
                "dimension_scores": r["judge_scores"],
                "weakest_dimension": (
                    min(r["judge_scores"], key=r["judge_scores"].get)
                    if r["judge_scores"] else "unknown"
                ),
                "justification_snippet": r["justification_snippet"],
            }
            for r in worst
        ]

    return results


# =========================================================
# SECTION 3: Dimension Failure Patterns
# =========================================================

def dimension_failure_analysis(all_merged: dict) -> dict:
    """
    Which scoring dimensions have the most failures (score < FAIL_THRESHOLD)?

    Interpretation:
    - If "legal_reasoning" fails most often → model lacks domain knowledge
    - If "actionability" fails most often → model gives vague recommendations
    - If "accuracy" fails → model gets facts wrong (most serious for legal use)

    Also: dimension score variance (are scores consistent or all over the place?)
    """
    print("[ERROR] Analyzing dimension failure patterns...")
    results = {}

    for method, records in all_merged.items():
        dim_stats = {}

        for dim in DIMENSIONS:
            all_vals = [r["judge_scores"].get(dim) for r in records
                        if r.get("judge_scores") and r["judge_scores"].get(dim) is not None]

            if not all_vals:
                continue

            arr = np.array(all_vals, dtype=float)
            below_threshold = sum(1 for v in all_vals if v < FAIL_THRESHOLD)

            dim_stats[dim] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr, ddof=1)), 4),
                "min": round(float(np.min(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "n_below_threshold": below_threshold,
                "pct_below_threshold": round(below_threshold / len(all_vals), 4),
            }

        # Rank dimensions by mean score (lowest first = weakest dimensions)
        ranked = sorted(dim_stats.items(), key=lambda x: x[1]["mean"])
        weakest_dim = ranked[0][0] if ranked else "unknown"
        strongest_dim = ranked[-1][0] if ranked else "unknown"

        # Dimension gap: spread between best and worst dimension means
        if dim_stats:
            dim_means = [v["mean"] for v in dim_stats.values()]
            dimension_gap = round(max(dim_means) - min(dim_means), 4)
        else:
            dimension_gap = 0.0

        results[method] = {
            "per_dimension": dim_stats,
            "weakest_dimension": weakest_dim,
            "strongest_dimension": strongest_dim,
            "dimension_gap": dimension_gap,
            "dimension_ranking": [d for d, _ in ranked],
        }

    return results


# =========================================================
# SECTION 4: Per-Category Failure Rates
# =========================================================

def category_failure_analysis(all_merged: dict) -> dict:
    """
    Which clause types have the highest failure rates?

    Legal domain insight: Indemnification and IP clauses are more nuanced
    than Termination or Governing Law clauses. If a method scores low on
    Indemnification consistently, it may need more training data for that
    clause type.

    Failure rate = % of examples scoring below FAIL_THRESHOLD overall.
    """
    print("[ERROR] Analyzing per-category failure rates...")
    results = {}

    # Collect all categories across all methods
    all_categories = set()
    for records in all_merged.values():
        for r in records:
            all_categories.add(r.get("clause_type", "unknown"))

    for method, records in all_merged.items():
        by_cat = defaultdict(list)
        for r in records:
            cat = r.get("clause_type", "unknown")
            by_cat[cat].append(r)

        cat_stats = {}
        for cat in sorted(all_categories):
            cat_records = by_cat.get(cat, [])
            if not cat_records:
                continue

            overalls = [r["overall"] for r in cat_records if r.get("overall") is not None]
            if not overalls:
                continue

            below = sum(1 for v in overalls if v < FAIL_THRESHOLD)
            arr = np.array(overalls, dtype=float)

            cat_stats[cat] = {
                "n": len(overalls),
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0), 4),
                "failure_count": below,
                "failure_rate": round(below / len(overalls), 4),
            }

        # Rank by failure rate (worst first)
        sorted_cats = sorted(cat_stats.items(), key=lambda x: -x[1]["failure_rate"])
        hardest_category = sorted_cats[0][0] if sorted_cats else "unknown"
        easiest_category = sorted_cats[-1][0] if sorted_cats else "unknown"

        results[method] = {
            "per_category": cat_stats,
            "hardest_category": hardest_category,
            "easiest_category": easiest_category,
        }

    return results


# =========================================================
# SECTION 5: JSON Parse Failure Analysis
# =========================================================

def json_failure_analysis(all_preds: dict) -> dict:
    """
    Deep-dive into JSON parse failures.

    RAG had ~5.6% failure rate; fine-tuned models were near-perfect.
    This matters clinically: a failed parse = no usable output.

    We look at:
    - Which clause types fail most?
    - What does the model produce when it fails? (raw output snippet)
    - Are failures random or clustered?
    """
    print("[ERROR] Analyzing JSON parse failures...")
    results = {}

    for method, preds in all_preds.items():
        failures = [p for p in preds if not p.get("json_parse_success")]
        successes = [p for p in preds if p.get("json_parse_success")]

        # Clause type breakdown of failures
        failure_by_type = defaultdict(int)
        success_by_type = defaultdict(int)

        for p in failures:
            ct = p.get("input", {}).get("clause_type", "unknown")
            failure_by_type[ct] += 1

        for p in successes:
            ct = p.get("input", {}).get("clause_type", "unknown")
            success_by_type[ct] += 1

        # Failure rates by clause type
        all_types = set(failure_by_type) | set(success_by_type)
        type_rates = {}
        for ct in sorted(all_types):
            n_fail = failure_by_type.get(ct, 0)
            n_succ = success_by_type.get(ct, 0)
            n_total = n_fail + n_succ
            type_rates[ct] = {
                "n_total": n_total,
                "n_failed": n_fail,
                "failure_rate": round(n_fail / n_total, 4) if n_total > 0 else 0.0,
            }

        # Latency of failures vs successes (did timeouts cause failures?)
        fail_latencies = [p.get("latency_ms", 0) for p in failures if p.get("latency_ms")]
        succ_latencies = [p.get("latency_ms", 0) for p in successes if p.get("latency_ms")]

        avg_fail_latency = (sum(fail_latencies) / len(fail_latencies)
                            if fail_latencies else None)
        avg_succ_latency = (sum(succ_latencies) / len(succ_latencies)
                            if succ_latencies else None)

        results[method] = {
            "n_total": len(preds),
            "n_failed": len(failures),
            "n_succeeded": len(successes),
            "failure_rate": round(len(failures) / len(preds), 4) if preds else 0.0,
            "by_clause_type": type_rates,
            "avg_latency_ms_on_failure": round(avg_fail_latency, 1) if avg_fail_latency else None,
            "avg_latency_ms_on_success": round(avg_succ_latency, 1) if avg_succ_latency else None,
            "failed_test_indices": [p["test_index"] for p in failures],
        }

    return results


# =========================================================
# SECTION 6: Cross-Method Failure Overlap
# =========================================================

def cross_method_overlap_analysis(all_merged: dict) -> dict:
    """
    Do all 4 methods fail on the same hard examples?

    Three types of failures:
    1. Universal failures (all methods score low) — intrinsically hard clauses
    2. Method-specific failures (one method is uniquely bad) — method weakness
    3. No overlap (random failures) — noise/judge variance

    Also computes score correlation between method pairs (Spearman rho).
    Higher correlation = methods make similar mistakes.
    """
    print("[ERROR] Computing cross-method failure overlap...")

    # Build test_index → {method: overall_score} mapping
    score_by_index = defaultdict(dict)
    for method, records in all_merged.items():
        for r in records:
            idx = r["test_index"]
            if r.get("overall") is not None:
                score_by_index[idx][method] = r["overall"]

    # Only keep examples scored by ALL methods
    available_methods = [m for m in METHODS if m in all_merged]
    common_indices = [
        idx for idx, scores in score_by_index.items()
        if all(m in scores for m in available_methods)
    ]

    print(f"  [OVERLAP] {len(common_indices)} examples scored by all methods")

    # Universal failures: all methods below threshold
    universal_failures = []
    for idx in common_indices:
        scores = score_by_index[idx]
        if all(scores.get(m, 10) < FAIL_THRESHOLD for m in available_methods):
            universal_failures.append({
                "test_index": idx,
                "scores": {m: scores[m] for m in available_methods},
                "mean_score": round(sum(scores[m] for m in available_methods) / len(available_methods), 3),
            })

    # Sort by mean score ascending
    universal_failures.sort(key=lambda x: x["mean_score"])

    # Method-specific failures: only one method is below threshold
    method_specific = defaultdict(list)
    for idx in common_indices:
        scores = score_by_index[idx]
        failing_methods = [m for m in available_methods if scores.get(m, 10) < FAIL_THRESHOLD]
        if len(failing_methods) == 1:
            method_specific[failing_methods[0]].append({
                "test_index": idx,
                "scores": {m: scores[m] for m in available_methods},
            })

    # Universal wins: all methods above 8.0
    universal_wins = []
    for idx in common_indices:
        scores = score_by_index[idx]
        if all(scores.get(m, 0) >= 8.0 for m in available_methods):
            universal_wins.append({
                "test_index": idx,
                "scores": {m: scores[m] for m in available_methods},
                "mean_score": round(sum(scores[m] for m in available_methods) / len(available_methods), 3),
            })
    universal_wins.sort(key=lambda x: -x["mean_score"])

    # Spearman rank correlations between method pairs
    correlations = {}
    for i, m1 in enumerate(available_methods):
        for m2 in available_methods[i + 1:]:
            shared = [idx for idx in common_indices
                      if m1 in score_by_index[idx] and m2 in score_by_index[idx]]
            x = [score_by_index[idx][m1] for idx in shared]
            y = [score_by_index[idx][m2] for idx in shared]

            if len(x) > 3:
                rho, p_val = spearmanr(x, y)
                correlations[f"{m1}_vs_{m2}"] = {
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(p_val), 6),
                    "n": len(shared),
                    "interpretation": (
                        "Strong agreement" if abs(rho) > 0.7 else
                        "Moderate agreement" if abs(rho) > 0.4 else
                        "Weak agreement"
                    ),
                }

    # Score variance per example (high variance = methods disagree)
    high_disagreement = []
    for idx in common_indices:
        scores = [score_by_index[idx].get(m, 0) for m in available_methods]
        variance = float(np.var(scores))
        if variance > 2.0:  # Disagreement threshold
            high_disagreement.append({
                "test_index": idx,
                "scores": {m: score_by_index[idx].get(m) for m in available_methods},
                "variance": round(variance, 3),
                "range": round(max(scores) - min(scores), 3),
            })
    high_disagreement.sort(key=lambda x: -x["variance"])

    return {
        "n_examples_all_methods": len(common_indices),
        "universal_failures": {
            "count": len(universal_failures),
            "threshold": FAIL_THRESHOLD,
            "examples": universal_failures[:10],
        },
        "method_specific_failures": {
            method: {
                "count": len(cases),
                "examples": cases[:5],
            }
            for method, cases in method_specific.items()
        },
        "universal_wins": {
            "count": len(universal_wins),
            "threshold": 8.0,
            "examples": universal_wins[:5],
        },
        "score_correlations": correlations,
        "high_disagreement_examples": {
            "count": len(high_disagreement),
            "variance_threshold": 2.0,
            "top_10": high_disagreement[:10],
        },
    }


# =========================================================
# SECTION 7: Risk Level Bias Analysis
# =========================================================

def risk_bias_analysis(all_merged: dict) -> dict:
    """
    Do methods systematically over- or under-predict risk severity?

    Over-prediction: Model says "High" when truth is "Medium" — creates
    unnecessary alarm, lawyer panic.

    Under-prediction: Model says "Low" when truth is "Critical" — misses
    serious risks, potential liability.

    Under-prediction is more dangerous in legal contexts.
    """
    print("[ERROR] Computing risk level bias analysis...")
    results = {}

    for method, records in all_merged.items():
        direction_counts = {"over_predicted": 0, "under_predicted": 0, "correct": 0, "unknown": 0}
        direction_amounts = []  # Positive = over-predicted, negative = under-predicted

        gt_distribution = defaultdict(int)
        pred_distribution = defaultdict(int)

        for r in records:
            gt = r.get("ground_truth_risk_level", "unknown").lower()
            pred = r.get("predicted_risk_level", "unknown").lower()

            gt_distribution[gt] += 1
            pred_distribution[pred] += 1

            if gt not in RISK_ORDER or pred not in RISK_ORDER:
                direction_counts["unknown"] += 1
                continue

            diff = RISK_ORDER[pred] - RISK_ORDER[gt]  # Positive = over, negative = under
            direction_amounts.append(diff)

            if diff > 0:
                direction_counts["over_predicted"] += 1
            elif diff < 0:
                direction_counts["under_predicted"] += 1
            else:
                direction_counts["correct"] += 1

        n_valid = len(direction_amounts)
        mean_bias = round(float(np.mean(direction_amounts)), 4) if direction_amounts else 0.0

        results[method] = {
            "n_evaluated": len(records),
            "n_valid_comparisons": n_valid,
            "direction_counts": dict(direction_counts),
            "over_prediction_rate": round(direction_counts["over_predicted"] / n_valid, 4) if n_valid else 0,
            "under_prediction_rate": round(direction_counts["under_predicted"] / n_valid, 4) if n_valid else 0,
            "exact_match_rate": round(direction_counts["correct"] / n_valid, 4) if n_valid else 0,
            "mean_bias": mean_bias,
            "bias_interpretation": (
                "Tends to over-predict risk (conservative)" if mean_bias > 0.1 else
                "Tends to under-predict risk (optimistic)" if mean_bias < -0.1 else
                "Well-calibrated (minimal systematic bias)"
            ),
            "ground_truth_distribution": dict(sorted(gt_distribution.items())),
            "predicted_distribution": dict(sorted(pred_distribution.items())),
        }

    return results


# =========================================================
# SECTION 8: Critical Failure Analysis (The Worst of the Worst)
# =========================================================

def critical_failure_analysis(all_merged: dict, all_preds: dict) -> dict:
    """
    Identify the most severe failure modes across methods:

    1. Missed Critical risks (GT=Critical, pred=Low/Medium) — most dangerous
    2. False Critical alarms (GT=Low, pred=Critical) — most alarming false positives
    3. Cases where BOTH risk level is wrong AND judge score is low — double failure

    These are the cases that matter most for legal risk assessment deployment.
    """
    print("[ERROR] Identifying critical failures...")
    results = {}

    for method, records in all_merged.items():
        # Missed critical risks: GT=Critical but predicted as Low or Medium
        missed_critical = [
            r for r in records
            if r.get("ground_truth_risk_level", "").lower() == "critical"
            and r.get("predicted_risk_level", "").lower() in ("low", "medium")
        ]

        # False alarms: GT=Low but predicted as Critical
        false_critical = [
            r for r in records
            if r.get("ground_truth_risk_level", "").lower() == "low"
            and r.get("predicted_risk_level", "").lower() == "critical"
        ]

        # Double failures: both risk level wrong AND low overall score
        double_failures = [
            r for r in records
            if r.get("ground_truth_risk_level", "").lower() != r.get("predicted_risk_level", "").lower()
            and r.get("overall", 10) < FAIL_THRESHOLD
        ]
        double_failures.sort(key=lambda x: x.get("overall", 10))

        def summarize_records(record_list, limit=5):
            return [
                {
                    "test_index": r["test_index"],
                    "clause_type": r["clause_type"],
                    "ground_truth_risk_level": r["ground_truth_risk_level"],
                    "predicted_risk_level": r["predicted_risk_level"],
                    "overall": r["overall"],
                    "clause_text_snippet": r["clause_text_snippet"],
                }
                for r in record_list[:limit]
            ]

        results[method] = {
            "missed_critical": {
                "count": len(missed_critical),
                "examples": summarize_records(missed_critical),
            },
            "false_critical_alarms": {
                "count": len(false_critical),
                "examples": summarize_records(false_critical),
            },
            "double_failures": {
                "count": len(double_failures),
                "examples": summarize_records(double_failures),
            },
        }

    return results


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("  Phase 3D: Error Analysis")
    print("  LegalRisk-LLM — Failure Mode Investigation")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[LOAD] Loading predictions and judge scores...")
    all_preds, all_scores_raw = load_all_data()

    # Build merged records (predictions + valid judge scores joined by test_index)
    all_merged = {}
    for method in METHODS:
        if method in all_preds and method in all_scores_raw:
            valid_scores = [s for s in all_scores_raw[method] if not s.get("eval_failed")]
            merged = merge_by_index(all_preds[method], valid_scores)
            all_merged[method] = merged
            print(f"  [MERGE] {method}: {len(merged)} fully-joined records")

    available = [m for m in METHODS if m in all_merged]
    print(f"\n  Available methods: {available}")

    # Run all analyses
    results = {
        "metadata": {
            "phase": "3D",
            "description": "Error analysis and failure mode investigation",
            "methods": available,
            "fail_threshold": FAIL_THRESHOLD,
            "n_examples_per_method": {m: len(all_merged[m]) for m in available},
        }
    }

    results["confusion_matrices"] = confusion_matrix_analysis(all_merged)
    results["worst_examples"] = worst_examples_analysis(all_merged, n=10)
    results["dimension_failures"] = dimension_failure_analysis(all_merged)
    results["category_failures"] = category_failure_analysis(all_merged)
    results["json_parse_failures"] = json_failure_analysis(all_preds)
    results["cross_method_overlap"] = cross_method_overlap_analysis(all_merged)
    results["risk_bias"] = risk_bias_analysis(all_merged)
    results["critical_failures"] = critical_failure_analysis(all_merged, all_preds)

    # Save to results/
    output_path = OUTPUT_DIR / "phase3d_error_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print_error_summary(results)

    print(f"\n[DONE] Full error analysis saved to: {output_path}")
    return results


def print_error_summary(results: dict):
    """Print human-readable error analysis summary."""
    print("\n" + "=" * 70)
    print("  ERROR ANALYSIS SUMMARY")
    print("=" * 70)

    available = results["metadata"]["methods"]

    # Confusion matrix accuracy
    print("\n  RISK LEVEL CLASSIFICATION:")
    print(f"  {'Method':8s}  {'Accuracy':>10s}  {'Macro-F1':>10s}  {'Worst Class':>15s}")
    print("  " + "-" * 50)
    for method in available:
        cm = results["confusion_matrices"].get(method, {})
        acc = cm.get("accuracy", 0) * 100
        f1 = cm.get("macro_f1", 0)
        # Find worst performing class
        per_class = cm.get("per_class_metrics", {})
        worst_class = min(per_class.items(), key=lambda x: x[1]["f1"])[0] if per_class else "N/A"
        print(f"  {method:8s}  {acc:>9.1f}%  {f1:>10.4f}  {worst_class:>15s}")

    # Dimension weaknesses
    print("\n  WEAKEST DIMENSION PER METHOD:")
    for method in available:
        dim_f = results["dimension_failures"].get(method, {})
        weakest = dim_f.get("weakest_dimension", "N/A")
        gap = dim_f.get("dimension_gap", 0)
        print(f"  {method:8s}  weakest={weakest:20s}  dimension_gap={gap:.3f}")

    # Per-category hardest clause types
    print("\n  HARDEST CLAUSE TYPE PER METHOD:")
    for method in available:
        cat_f = results["category_failures"].get(method, {})
        hardest = cat_f.get("hardest_category", "N/A")
        easiest = cat_f.get("easiest_category", "N/A")
        hardest_rate = cat_f.get("per_category", {}).get(hardest, {}).get("failure_rate", 0)
        print(f"  {method:8s}  hardest={hardest:18s} ({hardest_rate*100:.1f}% fail)  easiest={easiest}")

    # JSON failures
    print("\n  JSON PARSE FAILURES:")
    for method in available:
        jf = results["json_parse_failures"].get(method, {})
        n_fail = jf.get("n_failed", 0)
        rate = jf.get("failure_rate", 0) * 100
        print(f"  {method:8s}  {n_fail} failures ({rate:.1f}%)")

    # Risk bias
    print("\n  RISK LEVEL BIAS:")
    print(f"  {'Method':8s}  {'Over%':>7s}  {'Under%':>8s}  {'Correct%':>10s}  {'Mean Bias':>10s}")
    print("  " + "-" * 55)
    for method in available:
        rb = results["risk_bias"].get(method, {})
        over = rb.get("over_prediction_rate", 0) * 100
        under = rb.get("under_prediction_rate", 0) * 100
        correct = rb.get("exact_match_rate", 0) * 100
        bias = rb.get("mean_bias", 0)
        print(f"  {method:8s}  {over:>6.1f}%  {under:>7.1f}%  {correct:>9.1f}%  {bias:>+10.3f}")

    # Critical failures
    print("\n  SAFETY-CRITICAL FAILURES:")
    print(f"  {'Method':8s}  {'Missed Critical':>16s}  {'False Alarms':>13s}  {'Double Fail':>12s}")
    print("  " + "-" * 57)
    for method in available:
        cf = results["critical_failures"].get(method, {})
        mc = cf.get("missed_critical", {}).get("count", 0)
        fa = cf.get("false_critical_alarms", {}).get("count", 0)
        df = cf.get("double_failures", {}).get("count", 0)
        print(f"  {method:8s}  {mc:>16d}  {fa:>13d}  {df:>12d}")

    # Cross-method overlap
    overlap = results["cross_method_overlap"]
    univ_fail = overlap["universal_failures"]["count"]
    univ_win = overlap["universal_wins"]["count"]
    total = overlap["n_examples_all_methods"]
    print(f"\n  CROSS-METHOD OVERLAP (out of {total} shared examples):")
    print(f"  Universal failures (all < {FAIL_THRESHOLD}): {univ_fail}")
    print(f"  Universal wins (all >= 8.0):            {univ_win}")

    # Score correlations
    print("\n  SCORE CORRELATIONS (Spearman rho):")
    for pair, corr in overlap["score_correlations"].items():
        rho = corr["spearman_rho"]
        interp = corr["interpretation"]
        print(f"  {pair:25s}  rho={rho:.4f}  ({interp})")

    print("=" * 70)


if __name__ == "__main__":
    main()
