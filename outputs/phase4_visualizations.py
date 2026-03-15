"""
Phase 4: Visualizations for LegalRisk-LLM Evaluation Results

Generates 10 publication-ready charts comparing QLoRA, DoRA, IA3, and RAG:

 1. overall_comparison.png    - Bar chart with error bars (mean +/- std overall)
 2. radar_chart.png           - Spider/radar plot (5 dimensions x 4 methods)
 3. score_distributions.png  - Violin plots of overall score distributions
 4. dimension_heatmap.png    - Heatmap: methods x dimensions mean scores
 5. confusion_matrices.png   - 4-panel risk level confusion matrices
 6. category_performance.png - Grouped bar chart by clause type
 7. score_kde.png            - KDE density overlay of overall scores
 8. risk_bias.png            - Over/correct/under prediction stacked bars
 9. pairwise_correlation.png - Score correlation heatmap between methods
10. json_and_latency.png     - JSON success rates + latency box plots

No API calls. Pure matplotlib/seaborn. $0.00 cost.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# =========================================================
# CONFIG — Update these paths if needed
# =========================================================
BASE_DIR = Path(__file__).parent.parent  # Project root

PREDICTIONS_DIR  = BASE_DIR / "evaluation"
JUDGE_SCORES_DIR = BASE_DIR / "evaluation" / "judge_scores"
OUTPUT_DIR       = BASE_DIR / "results" / "visualizations"

METHODS     = ["qlora", "dora", "ia3", "rag"]
DIMENSIONS  = ["accuracy", "completeness", "legal_reasoning", "clarity", "actionability"]
RISK_LEVELS = ["low", "medium", "high", "critical"]

# One distinctive color per method — consistent across all charts
METHOD_COLORS = {
    "qlora": "#2196F3",   # Blue
    "dora":  "#4CAF50",   # Green
    "ia3":   "#FF9800",   # Orange
    "rag":   "#9C27B0",   # Purple
}

METHOD_LABELS = {
    "qlora": "QLoRA",
    "dora":  "DoRA",
    "ia3":   "IA3",
    "rag":   "RAG",
}

DIM_LABELS = {
    "accuracy":       "Accuracy",
    "completeness":   "Completeness",
    "legal_reasoning":"Legal\nReasoning",
    "clarity":        "Clarity",
    "actionability":  "Actionability",
}

# Chart style
sns.set_theme(style="whitegrid", font_scale=1.1)
CHART_DPI = 150


# =========================================================
# DATA LOADING
# =========================================================

def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all() -> tuple[dict, dict]:
    """Load all predictions and judge scores. Returns (all_preds, all_scores)."""
    all_preds  = {}
    all_scores = {}

    for method in METHODS:
        pred_path  = PREDICTIONS_DIR  / f"{method}_predictions.jsonl"
        score_path = JUDGE_SCORES_DIR / f"{method}_judge_scores.jsonl"

        if pred_path.exists():
            all_preds[method] = load_jsonl(pred_path)
            print(f"  [LOAD] {method}: {len(all_preds[method])} predictions")
        else:
            print(f"  [WARN] Missing: {pred_path}")

        if score_path.exists():
            raw = load_jsonl(score_path)
            all_scores[method] = [s for s in raw if not s.get("eval_failed")]
            print(f"  [LOAD] {method}: {len(all_scores[method])} judge scores")
        else:
            print(f"  [WARN] Missing: {score_path}")

    return all_preds, all_scores


def build_joined(all_preds: dict, all_scores: dict) -> dict:
    """
    Join predictions + judge scores by test_index per method.
    Returns dict[method] -> list of merged records.
    """
    joined = {}
    for method in METHODS:
        if method not in all_preds or method not in all_scores:
            continue
        pred_map  = {p["test_index"]: p for p in all_preds[method]}
        score_map = {s["test_index"]: s for s in all_scores[method]}
        common    = sorted(set(pred_map) & set(score_map))
        merged    = []
        for idx in common:
            p = pred_map[idx]
            s = score_map[idx]
            merged.append({
                "test_index":               idx,
                "clause_type":              s.get("clause_type", "unknown"),
                "ground_truth_risk_level":  s.get("ground_truth_risk_level", "unknown"),
                "predicted_risk_level":     s.get("predicted_risk_level", "unknown"),
                "json_parse_success":       p.get("json_parse_success", False),
                "latency_ms":               p.get("latency_ms", 0),
                "overall":                  s.get("overall"),
                "scores":                   s.get("scores", {}),
            })
        joined[method] = merged
    return joined


# =========================================================
# SAVE HELPER
# =========================================================

def save_fig(fig: plt.Figure, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {name}")


# =========================================================
# CHART 1: Overall Score Comparison (Bar + Error Bars)
# =========================================================

def chart_overall_comparison(all_scores: dict):
    """
    Horizontal bar chart showing mean overall score per method with ±1 std bars.

    Why horizontal? Method labels read cleanly left-to-right.
    Error bars give a sense of score variance (are scores consistent or noisy?).
    """
    means = []
    stds  = []
    labels = []
    colors = []

    for method in METHODS:
        if method not in all_scores:
            continue
        overalls = [s["overall"] for s in all_scores[method]]
        means.append(float(np.mean(overalls)))
        stds.append(float(np.std(overalls, ddof=1)))
        labels.append(METHOD_LABELS[method])
        colors.append(METHOD_COLORS[method])

    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(y_pos, means, xerr=stds, align="center",
                   color=colors, edgecolor="white", linewidth=0.8,
                   error_kw=dict(elinewidth=1.5, ecolor="#555555", capsize=5))

    # Value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.05, i, f"{mean:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Mean Overall Score (0–10)", fontsize=12)
    ax.set_title("Overall Judge Score by Method\n(mean ± 1 std, n=107 each)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.axvline(x=7, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Score = 7")
    ax.legend(fontsize=10)

    fig.tight_layout()
    save_fig(fig, "01_overall_comparison.png")


# =========================================================
# CHART 2: Radar / Spider Chart (5 Dimensions)
# =========================================================

def chart_radar(all_scores: dict):
    """
    Spider/radar chart comparing all 4 methods across the 5 scoring dimensions.

    Each axis = one scoring dimension (0–10).
    Filled polygons make it easy to see which method is best on each dimension
    and which has the most 'complete' coverage.
    """
    # Compute per-method dimension means
    dim_means = {}
    for method in METHODS:
        if method not in all_scores:
            continue
        dim_means[method] = {}
        for dim in DIMENSIONS:
            vals = [s["scores"][dim] for s in all_scores[method] if dim in s.get("scores", {})]
            dim_means[method][dim] = float(np.mean(vals)) if vals else 0.0

    N = len(DIMENSIONS)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Grid lines at 2, 4, 6, 8, 10
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="gray")

    # Category labels
    dim_label_list = [DIM_LABELS[d] for d in DIMENSIONS]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_label_list, fontsize=12)

    # Plot each method
    for method in METHODS:
        if method not in dim_means:
            continue
        values = [dim_means[method][d] for d in DIMENSIONS]
        values += values[:1]
        color = METHOD_COLORS[method]
        ax.plot(angles, values, color=color, linewidth=2.5, label=METHOD_LABELS[method])
        ax.fill(angles, values, color=color, alpha=0.08)
        # Mark data points
        ax.scatter(angles[:-1], values[:-1], color=color, s=50, zorder=5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=12)
    ax.set_title("Multi-Dimension Score Comparison\n(LLM Judge, 0–10 scale)", fontsize=14,
                 fontweight="bold", pad=25)

    fig.tight_layout()
    save_fig(fig, "02_radar_chart.png")


# =========================================================
# CHART 3: Score Distributions (Violin Plots)
# =========================================================

def chart_score_distributions(all_scores: dict):
    """
    Violin plots show the full distribution shape — not just the mean.

    A violin wider at the bottom = model often scores around 5-6.
    Bimodal violin (two wide spots) = model is inconsistent.
    Tight violin = model is consistent but may be mediocre.
    """
    data    = []
    methods = []

    for method in METHODS:
        if method not in all_scores:
            continue
        overalls = [s["overall"] for s in all_scores[method]]
        data.append(overalls)
        methods.append(METHOD_LABELS[method])

    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(data, positions=range(len(methods)),
                          showmeans=True, showmedians=True, showextrema=True)

    # Color each violin
    for i, (pc, method) in enumerate(zip(parts["bodies"], METHODS)):
        color = METHOD_COLORS[method]
        pc.set_facecolor(color)
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)

    for part_name in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
        if part_name in parts:
            parts[part_name].set_edgecolor("#333333")
            parts[part_name].set_linewidth(1.5)

    # Overlay individual data points (jittered)
    rng = np.random.default_rng(42)
    for i, (vals, method) in enumerate(zip(data, METHODS)):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(i + jitter, vals, s=8, color=METHOD_COLORS[method],
                   alpha=0.4, zorder=3)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=13)
    ax.set_ylabel("Overall Judge Score (0–10)", fontsize=12)
    ax.set_title("Score Distribution per Method\n(violin = density, line = mean, dots = examples)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.axhline(y=7, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(methods) - 0.5, 7.1, "Score=7", color="gray", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "03_score_distributions.png")


# =========================================================
# CHART 4: Dimension Heatmap (Methods x Dimensions)
# =========================================================

def chart_dimension_heatmap(all_scores: dict):
    """
    Heatmap where rows = methods and columns = scoring dimensions.
    Darker green = higher mean score.

    This is the single most information-dense chart — lets you see
    at a glance which method excels at which specific aspect.
    """
    all_dims = DIMENSIONS + ["overall"]

    # Build matrix: rows=methods, cols=dimensions
    matrix = []
    method_labels_list = []
    for method in METHODS:
        if method not in all_scores:
            continue
        row = []
        for dim in all_dims:
            if dim == "overall":
                vals = [s["overall"] for s in all_scores[method]]
            else:
                vals = [s["scores"][dim] for s in all_scores[method] if dim in s.get("scores", {})]
            row.append(round(float(np.mean(vals)), 2) if vals else 0.0)
        matrix.append(row)
        method_labels_list.append(METHOD_LABELS[method])

    matrix_np = np.array(matrix)

    col_labels = [DIM_LABELS.get(d, d).replace("\n", " ") for d in DIMENSIONS] + ["Overall"]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    heatmap = sns.heatmap(
        matrix_np,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        vmin=5.0,
        vmax=9.0,
        xticklabels=col_labels,
        yticklabels=method_labels_list,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        ax=ax,
        cbar_kws={"label": "Mean Score (0–10)", "shrink": 0.8},
    )

    ax.set_title("Mean Judge Scores: Methods × Dimensions", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.tick_params(axis="y", rotation=0, labelsize=12)

    fig.tight_layout()
    save_fig(fig, "04_dimension_heatmap.png")


# =========================================================
# CHART 5: Risk Level Confusion Matrices (4-Panel)
# =========================================================

def chart_confusion_matrices(joined: dict):
    """
    2x2 grid of confusion matrices — one per method.
    Rows = ground truth risk level, Cols = predicted risk level.
    Darker = more cases.

    Reveals: does QLoRA confuse High with Medium?
    Does RAG over-predict Critical?
    """
    available = [m for m in METHODS if m in joined]
    n = len(available)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, method in enumerate(available):
        ax = axes[ax_idx]
        records = joined[method]

        # Build 4x4 count matrix
        matrix = np.zeros((len(RISK_LEVELS), len(RISK_LEVELS)), dtype=int)
        for r in records:
            gt   = r.get("ground_truth_risk_level", "").lower()
            pred = r.get("predicted_risk_level", "").lower()
            if pred in ("failed", "unknown", ""):
                continue
            if gt in RISK_LEVELS and pred in RISK_LEVELS:
                row_i = RISK_LEVELS.index(gt)
                col_i = RISK_LEVELS.index(pred)
                matrix[row_i, col_i] += 1

        # Row-normalize for % display
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_pct = np.where(row_sums > 0, matrix / row_sums * 100, 0)

        labels_upper = [l.capitalize() for l in RISK_LEVELS]
        annot = np.array([[f"{matrix[i,j]}\n({matrix_pct[i,j]:.0f}%)"
                           for j in range(len(RISK_LEVELS))]
                          for i in range(len(RISK_LEVELS))])

        sns.heatmap(
            matrix_pct, annot=annot, fmt="", cmap="Blues",
            xticklabels=labels_upper, yticklabels=labels_upper,
            vmin=0, vmax=100, linewidths=0.5, linecolor="white",
            annot_kws={"fontsize": 9},
            ax=ax,
            cbar_kws={"label": "Row %"},
        )
        ax.set_title(f"{METHOD_LABELS[method]}", fontsize=13, fontweight="bold",
                     color=METHOD_COLORS[method])
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Ground Truth", fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    # Hide any unused axes
    for ax_idx in range(len(available), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Risk Level Confusion Matrices\n(row-normalized — shows where each GT class gets predicted)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "05_confusion_matrices.png")


# =========================================================
# CHART 6: Per-Category Performance (Grouped Bar Chart)
# =========================================================

def chart_category_performance(joined: dict):
    """
    Grouped bar chart: for each clause type, show all 4 methods' mean scores.

    Legal insight: models may generalize well on common clause types
    (termination, liability) but fail on rarer ones (indemnification, ip).
    This chart exposes domain-specific strengths and blind spots.
    """
    # Collect all clause types
    all_cats = set()
    for records in joined.values():
        for r in records:
            all_cats.add(r.get("clause_type", "unknown"))
    categories = sorted(all_cats)

    available = [m for m in METHODS if m in joined]
    n_cats = len(categories)
    n_methods = len(available)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(n_cats)

    for i, method in enumerate(available):
        records = joined[method]
        by_cat = defaultdict(list)
        for r in records:
            ct = r.get("clause_type", "unknown")
            if r.get("overall") is not None:
                by_cat[ct].append(r["overall"])

        means  = [float(np.mean(by_cat[cat])) if by_cat.get(cat) else 0 for cat in categories]
        errors = [float(np.std(by_cat[cat], ddof=1)) if len(by_cat.get(cat, [])) > 1 else 0
                  for cat in categories]

        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width=width * 0.9,
                      color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                      edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, means, yerr=errors, fmt="none",
                    ecolor="#444444", elinewidth=1, capsize=3)

    cat_labels = [c.replace("_", "\n") for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Mean Overall Score (0–10)", fontsize=12)
    ax.set_title("Performance by Clause Type\n(mean ± std across 107 test examples)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 10.5)
    ax.axhline(y=7, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.legend(fontsize=11, loc="lower right")

    fig.tight_layout()
    save_fig(fig, "06_category_performance.png")


# =========================================================
# CHART 7: KDE Density Plot (Score Distributions Overlaid)
# =========================================================

def chart_score_kde(all_scores: dict):
    """
    Kernel Density Estimation curves overlaid for all 4 methods.

    Unlike violin plots, KDE shows the smooth probability density.
    If two methods have very similar KDE curves, they perform equivalently.
    Vertical lines show the mean of each method.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in METHODS:
        if method not in all_scores:
            continue
        overalls = [s["overall"] for s in all_scores[method]]
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]

        # KDE using seaborn
        sns.kdeplot(overalls, ax=ax, color=color, linewidth=2.5,
                    label=label, fill=True, alpha=0.08)

        # Vertical mean line
        mean_val = float(np.mean(overalls))
        ax.axvline(x=mean_val, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(mean_val + 0.05, ax.get_ylim()[1] * 0.85 * (0.8 + 0.1 * METHODS.index(method)),
                f"{mean_val:.2f}", color=color, fontsize=9)

    ax.set_xlabel("Overall Score (0–10)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Density Distribution — All Methods Overlaid\n"
                 "(dashed lines = method means)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 10.5)
    ax.legend(fontsize=12)

    fig.tight_layout()
    save_fig(fig, "07_score_kde.png")


# =========================================================
# CHART 8: Risk Level Bias (Stacked Bar Chart)
# =========================================================

def chart_risk_bias(joined: dict):
    """
    Stacked horizontal bar chart: for each method, show % of predictions
    that were over-predicted, correct, or under-predicted.

    Risk order: Low(0) < Medium(1) < High(2) < Critical(3)
    Under-prediction is most dangerous in legal contexts — means the model
    missed a serious risk. Over-prediction creates false alarms.
    """
    risk_order = {level: i for i, level in enumerate(RISK_LEVELS)}

    available = [m for m in METHODS if m in joined]
    over_pcts, correct_pcts, under_pcts = [], [], []
    method_labels_list = []

    for method in available:
        records = joined[method]
        over = correct = under = 0

        for r in records:
            gt   = r.get("ground_truth_risk_level", "").lower()
            pred = r.get("predicted_risk_level", "").lower()
            if gt not in risk_order or pred not in risk_order:
                continue
            diff = risk_order[pred] - risk_order[gt]
            if diff > 0:
                over += 1
            elif diff < 0:
                under += 1
            else:
                correct += 1

        total = over + correct + under
        over_pcts.append(over / total * 100 if total else 0)
        correct_pcts.append(correct / total * 100 if total else 0)
        under_pcts.append(under / total * 100 if total else 0)
        method_labels_list.append(METHOD_LABELS[method])

    y_pos = np.arange(len(available))
    fig, ax = plt.subplots(figsize=(10, 5))

    bars_under   = ax.barh(y_pos, under_pcts, color="#EF5350", label="Under-predicted (missed risk)")
    bars_correct = ax.barh(y_pos, correct_pcts, left=under_pcts, color="#66BB6A", label="Correct")
    bars_over    = ax.barh(y_pos, over_pcts,
                           left=[u + c for u, c in zip(under_pcts, correct_pcts)],
                           color="#FFA726", label="Over-predicted (false alarm)")

    # Label each segment
    for i, (u, c, o) in enumerate(zip(under_pcts, correct_pcts, over_pcts)):
        if u > 5:
            ax.text(u / 2, i, f"{u:.0f}%", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        if c > 5:
            ax.text(u + c / 2, i, f"{c:.0f}%", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        if o > 5:
            ax.text(u + c + o / 2, i, f"{o:.0f}%", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_labels_list, fontsize=13)
    ax.set_xlabel("Percentage of Predictions (%)", fontsize=12)
    ax.set_title("Risk Level Prediction Bias\n(Under-prediction = most dangerous for legal use)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=11, loc="lower right")

    fig.tight_layout()
    save_fig(fig, "08_risk_bias.png")


# =========================================================
# CHART 9: Pairwise Score Correlation Heatmap
# =========================================================

def chart_pairwise_correlation(joined: dict):
    """
    Heatmap of Spearman rank correlations between all method pairs.

    High correlation (> 0.7) = methods make similar mistakes.
    Low correlation = methods fail on different examples.

    QLoRA vs DoRA should be highly correlated (same LoRA base).
    RAG vs fine-tuned should be weakly correlated (different approach).
    """
    available = [m for m in METHODS if m in joined]

    # Build {test_index: overall_score} per method
    index_score = {}
    for method in available:
        index_score[method] = {r["test_index"]: r["overall"]
                               for r in joined[method] if r.get("overall") is not None}

    # Find common indices
    common = sorted(
        set.intersection(*[set(index_score[m].keys()) for m in available])
    )

    # Build correlation matrix
    n = len(available)
    corr_matrix = np.zeros((n, n))
    for i, m1 in enumerate(available):
        for j, m2 in enumerate(available):
            x = [index_score[m1][idx] for idx in common]
            y = [index_score[m2][idx] for idx in common]
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(x, y)
                corr_matrix[i, j] = rho

    labels = [METHOD_LABELS[m] for m in available]

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, False)

    annot_vals = np.array([[f"{corr_matrix[i,j]:.3f}" for j in range(n)] for i in range(n)])

    sns.heatmap(
        corr_matrix, annot=annot_vals, fmt="",
        cmap="RdYlGn", vmin=-1, vmax=1,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        ax=ax,
        square=True,
        cbar_kws={"label": "Spearman rho", "shrink": 0.8},
    )
    ax.set_title(f"Pairwise Score Correlation (Spearman)\n"
                 f"n={len(common)} shared examples — how aligned are their errors?",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=0, labelsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=12)

    fig.tight_layout()
    save_fig(fig, "09_pairwise_correlation.png")


# =========================================================
# CHART 10: JSON Parse Success + Latency (Two-Panel)
# =========================================================

def chart_json_and_latency(all_preds: dict, joined: dict):
    """
    Two-panel chart:
    LEFT:  JSON parse success rate per method (horizontal bars)
    RIGHT: Inference latency distribution per method (box plot)

    Together these tell the full story of reliability + speed.
    Fine-tuned models: near-100% parse success, fast inference.
    RAG: ~94% parse success, slower (retrieval overhead).
    """
    available = [m for m in METHODS if m in all_preds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel 1: JSON Parse Success ---
    rates = []
    labels_list = []
    colors_list = []
    for method in available:
        preds = all_preds[method]
        successes = sum(1 for p in preds if p.get("json_parse_success"))
        rates.append(successes / len(preds) * 100 if preds else 0)
        labels_list.append(METHOD_LABELS[method])
        colors_list.append(METHOD_COLORS[method])

    y_pos = np.arange(len(available))
    bars = ax1.barh(y_pos, rates, color=colors_list, edgecolor="white", linewidth=0.8)

    for i, rate in enumerate(rates):
        ax1.text(rate - 1.5 if rate > 5 else rate + 0.5, i,
                 f"{rate:.1f}%", va="center", ha="right" if rate > 5 else "left",
                 fontsize=11, fontweight="bold",
                 color="white" if rate > 5 else "#333333")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels_list, fontsize=12)
    ax1.set_xlabel("JSON Parse Success Rate (%)", fontsize=11)
    ax1.set_title("JSON Parse Success Rate\n(failed parse = no usable output)", fontsize=12, fontweight="bold")
    ax1.set_xlim(85, 102)
    ax1.axvline(x=100, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # --- Panel 2: Latency Distribution ---
    latency_data = []
    latency_labels = []
    latency_colors = []

    for method in available:
        preds = all_preds[method]
        latencies = [p.get("latency_ms", 0) / 1000.0 for p in preds if p.get("latency_ms", 0) > 0]
        if latencies:
            latency_data.append(latencies)
            latency_labels.append(METHOD_LABELS[method])
            latency_colors.append(METHOD_COLORS[method])

    bp = ax2.boxplot(latency_data, vert=True, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     flierprops=dict(marker="o", markersize=4, alpha=0.4))

    for patch, color in zip(bp["boxes"], latency_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax2.set_xticklabels(latency_labels, fontsize=12)
    ax2.set_ylabel("Inference Latency (seconds)", fontsize=11)
    ax2.set_title("Inference Latency Distribution\n(RAG includes retrieval overhead)", fontsize=12, fontweight="bold")

    fig.suptitle("Reliability & Speed: JSON Parse Success and Inference Latency",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "10_json_and_latency.png")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("  Phase 4: Visualizations")
    print("  LegalRisk-LLM — 10 Charts")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[LOAD] Loading data...")
    all_preds, all_scores = load_all()
    joined = build_joined(all_preds, all_scores)

    available = [m for m in METHODS if m in all_scores]
    print(f"\n  Available methods: {available}")
    print(f"  Output directory: {OUTPUT_DIR}")

    print("\n[CHARTS] Generating 10 charts...")

    chart_overall_comparison(all_scores)
    chart_radar(all_scores)
    chart_score_distributions(all_scores)
    chart_dimension_heatmap(all_scores)
    chart_confusion_matrices(joined)
    chart_category_performance(joined)
    chart_score_kde(all_scores)
    chart_risk_bias(joined)
    chart_pairwise_correlation(joined)
    chart_json_and_latency(all_preds, joined)

    saved = list(OUTPUT_DIR.glob("*.png"))
    print(f"\n{'='*70}")
    print(f"  Done! {len(saved)} charts saved to:")
    print(f"  {OUTPUT_DIR}")
    print(f"{'='*70}")
    for p in sorted(saved):
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name:<40s}  {size_kb:>5d} KB")


if __name__ == "__main__":
    main()
