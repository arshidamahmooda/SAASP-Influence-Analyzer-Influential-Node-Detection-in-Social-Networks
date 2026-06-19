"""
viz_performance.py — Classification-style performance metrics for SAASP.

Provides:
  1. plot_roc_curves              – ROC curves + AUC for all methods
  2. plot_precision_recall_curves – PR curves for all methods
  3. plot_confusion_matrices      – Confusion matrices (top-K predicted vs GT)
  4. plot_f1_comparison           – F1-score bar chart
  5. plot_precision_at_k_comparison – Precision@K bar chart
  6. plot_recall_at_k_comparison  – Recall@K bar chart
  7. plot_accuracy_comparison     – Accuracy bar chart

Ground truth: nodes with SIR influence in top-K are "positive".
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, accuracy_score, recall_score,
    precision_score,
)

from viz_utils import (
    apply_publication_style, save_figure, ordered_methods,
    METHOD_COLORS, top_k_binary, DPI_PRINT,
)

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


def _build_gt_and_scores(method_scores, sir_ground_truth, sample_indices, top_k):
    """
    Build binary ground-truth vector and aligned score arrays.

    Returns
    -------
    gt_binary   : np.ndarray (n_sample,) binary labels
    score_dict  : dict  method → score array on sampled nodes
    methods     : ordered method list
    """
    gt_binary = top_k_binary(sir_ground_truth, k=top_k)
    methods   = ordered_methods(method_scores)
    score_dict = {}
    for m in methods:
        score_dict[m] = method_scores[m][sample_indices]
    return gt_binary, score_dict, methods


# ─── 1. ROC Curves ──────────────────────────────────────────────────────────

def plot_roc_curves(
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    dataset_name: str,
    output_dir: str,
    top_k: int = 10,
):
    """
    ROC curves + AUC for all methods. Returns dict method → AUC.
    """
    apply_publication_style()
    gt_binary, score_dict, methods = _build_gt_and_scores(
        method_scores, sir_ground_truth, sample_indices, top_k
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    auc_scores = {}

    for m in methods:
        s = score_dict[m]
        # normalise scores to [0,1]
        s_min, s_max = s.min(), s.max()
        s_norm = (s - s_min) / (s_max - s_min + 1e-12)
        fpr, tpr, _ = roc_curve(gt_binary, s_norm)
        roc_auc = auc(fpr, tpr)
        auc_scores[m] = roc_auc
        lw = 3.0 if "SAASP-adaptive" in m else 1.8
        ls = "--" if "fixed" in m else "-"
        ax.plot(fpr, tpr, color=METHOD_COLORS.get(m, "grey"),
                lw=lw, ls=ls, label=f"{m} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_roc_curves", DPI_PRINT)
    plt.close(fig)
    return auc_scores


# ─── 2. Precision–Recall Curves ─────────────────────────────────────────────

def plot_precision_recall_curves(
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    dataset_name: str,
    output_dir: str,
    top_k: int = 10,
):
    """PR curves + Average Precision for all methods."""
    apply_publication_style()
    gt_binary, score_dict, methods = _build_gt_and_scores(
        method_scores, sir_ground_truth, sample_indices, top_k
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ap_scores = {}

    for m in methods:
        s = score_dict[m]
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
        prec, rec, _ = precision_recall_curve(gt_binary, s_norm)
        ap = average_precision_score(gt_binary, s_norm)
        ap_scores[m] = ap
        lw = 3.0 if "SAASP-adaptive" in m else 1.8
        ls = "--" if "fixed" in m else "-"
        ax.plot(rec, prec, color=METHOD_COLORS.get(m, "grey"),
                lw=lw, ls=ls, label=f"{m} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision–Recall Curves — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_pr_curves", DPI_PRINT)
    plt.close(fig)
    return ap_scores


# ─── 3. Confusion Matrices ───────────────────────────────────────────────────

def plot_confusion_matrices(
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    dataset_name: str,
    output_dir: str,
    top_k: int = 10,
):
    """Grid of confusion matrices (one per method)."""
    apply_publication_style()
    gt_binary, score_dict, methods = _build_gt_and_scores(
        method_scores, sir_ground_truth, sample_indices, top_k
    )

    n_methods = len(methods)
    cols = 4
    rows = (n_methods + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = axes.flatten()

    for ax_idx, m in enumerate(methods):
        ax = axes[ax_idx]
        pred = top_k_binary(score_dict[m], k=top_k)
        cm = confusion_matrix(gt_binary, pred)
        if HAS_SNS:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-top", "Top-K"],
                        yticklabels=["Non-top", "Top-K"],
                        ax=ax, cbar=False, linewidths=0.5)
        else:
            ax.imshow(cm, cmap="Blues", aspect="auto")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-top", "Top-K"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-top", "Top-K"])
        ax.set_title(m, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Confusion Matrices — {dataset_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_confusion_matrices", DPI_PRINT)
    plt.close(fig)


# ─── 4–7. Bar chart metrics ──────────────────────────────────────────────────

def _compute_clf_metrics(method_scores, sir_ground_truth, sample_indices, top_k):
    """Return dict of dicts: method → {f1, accuracy, precision, recall}."""
    gt_binary, score_dict, methods = _build_gt_and_scores(
        method_scores, sir_ground_truth, sample_indices, top_k
    )
    out = {}
    for m in methods:
        pred = top_k_binary(score_dict[m], k=top_k)
        out[m] = {
            "f1":        f1_score(gt_binary, pred, zero_division=0),
            "accuracy":  accuracy_score(gt_binary, pred),
            "precision": precision_score(gt_binary, pred, zero_division=0),
            "recall":    recall_score(gt_binary, pred, zero_division=0),
        }
    return out, methods


def _bar_chart(metrics_dict, methods, metric_key, title, ylabel, output_dir,
               file_stem, colors):
    """Generic horizontal bar chart for a single metric."""
    apply_publication_style()
    values = [metrics_dict[m][metric_key] for m in methods]
    fig, ax = plt.subplots(figsize=(8, max(4, len(methods) * 0.55 + 1)))
    bars = ax.barh(methods, values, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, file_stem, DPI_PRINT)
    plt.close(fig)


def plot_f1_comparison(method_scores, sir_ground_truth, sample_indices,
                       dataset_name, output_dir, top_k=10):
    """F1-score bar chart for all methods."""
    met, methods = _compute_clf_metrics(method_scores, sir_ground_truth, sample_indices, top_k)
    colors = [METHOD_COLORS.get(m, "grey") for m in methods]
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    _bar_chart(met, methods, "f1",
               f"F1-Score Comparison — {dataset_name}", "F1 Score",
               output_dir, f"{safe}_f1_comparison", colors)
    return {m: met[m]["f1"] for m in methods}


def plot_accuracy_comparison(method_scores, sir_ground_truth, sample_indices,
                              dataset_name, output_dir, top_k=10):
    """Accuracy bar chart."""
    met, methods = _compute_clf_metrics(method_scores, sir_ground_truth, sample_indices, top_k)
    colors = [METHOD_COLORS.get(m, "grey") for m in methods]
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    _bar_chart(met, methods, "accuracy",
               f"Accuracy Comparison — {dataset_name}", "Accuracy",
               output_dir, f"{safe}_accuracy_comparison", colors)
    return {m: met[m]["accuracy"] for m in methods}


def plot_precision_at_k_comparison(precision_at_k: dict, dataset_name, output_dir):
    """Precision@K bar chart from pre-computed evaluator results."""
    apply_publication_style()
    from viz_utils import ordered_methods
    methods = ordered_methods(precision_at_k)
    values  = [precision_at_k[m] for m in methods]
    colors  = [METHOD_COLORS.get(m, "grey") for m in methods]

    fig, ax = plt.subplots(figsize=(8, max(4, len(methods) * 0.55 + 1)))
    bars = ax.barh(methods, values, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Precision@K", fontsize=11)
    ax.set_title(f"Precision@K Comparison — {dataset_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_precision_at_k", DPI_PRINT)
    plt.close(fig)


def plot_recall_at_k_comparison(method_scores, sir_ground_truth, sample_indices,
                                 dataset_name, output_dir, top_k=10):
    """Recall@K bar chart."""
    met, methods = _compute_clf_metrics(method_scores, sir_ground_truth, sample_indices, top_k)
    colors = [METHOD_COLORS.get(m, "grey") for m in methods]
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    _bar_chart(met, methods, "recall",
               f"Recall@K Comparison — {dataset_name}", "Recall@K",
               output_dir, f"{safe}_recall_at_k", colors)
    return {m: met[m]["recall"] for m in methods}


def plot_all_performance_metrics(
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    precision_at_k_dict: dict,
    dataset_name: str,
    output_dir: str,
    top_k: int = 10,
):
    """
    Convenience wrapper — runs all 7 performance metric plots.

    Returns dict of AUC scores (from ROC).
    """
    auc_scores = plot_roc_curves(
        method_scores, sir_ground_truth, sample_indices, dataset_name, output_dir, top_k)
    plot_precision_recall_curves(
        method_scores, sir_ground_truth, sample_indices, dataset_name, output_dir, top_k)
    plot_confusion_matrices(
        method_scores, sir_ground_truth, sample_indices, dataset_name, output_dir, top_k)
    plot_f1_comparison(
        method_scores, sir_ground_truth, sample_indices, dataset_name, output_dir, top_k)
    plot_accuracy_comparison(
        method_scores, sir_ground_truth, sample_indices, dataset_name, output_dir, top_k)
    plot_precision_at_k_comparison(precision_at_k_dict, dataset_name, output_dir)
    plot_recall_at_k_comparison(
        method_scores, sir_ground_truth, sample_indices, dataset_name, output_dir, top_k)
    return auc_scores
