"""
viz_heatmaps.py — Heatmap visualizations for SAASP research paper.

Provides:
  1. plot_method_correlation_heatmap  – pairwise Kendall-tau between all methods
  2. plot_tau_vs_lambda_heatmap       – Kendall tau grid (method × λ)
  3. plot_adaptive_weights_heatmap    – learned ξ weights per dataset
  4. plot_influence_spread_heatmap    – F(t) spread over time per method
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from viz_utils import (
    apply_publication_style, save_figure,
    ordered_methods, METHOD_COLORS, DPI_PRINT
)

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ─── 1. Method Correlation Heatmap ──────────────────────────────────────────

def plot_method_correlation_heatmap(
    method_scores: dict,
    dataset_name: str,
    output_dir: str,
):
    """
    Pairwise Kendall's τ correlation between all ranking methods.

    Saves as:  results/<dataset>_method_correlation_heatmap.{png,pdf,svg}
    """
    apply_publication_style()
    methods = ordered_methods(method_scores)
    n = len(methods)
    tau_matrix = np.zeros((n, n))

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                tau_matrix[i, j] = 1.0
            elif i < j:
                tau, _ = kendalltau(method_scores[m1], method_scores[m2])
                tau_matrix[i, j] = tau_matrix[j, i] = float(np.nan_to_num(tau))

    fig, ax = plt.subplots(figsize=(9, 7))

    if HAS_SNS:
        sns.heatmap(
            tau_matrix, annot=True, fmt=".2f",
            xticklabels=methods, yticklabels=methods,
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            ax=ax, annot_kws={"size": 9},
        )
    else:
        im = ax.imshow(tau_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(methods)
        fig.colorbar(im, ax=ax, label="Kendall's τ")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{tau_matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    ax.set_title(
        f"Kendall's τ Correlation Between Ranking Methods\n({dataset_name})",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, output_dir, f"{safe}_method_correlation_heatmap", DPI_PRINT)
    plt.close(fig)
    return tau_matrix, methods


# ─── 2. Kendall Tau vs Lambda Heatmap ───────────────────────────────────────

def plot_tau_vs_lambda_heatmap(
    tau_vs_lambda: dict,
    lambda_values,
    dataset_name: str,
    output_dir: str,
):
    """
    Heatmap of Kendall τ (rows = methods, columns = λ values).

    Saves as: results/<dataset>_tau_lambda_heatmap.{png,pdf,svg}
    """
    apply_publication_style()
    methods = ordered_methods(tau_vs_lambda)
    lam_labels = [f"{l:.3f}" for l in lambda_values]

    matrix = np.array([tau_vs_lambda[m] for m in methods])  # shape (n_methods, n_lambda)

    fig, ax = plt.subplots(figsize=(max(8, len(lambda_values) * 0.9), max(5, len(methods) * 0.6 + 1)))

    if HAS_SNS:
        sns.heatmap(
            matrix, annot=True, fmt=".2f",
            xticklabels=lam_labels, yticklabels=methods,
            cmap="YlOrRd", vmin=0, vmax=1,
            linewidths=0.4, linecolor="white",
            ax=ax, annot_kws={"size": 8},
        )
    else:
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(lambda_values))); ax.set_xticklabels(lam_labels, rotation=45)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
        fig.colorbar(im, ax=ax, label="Kendall's τ")

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    ax.set_xlabel("Infection Rate λ", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    ax.set_title(
        f"Kendall's τ vs Infection Rate λ\n({dataset_name})",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, output_dir, f"{safe}_tau_lambda_heatmap", DPI_PRINT)
    plt.close(fig)
    return matrix


# ─── 3. Adaptive Weights Heatmap ────────────────────────────────────────────

def plot_adaptive_weights_heatmap(
    xi_dict: dict,
    output_dir: str,
):
    """
    Heatmap of learned ξ1–ξ4 weights across datasets.

    Parameters
    ----------
    xi_dict : dict   dataset_name → np.ndarray of shape (4,)

    Saves as: results/adaptive_weights_heatmap.{png,pdf,svg}
    """
    apply_publication_style()
    components = ["ξ₁ (CLI)", "ξ₂ (CSLI)", "ξ₃ (CASPI)", "ξ₄ (CSI)"]
    datasets   = list(xi_dict.keys())
    matrix     = np.array([xi_dict[d] for d in datasets])   # (n_datasets, 4)

    fig, ax = plt.subplots(figsize=(7, max(3, len(datasets) * 0.9 + 1)))

    if HAS_SNS:
        sns.heatmap(
            matrix, annot=True, fmt=".4f",
            xticklabels=components, yticklabels=datasets,
            cmap="Blues", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white",
            ax=ax, annot_kws={"size": 10},
        )
    else:
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(components, rotation=25, ha="right")
        ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets)
        fig.colorbar(im, ax=ax)

    ax.set_title("Learned Adaptive Weights (ξ₁–ξ₄) per Dataset",
                 fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    save_figure(fig, output_dir, "adaptive_weights_heatmap", DPI_PRINT)
    plt.close(fig)
    return matrix


# ─── 4. Influence Spread Over Time Heatmap ──────────────────────────────────

def plot_influence_spread_heatmap(
    F_curves: dict,
    dataset_name: str,
    output_dir: str,
):
    """
    Heatmap showing F(t) spread for each method over time steps.
    Rows = methods, Columns = time steps.

    Saves as: results/<dataset>_influence_spread_heatmap.{png,pdf,svg}
    """
    apply_publication_style()
    methods = ordered_methods(F_curves)
    matrix  = np.array([F_curves[m] for m in methods])   # (n_methods, steps+1)
    # Normalize each method to [0,1] for visual clarity
    row_max = matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    matrix_norm = matrix / row_max

    T = matrix_norm.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, T * 0.55), max(4, len(methods) * 0.6 + 1)))

    if HAS_SNS:
        sns.heatmap(
            matrix_norm, annot=False,
            xticklabels=[str(t) for t in range(T)],
            yticklabels=methods,
            cmap="plasma", vmin=0, vmax=1,
            linewidths=0.2, linecolor="white",
            ax=ax, cbar_kws={"label": "Normalised F(t)"},
        )
    else:
        im = ax.imshow(matrix_norm, cmap="plasma", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(T)); ax.set_xticklabels(range(T))
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
        fig.colorbar(im, ax=ax, label="Normalised F(t)")

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    ax.set_xlabel("Time Step t", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    ax.set_title(
        f"Node Influence Spread Over Time\n({dataset_name})",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, output_dir, f"{safe}_influence_spread_heatmap", DPI_PRINT)
    plt.close(fig)
    return matrix_norm
