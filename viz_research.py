"""
viz_research.py — Research-paper-quality statistical plots for SAASP.

Provides:
  1. plot_score_boxplots        – Box plots of score distributions per method
  2. plot_score_violins         – Violin plots for method comparison
  3. plot_radar_chart           – Radar/spider chart comparing all methods
  4. plot_node_influence_hist   – Histogram of SAASP node influence
  5. plot_saasp_vs_sir_scatter  – Scatter: SAASP score vs SIR influence
  6. plot_pairplot              – Pairplot of all centrality scores
  7. plot_error_bar_sir         – Error bar plot over multiple SIR runs
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from viz_utils import (
    apply_publication_style, save_figure, ordered_methods,
    METHOD_COLORS, build_score_matrix, DPI_PRINT,
)

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False


# ─── 1. Box Plots ────────────────────────────────────────────────────────────

def plot_score_boxplots(
    method_scores: dict,
    dataset_name: str,
    output_dir: str,
):
    """Box plots of normalised score distributions for every method."""
    apply_publication_style()
    methods = ordered_methods(method_scores)
    # Normalise each method to [0,1]
    data = []
    for m in methods:
        s = method_scores[m].astype(float)
        mn, mx = s.min(), s.max()
        data.append((s - mn) / (mx - mn + 1e-12))

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(m, "grey"))
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Normalised Score", fontsize=11)
    ax.set_title(f"Score Distribution per Method — {dataset_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_score_boxplots", DPI_PRINT)
    plt.close(fig)


# ─── 2. Violin Plots ─────────────────────────────────────────────────────────

def plot_score_violins(
    method_scores: dict,
    dataset_name: str,
    output_dir: str,
):
    """Violin plots for method score distributions."""
    apply_publication_style()
    methods = ordered_methods(method_scores)
    data = []
    for m in methods:
        s = method_scores[m].astype(float)
        mn, mx = s.min(), s.max()
        data.append((s - mn) / (mx - mn + 1e-12))

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, positions=range(1, len(methods) + 1),
                          showmedians=True, showextrema=True)
    for i, (body, m) in enumerate(zip(parts["bodies"], methods)):
        body.set_facecolor(METHOD_COLORS.get(m, "grey"))
        body.set_alpha(0.75)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Normalised Score", fontsize=11)
    ax.set_title(f"Score Distribution Violins — {dataset_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_score_violins", DPI_PRINT)
    plt.close(fig)


# ─── 3. Radar Chart ──────────────────────────────────────────────────────────

def plot_radar_chart(
    metric_rows: dict,
    metric_names: list,
    dataset_name: str,
    output_dir: str,
):
    """
    Radar (spider) chart comparing methods across multiple metrics.

    Parameters
    ----------
    metric_rows : dict  method → list/array of values (one per metric)
    metric_names: list  names of the metrics (axes of the radar)
    """
    apply_publication_style()
    methods = ordered_methods(metric_rows)
    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names, fontsize=9)

    for m in methods:
        vals = list(metric_rows[m]) + [metric_rows[m][0]]  # close loop
        color = METHOD_COLORS.get(m, "grey")
        lw = 3 if "SAASP-adaptive" in m else 1.8
        ax.plot(angles, vals, color=color, linewidth=lw,
                linestyle="--" if "fixed" in m else "-", label=m)
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax.set_title(f"Multi-Metric Radar Chart — {dataset_name}",
                 fontsize=13, fontweight="bold", pad=18)

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_radar_chart", DPI_PRINT)
    plt.close(fig)


# ─── 4. Node Influence Histogram ─────────────────────────────────────────────

def plot_node_influence_hist(
    saasp_scores: np.ndarray,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    dataset_name: str,
    output_dir: str,
):
    """Overlapping histograms of SAASP scores and SIR influence (normalised)."""
    apply_publication_style()
    saasp_sample = saasp_scores[sample_indices]
    # Normalise both to [0,1]
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-12)

    s_norm  = norm(saasp_sample)
    gt_norm = norm(sir_ground_truth)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gt_norm, bins=30, alpha=0.6, color="#3498db", label="SIR Influence (GT)", density=True)
    ax.hist(s_norm,  bins=30, alpha=0.6, color="#c0392b", label="SAASP-adaptive Score", density=True)
    ax.set_xlabel("Normalised Influence Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Node Influence Distribution — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_influence_histogram", DPI_PRINT)
    plt.close(fig)


# ─── 5. SAASP vs SIR Scatter ─────────────────────────────────────────────────

def plot_saasp_vs_sir_scatter(
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    dataset_name: str,
    output_dir: str,
):
    """Scatter plot: SAASP-adaptive score vs SIR ground-truth influence."""
    apply_publication_style()
    from scipy.stats import kendalltau, pearsonr

    methods_to_plot = [m for m in ["SAASP-adaptive", "SAASP-fixed"] if m in method_scores]
    n_plots = len(methods_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    gt_norm = sir_ground_truth
    gt_norm = (gt_norm - gt_norm.min()) / (gt_norm.max() - gt_norm.min() + 1e-12)

    for ax, m in zip(axes, methods_to_plot):
        s = method_scores[m][sample_indices]
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
        tau, _ = kendalltau(s_norm, gt_norm)
        r, _   = pearsonr(s_norm, gt_norm)
        color  = METHOD_COLORS.get(m, "grey")

        ax.scatter(gt_norm, s_norm, c=color, alpha=0.55, s=18, edgecolors="none")
        # Trend line
        z = np.polyfit(gt_norm, s_norm, 1)
        p = np.poly1d(z)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, p(xs), "k--", linewidth=1.5, label=f"Kendall τ={tau:.3f}\nPearson r={r:.3f}")
        ax.set_xlabel("SIR Influence (Ground Truth)", fontsize=11)
        ax.set_ylabel(f"{m} Score", fontsize=11)
        ax.set_title(f"{m} vs SIR Influence\n({dataset_name})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_saasp_vs_sir_scatter", DPI_PRINT)
    plt.close(fig)


# ─── 6. Pairplot ─────────────────────────────────────────────────────────────

def plot_pairplot(
    method_scores: dict,
    dataset_name: str,
    output_dir: str,
    max_nodes: int = 500,
):
    """Pairplot of all centrality method scores (sampled for speed)."""
    if not (HAS_SNS and HAS_PD):
        print("  [SKIP] pairplot requires seaborn + pandas")
        return

    apply_publication_style()
    matrix, labels = build_score_matrix(method_scores, [])
    n = matrix.shape[0]
    if n > max_nodes:
        idx = np.random.RandomState(42).choice(n, size=max_nodes, replace=False)
        matrix = matrix[idx]

    df = pd.DataFrame(matrix, columns=labels)
    # Colour rows by SAASP-adaptive quintile
    if "SAASP-adaptive" in labels:
        col_idx = labels.index("SAASP-adaptive")
        quintile = pd.qcut(df.iloc[:, col_idx], q=5, labels=False, duplicates="drop")
        palette  = sns.color_palette("plasma", n_colors=5)
        hue      = quintile.astype(str)
    else:
        hue = None
        palette = None

    g = sns.pairplot(df, hue=hue, palette=palette,
                     plot_kws={"alpha": 0.35, "s": 10},
                     diag_kind="kde")
    g.fig.suptitle(f"Pairplot of Centrality Scores — {dataset_name}",
                   y=1.02, fontsize=13, fontweight="bold")

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(g.fig, output_dir, f"{safe}_pairplot", DPI_PRINT)
    plt.close("all")


# ─── 7. Error Bar Plot ────────────────────────────────────────────────────────

def plot_error_bar_sir(
    G,
    method_scores: dict,
    dataset_name: str,
    output_dir: str,
    top_k: int = 10,
    infection_rate: float = 0.1,
    n_repeats: int = 10,
    steps: int = 20,
):
    """
    Error bar plot of final SIR F(t) over multiple random seeds per method.

    Bars show mean ± std of F_final across n_repeats independent SIR runs.
    """
    from sir_model import run_sir_multiple
    apply_publication_style()
    methods = ordered_methods(method_scores)
    means, stds = [], []

    for m in methods:
        scores  = method_scores[m]
        top_idx = np.argsort(-scores)[:top_k]
        top_nodes = [list(G.nodes())[i] for i in top_idx
                     if i < G.number_of_nodes()]
        run_finals = []
        for seed in range(n_repeats):
            F = run_sir_multiple(G, top_nodes,
                                 infection_rate=infection_rate,
                                 steps=steps, num_runs=5,
                                 random_state=seed * 7 + 3)
            run_finals.append(F[-1])
        means.append(np.mean(run_finals))
        stds.append(np.std(run_finals))

    colors = [METHOD_COLORS.get(m, "grey") for m in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", width=0.6,
                  error_kw=dict(elinewidth=1.5, capthick=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Final Infected+Recovered F(T)", fontsize=11)
    ax.set_title(f"SIR Spreading — Mean ± Std over {n_repeats} Runs\n({dataset_name})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_error_bar_sir", DPI_PRINT)
    plt.close(fig)
    return dict(zip(methods, zip(means, stds)))
