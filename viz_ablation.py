"""
viz_ablation.py — Ablation study and node-removal visualizations.

Provides:
  1. plot_ablation_bars        – Bar chart of Kendall τ per ablation config
  2. plot_ablation_ft_curves   – F(t) curves per ablation config
  3. plot_ablation_component_contribution – Contribution of each component
  4. plot_node_removal_curves  – Before/after SIR spread curves
  5. plot_cascading_failure    – Cascading failure / spread-reduction bar
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz_utils import (
    apply_publication_style, save_figure,
    ABLATION_COLORS, COMPONENT_COLORS, DPI_PRINT,
)


# ─── 1. Ablation Bars ────────────────────────────────────────────────────────

def plot_ablation_bars(
    ablation_result: dict,
    dataset_name: str,
    output_dir: str,
):
    """
    Grouped bar chart comparing Kendall τ and Precision@K for each
    ablation configuration.
    """
    apply_publication_style()
    configs  = ablation_result["configs"]
    taus     = [ablation_result["kendall_taus"][c] for c in configs]
    p_at_k   = [ablation_result["precision_at_k"][c] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, taus, width, label="Kendall τ",
                   color=[ABLATION_COLORS.get(c, "#7f7f7f") for c in configs],
                   edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + width / 2, p_at_k, width, label="Precision@K",
                   color=[ABLATION_COLORS.get(c, "#7f7f7f") for c in configs],
                   edgecolor="white", alpha=0.55, hatch="//")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Ablation Study — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(taus), max(p_at_k)) * 1.18)
    plt.tight_layout()

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_ablation_bars", DPI_PRINT)
    plt.close(fig)


# ─── 2. Ablation F(t) Curves ─────────────────────────────────────────────────

def plot_ablation_ft_curves(
    ablation_result: dict,
    dataset_name: str,
    output_dir: str,
):
    """F(t) infection spread curves for each ablation configuration."""
    apply_publication_style()
    configs  = ablation_result["configs"]
    F_curves = ablation_result["F_curves"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in configs:
        F = F_curves[c]
        color = ABLATION_COLORS.get(c, "grey")
        lw = 3.0 if c == "Full SAASP" else 1.8
        ls = "-" if c == "Full SAASP" else "--"
        ax.plot(range(len(F)), F, label=c, color=color, linewidth=lw, linestyle=ls)

    ax.set_xlabel("Time Step t", fontsize=11)
    ax.set_ylabel("F(t) = Infected + Recovered", fontsize=11)
    ax.set_title(f"Ablation Study — F(t) Spread Curves\n({dataset_name})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_ablation_ft_curves", DPI_PRINT)
    plt.close(fig)


# ─── 3. Component Contribution ───────────────────────────────────────────────

def plot_component_contribution(
    ablation_result: dict,
    dataset_name: str,
    output_dir: str,
):
    """
    Waterfall-style bar showing performance gain from each component
    when added incrementally.
    """
    apply_publication_style()
    configs = ablation_result["configs"]
    taus    = [ablation_result["kendall_taus"][c] for c in configs]

    # Compute incremental gains
    gains = [taus[0]]
    for i in range(1, len(taus)):
        gains.append(taus[i] - taus[i - 1])

    labels = [
        "CLI\n(baseline)",
        "+CSLI",
        "+CASPI",
        "+CSI\n(Full)",
    ][:len(configs)]

    colors = ["#2ecc71" if g >= 0 else "#e74c3c" for g in gains]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, gains, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.002 if val >= 0 else -0.012),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔKendall τ (incremental gain)", fontsize=11)
    ax.set_title(f"Component Contribution to Kendall τ\n({dataset_name})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_component_contribution", DPI_PRINT)
    plt.close(fig)


# ─── 4. Node Removal Curves ──────────────────────────────────────────────────

def plot_node_removal_curves(
    removal_result: dict,
    dataset_name: str,
    output_dir: str,
):
    """Before vs after node removal SIR comparison plot."""
    apply_publication_style()
    F_orig = removal_result["F_original"]
    F_red  = removal_result["F_after_removal"]
    reduction = removal_result["spread_reduction"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t = range(len(F_orig))

    # Left: Overlay
    ax = axes[0]
    ax.plot(t, F_orig, color="#e74c3c", linewidth=2.5, label="Original graph")
    ax.fill_between(t, F_orig, alpha=0.12, color="#e74c3c")
    t2 = range(len(F_red))
    ax.plot(t2, F_red, color="#2ecc71", linewidth=2.5, linestyle="--",
            label="After top-K removal")
    ax.fill_between(t2, F_red, alpha=0.12, color="#2ecc71")
    ax.set_xlabel("Time Step t", fontsize=11)
    ax.set_ylabel("F(t) = Infected + Recovered", fontsize=11)
    ax.set_title(f"Before vs After Node Removal\n({dataset_name})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.annotate(f"Spread reduction:\n{reduction:.1%}",
                xy=(len(F_orig) - 1, F_orig[-1]),
                xytext=(len(F_orig) * 0.55, max(F_orig) * 0.6),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10, color="#c0392b", fontweight="bold")

    # Right: Bar chart of final values
    ax2 = axes[1]
    bars = ax2.bar(["Original", "Top-K Removed"],
                   [F_orig[-1], F_red[-1]],
                   color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.4)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=11)
    ax2.set_ylabel("Final F(T)", fontsize=11)
    ax2.set_title(f"Spread Reduction: {reduction:.1%}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_node_removal_curves", DPI_PRINT)
    plt.close(fig)


# ─── 5. Cascading Failure ────────────────────────────────────────────────────

def plot_cascading_failure(
    G,
    method_scores: dict,
    node_list: list,
    dataset_name: str,
    output_dir: str,
    top_k_max: int = 15,
    infection_rate: float = 0.1,
    steps: int = 20,
    num_sir_runs: int = 10,
):
    """
    Shows how progressively removing more top-k nodes reduces spread
    for SAASP-adaptive vs DC (degree centrality) as baseline.
    """
    from sir_model import run_sir_multiple
    apply_publication_style()

    compare_methods = {m: method_scores[m] for m in ["SAASP-adaptive", "DC"]
                       if m in method_scores}
    k_range = range(1, top_k_max + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for m, scores in compare_methods.items():
        reductions = []
        ranking = np.argsort(-scores)
        seed_nodes = [node_list[ranking[i]] for i in range(min(3, len(ranking)))]

        F_base = run_sir_multiple(G, seed_nodes,
                                  infection_rate=infection_rate,
                                  steps=steps, num_runs=num_sir_runs,
                                  random_state=0)
        f_base = F_base[-1]

        for k in k_range:
            top_nodes = [node_list[ranking[i]] for i in range(k)]
            G_r = G.copy()
            for v in top_nodes:
                if G_r.has_node(v):
                    G_r.remove_node(v)
            seeds_r = [v for v in seed_nodes if G_r.has_node(v)]
            if not seeds_r:
                seeds_r = [list(G_r.nodes())[0]] if G_r.nodes() else []
            if not seeds_r:
                reductions.append(1.0)
                continue
            F_r = run_sir_multiple(G_r, seeds_r,
                                   infection_rate=infection_rate,
                                   steps=steps, num_runs=num_sir_runs,
                                   random_state=0)
            f_red = F_r[-1]
            reductions.append((f_base - f_red) / (f_base + 1e-12))

        lw = 3.0 if "SAASP" in m else 1.8
        ax.plot(list(k_range), reductions,
                color=from_viz_utils_color(m), linewidth=lw,
                marker="o", markersize=4, label=m)

    ax.set_xlabel("Number of Removed Top-K Nodes", fontsize=11)
    ax.set_ylabel("Spread Reduction (fraction)", fontsize=11)
    ax.set_title(f"Cascading Failure Analysis — {dataset_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_cascading_failure", DPI_PRINT)
    plt.close(fig)


def from_viz_utils_color(method):
    from viz_utils import METHOD_COLORS
    return METHOD_COLORS.get(method, "grey")
