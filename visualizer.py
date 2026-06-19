"""
visualizer.py — Interactive Pyvis graph + matplotlib plots.

Creates:
  1. Static matplotlib figure (results/network_analysis.png)
  2. Interactive Pyvis HTML for each dataset

Improvements (v2):
  - Top-5 nodes are explicitly highlighted with star markers and labels
  - Influence scores shown in bar-chart titles and Pyvis tooltips
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("[WARNING] pyvis not installed. Interactive HTML will be skipped.")


# Color scheme for methods
METHOD_COLORS = {
    "DC":             "#3498db",    # blue
    "BC":             "#e67e22",    # orange
    "CC":             "#2ecc71",    # green
    "CI":             "#e74c3c",    # red
    "PR":             "#1abc9c",    # teal
    "KS":             "#95a5a6",    # grey
    "SAASP-fixed":    "#9b59b6",    # purple
    "SAASP-adaptive": "#2c3e50",    # dark (bold)
}

METHOD_LINEWIDTHS = {
    "DC":             1.5,
    "BC":             1.5,
    "CC":             1.5,
    "CI":             1.5,
    "PR":             1.5,
    "KS":             1.5,
    "SAASP-fixed":    2.0,
    "SAASP-adaptive": 3.0,
}


def plot_ft_curves(ax, F_curves: dict, title: str):
    """Plot F(t) infection spread curves for all methods on one axes."""
    for method, F in F_curves.items():
        color = METHOD_COLORS.get(method, "grey")
        lw = METHOD_LINEWIDTHS.get(method, 1.5)
        ls = "--" if method == "SAASP-fixed" else "-"
        ax.plot(range(len(F)), F, label=method, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlabel("Time step t", fontsize=10)
    ax.set_ylabel("F(t) = Infected + Recovered", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_kendall_vs_lambda(ax, tau_results: dict, lambda_values, title: str):
    """Plot Kendall's tau vs infection probability lambda."""
    for method, taus in tau_results.items():
        color = METHOD_COLORS.get(method, "grey")
        lw = METHOD_LINEWIDTHS.get(method, 1.5)
        ls = "--" if method == "SAASP-fixed" else "-"
        ax.plot(lambda_values, taus, label=method, color=color,
                linewidth=lw, linestyle=ls, marker="o", markersize=3)
    ax.set_xlabel("Infection rate λ", fontsize=10)
    ax.set_ylabel("Kendall's τ", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_top10_bar(
    ax,
    top_nodes: dict,
    method_scores: dict,
    node_list: list,
    title: str,
    highlight_method: str = "SAASP-adaptive",
):
    """
    Bar chart of top-10 node scores for the highlight method.
    Top-5 nodes are explicitly highlighted with star markers and score labels.
    """
    scores = method_scores[highlight_method]
    ranking = np.argsort(-scores)[:10]
    top_ids = [node_list[i] for i in ranking]
    top_scores = scores[ranking]

    # Color: red for top-5, orange for 6-10
    colors = ["#e74c3c" if i < 5 else "#f39c12" for i in range(10)]

    bars = ax.bar(range(10), top_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(nid) for nid in top_ids], fontsize=7, rotation=45)
    ax.set_xlabel("Node ID", fontsize=10)
    ax.set_ylabel(f"{highlight_method} Score", fontsize=10)
    ax.set_title(f"{title}\n(★ = top-5 influential nodes)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate top-5 bars with their scores and a star marker
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        if i < 5:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(top_scores) * 0.01,
                f"★\n{score:.3f}",
                ha="center", va="bottom", fontsize=6, color="#c0392b",
                fontweight="bold",
            )


def create_static_plots(
    email_results: dict,
    fb_results: dict,
    email_tau_results: dict,
    fb_tau_results: dict,
    email_lambda_values,
    fb_lambda_values,
    email_method_scores: dict,
    fb_method_scores: dict,
    email_node_list: list,
    fb_node_list: list,
    output_dir: str = "results",
):
    """
    Create the 2x3 static matplotlib figure.

    Top row: Email network
    Bottom row: Facebook network
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SAASP Centrality Analysis — Comparison of Methods",
                 fontsize=14, fontweight="bold", y=0.98)

    # Top row: Email
    plot_ft_curves(axes[0, 0], email_results["F_curves"],
                   "Email-Eu-core: F(t) Spread Curves")
    plot_kendall_vs_lambda(axes[0, 1], email_tau_results, email_lambda_values,
                           "Email-Eu-core: Kendall τ vs λ")
    plot_top10_bar(axes[0, 2], email_results["top_nodes"],
                   email_method_scores, email_node_list,
                   "Email-Eu-core: Top-10 Nodes (SAASP-adaptive)")

    # Bottom row: Facebook
    plot_ft_curves(axes[1, 0], fb_results["F_curves"],
                   "Facebook: F(t) Spread Curves")
    plot_kendall_vs_lambda(axes[1, 1], fb_tau_results, fb_lambda_values,
                           "Facebook: Kendall τ vs λ")
    plot_top10_bar(axes[1, 2], fb_results["top_nodes"],
                   fb_method_scores, fb_node_list,
                   "Facebook: Top-10 Nodes (SAASP-adaptive)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(output_dir, "network_analysis.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Static plots saved to: {outpath}")


def create_interactive_html(
    G: nx.Graph,
    node_list: list,
    scores: np.ndarray,
    dataset_name: str,
    output_dir: str = "results",
    max_display_nodes: int = 500,
):
    """
    Create interactive Pyvis HTML visualization.

    Top-5 nodes are drawn larger and with a gold border + star in the label.
    Influence score is shown in every node's tooltip.

    For large graphs (>2000 nodes), extract a subgraph around
    the top-5 influential nodes.
    """
    if not HAS_PYVIS:
        print(f"  [SKIP] Pyvis not available for {dataset_name}")
        return

    os.makedirs(output_dir, exist_ok=True)

    node_score_map = {node_list[i]: scores[i] for i in range(len(node_list))}

    # Select subgraph if needed
    if G.number_of_nodes() > 2000:
        print(f"  Large graph ({G.number_of_nodes()} nodes), selecting subgraph...")
        ranking = np.argsort(-scores)
        top5_nodes = [node_list[i] for i in ranking[:5]]
        sub_nodes = set()
        for seed in top5_nodes:
            ego = nx.ego_graph(G, seed, radius=2)
            sub_nodes.update(ego.nodes())
            if len(sub_nodes) >= max_display_nodes:
                break
        sub_nodes = list(sub_nodes)[:max_display_nodes]
        sub_G = G.subgraph(sub_nodes).copy()
    else:
        sub_G = G

    # Normalize scores for sizing
    sub_scores = np.array([node_score_map.get(v, 0) for v in sub_G.nodes()])
    if sub_scores.max() - sub_scores.min() > 1e-12:
        norm_scores = (sub_scores - sub_scores.min()) / (sub_scores.max() - sub_scores.min())
    else:
        norm_scores = np.zeros_like(sub_scores)

    # Rank tiers in the FULL graph
    full_ranking  = np.argsort(-scores)
    top5_set  = set(node_list[i] for i in full_ranking[:5])
    top10_set = set(node_list[i] for i in full_ranking[:10])
    top50_set = set(node_list[i] for i in full_ranking[:50])

    # Build Pyvis network
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        heading=f"SAASP Node Influence — {dataset_name}",
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)

    for idx, v in enumerate(sub_G.nodes()):
        score  = node_score_map.get(v, 0)
        n_score = norm_scores[idx]

        if v in top5_set:
            color  = "#FFD700"   # gold — top-5
            size   = 10 + n_score * 40
            label  = f"★ {v}"
            border = "#FF8C00"
        elif v in top10_set:
            color  = "#e74c3c"   # red — top-10
            size   = 8 + n_score * 32
            label  = str(v)
            border = "#c0392b"
        elif v in top50_set:
            color  = "#f39c12"   # orange — top-50
            size   = 6 + n_score * 20
            label  = str(v)
            border = "#e67e22"
        else:
            color  = "#85c1e9"   # light blue — rest
            size   = 4 + n_score * 12
            label  = str(v)
            border = "#2980b9"

        net.add_node(
            int(v),
            label=label,
            size=float(size),
            color={"background": color, "border": border},
            title=(
                f"<b>Node {v}</b><br>"
                f"SAASP Score: <b>{score:.4f}</b><br>"
                f"Rank: {list(full_ranking).index(node_list.index(v)) + 1 if v in node_score_map else '?'}"
            ),
        )

    for u, v in sub_G.edges():
        net.add_edge(int(u), int(v), color="#555555", width=0.5)

    outpath = os.path.join(output_dir, f"{safe_name}_network.html")
    net.save_graph(outpath)
    print(f"  Interactive HTML saved to: {outpath}")
