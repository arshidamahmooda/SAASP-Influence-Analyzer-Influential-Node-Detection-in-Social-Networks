"""
viz_network.py — Advanced network visualization for SAASP.

Provides:
  1. plot_network_influence_static   – Matplotlib: nodes coloured & sized by SAASP,
                                       top-10 highlighted, community coloring option
  2. create_advanced_interactive_html – Enhanced PyVis with hover tooltips,
                                        community colours, top-10 highlights
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import networkx as nx

from viz_utils import apply_publication_style, save_figure, DPI_PRINT

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

try:
    import community as community_louvain   # python-louvain
    HAS_LOUVAIN = True
except ImportError:
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        HAS_LOUVAIN = False
    except Exception:
        HAS_LOUVAIN = False


# ─── Community detection helper ──────────────────────────────────────────────

def _detect_communities(G):
    """Return dict node → community_id. Falls back to nx greedy if louvain missing."""
    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(G)
        return partition
    else:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))
        partition = {}
        for cid, comm in enumerate(comms):
            for node in comm:
                partition[node] = cid
        return partition


def _community_color(cid, n_communities, cmap_name="tab20"):
    cmap = cm.get_cmap(cmap_name, max(n_communities, 2))
    rgba = cmap(cid % n_communities)
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


# ─── 1. Static network influence plot ────────────────────────────────────────

def plot_network_influence_static(
    G: nx.Graph,
    node_list: list,
    saasp_scores: np.ndarray,
    dataset_name: str,
    output_dir: str,
    top_k: int = 10,
    max_nodes: int = 300,
    use_community_coloring: bool = True,
):
    """
    Static matplotlib network plot:
      - Node colour = community (if community coloring) else SAASP score
      - Node size   = SAASP score
      - Top-10 nodes outlined in gold
    """
    apply_publication_style()
    node_score_map = {node_list[i]: saasp_scores[i] for i in range(len(node_list))}
    ranking = np.argsort(-saasp_scores)
    top10   = set(node_list[i] for i in ranking[:top_k])

    # Subsample for large graphs
    if G.number_of_nodes() > max_nodes:
        seed_nodes = [node_list[i] for i in ranking[:5]]
        sub_nodes  = set()
        for s in seed_nodes:
            ego = nx.ego_graph(G, s, radius=2)
            sub_nodes.update(ego.nodes())
            if len(sub_nodes) >= max_nodes:
                break
        sub_nodes = list(sub_nodes)[:max_nodes]
        G_plot = G.subgraph(sub_nodes).copy()
    else:
        G_plot = G

    nodes = list(G_plot.nodes())
    scores_plot = np.array([node_score_map.get(v, 0.0) for v in nodes])
    s_min, s_max = scores_plot.min(), scores_plot.max()
    norm_scores  = (scores_plot - s_min) / (s_max - s_min + 1e-12)

    # Community colours or score colour
    if use_community_coloring and G_plot.number_of_nodes() > 0:
        partition    = _detect_communities(G_plot)
        n_comm       = max(partition.values()) + 1 if partition else 1
        node_colors  = [_community_color(partition.get(v, 0), n_comm) for v in nodes]
    else:
        cmap = cm.plasma
        norm = Normalize(vmin=0, vmax=1)
        node_colors = [cmap(norm(s)) for s in norm_scores]

    node_sizes = 80 + norm_scores * 600

    edge_colors = ["#cccccc"] * G_plot.number_of_edges()
    edgecolors  = ["#FFD700" if v in top10 else "none" for v in nodes]
    linewidths  = [3.0 if v in top10 else 0.0 for v in nodes]

    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        pos = nx.spring_layout(G_plot, seed=42, k=1.5 / np.sqrt(len(nodes) + 1))
    except Exception:
        pos = nx.random_layout(G_plot, seed=42)

    nx.draw_networkx_edges(G_plot, pos, ax=ax, alpha=0.2,
                           edge_color=edge_colors, width=0.5)
    nx.draw_networkx_nodes(G_plot, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors=edgecolors,
                           linewidths=linewidths)

    # Label only top-10
    labels = {v: str(v) for v in nodes if v in top10}
    nx.draw_networkx_labels(G_plot, pos, labels=labels, ax=ax,
                            font_size=7, font_color="black", font_weight="bold")

    # Colorbar for SAASP score (when not community mode)
    if not use_community_coloring:
        sm = cm.ScalarMappable(cmap=cm.plasma, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="SAASP-adaptive Score (normalised)", shrink=0.7)

    ax.set_title(f"Network Influence Map — {dataset_name}\n"
                 f"(Gold outline = Top-{top_k} nodes)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe}_network_influence", DPI_PRINT)
    plt.close(fig)


# ─── 2. Enhanced Interactive HTML ────────────────────────────────────────────

def create_advanced_interactive_html(
    G: nx.Graph,
    node_list: list,
    saasp_scores: np.ndarray,
    all_methods: dict,
    dataset_name: str,
    output_dir: str,
    max_display_nodes: int = 600,
):
    """
    Enhanced PyVis interactive HTML:
      - Community-based node colouring
      - Node size ∝ SAASP score
      - Top-10 gold, top-50 orange
      - Hover tooltip shows all method scores + community ID
    """
    if not HAS_PYVIS:
        print(f"  [SKIP] pyvis not installed for {dataset_name}")
        return

    os.makedirs(output_dir, exist_ok=True)
    node_score_map = {node_list[i]: saasp_scores[i] for i in range(len(node_list))}
    ranking   = np.argsort(-saasp_scores)
    top10_set = set(node_list[i] for i in ranking[:10])
    top50_set = set(node_list[i] for i in ranking[:50])

    # Select subgraph
    if G.number_of_nodes() > max_display_nodes:
        seed_nodes = [node_list[i] for i in ranking[:5]]
        sub_nodes  = set()
        for s in seed_nodes:
            ego = nx.ego_graph(G, s, radius=2)
            sub_nodes.update(ego.nodes())
            if len(sub_nodes) >= max_display_nodes:
                break
        sub_G = G.subgraph(list(sub_nodes)[:max_display_nodes]).copy()
    else:
        sub_G = G

    # Community detection
    partition = _detect_communities(sub_G)
    n_comm    = max(partition.values()) + 1 if partition else 1

    # Normalise scores
    sub_scores = np.array([node_score_map.get(v, 0) for v in sub_G.nodes()])
    s_min, s_max = sub_scores.min(), sub_scores.max()
    norm_scores  = (sub_scores - s_min) / (s_max - s_min + 1e-12)

    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    net = Network(
        height="800px", width="100%",
        bgcolor="#111827", font_color="white",
        heading=f"SAASP Influence Network — {dataset_name}",
    )
    net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=120)
    net.set_options("""
    {
      "interaction": {"tooltipDelay": 100, "hideEdgesOnDrag": true},
      "physics": {"stabilization": {"iterations": 150}}
    }
    """)

    for idx, v in enumerate(sub_G.nodes()):
        score  = node_score_map.get(v, 0)
        n_sc   = float(norm_scores[idx])
        cid    = partition.get(v, 0)
        comm_c = _community_color(cid, n_comm)

        if v in top10_set:
            color  = "#FFD700"
            size   = 12 + n_sc * 45
            label  = f"★ {v}"
            border = "#FF8C00"
        elif v in top50_set:
            color  = "#f39c12"
            size   = 7 + n_sc * 25
            label  = str(v)
            border = "#e67e22"
        else:
            color  = comm_c
            size   = 4 + n_sc * 14
            label  = str(v)
            border = comm_c

        # Build tooltip with all method scores
        tooltip_lines = [f"<b>Node {v}</b>", f"Community: {cid}",
                         f"SAASP Score: <b>{score:.4f}</b>", "—"]
        for m, scores_arr in all_methods.items():
            if m == "SAASP-adaptive":
                continue
            try:
                idx_in_list = node_list.index(v)
                tooltip_lines.append(f"{m}: {scores_arr[idx_in_list]:.4f}")
            except (ValueError, IndexError):
                pass

        net.add_node(
            int(v), label=label, size=float(size),
            color={"background": color, "border": border,
                   "highlight": {"background": "#ffffff", "border": "#FFD700"}},
            title="<br>".join(tooltip_lines),
        )

    for u, v in sub_G.edges():
        net.add_edge(int(u), int(v), color="#374151", width=0.6)

    outpath = os.path.join(output_dir, f"{safe}_advanced_network.html")
    net.save_graph(outpath)
    print(f"  Advanced interactive HTML saved to: {outpath}")
