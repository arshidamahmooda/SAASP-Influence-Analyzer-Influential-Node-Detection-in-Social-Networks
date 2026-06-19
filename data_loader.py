"""
data_loader.py — Load and preprocess both graph datasets.

Handles gzipped edge-list files for Email-Eu-core and Facebook combined
networks. Returns NetworkX Graph objects with basic statistics.

Improvements (v2):
  - Largest connected component extraction ensures a consistent, connected
    graph structure before any downstream computation.
"""

import gzip
import os
import networkx as nx


def load_graph(filepath: str, name: str = "Graph") -> nx.Graph:
    """
    Load a graph from a gzipped edge-list file.

    The returned graph is the largest connected component (LCC) of the
    original edge list with self-loops removed.

    Parameters
    ----------
    filepath : str
        Path to the .gz edge-list file.
    name : str
        Human-readable name for the dataset.

    Returns
    -------
    G : nx.Graph
        Undirected graph — largest connected component.
    """
    with gzip.open(filepath, "rt") as fh:
        G_raw = nx.read_edgelist(fh, comments="#", nodetype=int)
    G_raw.name = name
    # Remove self-loops
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))

    # Keep only the largest connected component for a consistent structure
    lcc_nodes = max(nx.connected_components(G_raw), key=len)
    G = G_raw.subgraph(lcc_nodes).copy()
    G.name = name

    dropped = G_raw.number_of_nodes() - G.number_of_nodes()
    if dropped > 0:
        print(f"  [{name}] Kept LCC: {G.number_of_nodes()} nodes "
              f"({dropped} nodes in smaller components dropped)")

    return G


def print_graph_stats(G: nx.Graph) -> dict:
    """
    Print and return basic statistics for a graph.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_deg = sum(degrees) / n if n > 0 else 0
    max_deg = max(degrees) if degrees else 0
    stats = {
        "name": G.name,
        "nodes": n,
        "edges": m,
        "avg_degree": round(avg_deg, 2),
        "max_degree": max_deg,
    }
    print(f"\n{'='*50}")
    print(f"  Dataset: {stats['name']}")
    print(f"  Nodes:       {stats['nodes']:,}")
    print(f"  Edges:       {stats['edges']:,}")
    print(f"  Avg degree:  {stats['avg_degree']}")
    print(f"  Max degree:  {stats['max_degree']}")
    print(f"{'='*50}")
    return stats


def load_all_datasets(data_dir: str) -> list:
    """
    Load both Email-Eu-core and Facebook combined datasets.

    Parameters
    ----------
    data_dir : str
        Directory containing the .gz files.

    Returns
    -------
    datasets : list of (nx.Graph, dict)
        Each entry is (graph, stats_dict).
    """
    datasets = []
    files = [
        ("email-Eu-core.txt.gz", "Email-Eu-core"),
        ("facebook_combined.txt.gz", "Facebook"),
    ]
    for fname, name in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARNING] Dataset file not found: {fpath}")
            continue
        G = load_graph(fpath, name)
        stats = print_graph_stats(G)
        datasets.append((G, stats))
    return datasets
