"""
feature_extractor.py — Node feature vectors (IMPROVED).

Original 3 features: [own_degree, avg_neighbor_degree, max_neighbor_degree]
Improved adds 3 more: [clustering_coefficient, k_core_index, local_efficiency]
v2 adds 1 more:       [pagerank]
All 7 features are normalized to [0,1] per column.
"""

import numpy as np
import networkx as nx
from tqdm import tqdm


_LE_MAX_SOURCES = 50  # max BFS sources for local efficiency estimation
_LE_RNG = np.random.RandomState(42)


def _local_efficiency(G: nx.Graph, node, neighbors: list) -> float:
    """
    Compute local efficiency for a node.

    local_efficiency(vi) = (1 / ki*(ki-1)) * sum of 1/d(vj,vk)
    for all pairs vj,vk in neighbors of vi.
    Returns 0 if degree < 2.
    For high-degree nodes, samples BFS sources for speed.
    """
    ki = len(neighbors)
    if ki < 2:
        return 0.0
    # Build subgraph of neighbors only (ego-graph minus the node itself)
    sub = G.subgraph(neighbors)

    # Sample BFS sources for high-degree nodes
    if ki <= _LE_MAX_SOURCES:
        sources = neighbors
    else:
        idx = _LE_RNG.choice(ki, size=_LE_MAX_SOURCES, replace=False)
        sources = [neighbors[i] for i in idx]

    total = 0.0
    n_pairs = 0
    for vj in sources:
        # BFS from vj within the subgraph
        distances = {vj: 0}
        queue = [vj]
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            d_curr = distances[curr]
            for nb in sub.neighbors(curr):
                if nb not in distances:
                    distances[nb] = d_curr + 1
                    queue.append(nb)
        # Sum 1/d for all other neighbors reachable from vj
        for vk in neighbors:
            if vk == vj:
                continue
            if vk in distances and distances[vk] > 0:
                total += 1.0 / distances[vk]
                n_pairs += 1

    if n_pairs == 0:
        return 0.0
    # Average efficiency across sampled pairs
    return total / n_pairs


def extract_features(G: nx.Graph, verbose: bool = True) -> tuple:
    """
    Extract 7-dimensional feature vector for every node.

    Features:
      0: own degree
      1: average neighbor degree
      2: max neighbor degree
      3: clustering coefficient
      4: k-core index
      5: local efficiency
      6: PageRank score

    Returns
    -------
    node_list : list
        Ordered list of node IDs.
    X : np.ndarray, shape (n, 7)
        Normalized feature matrix.
    X_raw : np.ndarray, shape (n, 7)
        Raw (unnormalized) feature matrix.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    n_features = 7
    X_raw = np.zeros((n, n_features), dtype=np.float64)

    # Precompute clustering coefficients, k-core numbers, and PageRank
    clustering   = nx.clustering(G)
    core_numbers = nx.core_number(G)
    pagerank     = nx.pagerank(G, alpha=0.85)

    iterator = enumerate(nodes)
    if verbose:
        iterator = tqdm(list(iterator), desc="Extracting features", unit="node")

    for idx, v in iterator:
        neighbors = list(G.neighbors(v))
        ki = len(neighbors)

        # Feature 0: own degree
        X_raw[idx, 0] = ki

        # Feature 1: average neighbor degree
        # Feature 2: max neighbor degree
        if ki > 0:
            neighbor_degs = [G.degree(u) for u in neighbors]
            X_raw[idx, 1] = np.mean(neighbor_degs)
            X_raw[idx, 2] = np.max(neighbor_degs)
        else:
            X_raw[idx, 1] = 0.0
            X_raw[idx, 2] = 0.0

        # Feature 3: clustering coefficient
        X_raw[idx, 3] = clustering[v]

        # Feature 4: k-core index
        X_raw[idx, 4] = core_numbers[v]

        # Feature 5: local efficiency
        X_raw[idx, 5] = _local_efficiency(G, v, neighbors)

        # Feature 6: PageRank
        X_raw[idx, 6] = pagerank[v]

    # Normalize each column to [0, 1]
    X = X_raw.copy()
    for col in range(n_features):
        col_min = X[:, col].min()
        col_max = X[:, col].max()
        if col_max - col_min > 1e-12:
            X[:, col] = (X[:, col] - col_min) / (col_max - col_min)
        else:
            X[:, col] = 0.0

    return nodes, X, X_raw
