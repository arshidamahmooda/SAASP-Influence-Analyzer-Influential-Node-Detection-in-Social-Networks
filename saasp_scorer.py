"""
saasp_scorer.py — Compute all 4 influence components + total SAASP score.

Components:
  CLI   — Local influence (degree ratio)
  CSLI  — Semi-local influence (neighbor influence with damping)
  CASPI — ASP-based influence (average shortest path disruption)
  CSI   — Semantic influence (from augmented graph GA)

Optimizations (v2):
  - L-hop neighbor cache: BFS precomputed once per node, reused in CSLI
  - CASPI parallelized via joblib Parallel + delayed
  - Normalization performed once globally after all components are ready
"""

import numpy as np
import networkx as nx
from scipy import sparse
from tqdm import tqdm
from joblib import Parallel, delayed

from local_subgraphs import extract_local_subgraph, extract_local_subgraph_without_node


_ASP_RNG = np.random.RandomState(42)
_MAX_ASP_SOURCES = 50  # max BFS sources for ASP estimation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fast_asp(G_sub: nx.Graph) -> float:
    """
    Estimate average shortest path length using sampled BFS.
    For disconnected graphs, use only the largest connected component.
    For subgraphs with >_MAX_ASP_SOURCES nodes, sample sources for speed.
    Returns 0 if the graph has fewer than 2 nodes.
    """
    nodes = list(G_sub.nodes())
    if len(nodes) < 2:
        return 0.0

    # Find largest connected component via BFS
    visited_global = set()
    largest_cc = []
    for start in nodes:
        if start in visited_global:
            continue
        cc = []
        queue = [start]
        visited_global.add(start)
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            cc.append(curr)
            for nb in G_sub.neighbors(curr):
                if nb not in visited_global:
                    visited_global.add(nb)
                    queue.append(nb)
        if len(cc) > len(largest_cc):
            largest_cc = cc

    if len(largest_cc) < 2:
        return 0.0

    cc_set = set(largest_cc)
    n_cc = len(largest_cc)

    # Sample sources if subgraph is large
    if n_cc <= _MAX_ASP_SOURCES:
        sources = largest_cc
    else:
        idx = _ASP_RNG.choice(n_cc, size=_MAX_ASP_SOURCES, replace=False)
        sources = [largest_cc[i] for i in idx]

    total_dist = 0
    n_pairs = 0

    for src in sources:
        dist = {src: 0}
        queue = [src]
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            d_curr = dist[curr]
            for nb in G_sub.neighbors(curr):
                if nb in cc_set and nb not in dist:
                    dist[nb] = d_curr + 1
                    queue.append(nb)
        total_dist += sum(dist.values())
        n_pairs += len(dist) - 1  # exclude self

    if n_pairs == 0:
        return 0.0
    return total_dist / n_pairs


def _build_lhop_cache(G: nx.Graph, node_list: list, L: int) -> dict:
    """
    Precompute L-hop BFS neighborhoods for all nodes in node_list.

    Returns
    -------
    cache : dict
        cache[v] = list of sets, where cache[v][l-1] = set of nodes
        at exactly hop-distance l from v (for l = 1..L).
    """
    cache = {}
    for v in node_list:
        layers = []
        visited = {v}
        frontier = {v}
        for _ in range(L):
            next_frontier = set()
            for u in frontier:
                for w in G.neighbors(u):
                    if w not in visited:
                        visited.add(w)
                        next_frontier.add(w)
            layers.append(next_frontier)
            frontier = next_frontier
        cache[v] = layers  # layers[l-1] = nodes at exactly distance l
    return cache


# ---------------------------------------------------------------------------
# Component: CLI
# ---------------------------------------------------------------------------

def compute_cli(G: nx.Graph, node_list: list) -> np.ndarray:
    """
    CLI(vi) = ki / kmax — Local influence via degree ratio.
    """
    degrees = np.array([G.degree(v) for v in node_list], dtype=np.float64)
    kmax = degrees.max()
    if kmax < 1:
        kmax = 1.0
    return degrees / kmax


# ---------------------------------------------------------------------------
# Component: CSLI  (uses precomputed L-hop cache)
# ---------------------------------------------------------------------------

def compute_csli(
    G: nx.Graph,
    node_list: list,
    GA: sparse.csr_matrix,
    node_to_idx: dict,
    L: int = 3,
    lam: float = 0.85,
    lhop_cache: dict = None,
) -> np.ndarray:
    """
    CSLI(vi) = sum over l=1..L of:
      lambda^l * sum over vj in Gamma_l(vi) of: (wij * kj) / d(vi,vj)

    where wij = GA edge weight if exists, else 1.
    d(vi,vj) = hop distance l.

    Parameters
    ----------
    lhop_cache : dict, optional
        Precomputed L-hop cache from _build_lhop_cache().
        If None, cache is built internally (less efficient for repeated calls).
    """
    if lhop_cache is None:
        lhop_cache = _build_lhop_cache(G, node_list, L)

    n = len(node_list)
    csli = np.zeros(n, dtype=np.float64)

    for idx, v in enumerate(node_list):
        total = 0.0
        layers = lhop_cache[v]  # layers[l-1] = set of nodes at distance l
        i_idx = node_to_idx.get(v)

        for l_minus1, hop_set in enumerate(layers):
            l = l_minus1 + 1
            lam_l = lam ** l
            for vj in hop_set:
                kj = G.degree(vj)
                j_idx = node_to_idx.get(vj)
                if i_idx is not None and j_idx is not None:
                    wij = GA[i_idx, j_idx]
                    if wij == 0:
                        wij = 1.0
                else:
                    wij = 1.0
                total += lam_l * (wij * kj) / l

        csli[idx] = total

    return csli


# ---------------------------------------------------------------------------
# Component: CASPI  (parallelized)
# ---------------------------------------------------------------------------

def _caspi_single(v, G: nx.Graph, L: int) -> float:
    """
    Compute CASPI for a single node v.
    Isolated helper for joblib parallelism.
    """
    GL = extract_local_subgraph(G, v, L)
    asp_full = _fast_asp(GL)

    if asp_full < 1e-12:
        return 0.0

    GL_minus = GL.copy()
    GL_minus.remove_node(v)
    asp_minus = _fast_asp(GL_minus)

    return abs(asp_full - asp_minus) / asp_full


def compute_caspi(
    G: nx.Graph,
    node_list: list,
    L: int = 3,
    verbose: bool = True,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    CASPI(vi) = |ASP[GL(vi)] - ASP[GL(vi) \\ vi]| / ASP[GL(vi)]

    Measures how much removing vi disrupts the average shortest path
    in its local subgraph.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers for joblib (-1 = all CPUs).
    """
    n = len(node_list)

    if verbose:
        print(f"  Running CASPI for {n} nodes (parallel, n_jobs={n_jobs})...")

    caspi_values = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_caspi_single)(v, G, L)
        for v in (tqdm(node_list, desc="Computing CASPI", unit="node") if verbose else node_list)
    )

    return np.array(caspi_values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Component: CSI
# ---------------------------------------------------------------------------

def compute_csi(
    GA: sparse.csr_matrix,
    node_list: list,
) -> np.ndarray:
    """
    CSI(vi) = sum over vj of GA(vi, vj) — Semantic influence from GA.
    """
    row_sums = np.array(GA.sum(axis=1)).flatten()
    return row_sums


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def compute_saasp_scores(
    G: nx.Graph,
    node_list: list,
    GA: sparse.csr_matrix,
    node_to_idx: dict,
    L: int = 3,
    xi: np.ndarray = None,
    verbose: bool = True,
    n_jobs: int = -1,
) -> dict:
    """
    Compute all 4 SAASP components and the total score.

    Optimizations applied:
      - L-hop cache built once and shared between CSLI and any other user.
      - CASPI is parallelized via joblib.
      - Normalization happens once globally after all components are ready.

    Parameters
    ----------
    G : nx.Graph
    node_list : list
    GA : sparse.csr_matrix
    node_to_idx : dict
    L : int
        Hop distance for local subgraphs (default 3).
    xi : np.ndarray, shape (4,)
        Weights [xi1, xi2, xi3, xi4]. Default [0.25]*4.
    verbose : bool
    n_jobs : int
        Parallel workers for CASPI (-1 = all CPUs).

    Returns
    -------
    results : dict with keys:
        'CLI', 'CSLI', 'CASPI', 'CSI' — raw component arrays
        'CLI_norm', 'CSLI_norm', 'CASPI_norm', 'CSI_norm' — normalized arrays
        'scores' — total SAASP score array
        'xi' — weights used
        'node_list' — ordered node list
    """
    if xi is None:
        xi = np.array([0.25, 0.25, 0.25, 0.25])

    print(f"  Computing CLI...")
    cli = compute_cli(G, node_list)

    # Build L-hop cache once — reused by CSLI (and optionally by callers)
    print(f"  Building L-hop neighbor cache (L={L})...")
    lhop_cache = _build_lhop_cache(G, node_list, L)

    print(f"  Computing CSLI (L={L}, cached)...")
    csli = compute_csli(G, node_list, GA, node_to_idx, L=L, lhop_cache=lhop_cache)

    print(f"  Computing CASPI (L={L}, parallel)...")
    caspi = compute_caspi(G, node_list, L=L, verbose=verbose, n_jobs=n_jobs)

    print(f"  Computing CSI...")
    csi = compute_csi(GA, node_list)

    # Normalize each component to [0, 1] — single global pass
    def _normalize(arr: np.ndarray) -> np.ndarray:
        cmin, cmax = arr.min(), arr.max()
        if cmax - cmin > 1e-12:
            return (arr - cmin) / (cmax - cmin)
        return np.zeros_like(arr)

    cli_norm   = _normalize(cli)
    csli_norm  = _normalize(csli)
    caspi_norm = _normalize(caspi)
    csi_norm   = _normalize(csi)

    # Total score
    scores = (
        xi[0] * cli_norm
        + xi[1] * csli_norm
        + xi[2] * caspi_norm
        + xi[3] * csi_norm
    )

    return {
        "CLI": cli,
        "CSLI": csli,
        "CASPI": caspi,
        "CSI": csi,
        "CLI_norm": cli_norm,
        "CSLI_norm": csli_norm,
        "CASPI_norm": caspi_norm,
        "CSI_norm": csi_norm,
        "scores": scores,
        "xi": xi.copy(),
        "node_list": node_list,
        "lhop_cache": lhop_cache,  # expose cache for downstream reuse
    }


def compute_saasp_from_components(
    cli_norm: np.ndarray,
    csli_norm: np.ndarray,
    caspi_norm: np.ndarray,
    csi_norm: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    """
    Compute total SAASP score from pre-computed normalized components.
    """
    return (
        xi[0] * cli_norm
        + xi[1] * csli_norm
        + xi[2] * caspi_norm
        + xi[3] * csi_norm
    )
