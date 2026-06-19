"""
evaluator.py — Kendall's tau comparison vs DC/BC/CC/CI/PageRank/K-shell baselines.

Implements baseline centrality methods and comparison tools.

Improvements (v2):
  - Added PageRank baseline
  - Added K-shell (core number) baseline
  - Added precision@K metric
  - compute_all_baselines() now includes all 6 methods
"""

import numpy as np
import networkx as nx
from scipy.stats import kendalltau
import pandas as pd

from sir_model import run_sir_multiple


# ---------------------------------------------------------------------------
# Baseline centrality methods
# ---------------------------------------------------------------------------

def compute_degree_centrality(G: nx.Graph, node_list: list) -> np.ndarray:
    """DC: ki / (n-1)"""
    dc = nx.degree_centrality(G)
    return np.array([dc[v] for v in node_list], dtype=np.float64)


def compute_betweenness_centrality(G: nx.Graph, node_list: list) -> np.ndarray:
    """BC: normalized betweenness centrality."""
    bc = nx.betweenness_centrality(G, normalized=True)
    return np.array([bc[v] for v in node_list], dtype=np.float64)


def compute_closeness_centrality(G: nx.Graph, node_list: list) -> np.ndarray:
    """CC: closeness centrality."""
    cc = nx.closeness_centrality(G)
    return np.array([cc[v] for v in node_list], dtype=np.float64)


def compute_collective_influence(
    G: nx.Graph, node_list: list, r: int = 2
) -> np.ndarray:
    """
    CI(vi) = (ki-1) * sum over vj in Ball(vi, r) of (kj-1)

    Ball(vi, r) = nodes at exactly distance r from vi.
    """
    n = len(node_list)
    ci = np.zeros(n, dtype=np.float64)

    for idx, v in enumerate(node_list):
        ki = G.degree(v)
        # BFS to find nodes at exactly distance r
        visited = {v}
        frontier = {v}
        for _ in range(r):
            next_frontier = set()
            for u in frontier:
                for w in G.neighbors(u):
                    if w not in visited:
                        visited.add(w)
                        next_frontier.add(w)
            frontier = next_frontier
        # frontier now contains nodes at exactly distance r
        ball_sum = sum(G.degree(vj) - 1 for vj in frontier)
        ci[idx] = (ki - 1) * ball_sum

    return ci


def compute_pagerank(
    G: nx.Graph,
    node_list: list,
    alpha: float = 0.85,
) -> np.ndarray:
    """
    PageRank centrality.

    Parameters
    ----------
    alpha : float
        Damping parameter (default 0.85).

    Returns
    -------
    pr : np.ndarray
        PageRank score for each node in node_list order.
    """
    pr_dict = nx.pagerank(G, alpha=alpha)
    return np.array([pr_dict[v] for v in node_list], dtype=np.float64)


def compute_kshell(G: nx.Graph, node_list: list) -> np.ndarray:
    """
    K-shell decomposition (core number).

    Each node's score = its k-core index.

    Returns
    -------
    ks : np.ndarray
        K-shell (core number) for each node in node_list order.
    """
    core_nums = nx.core_number(G)
    return np.array([core_nums[v] for v in node_list], dtype=np.float64)


def compute_all_baselines(G: nx.Graph, node_list: list) -> dict:
    """
    Compute all baseline centrality methods.

    Returns dict mapping method name to score array.
    Includes: DC, BC, CC, CI, PageRank, K-shell.
    """
    print("  Computing baselines...")
    print("    DC (degree centrality)...")
    dc = compute_degree_centrality(G, node_list)

    print("    BC (betweenness centrality)...")
    bc = compute_betweenness_centrality(G, node_list)

    print("    CC (closeness centrality)...")
    cc = compute_closeness_centrality(G, node_list)

    print("    CI (collective influence, r=2)...")
    ci = compute_collective_influence(G, node_list, r=2)

    print("    PR (PageRank, alpha=0.85)...")
    pr = compute_pagerank(G, node_list)

    print("    KS (K-shell / core number)...")
    ks = compute_kshell(G, node_list)

    return {"DC": dc, "BC": bc, "CC": cc, "CI": ci, "PR": pr, "KS": ks}


# ---------------------------------------------------------------------------
# Precision@K metric
# ---------------------------------------------------------------------------

def precision_at_k(
    predicted_scores: np.ndarray,
    ground_truth_scores: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute Precision@K: fraction of top-K predicted nodes that are also
    in the top-K ground-truth nodes.

    Parameters
    ----------
    predicted_scores : np.ndarray
        Score array for the method being evaluated (same order as node_list).
    ground_truth_scores : np.ndarray
        Ground-truth score array (e.g., SIR single-node influence).
    k : int
        Number of top nodes to consider.

    Returns
    -------
    precision : float
        |top-K predicted ∩ top-K ground-truth| / K
    """
    k = min(k, len(predicted_scores))
    pred_top_k  = set(np.argsort(-predicted_scores)[:k])
    truth_top_k = set(np.argsort(-ground_truth_scores)[:k])
    return len(pred_top_k & truth_top_k) / k


def compute_precision_at_k_all(
    method_scores: dict,
    ground_truth_scores: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Compute Precision@K for every method in method_scores.

    Returns
    -------
    p_at_k : dict
        method_name -> precision@K float.
    """
    return {
        name: precision_at_k(scores, ground_truth_scores, k=k)
        for name, scores in method_scores.items()
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_methods(
    G: nx.Graph,
    node_list: list,
    method_scores: dict,
    infection_rate: float = 0.1,
    top_k: int = 10,
    steps: int = 20,
    num_sir_runs: int = 25,
    random_state: int = 42,
) -> dict:
    """
    Evaluate all methods via SIR F(t) curves, Kendall's tau, and Precision@K.

    Parameters
    ----------
    method_scores : dict
        Mapping method_name -> score array (same order as node_list).

    Returns
    -------
    results : dict with keys:
        'F_curves'        — method_name -> F(t) array
        'kendall_taus'    — method_name -> tau value
        'precision_at_k'  — method_name -> precision@top_k float
        'top_nodes'       — method_name -> list of top-K node IDs
        'sir_ground_truth'— SIR F_final per node (used for tau & precision)
        'sample_indices'  — indices of sampled nodes
    """
    results = {
        "F_curves": {},
        "kendall_taus": {},
        "precision_at_k": {},
        "top_nodes": {},
    }

    # For each method, get top-K nodes and run SIR
    for method_name, scores in method_scores.items():
        ranking = np.argsort(-scores)
        top_indices = ranking[:top_k]
        top_nodes = [node_list[i] for i in top_indices]
        results["top_nodes"][method_name] = top_nodes

        # Run SIR with top-K nodes as seeds
        F_avg = run_sir_multiple(
            G, top_nodes,
            infection_rate=infection_rate,
            recovery_rate=1.0,
            steps=steps,
            num_runs=num_sir_runs,
            random_state=random_state,
        )
        results["F_curves"][method_name] = F_avg

    # Compute Kendall's tau and Precision@K using per-node SIR influence
    from sir_model import sir_node_influence

    print("  Computing SIR ground-truth influence for Kendall's tau...")
    n = len(node_list)
    rng = np.random.RandomState(random_state)
    if n > 500:
        sample_size = min(200, n)
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(n)

    sample_nodes = [node_list[i] for i in sample_idx]
    sir_scores = sir_node_influence(
        G, sample_nodes,
        infection_rate=infection_rate,
        num_runs=10,
        random_state=random_state,
    )

    results["sir_ground_truth"] = sir_scores
    results["sample_indices"] = sample_idx

    # Build a full-length ground-truth array for Precision@K
    gt_full = np.zeros(n, dtype=np.float64)
    gt_full[sample_idx] = sir_scores

    for method_name, scores in method_scores.items():
        method_sample = scores[sample_idx]
        tau, pval = kendalltau(method_sample, sir_scores)
        if np.isnan(tau):
            tau = 0.0
        results["kendall_taus"][method_name] = tau

        # Precision@K on the sampled subspace (consistent with tau calc)
        p_k = precision_at_k(method_sample, sir_scores, k=min(top_k, len(sample_idx)))
        results["precision_at_k"][method_name] = p_k

    return results


# ---------------------------------------------------------------------------
# Kendall tau vs lambda sweep
# ---------------------------------------------------------------------------

def kendall_tau_vs_lambda(
    G: nx.Graph,
    node_list: list,
    method_scores: dict,
    lambda_values: np.ndarray = None,
    steps: int = 20,
    num_sir_runs: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Compute Kendall's tau vs infection probability lambda for all methods.

    Returns dict: method_name -> array of tau values for each lambda.
    """
    if lambda_values is None:
        lambda_values = np.linspace(0.01, 0.1, 10)

    from sir_model import sir_node_influence

    n = len(node_list)
    rng = np.random.RandomState(random_state)
    if n > 500:
        sample_size = min(150, n)
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(n)

    sample_nodes = [node_list[i] for i in sample_idx]

    tau_results = {name: [] for name in method_scores}

    for lam in lambda_values:
        print(f"    lambda = {lam:.3f}")
        sir_scores = sir_node_influence(
            G, sample_nodes,
            infection_rate=lam,
            num_runs=num_sir_runs,
            random_state=random_state,
        )
        for method_name, scores in method_scores.items():
            method_sample = scores[sample_idx]
            tau, _ = kendalltau(method_sample, sir_scores)
            if np.isnan(tau):
                tau = 0.0
            tau_results[method_name].append(tau)

    return tau_results, lambda_values


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict, dataset_name: str):
    """
    Print a formatted comparison table of all methods including Precision@K.
    """
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE — {dataset_name}")
    print(f"{'='*80}")

    rows = []
    for method in results["kendall_taus"]:
        tau   = results["kendall_taus"][method]
        p_k   = results["precision_at_k"].get(method, float("nan"))
        top   = results["top_nodes"][method]
        f_final = results["F_curves"][method][-1]
        rows.append({
            "Method":     method,
            "Kendall τ":  f"{tau:.4f}",
            "Precision@K": f"{p_k:.4f}",
            "F(t=final)": f"{f_final:.1f}",
            "Top-5 Nodes": str(top[:5]),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    # Print top-10 nodes per method
    print(f"  TOP-10 NODES PER METHOD — {dataset_name}")
    print(f"  {'-'*60}")
    for method in results["top_nodes"]:
        top10 = results["top_nodes"][method][:10]
        print(f"  {method:20s}: {top10}")
    print()
