"""
experiments.py — Advanced experimental modules for research-grade evaluation.

Provides three independent experiment types:
  1. node_removal_experiment  — Remove top-K SAASP nodes and compare SIR spread
  2. ablation_study           — Evaluate SAASP with progressively more components
  3. metric_comparison        — Compare all methods on Kendall τ, Precision@K, F(T)

All functions return structured result dicts and can be called from main.py
or independently.
"""

import numpy as np
import networkx as nx
import pandas as pd

from sir_model import run_sir_multiple, sir_node_influence, final_infected_ratio
from saasp_scorer import compute_saasp_from_components
from evaluator import precision_at_k
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# 1. Node Removal Experiment
# ---------------------------------------------------------------------------

def node_removal_experiment(
    G: nx.Graph,
    node_list: list,
    saasp_scores: np.ndarray,
    infection_rate: float = 0.1,
    top_k: int = 10,
    steps: int = 20,
    num_sir_runs: int = 20,
    random_state: int = 42,
) -> dict:
    """
    Compare SIR spreading before and after removing top-K SAASP nodes.

    Procedure:
      1. Run SIR on the original graph G with a random seed set.
      2. Remove top-K SAASP nodes from G.
      3. Re-run SIR on G' with the same seed set.
      4. Compute spread reduction = (F_orig - F_reduced) / F_orig.

    Parameters
    ----------
    G : nx.Graph
    node_list : list
    saasp_scores : np.ndarray
        SAASP adaptive score for each node in node_list.
    top_k : int
        Number of top-SAASP nodes to remove.
    steps : int
    num_sir_runs : int
    random_state : int

    Returns
    -------
    result : dict
        'F_original'       — averaged F(t) on G
        'F_after_removal'  — averaged F(t) on G' (top-K nodes removed)
        'removed_nodes'    — list of removed node IDs
        'spread_reduction' — fraction reduction in final F(t)
        'ratio_original'   — final infected ratio on G
        'ratio_reduced'    — final infected ratio on G'
    """
    rng = np.random.RandomState(random_state)
    n = G.number_of_nodes()

    # Pick random seeds (not from top-K, to measure organic spread)
    ranking    = np.argsort(-saasp_scores)
    top_k_idx  = ranking[:top_k]
    top_k_nodes = [node_list[i] for i in top_k_idx]
    top_k_set   = set(top_k_nodes)

    # Sample seed nodes outside the top-K
    non_top_nodes = [v for v in node_list if v not in top_k_set]
    seed_size  = max(1, min(5, len(non_top_nodes)))
    seed_nodes = list(rng.choice(non_top_nodes, size=seed_size, replace=False))

    print(f"  [NodeRemoval] Seed nodes: {seed_nodes}")
    print(f"  [NodeRemoval] Removing top-{top_k} nodes: {top_k_nodes}")

    # Run SIR on original graph
    F_original = run_sir_multiple(
        G, seed_nodes,
        infection_rate=infection_rate,
        steps=steps,
        num_runs=num_sir_runs,
        random_state=random_state,
    )

    # Build reduced graph (remove top-K nodes)
    G_reduced = G.copy()
    for v in top_k_nodes:
        if G_reduced.has_node(v):
            G_reduced.remove_node(v)

    # Adjust seed nodes to only those still in G_reduced
    seed_nodes_reduced = [v for v in seed_nodes if G_reduced.has_node(v)]
    if not seed_nodes_reduced:
        seed_nodes_reduced = list(rng.choice(
            list(G_reduced.nodes()), size=1, replace=False
        ))

    F_reduced = run_sir_multiple(
        G_reduced, seed_nodes_reduced,
        infection_rate=infection_rate,
        steps=steps,
        num_runs=num_sir_runs,
        random_state=random_state,
    )

    f_orig  = float(F_original[-1])
    f_red   = float(F_reduced[-1])
    spread_reduction = (f_orig - f_red) / f_orig if f_orig > 1e-12 else 0.0

    ratio_original = final_infected_ratio(F_original, n)
    ratio_reduced  = final_infected_ratio(F_reduced, G_reduced.number_of_nodes())

    print(f"  [NodeRemoval] F_final original: {f_orig:.2f} "
          f"({ratio_original:.2%} of nodes)")
    print(f"  [NodeRemoval] F_final reduced:  {f_red:.2f} "
          f"({ratio_reduced:.2%} of remaining nodes)")
    print(f"  [NodeRemoval] Spread reduction: {spread_reduction:.2%}")

    return {
        "F_original":       F_original,
        "F_after_removal":  F_reduced,
        "removed_nodes":    top_k_nodes,
        "spread_reduction": spread_reduction,
        "ratio_original":   ratio_original,
        "ratio_reduced":    ratio_reduced,
    }


# ---------------------------------------------------------------------------
# 2. Ablation Study
# ---------------------------------------------------------------------------

def ablation_study(
    G: nx.Graph,
    node_list: list,
    cli_norm: np.ndarray,
    csli_norm: np.ndarray,
    caspi_norm: np.ndarray,
    csi_norm: np.ndarray,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    top_k: int = 10,
    infection_rate: float = 0.1,
    steps: int = 20,
    num_sir_runs: int = 20,
    random_state: int = 42,
) -> dict:
    """
    Ablation study: progressively add SAASP components and measure performance.

    Configurations evaluated:
      - 'CLI only'      — equal weight only on CLI (xi=[1,0,0,0])
      - 'CLI+CSLI'      — equal weights on CLI and CSLI (xi=[0.5,0.5,0,0])
      - 'CLI+CSLI+CASPI'— equal on first 3 (xi=[1/3,1/3,1/3,0])
      - 'Full SAASP'    — all 4 equally (xi=[0.25]*4)

    Parameters
    ----------
    G : nx.Graph
    node_list : list
    cli_norm, csli_norm, caspi_norm, csi_norm : np.ndarray
        Precomputed normalized component arrays.
    sir_ground_truth : np.ndarray
        SIR influence scores for `sample_indices` nodes.
    sample_indices : np.ndarray
        Indices into node_list for the sample used in evaluation.
    top_k : int
        Number of top nodes used for SIR seeding.

    Returns
    -------
    result : dict
        'configs'          — list of config names
        'kendall_taus'     — dict config_name -> tau
        'precision_at_k'   — dict config_name -> p@k
        'F_curves'         — dict config_name -> F(t) array
        'table'            — pandas DataFrame summary
    """
    configs = {
        "CLI only":       np.array([1.0, 0.0, 0.0, 0.0]),
        "CLI+CSLI":       np.array([0.5, 0.5, 0.0, 0.0]),
        "CLI+CSLI+CASPI": np.array([1/3, 1/3, 1/3, 0.0]),
        "Full SAASP":     np.array([0.25, 0.25, 0.25, 0.25]),
    }

    taus     = {}
    p_at_k   = {}
    F_curves = {}

    for name, xi in configs.items():
        scores = compute_saasp_from_components(
            cli_norm, csli_norm, caspi_norm, csi_norm, xi
        )

        # Kendall tau on sample
        sample_scores = scores[sample_indices]
        tau, _ = kendalltau(sample_scores, sir_ground_truth)
        taus[name] = float(np.nan_to_num(tau))

        # Precision@K on sample
        p_at_k[name] = precision_at_k(sample_scores, sir_ground_truth, k=top_k)

        # SIR F(t) with top-K seeds
        ranking  = np.argsort(-scores)[:top_k]
        top_nodes = [node_list[i] for i in ranking]
        F = run_sir_multiple(
            G, top_nodes,
            infection_rate=infection_rate,
            steps=steps,
            num_runs=num_sir_runs,
            random_state=random_state,
        )
        F_curves[name] = F

    # Build summary DataFrame
    rows = []
    for name in configs:
        rows.append({
            "Config":      name,
            "Kendall τ":   f"{taus[name]:.4f}",
            "Precision@K": f"{p_at_k[name]:.4f}",
            "F(t_final)":  f"{F_curves[name][-1]:.2f}",
        })
    table = pd.DataFrame(rows)

    print("\n  ABLATION STUDY RESULTS")
    print("  " + "-" * 60)
    print(table.to_string(index=False))
    print()

    return {
        "configs":        list(configs.keys()),
        "xi_configs":     configs,
        "kendall_taus":   taus,
        "precision_at_k": p_at_k,
        "F_curves":       F_curves,
        "table":          table,
    }


# ---------------------------------------------------------------------------
# 3. Metric Comparison
# ---------------------------------------------------------------------------

def metric_comparison(
    G: nx.Graph,
    node_list: list,
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    infection_rate: float = 0.1,
    top_k: int = 10,
    steps: int = 20,
    num_sir_runs: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Side-by-side comparison of all methods on:
      - Kendall tau
      - Precision@K
      - F(t_final) when seeding top-K nodes
      - final_infected_ratio
      - time_to_saturation (95%)
      - peak_infection_time

    Parameters
    ----------
    method_scores : dict
        method_name -> full-length score array.
    sir_ground_truth : np.ndarray
        SIR scores for the sampled nodes.
    sample_indices : np.ndarray
        Indices into node_list corresponding to sir_ground_truth.

    Returns
    -------
    df : pd.DataFrame
        One row per method with all metrics.
    """
    from sir_model import (
        final_infected_ratio as fir,
        time_to_saturation,
        peak_infection_time,
    )

    rows = []
    n_total = G.number_of_nodes()

    for method_name, scores in method_scores.items():
        sample_scores = scores[sample_indices]
        tau, _ = kendalltau(sample_scores, sir_ground_truth)
        tau = float(np.nan_to_num(tau))
        pk  = precision_at_k(sample_scores, sir_ground_truth, k=top_k)

        # Top-K seeding SIR
        top_idx   = np.argsort(-scores)[:top_k]
        top_nodes = [node_list[i] for i in top_idx]
        F = run_sir_multiple(
            G, top_nodes,
            infection_rate=infection_rate,
            steps=steps,
            num_runs=num_sir_runs,
            random_state=random_state,
        )

        rows.append({
            "Method":           method_name,
            "Kendall τ":        round(tau, 4),
            "Precision@K":      round(pk, 4),
            "F(t_final)":       round(float(F[-1]), 2),
            "Infected ratio":   round(fir(F, n_total), 4),
            "T_saturation(95%)":time_to_saturation(F, threshold=0.95),
            "T_peak_infect":    peak_infection_time(F),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Kendall τ", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Utility: save scores to CSV
# ---------------------------------------------------------------------------

def save_scores_to_csv(
    node_list: list,
    method_scores: dict,
    filepath: str,
):
    """
    Export all method scores per node to a CSV file.

    Parameters
    ----------
    node_list : list
        Ordered list of node IDs.
    method_scores : dict
        method_name -> score array.
    filepath : str
        Output CSV path.
    """
    data = {"node_id": node_list}
    for method_name, scores in method_scores.items():
        data[method_name] = scores
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"  Scores saved to: {filepath}")
    return df
