"""
adaptive_weights.py — Learn optimal ξ1–ξ4 via gradient descent.

Replaces the hardcoded xi = [0.25, 0.25, 0.25, 0.25] with learned weights
that maximize Kendall's tau correlation between SAASP ranking and SIR
ground-truth ranking.

Uses scipy.optimize.minimize with L-BFGS-B method.

Improvements (v2):
  - L2 regularization term in loss function for stability:
      loss = -tau + lambda_reg * sum(xi^2)
  - Learned weights are saved to disk (learned_weights.npy)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau
from tqdm import tqdm

from sir_model import sir_node_influence
from saasp_scorer import compute_saasp_from_components


def _xi_to_weights(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw unconstrained values to valid weights (positive, sum to 1)
    using softmax transformation.
    """
    exp_vals = np.exp(raw - raw.max())  # numerical stability
    return exp_vals / exp_vals.sum()


def learn_adaptive_weights(
    G,
    node_list: list,
    cli_norm: np.ndarray,
    csli_norm: np.ndarray,
    caspi_norm: np.ndarray,
    csi_norm: np.ndarray,
    sample_fraction: float = 0.2,
    infection_rate: float = 0.1,
    recovery_rate: float = 1.0,
    steps: int = 20,
    num_sir_runs: int = 10,
    random_state: int = 42,
    lambda_reg: float = 1e-3,
    save_weights: bool = True,
    weights_path: str = "learned_weights.npy",
    verbose: bool = True,
) -> dict:
    """
    Learn optimal xi weights by maximizing Kendall's tau with SIR ground truth.

    Parameters
    ----------
    G : nx.Graph
    node_list : list
        Full ordered node list.
    cli_norm, csli_norm, caspi_norm, csi_norm : np.ndarray
        Normalized SAASP component arrays (full graph).
    sample_fraction : float
        Fraction of nodes to sample for training (default 0.2).
    lambda_reg : float
        L2 regularization coefficient applied to the softmax weights.
        Higher values keep weights more uniform; default 1e-3.
    save_weights : bool
        If True, save learned weights to `weights_path` as .npy file.
    weights_path : str
        Path to save the learned weights array.

    Returns
    -------
    result : dict with keys:
        'xi_learned'         — learned weights (shape 4,)
        'tau_learned'        — Kendall's tau with learned weights (on sample)
        'tau_fixed'          — Kendall's tau with fixed weights (on sample)
        'sir_influence'      — SIR influence scores for sampled nodes
        'sample_indices'     — indices of sampled nodes
        'optimization_result'— raw scipy result object
    """
    rng = np.random.RandomState(random_state)
    n = len(node_list)

    # Sample nodes
    sample_size = max(10, int(n * sample_fraction))
    sample_indices = rng.choice(n, size=sample_size, replace=False)
    sample_indices.sort()

    sample_nodes = [node_list[i] for i in sample_indices]
    cli_sample   = cli_norm[sample_indices]
    csli_sample  = csli_norm[sample_indices]
    caspi_sample = caspi_norm[sample_indices]
    csi_sample   = csi_norm[sample_indices]

    # Get SIR ground truth for sampled nodes
    if verbose:
        print(f"  Running SIR simulations for {sample_size} sampled nodes...")

    sir_influence = sir_node_influence(
        G, sample_nodes,
        infection_rate=infection_rate,
        recovery_rate=recovery_rate,
        steps=steps,
        num_runs=num_sir_runs,
        random_state=random_state,
    )

    # Define loss function: negative Kendall's tau + L2 regularization
    def loss_fn(raw_xi):
        xi = _xi_to_weights(raw_xi)
        scores = compute_saasp_from_components(
            cli_sample, csli_sample, caspi_sample, csi_sample, xi
        )
        tau, _ = kendalltau(scores, sir_influence)
        if np.isnan(tau):
            tau = 0.0
        # L2 penalty: encourages smoother weight distribution
        l2_penalty = lambda_reg * float(np.sum(xi ** 2))
        return -tau + l2_penalty  # minimize negative tau + penalty

    # Optimize using L-BFGS-B (raw unconstrained space; softmax handles constraints)
    initial_raw = np.zeros(4)  # softmax(0,0,0,0) = [0.25]*4

    if verbose:
        print(f"  Optimizing weights via L-BFGS-B (λ_reg={lambda_reg})...")

    result = minimize(
        loss_fn,
        initial_raw,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-8},
    )

    xi_learned = _xi_to_weights(result.x)

    # Save learned weights to disk for reproducibility
    if save_weights:
        np.save(weights_path, xi_learned)
        if verbose:
            print(f"  Learned weights saved to: {weights_path}")

    # Compute taus for comparison
    scores_fixed = compute_saasp_from_components(
        cli_sample, csli_sample, caspi_sample, csi_sample,
        np.array([0.25, 0.25, 0.25, 0.25]),
    )
    tau_fixed, _ = kendalltau(scores_fixed, sir_influence)

    scores_learned = compute_saasp_from_components(
        cli_sample, csli_sample, caspi_sample, csi_sample, xi_learned,
    )
    tau_learned, _ = kendalltau(scores_learned, sir_influence)

    if verbose:
        print(f"  Learned xi: [{', '.join(f'{x:.4f}' for x in xi_learned)}]")
        print(f"  Tau (fixed):    {tau_fixed:.4f}")
        print(f"  Tau (adaptive): {tau_learned:.4f}")
        improvement = ((tau_learned - tau_fixed) / abs(tau_fixed) * 100
                       if abs(tau_fixed) > 1e-8 else 0.0)
        print(f"  Improvement:    {improvement:+.2f}%")

    return {
        "xi_learned": xi_learned,
        "tau_learned": tau_learned,
        "tau_fixed": tau_fixed,
        "sir_influence": sir_influence,
        "sample_indices": sample_indices,
        "optimization_result": result,
    }
