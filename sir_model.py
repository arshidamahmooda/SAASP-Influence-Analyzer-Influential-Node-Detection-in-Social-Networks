"""
sir_model.py — SIR epidemic simulation for evaluation.

Discrete-time SIR model:
  - infection_rate (lambda): probability of infecting each susceptible neighbor
  - recovery_rate (mu): probability of recovering per step
  - Seed top-K nodes as initially infected
  - Track F(t) = infected + recovered at each step
  - Run multiple simulations and average

Additional metrics (v2):
  - final_infected_ratio(F, total_nodes)
  - time_to_saturation(F, threshold)
  - peak_infection_time(F)
"""

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def sir_simulation(
    G: nx.Graph,
    seed_nodes: list,
    infection_rate: float = 0.1,
    recovery_rate: float = 1.0,
    steps: int = 20,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """
    Run a single SIR simulation.

    Parameters
    ----------
    G : nx.Graph
    seed_nodes : list
        Initially infected nodes.
    infection_rate : float
        Probability of transmitting infection to each susceptible neighbor.
    recovery_rate : float
        Probability of recovery per step.
    steps : int
        Number of time steps.
    rng : np.random.RandomState
        Random state for reproducibility.

    Returns
    -------
    F : np.ndarray, shape (steps+1,)
        F(t) = number of infected + recovered nodes at each time step.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # States: 0=Susceptible, 1=Infected, 2=Recovered
    state = {}
    for node in G.nodes():
        state[node] = 0
    for node in seed_nodes:
        if node in state:
            state[node] = 1

    F = np.zeros(steps + 1, dtype=np.float64)
    infected_count = sum(1 for s in state.values() if s == 1)
    recovered_count = 0
    F[0] = infected_count + recovered_count

    for t in range(1, steps + 1):
        new_infections = set()
        new_recoveries = set()

        # Infection step
        for node in list(G.nodes()):
            if state[node] == 1:  # Infected
                for neighbor in G.neighbors(node):
                    if state[neighbor] == 0:  # Susceptible
                        if rng.random() < infection_rate:
                            new_infections.add(neighbor)

        # Recovery step
        for node in list(G.nodes()):
            if state[node] == 1:  # Infected
                if rng.random() < recovery_rate:
                    new_recoveries.add(node)

        # Apply state changes
        for node in new_infections:
            if state[node] == 0:  # still susceptible (not yet recovered)
                state[node] = 1
        for node in new_recoveries:
            state[node] = 2

        infected_count = sum(1 for s in state.values() if s == 1)
        recovered_count = sum(1 for s in state.values() if s == 2)
        F[t] = infected_count + recovered_count

    return F


def run_sir_multiple(
    G: nx.Graph,
    seed_nodes: list,
    infection_rate: float = 0.1,
    recovery_rate: float = 1.0,
    steps: int = 20,
    num_runs: int = 25,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run SIR simulation multiple times and return the average F(t) curve.

    Returns
    -------
    F_avg : np.ndarray, shape (steps+1,)
        Averaged F(t) over all runs.
    """
    rng = np.random.RandomState(random_state)
    all_F = np.zeros((num_runs, steps + 1), dtype=np.float64)

    for run in range(num_runs):
        seed = rng.randint(0, 2**31)
        run_rng = np.random.RandomState(seed)
        all_F[run] = sir_simulation(
            G, seed_nodes, infection_rate, recovery_rate, steps, run_rng
        )

    return all_F.mean(axis=0)


def sir_node_influence(
    G: nx.Graph,
    node_list: list,
    infection_rate: float = 0.1,
    recovery_rate: float = 1.0,
    steps: int = 20,
    num_runs: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute ground-truth influence for each node by running SIR
    with that node as the sole initial seed.

    Returns F_final = F(t=steps) for each node, averaged over num_runs.
    Used as ground truth for adaptive weight learning.
    """
    rng = np.random.RandomState(random_state)
    n = len(node_list)
    influence = np.zeros(n, dtype=np.float64)

    for idx, node in enumerate(node_list):
        total = 0.0
        for run in range(num_runs):
            seed = rng.randint(0, 2**31)
            run_rng = np.random.RandomState(seed)
            F = sir_simulation(
                G, [node], infection_rate, recovery_rate, steps, run_rng
            )
            total += F[-1]  # F(t=steps)
        influence[idx] = total / num_runs

    return influence


# ---------------------------------------------------------------------------
# Additional SIR metrics (v2)
# ---------------------------------------------------------------------------

def final_infected_ratio(F: np.ndarray, total_nodes: int) -> float:
    """
    Return the fraction of total nodes that were infected/recovered
    by the end of the simulation.

    Parameters
    ----------
    F : np.ndarray
        F(t) curve from sir_simulation or run_sir_multiple.
    total_nodes : int
        Total number of nodes in the graph.

    Returns
    -------
    ratio : float
        F(t_final) / total_nodes.  Range [0, 1].
    """
    if total_nodes == 0:
        return 0.0
    return float(F[-1]) / total_nodes


def time_to_saturation(F: np.ndarray, threshold: float = 0.95) -> int:
    """
    Return the first time step at which F(t) reaches `threshold` fraction
    of its final value F(t_final).

    If F never reaches the threshold, returns the last time step index.

    Parameters
    ----------
    F : np.ndarray
        F(t) curve.
    threshold : float
        Fraction of F_final to be considered saturated (default 0.95).

    Returns
    -------
    t_sat : int
        Index of the first time step where F[t] >= threshold * F[-1].
    """
    f_final = F[-1]
    if f_final < 1e-12:
        return len(F) - 1
    target = threshold * f_final
    for t, val in enumerate(F):
        if val >= target:
            return t
    return len(F) - 1


def peak_infection_time(F: np.ndarray) -> int:
    """
    Return the time step at which the *incremental* increase in F(t)
    is largest — i.e., peak new infections per step.

    For a monotone F(t) this is the step with maximum ΔF.

    Parameters
    ----------
    F : np.ndarray
        F(t) curve.

    Returns
    -------
    t_peak : int
        Time step index with the highest ΔF = F[t] - F[t-1].
        Returns 0 if the curve has fewer than 2 entries.
    """
    if len(F) < 2:
        return 0
    delta = np.diff(F)          # shape (steps,)
    return int(np.argmax(delta)) + 1  # +1 because diff shifts by 1
