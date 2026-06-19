# SAASP Research-Grade Refactor — Implementation Plan

## Overview

Full upgrade of the SAASP centrality codebase from its current 9-file layout into a
clean, modular, research-quality Python project. All existing logic is preserved and
extended; every module is rewritten with proper docstrings, type hints, NumPy
vectorization, and caching.

---

## Proposed Changes

### New File Structure

```
saasp_project/
├── data_loader.py        [MODIFY]  minor quality + type hint improvements
├── saasp_model.py        [NEW]     replaces saasp_scorer.py + local_subgraphs.py + augmented_graph.py
├── weight_learning.py    [NEW]     replaces adaptive_weights.py; adds grid search
├── sir_simulation.py     [NEW]     replaces sir_model.py; vectorized + new metrics
├── evaluation.py         [NEW]     replaces evaluator.py; new baselines + precision_at_k
├── experiment.py         [NEW]     pipeline + node-removal experiment + CSV export
├── visualization.py      [NEW]     replaces visualizer.py; plotting separated from logic
├── main.py               [MODIFY]  clean orchestration entry point
├── requirements.txt      [MODIFY]  add scipy, tqdm, pandas, pyvis, networkx
│
│── [OLD FILES — kept for safety but no longer imported]
│   saasp_scorer.py
│   adaptive_weights.py
│   sir_model.py
│   evaluator.py
│   feature_extractor.py
│   augmented_graph.py
│   local_subgraphs.py
│   visualizer.py
```

---

### Module Details

#### [NEW] `saasp_model.py`
Merges `saasp_scorer.py`, `augmented_graph.py`, `local_subgraphs.py`, `feature_extractor.py`.

Key improvements:
- `compute_l_hop_neighbors(G, node_list, L)` — precomputes **all** L-hop neighbor sets once, cached in a dict; reused by both `compute_csli()` and `compute_caspi()`, eliminating repeated BFS.
- `compute_cli()` — fully vectorized via `np.array()`.
- `compute_csli()` — reuses cached hop layers; sparse GA access via prebuilt `node_to_idx`.
- `compute_caspi()` — reuses cached subgraph node sets; only builds the subgraph once per node.
- `compute_csi()` — unchanged (already O(nnz)).
- `normalize_minmax()` — shared utility, avoids repeated inline normalization.
- `compute_saasp_scores()` — master function returning full result dict.
- `compute_saasp_from_components()` — lightweight score combiner.
- Augmented graph construction consolidated inside this module.
- Feature extraction (6-dim) also moved here with BFS-based local efficiency.

#### [NEW] `weight_learning.py`
Replaces `adaptive_weights.py`. Main public API:

```python
optimize_weights(
    G, node_list,
    cli_norm, csli_norm, caspi_norm, csi_norm,
    method="grid",           # "grid" | "lbfgs"
    sample_fraction=0.2,
    infection_rate=0.1,
    recovery_rate=1.0,
    steps=20,
    num_sir_runs=10,
    random_state=42,
    verbose=True,
) -> dict
```

- `method="grid"`: exhaustive 4-D grid search over ξ values in `{0.1, 0.2, …, 0.7}` constrained to sum=1. Fast, interpretable, no gradient needed.
- `method="lbfgs"`: L-BFGS-B with softmax reparametrization (existing approach, kept clean).
- Both paths return the same result dict: `{xi_learned, tau_learned, tau_fixed, sir_influence, sample_indices}`.
- Helper `_xi_to_weights(raw)` shared between methods.

#### [NEW] `sir_simulation.py`
Replaces `sir_model.py`. Vectorized NumPy SIR with adjacency array for speed.

New public API functions:
```python
sir_simulation(G, seed_nodes, beta, gamma, steps, rng) -> dict
    # returns {"S": array, "I": array, "R": array, "F": array}

run_sir_multiple(G, seed_nodes, beta, gamma, steps, num_runs, random_state) -> dict
    # returns time-series mean + std arrays

sir_node_influence(G, node_list, beta, gamma, steps, num_runs, random_state) -> np.ndarray

final_infection_ratio(F_curve, N) -> float
    # F(T) / N — fraction of population ever infected

time_to_saturation(F_curve, threshold=0.95) -> int
    # first t where F(t) >= threshold * F(T)

precision_at_k(ranked_list, ground_truth_top_k) -> float
    # |intersection| / k
```

Internal vectorization: replace per-node Python loops with NumPy state arrays
(`S_arr`, `I_arr`, `R_arr`) and adjacency list lookups using prebuilt `adj` dict.

#### [NEW] `evaluation.py`
Replaces `evaluator.py`. Full baseline suite + new metrics.

New baselines added vs current code:
| Method | Status |
|---|---|
| Degree Centrality (DC) | ✅ existing |
| Betweenness Centrality (BC) | ✅ existing |
| Closeness Centrality (CC) | ✅ existing |
| Collective Influence (CI) | ✅ existing |
| **PageRank** | 🆕 new |
| **K-Shell (k-core)** | 🆕 new |

New metric functions added:
- `final_infection_ratio()` — delegates to `sir_simulation.py`
- `time_to_saturation()` — delegates to `sir_simulation.py`
- `precision_at_k()` — delegates to `sir_simulation.py`
- `evaluate_methods()` — now returns extended results dict including all new metrics.
- `print_comparison_table()` — extended to show PageRank, K-Shell columns.

#### [NEW] `experiment.py`
New module. Public API:

```python
run_experiment_pipeline(G, stats, results_dir, config=None) -> dict
    # Full end-to-end pipeline: features → GA → SAASP → weights → eval → plots → CSV

remove_top_k_and_simulate(G, scores, node_list, k, beta, gamma, steps, num_runs) -> dict
    # Returns {"before": F_avg, "after": F_avg, "removed_nodes": list}
    # Compares spread before vs after removing top-K nodes

save_results_csv(eval_results, output_path)
    # Saves method comparison table as CSV
```

#### [NEW] `visualization.py`
Replaces `visualizer.py`. **Plotting is strictly separated from logic.**

- All plot functions take pre-computed data; no computation inside.
- New: `plot_node_removal_comparison(ax, before_F, after_F, title)`.
- New: `plot_precision_at_k_bar(ax, results, title)`.
- New: `plot_baseline_heatmap(ax, eval_results, title)` — heatmap of τ across methods and λ values.
- `save_results_csv()` moved to `experiment.py`.

#### [MODIFY] `data_loader.py`
- Add type hints throughout.
- Return `density` in stats dict.
- No logic changes.

#### [MODIFY] `main.py`
- Import from new modules only.
- Clean `main()` entry point calling `run_experiment_pipeline()` from `experiment.py`.
- Full summary table at end.

---

## Verification Plan

### Automated Checks
```powershell
# 1. Syntax check all new files
python -c "import saasp_model, weight_learning, sir_simulation, evaluation, experiment, visualization"

# 2. Run end-to-end pipeline
python main.py
```

### Expected Outputs
- `results/network_analysis.png` — 2×3 or 2×4 plot grid
- `results/email_eu_core_results.csv` — comparison table
- `results/facebook_results.csv`
- `results/<dataset>_network.html` — Pyvis interactive

### Manual Checks
- All 6 methods (DC, BC, CC, CI, PageRank, K-Shell, SAASP-fixed, SAASP-adaptive) appear in comparison table
- `precision_at_k`, `final_infection_ratio`, `time_to_saturation` columns in CSV
- Node removal experiment prints "before" vs "after" F(T) values
- No import errors or circular dependencies

---

## Open Questions

> [!IMPORTANT]
> **Grid search granularity**: A 4-D grid over `{0.1, 0.2, …, 0.7}` (step 0.1, sum=1) yields ~84 valid combinations — fast but coarse. Should I use a finer step (0.05, ~1140 combos) at the cost of ~10× more SIR evaluations? Default will be 0.1 step.

> [!NOTE]
> The running `python main.py` terminal process (now >1h) is from the **old code**. The new pipeline should be significantly faster due to L-hop caching and vectorized SIR. You may want to kill it before running the new code.
