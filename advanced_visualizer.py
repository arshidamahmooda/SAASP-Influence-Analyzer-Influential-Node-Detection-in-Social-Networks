"""
advanced_visualizer.py — Master orchestrator for all advanced SAASP visualizations.

Call run_all_advanced_visualizations() from main.py after the pipeline completes.

Sections orchestrated:
  A. Heatmaps          (viz_heatmaps)
  B. Performance plots (viz_performance)
  C. Research plots    (viz_research)
  D. Ablation plots    (viz_ablation)
  E. Network plots     (viz_network)
  F. Export            (viz_export)
"""

import os
import numpy as np

# ── sub-modules ──────────────────────────────────────────────────────────────
from viz_heatmaps import (
    plot_method_correlation_heatmap,
    plot_tau_vs_lambda_heatmap,
    plot_adaptive_weights_heatmap,
    plot_influence_spread_heatmap,
)
from viz_performance import plot_all_performance_metrics
from viz_research import (
    plot_score_boxplots,
    plot_score_violins,
    plot_radar_chart,
    plot_node_influence_hist,
    plot_saasp_vs_sir_scatter,
    plot_pairplot,
    plot_error_bar_sir,
)
from viz_ablation import (
    plot_ablation_bars,
    plot_ablation_ft_curves,
    plot_component_contribution,
    plot_node_removal_curves,
    plot_cascading_failure,
)
from viz_network import (
    plot_network_influence_static,
    create_advanced_interactive_html,
)
from viz_export import (
    export_all_scores_csv,
    export_metrics_excel,
    export_ranked_nodes_table,
    export_adaptive_weights_table,
    export_statistical_significance,
    export_comparison_latex,
)


# ─── Helper: build radar metrics dict ────────────────────────────────────────

def _build_radar_metrics(eval_results: dict, auc_scores: dict, method_scores: dict):
    """
    Assemble normalised metric vectors for the radar chart.

    Metrics used: Kendall τ, Precision@K, AUC, F(t_final) normalised.
    Returns (metric_rows dict, metric_names list).
    """
    methods = list(eval_results["kendall_taus"].keys())
    # Max values for normalisation
    taus   = eval_results["kendall_taus"]
    p_at_k = eval_results["precision_at_k"]
    f_fin  = {m: eval_results["F_curves"][m][-1] for m in methods}

    max_tau   = max(abs(v) for v in taus.values()) or 1.0
    max_f     = max(f_fin.values()) or 1.0
    max_pak   = max(p_at_k.values()) or 1.0
    max_auc   = max(auc_scores.values()) if auc_scores else 1.0

    metric_rows = {}
    for m in methods:
        metric_rows[m] = [
            max(taus[m], 0) / max_tau,
            p_at_k.get(m, 0) / max_pak,
            auc_scores.get(m, 0.5) / max(max_auc, 1.0),
            f_fin[m] / max_f,
        ]
    metric_names = ["Kendall τ", "Precision@K", "AUC", "F(T) spread"]
    return metric_rows, metric_names


# ─── Main entry point ────────────────────────────────────────────────────────

def run_all_advanced_visualizations(
    G,
    dataset_name: str,
    node_list: list,
    all_methods: dict,
    eval_results: dict,
    tau_vs_lambda: dict,
    lambda_values,
    scores_adaptive: np.ndarray,
    xi_learned: np.ndarray,
    ablation_result: dict,
    removal_result: dict,
    metric_df,
    output_dir: str,
    top_k: int = 10,
    infection_rate: float = 0.1,
    # Accumulated cross-dataset dicts (populated externally and passed in)
    xi_all_datasets: dict = None,
):
    """
    Run all advanced visualizations for a single dataset.

    Parameters match the keys returned by run_pipeline_for_dataset() in main.py.

    xi_all_datasets : dict   dataset_name → xi_learned (for weights heatmap).
                             Pass None to skip the cross-dataset weights heatmap.
    """
    sir_gt      = eval_results["sir_ground_truth"]
    sample_idx  = eval_results["sample_indices"]
    p_at_k_dict = eval_results["precision_at_k"]
    safe        = dataset_name.lower().replace(" ", "_").replace("-", "_")

    print(f"\n  {'─'*55}")
    print(f"  ADVANCED VISUALIZATIONS — {dataset_name}")
    print(f"  {'─'*55}")

    # ── A. Heatmaps ──────────────────────────────────────────────────────────
    print("\n  [A] Generating heatmaps...")

    _safe_call("Method Correlation Heatmap",
               plot_method_correlation_heatmap,
               all_methods, dataset_name, output_dir)

    _safe_call("Tau vs Lambda Heatmap",
               plot_tau_vs_lambda_heatmap,
               tau_vs_lambda, lambda_values, dataset_name, output_dir)

    if xi_all_datasets:
        _safe_call("Adaptive Weights Heatmap",
                   plot_adaptive_weights_heatmap,
                   xi_all_datasets, output_dir)

    _safe_call("Influence Spread Heatmap",
               plot_influence_spread_heatmap,
               eval_results["F_curves"], dataset_name, output_dir)

    # ── B. Performance Metrics ────────────────────────────────────────────────
    print("\n  [B] Generating performance metric plots...")

    auc_scores = _safe_call("All Performance Metrics",
                             plot_all_performance_metrics,
                             all_methods, sir_gt, sample_idx,
                             p_at_k_dict, dataset_name, output_dir, top_k) or {}

    # ── C. Research Plots ────────────────────────────────────────────────────
    print("\n  [C] Generating research paper plots...")

    _safe_call("Score Boxplots",
               plot_score_boxplots,
               all_methods, dataset_name, output_dir)

    _safe_call("Score Violins",
               plot_score_violins,
               all_methods, dataset_name, output_dir)

    radar_metrics, radar_names = _build_radar_metrics(eval_results, auc_scores, all_methods)
    _safe_call("Radar Chart",
               plot_radar_chart,
               radar_metrics, radar_names, dataset_name, output_dir)

    _safe_call("Influence Histogram",
               plot_node_influence_hist,
               scores_adaptive, sir_gt, sample_idx, dataset_name, output_dir)

    _safe_call("SAASP vs SIR Scatter",
               plot_saasp_vs_sir_scatter,
               all_methods, sir_gt, sample_idx, dataset_name, output_dir)

    _safe_call("Pairplot",
               plot_pairplot,
               all_methods, dataset_name, output_dir)

    _safe_call("Error Bar SIR",
               plot_error_bar_sir,
               G, all_methods, dataset_name, output_dir,
               top_k, infection_rate, 8, 20)

    # ── D. Ablation ───────────────────────────────────────────────────────────
    print("\n  [D] Generating ablation & node-removal plots...")

    _safe_call("Ablation Bars",
               plot_ablation_bars,
               ablation_result, dataset_name, output_dir)

    _safe_call("Ablation F(t) Curves",
               plot_ablation_ft_curves,
               ablation_result, dataset_name, output_dir)

    _safe_call("Component Contribution",
               plot_component_contribution,
               ablation_result, dataset_name, output_dir)

    _safe_call("Node Removal Curves",
               plot_node_removal_curves,
               removal_result, dataset_name, output_dir)

    _safe_call("Cascading Failure",
               plot_cascading_failure,
               G, all_methods, node_list,
               dataset_name, output_dir, 10, infection_rate, 20, 8)

    # ── E. Network ───────────────────────────────────────────────────────────
    print("\n  [E] Generating network visualizations...")

    _safe_call("Static Network Influence",
               plot_network_influence_static,
               G, node_list, scores_adaptive, dataset_name, output_dir, top_k)

    _safe_call("Advanced Interactive HTML",
               create_advanced_interactive_html,
               G, node_list, scores_adaptive, all_methods,
               dataset_name, output_dir)

    # ── F. Exports ────────────────────────────────────────────────────────────
    print("\n  [F] Exporting tables and data files...")

    _safe_call("All Scores CSV",
               export_all_scores_csv,
               node_list, all_methods, sir_gt, sample_idx, output_dir, dataset_name)

    _safe_call("Metrics Excel",
               export_metrics_excel,
               metric_df, ablation_result, xi_all_datasets or {dataset_name: xi_learned},
               auc_scores, output_dir, dataset_name)

    _safe_call("Ranked Nodes Table",
               export_ranked_nodes_table,
               node_list, all_methods, sir_gt, sample_idx, output_dir, dataset_name)

    _safe_call("Statistical Significance",
               export_statistical_significance,
               all_methods, sir_gt, sample_idx, output_dir, dataset_name)

    _safe_call("LaTeX Table",
               export_comparison_latex,
               metric_df, output_dir, dataset_name)

    print(f"\n  ✓ Advanced visualizations complete for {dataset_name}")
    print(f"    All files saved to: {output_dir}/")


# ─── Cross-dataset weights heatmap (call after all datasets processed) ───────

def run_cross_dataset_weights_heatmap(xi_all_datasets: dict, output_dir: str):
    """Call once after all datasets to produce the multi-dataset weights heatmap."""
    if len(xi_all_datasets) < 1:
        return
    _safe_call("Adaptive Weights Heatmap (cross-dataset)",
               plot_adaptive_weights_heatmap,
               xi_all_datasets, output_dir)


# ─── Adaptive weights CSV (cross-dataset summary) ────────────────────────────

def run_adaptive_weights_export(xi_all_datasets: dict, output_dir: str):
    """Export combined adaptive weights table across all datasets."""
    _safe_call("Adaptive Weights Summary CSV",
               export_adaptive_weights_table,
               xi_all_datasets, output_dir)


# ─── Internal helper ─────────────────────────────────────────────────────────

def _safe_call(name: str, fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) with error catching so one failing plot
    never aborts the whole pipeline.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  [WARN] {name} failed: {type(e).__name__}: {e}")
        return None
