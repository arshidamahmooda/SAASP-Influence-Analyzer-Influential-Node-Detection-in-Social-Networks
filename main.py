# -*- coding: utf-8 -*-
"""
main.py -- Entry point: runs the full SAASP pipeline.

Pipeline:
  1. Load both datasets, print basic stats
  2. For each dataset:
     a. Extract features (7-dim improved)
     b. Build augmented graph GA
     c. Compute all 4 SAASP components for every node
     d. Score with fixed weights (xi=0.25 each)
     e. Learn adaptive weights via SIR ground truth
     f. Score with adaptive weights
     g. Evaluate all methods, print comparison table
     h. Run advanced experiments (node removal, ablation, metric comparison)
     i. Export node scores to CSV
     j. Generate and save all plots
  3. Print final summary

Improvements (v2):
  - Node removal experiment integrated
  - Ablation study integrated
  - Full metric comparison table (τ, P@K, F(T), saturation, peak)
  - Results exported to CSV
"""

import os
import sys
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_datasets
from feature_extractor import extract_features
from augmented_graph import build_augmented_graph
from saasp_scorer import compute_saasp_scores, compute_saasp_from_components
from adaptive_weights import learn_adaptive_weights
from evaluator import (
    compute_all_baselines,
    evaluate_methods,
    kendall_tau_vs_lambda,
    print_comparison_table,
)
from visualizer import create_static_plots, create_interactive_html
from experiments import (
    node_removal_experiment,
    ablation_study,
    metric_comparison,
    save_scores_to_csv,
)
from advanced_visualizer import (
    run_all_advanced_visualizations,
    run_cross_dataset_weights_heatmap,
    run_adaptive_weights_export,
)


def run_pipeline_for_dataset(G, stats, results_dir):
    """
    Run the full SAASP pipeline for a single dataset.

    Returns all results needed for cross-dataset comparison and visualization.
    """
    dataset_name = stats["name"]
    n = stats["nodes"]

    print(f"\n{'#'*60}")
    print(f"  PROCESSING: {dataset_name}")
    print(f"{'#'*60}")

    # Use L=2 for CASPI to keep runtime practical in dense networks
    L = 2
    print(f"  Using L={L} for local subgraphs")

    # Step 1: Extract features (7-dim improved)
    print(f"\n--- Step 1: Feature Extraction ---")
    t0 = time.time()
    node_list, X, X_raw = extract_features(G, verbose=True)
    print(f"  Features extracted in {time.time() - t0:.1f}s")
    print(f"  Feature matrix shape: {X.shape}")

    # Step 2: Build augmented graph GA
    print(f"\n--- Step 2: Augmented Graph Construction ---")
    t0 = time.time()
    GA, node_to_idx = build_augmented_graph(G, node_list, X, alpha=0.6, top_k=10)
    print(f"  GA built in {time.time() - t0:.1f}s")

    # Step 3: Compute all 4 SAASP components
    print(f"\n--- Step 3: SAASP Component Computation ---")
    t0 = time.time()
    fixed_xi = np.array([0.25, 0.25, 0.25, 0.25])
    saasp_results = compute_saasp_scores(
        G, node_list, GA, node_to_idx, L=L, xi=fixed_xi, verbose=True
    )
    print(f"  SAASP computed in {time.time() - t0:.1f}s")

    scores_fixed = saasp_results["scores"]

    # Step 4: Learn adaptive weights
    print(f"\n--- Step 4: Adaptive Weight Learning ---")
    t0 = time.time()
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    weights_path = os.path.join(results_dir, f"{safe_name}_learned_weights.npy")
    adaptive_result = learn_adaptive_weights(
        G, node_list,
        saasp_results["CLI_norm"],
        saasp_results["CSLI_norm"],
        saasp_results["CASPI_norm"],
        saasp_results["CSI_norm"],
        sample_fraction=0.2,
        infection_rate=0.1,
        random_state=42,
        lambda_reg=1e-3,
        save_weights=True,
        weights_path=weights_path,
        verbose=True,
    )
    xi_learned = adaptive_result["xi_learned"]
    print(f"  Adaptive weights learned in {time.time() - t0:.1f}s")

    # Score with adaptive weights on full graph
    scores_adaptive = compute_saasp_from_components(
        saasp_results["CLI_norm"],
        saasp_results["CSLI_norm"],
        saasp_results["CASPI_norm"],
        saasp_results["CSI_norm"],
        xi_learned,
    )

    # Step 5: Compute baselines (now includes PR + KS)
    print(f"\n--- Step 5: Baseline Methods ---")
    t0 = time.time()
    baselines = compute_all_baselines(G, node_list)
    print(f"  Baselines computed in {time.time() - t0:.1f}s")

    # Combine all methods
    all_methods = {}
    all_methods.update(baselines)
    all_methods["SAASP-fixed"]    = scores_fixed
    all_methods["SAASP-adaptive"] = scores_adaptive

    # Step 6: Evaluate (includes Precision@K)
    print(f"\n--- Step 6: Evaluation ---")
    t0 = time.time()
    eval_results = evaluate_methods(
        G, node_list, all_methods,
        infection_rate=0.1, top_k=10,
        random_state=42,
    )
    print(f"  Evaluation completed in {time.time() - t0:.1f}s")

    # Print comparison table
    print_comparison_table(eval_results, dataset_name)

    # Step 7: Kendall tau vs lambda
    print(f"\n--- Step 7: Kendall τ vs λ ---")
    t0 = time.time()
    lambda_values = np.linspace(0.01, 0.1, 10)
    tau_vs_lambda, lam_vals = kendall_tau_vs_lambda(
        G, node_list, all_methods,
        lambda_values=lambda_values,
        random_state=42,
    )
    print(f"  Tau vs lambda computed in {time.time() - t0:.1f}s")

    # Step 8: Export scores to CSV
    print(f"\n--- Step 8: Export Scores to CSV ---")
    csv_path = os.path.join(results_dir, f"{safe_name}_scores.csv")
    save_scores_to_csv(node_list, all_methods, csv_path)

    # Step 9: Advanced Experiments
    print(f"\n--- Step 9: Node Removal Experiment ---")
    t0 = time.time()
    removal_result = node_removal_experiment(
        G, node_list, scores_adaptive,
        infection_rate=0.1, top_k=10,
        steps=20, num_sir_runs=15,
        random_state=42,
    )
    print(f"  Node removal experiment done in {time.time() - t0:.1f}s")

    print(f"\n--- Step 10: Ablation Study ---")
    t0 = time.time()
    ablation_result = ablation_study(
        G, node_list,
        saasp_results["CLI_norm"],
        saasp_results["CSLI_norm"],
        saasp_results["CASPI_norm"],
        saasp_results["CSI_norm"],
        sir_ground_truth=eval_results["sir_ground_truth"],
        sample_indices=eval_results["sample_indices"],
        top_k=10,
        infection_rate=0.1,
        steps=20, num_sir_runs=15,
        random_state=42,
    )
    print(f"  Ablation study done in {time.time() - t0:.1f}s")

    print(f"\n--- Step 11: Full Metric Comparison ---")
    t0 = time.time()
    metric_df = metric_comparison(
        G, node_list, all_methods,
        sir_ground_truth=eval_results["sir_ground_truth"],
        sample_indices=eval_results["sample_indices"],
        infection_rate=0.1, top_k=10,
        steps=20, num_sir_runs=15,
        random_state=42,
    )
    print(f"\n  FULL METRIC COMPARISON — {dataset_name}")
    print(metric_df.to_string(index=False))
    # Save metric table to CSV
    metric_csv = os.path.join(results_dir, f"{safe_name}_metrics.csv")
    metric_df.to_csv(metric_csv, index=False)
    print(f"  Metrics saved to: {metric_csv}")
    print(f"  Metric comparison done in {time.time() - t0:.1f}s")

    # Step 12: Interactive visualization (original)
    print(f"\n--- Step 12: Interactive Visualization ---")
    create_interactive_html(
        G, node_list, scores_adaptive, dataset_name,
        output_dir=results_dir,
    )

    return {
        "node_list":       node_list,
        "all_methods":     all_methods,
        "eval_results":    eval_results,
        "tau_vs_lambda":   tau_vs_lambda,
        "lambda_values":   lam_vals,
        "scores_fixed":    scores_fixed,
        "scores_adaptive": scores_adaptive,
        "xi_learned":      xi_learned,
        "adaptive_result": adaptive_result,
        "saasp_results":   saasp_results,
        "removal_result":  removal_result,
        "ablation_result": ablation_result,
        "metric_df":       metric_df,
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("  SAASP Centrality Analysis — Research-Grade Implementation v2")
    print("  Based on Meng & Rezaeipanah (2025)")
    print("=" * 60)

    # Paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.dirname(project_dir)  # parent dir contains .gz files
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load datasets
    print("\n" + "=" * 60)
    print("  STEP 1: Loading Datasets")
    print("=" * 60)
    datasets = load_all_datasets(data_dir)

    if not datasets:
        print("[ERROR] No datasets loaded. Exiting.")
        return

    # Process each dataset
    all_pipeline_results = {}
    xi_all_datasets      = {}   # accumulated for cross-dataset weights heatmap
    graph_by_name        = {}   # keep graph objects for advanced viz

    for G_iter, stats in datasets:
        pipeline_result = run_pipeline_for_dataset(G_iter, stats, results_dir)
        ds_name = stats["name"]
        all_pipeline_results[ds_name] = pipeline_result
        xi_all_datasets[ds_name]      = pipeline_result["xi_learned"]
        graph_by_name[ds_name]        = G_iter

    # Generate combined static plots
    print("\n" + "=" * 60)
    print("  GENERATING COMBINED STATIC PLOTS")
    print("=" * 60)

    dataset_names = list(all_pipeline_results.keys())

    if len(dataset_names) >= 2:
        email_key = dataset_names[0]
        fb_key    = dataset_names[1]

        email_r = all_pipeline_results[email_key]
        fb_r    = all_pipeline_results[fb_key]

        create_static_plots(
            email_results=email_r["eval_results"],
            fb_results=fb_r["eval_results"],
            email_tau_results=email_r["tau_vs_lambda"],
            fb_tau_results=fb_r["tau_vs_lambda"],
            email_lambda_values=email_r["lambda_values"],
            fb_lambda_values=fb_r["lambda_values"],
            email_method_scores=email_r["all_methods"],
            fb_method_scores=fb_r["all_methods"],
            email_node_list=email_r["node_list"],
            fb_node_list=fb_r["node_list"],
            output_dir=results_dir,
        )
    elif len(dataset_names) == 1:
        key = dataset_names[0]
        r   = all_pipeline_results[key]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualizer import plot_ft_curves, plot_kendall_vs_lambda, plot_top10_bar

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plot_ft_curves(axes[0], r["eval_results"]["F_curves"],
                       f"{key}: F(t) Spread Curves")
        plot_kendall_vs_lambda(axes[1], r["tau_vs_lambda"], r["lambda_values"],
                               f"{key}: Kendall τ vs λ")
        plot_top10_bar(axes[2], r["eval_results"]["top_nodes"],
                       r["all_methods"], r["node_list"],
                       f"{key}: Top-10 Nodes")
        plt.tight_layout()
        outpath = os.path.join(results_dir, "network_analysis.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Static plots saved to: {outpath}")

    # ── Advanced visualizations (run after all datasets are processed) ──────
    print("\n" + "=" * 60)
    print("  ADVANCED VISUALIZATIONS")
    print("=" * 60)

    for ds_name, result in all_pipeline_results.items():
        G_ds = graph_by_name[ds_name]
        run_all_advanced_visualizations(
            G              = G_ds,
            dataset_name   = ds_name,
            node_list      = result["node_list"],
            all_methods    = result["all_methods"],
            eval_results   = result["eval_results"],
            tau_vs_lambda  = result["tau_vs_lambda"],
            lambda_values  = result["lambda_values"],
            scores_adaptive= result["scores_adaptive"],
            xi_learned     = result["xi_learned"],
            ablation_result= result["ablation_result"],
            removal_result = result["removal_result"],
            metric_df      = result["metric_df"],
            output_dir     = results_dir,
            top_k          = 10,
            infection_rate = 0.1,
            xi_all_datasets= xi_all_datasets,
        )

    # Cross-dataset summary exports
    run_cross_dataset_weights_heatmap(xi_all_datasets, results_dir)
    run_adaptive_weights_export(xi_all_datasets, results_dir)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)

    for name, result in all_pipeline_results.items():
        xi          = result["xi_learned"]
        tau_fixed   = result["adaptive_result"]["tau_fixed"]
        tau_adaptive = result["adaptive_result"]["tau_learned"]

        improvement = (
            (tau_adaptive - tau_fixed) / abs(tau_fixed) * 100
            if abs(tau_fixed) > 1e-8 else 0.0
        )

        removal = result["removal_result"]

        print(f"\n  {name}:")
        print(f"    Learned weights: xi1={xi[0]:.4f}, xi2={xi[1]:.4f}, "
              f"xi3={xi[2]:.4f}, xi4={xi[3]:.4f}")
        print(f"    Kendall tau (fixed weights):    {tau_fixed:.4f}")
        print(f"    Kendall tau (adaptive weights): {tau_adaptive:.4f}")
        print(f"    Improvement: {improvement:+.2f}%")
        print(f"    Node removal spread reduction: "
              f"{removal['spread_reduction']:.2%}")

    # Average improvement
    improvements = []
    for name, result in all_pipeline_results.items():
        tau_f = result["adaptive_result"]["tau_fixed"]
        tau_a = result["adaptive_result"]["tau_learned"]
        if abs(tau_f) > 1e-8:
            improvements.append((tau_a - tau_f) / abs(tau_f) * 100)

    if improvements:
        avg_imp = np.mean(improvements)
        print(f"\n  Average Kendall tau improvement of adaptive vs fixed SAASP: "
              f"{avg_imp:+.2f}%")

    print(f"\n  Results saved to: {results_dir}/")
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
