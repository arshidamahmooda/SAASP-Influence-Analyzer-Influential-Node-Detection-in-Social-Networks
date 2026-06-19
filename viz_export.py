"""
viz_export.py — Export utilities: CSV, Excel, and thesis-quality tables.

Provides:
  1. export_all_scores_csv      – All method scores per node → CSV
  2. export_metrics_excel       – Full metrics DataFrame → Excel workbook
  3. export_ranked_nodes_table  – Ranked node table per method → CSV + Excel
  4. export_adaptive_weights_table – ξ weights summary → CSV
  5. export_statistical_significance – Pairwise Kendall τ significance table
  6. export_comparison_latex    – LaTeX-formatted comparison table
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

from viz_utils import ordered_methods


# ─── 1. All Scores CSV ───────────────────────────────────────────────────────

def export_all_scores_csv(
    node_list: list,
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    output_dir: str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Export a CSV with node_id, all centrality scores, and SIR ground truth.

    Columns: node_id, DC, BC, CC, CI, PR, KS, SAASP-fixed, SAASP-adaptive,
             sir_influence (NaN for non-sampled nodes)
    """
    os.makedirs(output_dir, exist_ok=True)
    data = {"node_id": node_list}
    methods = ordered_methods(method_scores)
    for m in methods:
        data[m] = method_scores[m]

    # Add SIR influence (only for sampled nodes)
    sir_full = np.full(len(node_list), np.nan)
    sir_full[sample_indices] = sir_ground_truth
    data["SIR_influence"] = sir_full

    df = pd.DataFrame(data)
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe}_all_scores.csv")
    df.to_csv(path, index=False)
    print(f"  Scores CSV saved: {path}")
    return df


# ─── 2. Excel Workbook ───────────────────────────────────────────────────────

def export_metrics_excel(
    metric_df: pd.DataFrame,
    ablation_result: dict,
    xi_dict: dict,
    auc_scores: dict,
    output_dir: str,
    dataset_name: str,
):
    """
    Export a multi-sheet Excel workbook containing:
      Sheet 1: Full metric comparison table
      Sheet 2: Ablation study results
      Sheet 3: Adaptive weights per dataset
      Sheet 4: AUC scores

    Requires openpyxl.
    """
    try:
        import openpyxl
    except ImportError:
        print("  [SKIP] openpyxl not installed; Excel export skipped.")
        return

    os.makedirs(output_dir, exist_ok=True)
    safe  = dataset_name.lower().replace(" ", "_").replace("-", "_")
    path  = os.path.join(output_dir, f"{safe}_results_summary.xlsx")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Sheet 1: Metric comparison
        metric_df.to_excel(writer, sheet_name="Metric Comparison", index=False)

        # Sheet 2: Ablation
        if ablation_result:
            configs  = ablation_result["configs"]
            abl_rows = []
            for c in configs:
                abl_rows.append({
                    "Config":        c,
                    "Kendall τ":     round(ablation_result["kendall_taus"][c], 4),
                    "Precision@K":   round(ablation_result["precision_at_k"][c], 4),
                    "F(t_final)":    round(float(ablation_result["F_curves"][c][-1]), 2),
                })
            pd.DataFrame(abl_rows).to_excel(writer, sheet_name="Ablation Study", index=False)

        # Sheet 3: Adaptive weights
        if xi_dict:
            components = ["xi1_CLI", "xi2_CSLI", "xi3_CASPI", "xi4_CSI"]
            rows = []
            for ds_name, xi in xi_dict.items():
                row = {"Dataset": ds_name}
                for comp, val in zip(components, xi):
                    row[comp] = round(float(val), 6)
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name="Adaptive Weights", index=False)

        # Sheet 4: AUC
        if auc_scores:
            auc_rows = [{"Method": m, "AUC": round(v, 4)}
                        for m, v in auc_scores.items()]
            pd.DataFrame(auc_rows).to_excel(writer, sheet_name="AUC Scores", index=False)

    print(f"  Excel workbook saved: {path}")


# ─── 3. Ranked Nodes Table ───────────────────────────────────────────────────

def export_ranked_nodes_table(
    node_list: list,
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    output_dir: str,
    dataset_name: str,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Export a table showing top-K nodes for each method side by side.
    """
    os.makedirs(output_dir, exist_ok=True)
    methods = ordered_methods(method_scores)
    data    = {"Rank": list(range(1, top_k + 1))}

    # Ground-truth top-K
    sir_full = np.zeros(len(node_list))
    sir_full[sample_indices] = sir_ground_truth
    gt_ranking = np.argsort(-sir_full)
    data["SIR_GT (Top Nodes)"] = [node_list[i] for i in gt_ranking[:top_k]]

    for m in methods:
        ranking = np.argsort(-method_scores[m])
        data[m] = [node_list[i] for i in ranking[:top_k]]

    df   = pd.DataFrame(data)
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    csv_path = os.path.join(output_dir, f"{safe}_ranked_nodes.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Ranked nodes table saved: {csv_path}")
    return df


# ─── 4. Adaptive Weights Summary ─────────────────────────────────────────────

def export_adaptive_weights_table(
    xi_dict: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Export learned ξ weights across all datasets to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for ds_name, xi in xi_dict.items():
        rows.append({
            "Dataset":    ds_name,
            "ξ1 (CLI)":   round(float(xi[0]), 6),
            "ξ2 (CSLI)":  round(float(xi[1]), 6),
            "ξ3 (CASPI)": round(float(xi[2]), 6),
            "ξ4 (CSI)":   round(float(xi[3]), 6),
            "Sum":         round(float(np.sum(xi)), 6),
            "Dominant":   ["ξ1","ξ2","ξ3","ξ4"][int(np.argmax(xi))],
        })
    df   = pd.DataFrame(rows)
    path = os.path.join(output_dir, "adaptive_weights_summary.csv")
    df.to_csv(path, index=False)
    print(f"  Adaptive weights summary saved: {path}")
    return df


# ─── 5. Statistical Significance ─────────────────────────────────────────────

def export_statistical_significance(
    method_scores: dict,
    sir_ground_truth: np.ndarray,
    sample_indices: np.ndarray,
    output_dir: str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank test p-values between method scores
    on the sampled nodes vs SIR ground truth.

    Returns a DataFrame with columns: Method_A, Method_B, p_value, significant.
    """
    os.makedirs(output_dir, exist_ok=True)
    methods = ordered_methods(method_scores)
    rows = []

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if j <= i:
                continue
            s1 = method_scores[m1][sample_indices]
            s2 = method_scores[m2][sample_indices]
            diff = s1 - s2
            if np.all(diff == 0):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(s1, s2, zero_method="wilcox")
                except Exception:
                    p = float("nan")
            rows.append({
                "Method_A":    m1,
                "Method_B":    m2,
                "p_value":     round(float(p), 6),
                "significant": "Yes" if (not np.isnan(p) and p < 0.05) else "No",
            })

    df   = pd.DataFrame(rows)
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe}_significance_table.csv")
    df.to_csv(path, index=False)
    print(f"  Statistical significance table saved: {path}")
    return df


# ─── 6. LaTeX Comparison Table ───────────────────────────────────────────────

def export_comparison_latex(
    metric_df: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
):
    """
    Write a LaTeX booktabs table from the metric comparison DataFrame.
    Saves as <dataset>_comparison_table.tex
    """
    os.makedirs(output_dir, exist_ok=True)
    safe = dataset_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe}_comparison_table.tex")

    try:
        latex_str = metric_df.to_latex(
            index=False,
            float_format="{:.4f}".format,
            caption=f"Performance Comparison of Centrality Methods ({dataset_name})",
            label=f"tab:{safe}_comparison",
            escape=True,
        )
        # Add booktabs rules
        latex_str = latex_str.replace(r"\hline", r"\midrule", 1)
        latex_str = latex_str.replace(r"\begin{tabular}", r"\begin{tabular}", 1)

        with open(path, "w", encoding="utf-8") as f:
            f.write("% Requires \\usepackage{booktabs} in preamble\n")
            f.write(latex_str)
        print(f"  LaTeX table saved: {path}")
    except Exception as e:
        print(f"  [WARN] LaTeX export failed: {e}")
