"""
viz_utils.py — Shared plotting utilities for SAASP advanced visualizations.

Provides reusable helpers for:
  - Publication-quality figure setup
  - Consistent color palettes
  - Multi-format figure saving (PNG, PDF, SVG)
  - Excel export utilities
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print("[WARNING] seaborn not installed. Some plots will be simplified.")

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# ── Consistent colour palette ────────────────────────────────────────────────
METHOD_COLORS = {
    "DC":             "#3498db",
    "BC":             "#e67e22",
    "CC":             "#2ecc71",
    "CI":             "#e74c3c",
    "PR":             "#1abc9c",
    "KS":             "#95a5a6",
    "SAASP-fixed":    "#9b59b6",
    "SAASP-adaptive": "#c0392b",
}

METHOD_ORDER = ["DC", "BC", "CC", "CI", "PR", "KS", "SAASP-fixed", "SAASP-adaptive"]

COMPONENT_COLORS = {
    "CLI":   "#2ecc71",
    "CSLI":  "#3498db",
    "CASPI": "#e67e22",
    "CSI":   "#9b59b6",
}

ABLATION_COLORS = {
    "CLI only":        "#95a5a6",
    "CLI+CSLI":        "#3498db",
    "CLI+CSLI+CASPI":  "#e67e22",
    "Full SAASP":      "#c0392b",
}

DPI_SCREEN = 150
DPI_PRINT  = 300

# ── Global style ─────────────────────────────────────────────────────────────

def apply_publication_style():
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#f8f9fa",
        "axes.edgecolor":    "#cccccc",
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "grid.color":        "#e0e0e0",
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.framealpha": 0.85,
        "lines.linewidth":   1.8,
    })

# ── Save helper ──────────────────────────────────────────────────────────────

def save_figure(fig, output_dir: str, base_name: str, dpi: int = DPI_PRINT):
    """
    Save figure in PNG, PDF, and SVG formats.

    Parameters
    ----------
    fig       : matplotlib Figure
    output_dir: directory to save into (created if missing)
    base_name : filename stem (no extension)
    dpi       : resolution for raster formats
    """
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(output_dir, f"{base_name}.{ext}")
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        except Exception as e:
            print(f"  [WARN] Could not save {path}: {e}")
    print(f"  Saved: {base_name}.{{png,pdf,svg}} → {output_dir}/")

# ── Colour helpers ───────────────────────────────────────────────────────────

def method_color_list(methods):
    """Return a list of colours aligned to method order."""
    return [METHOD_COLORS.get(m, "#7f7f7f") for m in methods]

def ordered_methods(method_scores: dict):
    """Return method names in canonical order."""
    present = set(method_scores.keys())
    return [m for m in METHOD_ORDER if m in present] + \
           [m for m in method_scores if m not in METHOD_ORDER]

# ── Score matrix helper ───────────────────────────────────────────────────────

def build_score_matrix(method_scores: dict, node_list: list):
    """
    Build a (n_nodes × n_methods) array and matching labels.

    Returns
    -------
    matrix : np.ndarray  shape (n_nodes, n_methods)
    labels : list of method names
    """
    labels = ordered_methods(method_scores)
    matrix = np.column_stack([method_scores[m] for m in labels])
    return matrix, labels

# ── Binary ground-truth helper ────────────────────────────────────────────────

def top_k_binary(scores: np.ndarray, k: int) -> np.ndarray:
    """Return a binary array: 1 if node is in top-k by score, else 0."""
    idx = np.argsort(-scores)[:k]
    binary = np.zeros(len(scores), dtype=int)
    binary[idx] = 1
    return binary
