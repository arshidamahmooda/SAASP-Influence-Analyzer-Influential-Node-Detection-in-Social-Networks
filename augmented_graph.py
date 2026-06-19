"""
augmented_graph.py — Build refined, auxiliary, enhanced, and final GA.

Implements the augmented graph construction from the SAASP algorithm:
  WR = (1/kavg) * (I + A)             — Refined graph
  WU = alpha * Wk + (1-alpha) * WR    — Auxiliary graph
  WE = (1 - Wk) ⊙ (1 - WU)           — Enhanced graph
  GA = 0.33*WR + 0.33*WU + 0.33*WE   — Final augmented graph (top-k sparsified)

Improvements (v2):
  - Replaced custom cdist cosine loop with sklearn pairwise cosine_similarity
    for vectorized, memory-chunked computation.
"""

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity


def _cosine_similarity_topk(X: np.ndarray, k: int = 10) -> sparse.csr_matrix:
    """
    Compute cosine similarity between all row pairs in X using
    sklearn.metrics.pairwise.cosine_similarity, keep only the top-k
    values per row, and normalize by the global max.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Normalized feature matrix.
    k : int
        Number of top similarities to retain per row.

    Returns
    -------
    Wk : scipy.sparse.csr_matrix, shape (n, n)
        Sparse, normalized cosine similarity matrix.
    """
    n = X.shape[0]
    chunk_size = 500
    rows_out, cols_out, vals_out = [], [], []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # sklearn gives a (chunk, n) similarity matrix — fully vectorized
        sim_chunk = sk_cosine_similarity(X[start:end], X)  # shape (chunk, n)
        np.clip(sim_chunk, 0, 1, out=sim_chunk)

        for i_local in range(end - start):
            i_global = start + i_local
            sim_row = sim_chunk[i_local].copy()
            sim_row[i_global] = 0.0  # no self-similarity
            # Select top-k indices
            if k < n:
                topk_idx = np.argpartition(sim_row, -k)[-k:]
            else:
                topk_idx = np.arange(n)
            for j in topk_idx:
                if sim_row[j] > 0:
                    rows_out.append(i_global)
                    cols_out.append(j)
                    vals_out.append(sim_row[j])

    Wk = sparse.csr_matrix((vals_out, (rows_out, cols_out)), shape=(n, n))
    # Normalize by global max
    max_val = Wk.max()
    if max_val > 1e-12:
        Wk = Wk / max_val
    return Wk


def _sparsify_topk(M: sparse.csr_matrix, k: int = 10) -> sparse.csr_matrix:
    """
    Keep only the top-k values per row in a sparse matrix.
    """
    n = M.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        row_start = M.indptr[i]
        row_end   = M.indptr[i + 1]
        row_indices = M.indices[row_start:row_end]
        row_data    = M.data[row_start:row_end]

        if len(row_data) <= k:
            for j, v in zip(row_indices, row_data):
                rows.append(i)
                cols.append(j)
                vals.append(v)
        else:
            topk_pos = np.argpartition(row_data, -k)[-k:]
            for pos in topk_pos:
                rows.append(i)
                cols.append(row_indices[pos])
                vals.append(row_data[pos])

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_augmented_graph(
    G,
    node_list: list,
    X: np.ndarray,
    alpha: float = 0.6,
    top_k: int = 10,
) -> sparse.csr_matrix:
    """
    Build the final augmented graph GA.

    Parameters
    ----------
    G : nx.Graph
        Original graph.
    node_list : list
        Ordered list of node IDs (defines matrix row/col ordering).
    X : np.ndarray, shape (n, d)
        Normalized feature matrix.
    alpha : float
        Blending parameter for auxiliary graph (default 0.6).
    top_k : int
        Number of top values to keep per row in GA (default 10).

    Returns
    -------
    GA : scipy.sparse.csr_matrix, shape (n, n)
        Final augmented graph adjacency matrix.
    node_to_idx : dict
        Mapping from node ID to matrix index.
    """
    n = len(node_list)
    node_to_idx = {v: i for i, v in enumerate(node_list)}

    # Build adjacency matrix A
    rows_a, cols_a, vals_a = [], [], []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            rows_a.extend([i, j])
            cols_a.extend([j, i])
            vals_a.extend([1.0, 1.0])
    A = sparse.csr_matrix((vals_a, (rows_a, cols_a)), shape=(n, n))

    # Average degree
    degrees = np.array(A.sum(axis=1)).flatten()
    kavg = degrees.mean()
    if kavg < 1e-12:
        kavg = 1.0

    # Step 1: Refined graph WR = (1/kavg) * (I + A)
    I  = sparse.eye(n, format="csr")
    WR = (1.0 / kavg) * (I + A)

    # Step 2: Cosine similarity matrix Wk (top-k per row, normalized)
    # Uses sklearn for fast vectorized computation instead of manual cdist loop
    print("  Computing cosine similarity matrix Wk (sklearn)...")
    Wk = _cosine_similarity_topk(X, k=top_k)

    # Step 3: Auxiliary graph WU = alpha * Wk + (1-alpha) * WR
    WU = alpha * Wk + (1.0 - alpha) * WR

    # Step 4: Enhanced graph WE = (1 - Wk) ⊙ (1 - WU)
    # We only compute WE on positions where Wk or WU are nonzero.
    union_mask = (Wk != 0) + (WU != 0)  # boolean sparse OR

    rows_e, cols_e, vals_e = [], [], []
    union_coo = union_mask.tocoo()

    Wk_lil = Wk.tolil()
    WU_lil = WU.tolil()

    for r, c in zip(union_coo.row, union_coo.col):
        wk_val = Wk_lil[r, c]
        wu_val = WU_lil[r, c]
        we_val = (1.0 - wk_val) * (1.0 - wu_val)
        if abs(we_val) > 1e-12:
            rows_e.append(r)
            cols_e.append(c)
            vals_e.append(we_val)

    WE = sparse.csr_matrix((vals_e, (rows_e, cols_e)), shape=(n, n))

    # Step 5: Final augmented graph GA = 0.33*WR + 0.33*WU + 0.33*WE
    GA = (1.0 / 3.0) * WR + (1.0 / 3.0) * WU + (1.0 / 3.0) * WE

    # Sparsify: keep only top-k values per row
    GA = _sparsify_topk(GA, k=top_k)

    print(f"  GA constructed: {GA.nnz} nonzero entries")
    return GA, node_to_idx
