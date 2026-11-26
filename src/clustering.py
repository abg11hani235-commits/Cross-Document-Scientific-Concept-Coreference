"""Topic-scoped cosine similarity clustering for scientific concept mentions."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

ClusteringAlgorithm = Literal["connected_components", "agglomerative"]


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safely L2-normalize rows of a 2D array; zero vectors remain zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > eps, norms, 1.0)
    normalized = vectors / safe_norms
    zero_mask = (norms <= eps).ravel()
    if np.any(zero_mask):
        normalized = normalized.copy()
        normalized[zero_mask] = 0.0
    return normalized.astype(np.float32, copy=False)


def _canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel cluster IDs deterministically by first appearance (0, 1, 2, ...)."""
    mapping: dict[int, int] = {}
    out = np.empty_like(labels, dtype=np.int64)
    for idx, lbl in enumerate(labels):
        val = int(lbl)
        if val not in mapping:
            mapping[val] = len(mapping)
        out[idx] = mapping[val]
    return out


def cluster_topic_connected_components(
    embeddings: np.ndarray,
    threshold: float,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Cluster mentions in a single topic via cosine similarity threshold graph.

    Builds an undirected graph where edge ``(i, j)`` exists iff
    ``cos(e_i, e_j) >= threshold`` and extracts connected components.

    Note on transitivity:
        Connected-component clustering is transitive: if ``sim(A, B) >= threshold``
        and ``sim(B, C) >= threshold``, then ``A`` and ``C`` are placed in the same
        cluster even when ``sim(A, C) < threshold``. This can recover indirect
        surface-form chains but also introduces false-merge chaining risks at lower
        thresholds.

    Computes similarities in row chunks of ``batch_size`` to avoid materializing
    a full dense float32 ``N x N`` matrix for very large topics.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    normed = l2_normalize(embeddings)
    row_norms = np.linalg.norm(normed, axis=1)
    valid_mask = row_norms > 0.5

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim_block = normed[start:end] @ normed.T
        # Only connect valid non-zero vectors
        block_valid = valid_mask[start:end, None] & valid_mask[None, :]
        adj_block = (sim_block >= threshold) & block_valid
        r_idx, c_idx = np.nonzero(adj_block)
        if r_idx.size > 0:
            rows_list.append(r_idx + start)
            cols_list.append(c_idx)

    if rows_list:
        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        data = np.ones(all_rows.shape[0], dtype=np.int8)
        graph = csr_matrix((data, (all_rows, all_cols)), shape=(n, n))
    else:
        graph = csr_matrix((n, n), dtype=np.int8)

    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return _canonicalize_labels(labels)


def cluster_topic_agglomerative(
    embeddings: np.ndarray,
    threshold: float,
    *,
    linkage: str = "average",
) -> np.ndarray:
    """Cluster mentions in a single topic via Agglomerative Clustering on cosine distance.

    Uses ``distance_threshold = 1.0 - threshold`` with ``metric='cosine'``.
    Unlike connected components, average-linkage agglomerative clustering requires
    the mean pairwise similarity between merged clusters to exceed ``threshold``,
    reducing transitive chaining.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    normed = l2_normalize(embeddings)
    row_norms = np.linalg.norm(normed, axis=1)
    valid_indices = np.flatnonzero(row_norms > 0.5)

    labels = np.full(n, -1, dtype=np.int64)
    if valid_indices.size == 1:
        labels[valid_indices[0]] = 0
    elif valid_indices.size > 1:
        sub = normed[valid_indices]
        # Cosine distance matrix bounded to [0, 2]
        cos_sim = np.clip(sub @ sub.T, -1.0, 1.0)
        cos_dist = np.maximum(0.0, 1.0 - cos_sim)
        np.fill_diagonal(cos_dist, 0.0)
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=1.0 - float(threshold),
        )
        sub_labels = model.fit_predict(cos_dist)
        labels[valid_indices] = sub_labels

    next_lbl = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    for idx in range(n):
        if labels[idx] < 0:
            labels[idx] = next_lbl
            next_lbl += 1

    return _canonicalize_labels(labels)


def cluster_mentions_by_similarity(
    mentions_df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    threshold: float,
    algorithm: ClusteringAlgorithm = "connected_components",
    agglomerative_linkage: str = "average",
    batch_size: int = 512,
) -> pd.DataFrame:
    """Assign topic-scoped predicted coreference clusters using embedding similarity.

    Similarity computations and clustering are strictly confined within each
    ``(split, topic_id)`` group so mentions from different topics are never merged.
    """
    if len(mentions_df) != embeddings.shape[0]:
        raise ValueError(
            f"Length mismatch: mentions_df has {len(mentions_df)} rows, "
            f"embeddings has {embeddings.shape[0]} rows."
        )

    if mentions_df.empty:
        out = mentions_df.copy()
        out["gold_cluster_id"] = pd.Series(dtype="object")
        out["pred_cluster_id"] = pd.Series(dtype="object")
        out["method"] = pd.Series(dtype="object")
        return out

    # Work on positional indices for fast slicing into embeddings
    pos_df = mentions_df.reset_index(drop=True)
    pred_ids: list[str] = [""] * len(pos_df)

    for (split_val, topic_val), grp in pos_df.groupby(["split", "topic_id"], sort=False):
        idx_arr = grp.index.to_numpy()
        topic_emb = embeddings[idx_arr]

        if algorithm == "connected_components":
            local_labels = cluster_topic_connected_components(
                topic_emb, threshold=threshold, batch_size=batch_size
            )
        elif algorithm == "agglomerative":
            local_labels = cluster_topic_agglomerative(
                topic_emb, threshold=threshold, linkage=agglomerative_linkage
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        for pos, cid in zip(idx_arr, local_labels):
            pred_ids[int(pos)] = f"{split_val}::t{int(topic_val)}::{algorithm}_{int(cid)}"

    result = pos_df[
        [
            "mention_id",
            "split",
            "topic_id",
            "paper_id",
            "paper_title",
            "mention_text",
            "normalized_text",
            "start",
            "end",
            "token_length",
            "cluster_id",
            "parent_cluster_id",
        ]
    ].copy()
    result = result.rename(columns={"cluster_id": "gold_cluster_id"})
    result["pred_cluster_id"] = pred_ids
    result["method"] = f"embedding_{algorithm}"

    logger.debug(
        "Similarity clustering (%s @ %.3f) grouped %d mentions into %d clusters.",
        algorithm,
        threshold,
        len(result),
        result["pred_cluster_id"].nunique(),
    )
    return result
