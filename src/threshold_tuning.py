"""Validation-only similarity threshold selection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.clustering import ClusteringAlgorithm, cluster_mentions_by_similarity
from src.evaluation import compute_pairwise_metrics

logger = logging.getLogger(__name__)


def build_threshold_grid(
    min_threshold: float = 0.50,
    max_threshold: float = 0.95,
    step: float = 0.02,
) -> list[float]:
    """Construct a deterministic rounded threshold grid."""
    if step <= 0:
        raise ValueError(f"Threshold step must be positive, got {step}.")
    if min_threshold > max_threshold:
        raise ValueError(
            f"min_threshold ({min_threshold}) cannot exceed max_threshold ({max_threshold})."
        )
    values: list[float] = []
    cur = min_threshold
    while cur <= max_threshold + 1e-9:
        values.append(round(float(cur), 4))
        cur += step
    if round(max_threshold, 4) not in values:
        values.append(round(float(max_threshold), 4))
    return values


def select_best_threshold(
    results_df: pd.DataFrame,
    *,
    tie_breaker: str = "higher_precision",
) -> dict[str, Any]:
    """Select optimal threshold deterministically using validation F1.

    Args:
        results_df: DataFrame of threshold search metrics from validation.
        tie_breaker: Rule for breaking F1 ties (``higher_precision`` or ``higher_threshold``).

    Returns:
        Dictionary representing the winning row.
    """
    if results_df.empty:
        raise ValueError("Cannot select threshold from empty results DataFrame.")

    if tie_breaker == "higher_precision":
        ordered = results_df.sort_values(
            by=["f1", "precision", "threshold"],
            ascending=[False, False, False],
        )
    elif tie_breaker == "higher_threshold":
        ordered = results_df.sort_values(
            by=["f1", "threshold", "precision"],
            ascending=[False, False, False],
        )
    else:
        raise ValueError(f"Unsupported tie_breaker: {tie_breaker}")

    best_row = ordered.iloc[0].to_dict()
    return {
        k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
        for k, v in best_row.items()
    }


def tune_threshold_on_validation(
    val_mentions_df: pd.DataFrame,
    val_embeddings: np.ndarray,
    *,
    thresholds: list[float] | None = None,
    algorithm: ClusteringAlgorithm = "connected_components",
    agglomerative_linkage: str = "average",
    batch_size: int = 512,
    tie_breaker: str = "higher_precision",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run threshold grid search strictly on validation data and freeze the winner."""
    if not (val_mentions_df["split"] == "validation").all():
        raise ValueError("Threshold tuning must only receive 'validation' split records.")

    grid = thresholds if thresholds is not None else build_threshold_grid()
    rows: list[dict[str, Any]] = []

    for thr in tqdm(grid, desc="Tuning threshold on validation", unit="thr"):
        preds = cluster_mentions_by_similarity(
            val_mentions_df,
            val_embeddings,
            threshold=thr,
            algorithm=algorithm,
            agglomerative_linkage=agglomerative_linkage,
            batch_size=batch_size,
        )
        metrics = compute_pairwise_metrics(preds)
        rows.append(
            {
                "threshold": thr,
                "algorithm": algorithm,
                **metrics,
            }
        )

    results_df = pd.DataFrame(rows)
    best = select_best_threshold(results_df, tie_breaker=tie_breaker)
    logger.info(
        "Selected validation threshold=%.4f (Val F1=%.4f, P=%.4f, R=%.4f)",
        best["threshold"],
        best["f1"],
        best["precision"],
        best["recall"],
    )
    return results_df, best
