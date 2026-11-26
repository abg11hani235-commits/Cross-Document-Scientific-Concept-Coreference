"""Pairwise coreference evaluation metrics (Precision, Recall, F1)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import pandas as pd


def _comb2(n: int) -> int:
    """Return n choose 2."""
    return n * (n - 1) // 2


def compute_pairwise_counts_for_scope(
    gold_labels: Sequence[Any],
    pred_labels: Sequence[Any],
) -> tuple[int, int, int]:
    """Compute exact pairwise (TP, FP, FN) within a single evaluation scope in O(N).

    Uses cluster contingency counts rather than explicit O(N^2) pair enumeration:
      - TP = sum_{(g, p)} C(count(g, p), 2)
      - FP = sum_p C(|p|, 2) - TP
      - FN = sum_g C(|g|, 2) - TP
    """
    if len(gold_labels) != len(pred_labels):
        raise ValueError(
            f"Length mismatch between gold ({len(gold_labels)}) and pred ({len(pred_labels)})."
        )

    gold_counts = Counter(gold_labels)
    pred_counts = Counter(pred_labels)
    joint_counts = Counter(zip(gold_labels, pred_labels))

    tp = sum(_comb2(c) for c in joint_counts.values())
    total_pred_pairs = sum(_comb2(c) for c in pred_counts.values())
    total_gold_pairs = sum(_comb2(c) for c in gold_counts.values())

    fp = total_pred_pairs - tp
    fn = total_gold_pairs - tp
    return tp, fp, fn


def compute_pairwise_metrics(
    predictions_df: pd.DataFrame,
    *,
    gold_col: str = "gold_cluster_id",
    pred_col: str = "pred_cluster_id",
    scope_col: str = "topic_id",
) -> dict[str, Any]:
    """Evaluate pairwise coreference Precision, Recall, and F1 within topic scope.

    Pairs are only formed between mentions sharing the same ``scope_col`` (topic),
    matching SciCo's topic-level coreference annotation scope. Handles zero
    denominators safely by returning ``0.0``.
    """
    required = {gold_col, pred_col, scope_col}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df missing required columns: {sorted(missing)}")

    if predictions_df.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "num_gold_clusters": 0,
            "num_pred_clusters": 0,
            "num_mentions": 0,
            "num_topics": 0,
        }

    for col in (scope_col, gold_col, pred_col):
        if predictions_df[col].isna().any():
            raise ValueError(f"predictions_df contains missing/null values in column '{col}'.")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for _, grp in predictions_df.groupby(scope_col, sort=False):
        tp, fp, fn = compute_pairwise_counts_for_scope(
            grp[gold_col].tolist(),
            grp[pred_col].tolist(),
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "num_gold_clusters": int(predictions_df[gold_col].nunique()),
        "num_pred_clusters": int(predictions_df[pred_col].nunique()),
        "num_mentions": int(len(predictions_df)),
        "num_topics": int(predictions_df[scope_col].nunique()),
    }
