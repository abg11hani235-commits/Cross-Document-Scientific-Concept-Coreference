"""Exact normalized string matching baseline for cross-document coreference."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def cluster_exact_matching(mentions_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster mentions within each topic by exact normalized string equality.

    Mentions from different topics are never merged. Empty or whitespace-only
    normalized strings are assigned unique singleton clusters so they are never
    falsely merged together.

    Args:
        mentions_df: Extracted mentions DataFrame containing ``mention_id``,
            ``split``, ``topic_id``, ``paper_id``, ``mention_text``,
            ``normalized_text``, and ``cluster_id`` (gold).

    Returns:
        DataFrame with predictions (``pred_cluster_id``) aligned to ``mentions_df``.
    """
    required = {
        "mention_id",
        "split",
        "topic_id",
        "paper_id",
        "mention_text",
        "normalized_text",
        "cluster_id",
    }
    missing = required - set(mentions_df.columns)
    if missing:
        raise ValueError(f"mentions_df is missing required columns: {sorted(missing)}")

    if mentions_df.empty:
        out = mentions_df.copy()
        out["gold_cluster_id"] = pd.Series(dtype="object")
        out["pred_cluster_id"] = pd.Series(dtype="object")
        out["method"] = pd.Series(dtype="object")
        return out

    pred_ids: list[str] = [""] * len(mentions_df)

    # Group strictly within each (split, topic_id) boundary
    for (split_val, topic_val), grp in mentions_df.groupby(["split", "topic_id"], sort=False):
        norm_to_cid: dict[str, str] = {}
        next_cid = 0
        for row_pos, norm_text in zip(grp.index, grp["normalized_text"]):
            key = "" if norm_text is None else str(norm_text).strip()
            if not key:
                # Empty/invalid normalized text becomes a unique singleton
                pred_ids[row_pos] = f"{split_val}::t{int(topic_val)}::exact_{next_cid}"
                next_cid += 1
                continue

            if key not in norm_to_cid:
                norm_to_cid[key] = f"{split_val}::t{int(topic_val)}::exact_{next_cid}"
                next_cid += 1
            pred_ids[row_pos] = norm_to_cid[key]

    result = mentions_df[
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
    result["method"] = "exact_matching"

    logger.info(
        "Exact matching clustered %d mentions into %d predicted clusters.",
        len(result),
        result["pred_cluster_id"].nunique(),
    )
    return result
