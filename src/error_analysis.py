"""Error analysis for false merges (FP) and missed matches (FN)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.clustering import l2_normalize

logger = logging.getLogger(__name__)


def _infer_error_cause(
    text1: str,
    text2: str,
    norm1: str,
    norm2: str,
    paper1: str,
    paper2: str,
    cos_sim: float | None,
    threshold: float | None,
    category: str,
) -> str:
    """Assign an evidence-based heuristic cause label to an error pair."""
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    t1_alpha = "".join(ch for ch in text1 if ch.isalpha())
    t2_alpha = "".join(ch for ch in text2 if ch.isalpha())
    is_acronym_pair = (
        (t1_alpha.isupper() and 2 <= len(t1_alpha) <= 8 and len(tokens2) >= 2)
        or (t2_alpha.isupper() and 2 <= len(t2_alpha) <= 8 and len(tokens1) >= 2)
    )

    if category == "false_merge":
        if threshold is not None and cos_sim is not None and cos_sim < threshold - 1e-5:
            return "transitive_chaining_below_threshold"
        if tokens1 & tokens2:
            return "lexical_overlap_different_concept"
        if paper1 == paper2:
            return "same_paper_high_semantic_similarity"
        return "high_semantic_similarity_distinct_concepts"

    # category == "missed_match"
    if is_acronym_pair:
        return "acronym_vs_expanded_or_variant"
    if norm1 == norm2:
        return "unexpected_split_identical_normalized"
    if tokens1 & tokens2:
        return "partial_lexical_overlap_below_threshold"
    if paper1 != paper2:
        return "cross_paper_terminology_variation"
    return "different_surface_expression"


def analyze_coreference_errors(
    predictions_df: pd.DataFrame,
    *,
    embeddings: np.ndarray | None = None,
    threshold: float | None = None,
    max_examples_per_category: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract false merges (FP pairs) and missed matches (FN pairs) within topics.

    Args:
        predictions_df: DataFrame with ``topic_id``, ``paper_id``, ``mention_id``,
            ``mention_text``, ``normalized_text``, ``gold_cluster_id``, ``pred_cluster_id``.
        embeddings: Optional embedding matrix aligned with ``predictions_df`` rows.
        threshold: Optional similarity threshold used to produce ``predictions_df``
            (used to detect transitive connected-component merges).
        max_examples_per_category: Maximum rows exported per error category CSV.

    Returns:
        Tuple of ``(false_merges_df, missed_matches_df, summary_dict)``.
    """
    pos_df = predictions_df.reset_index(drop=True)
    norm_emb = l2_normalize(embeddings) if embeddings is not None else None

    false_merges: list[dict[str, Any]] = []
    missed_matches: list[dict[str, Any]] = []
    total_fp_pairs = 0
    total_fn_pairs = 0

    for topic_id, grp in pos_df.groupby("topic_id", sort=False):
        idxs = grp.index.to_numpy()
        n = len(idxs)
        if n < 2:
            continue

        gold = grp["gold_cluster_id"].tolist()
        pred = grp["pred_cluster_id"].tolist()
        mids = grp["mention_id"].tolist()
        pids = grp["paper_id"].tolist()
        texts = grp["mention_text"].tolist()
        norms = grp["normalized_text"].tolist()

        # Index members by predicted cluster for FP enumeration
        by_pred: dict[str, list[int]] = {}
        by_gold: dict[str, list[int]] = {}
        for local_i in range(n):
            by_pred.setdefault(pred[local_i], []).append(local_i)
            by_gold.setdefault(gold[local_i], []).append(local_i)

        # 1. False Merges: same pred cluster, different gold cluster
        for members in by_pred.values():
            if len(members) < 2:
                continue
            for a_pos in range(len(members)):
                i = members[a_pos]
                for b_pos in range(a_pos + 1, len(members)):
                    j = members[b_pos]
                    if gold[i] != gold[j]:
                        total_fp_pairs += 1
                        cos_sim = (
                            float(np.dot(norm_emb[idxs[i]], norm_emb[idxs[j]]))
                            if norm_emb is not None
                            else None
                        )
                        false_merges.append(
                            {
                                "error_category": "false_merge",
                                "topic_id": int(topic_id),
                                "mention_id_1": mids[i],
                                "mention_id_2": mids[j],
                                "paper_id_1": pids[i],
                                "paper_id_2": pids[j],
                                "same_paper": pids[i] == pids[j],
                                "mention_text_1": texts[i],
                                "mention_text_2": texts[j],
                                "normalized_text_1": norms[i],
                                "normalized_text_2": norms[j],
                                "gold_cluster_id_1": gold[i],
                                "gold_cluster_id_2": gold[j],
                                "pred_cluster_id_1": pred[i],
                                "pred_cluster_id_2": pred[j],
                                "cosine_similarity": round(cos_sim, 6) if cos_sim is not None else None,
                                "likely_cause": _infer_error_cause(
                                    texts[i],
                                    texts[j],
                                    norms[i],
                                    norms[j],
                                    pids[i],
                                    pids[j],
                                    cos_sim,
                                    threshold,
                                    "false_merge",
                                ),
                            }
                        )

        # 2. Missed Matches: same gold cluster, different pred cluster
        for members in by_gold.values():
            if len(members) < 2:
                continue
            for a_pos in range(len(members)):
                i = members[a_pos]
                for b_pos in range(a_pos + 1, len(members)):
                    j = members[b_pos]
                    if pred[i] != pred[j]:
                        total_fn_pairs += 1
                        cos_sim = (
                            float(np.dot(norm_emb[idxs[i]], norm_emb[idxs[j]]))
                            if norm_emb is not None
                            else None
                        )
                        missed_matches.append(
                            {
                                "error_category": "missed_match",
                                "topic_id": int(topic_id),
                                "mention_id_1": mids[i],
                                "mention_id_2": mids[j],
                                "paper_id_1": pids[i],
                                "paper_id_2": pids[j],
                                "same_paper": pids[i] == pids[j],
                                "mention_text_1": texts[i],
                                "mention_text_2": texts[j],
                                "normalized_text_1": norms[i],
                                "normalized_text_2": norms[j],
                                "gold_cluster_id_1": gold[i],
                                "gold_cluster_id_2": gold[j],
                                "pred_cluster_id_1": pred[i],
                                "pred_cluster_id_2": pred[j],
                                "cosine_similarity": round(cos_sim, 6) if cos_sim is not None else None,
                                "likely_cause": _infer_error_cause(
                                    texts[i],
                                    texts[j],
                                    norms[i],
                                    norms[j],
                                    pids[i],
                                    pids[j],
                                    cos_sim,
                                    threshold,
                                    "missed_match",
                                ),
                            }
                        )

    fm_df = pd.DataFrame.from_records(false_merges)
    mm_df = pd.DataFrame.from_records(missed_matches)

    summary: dict[str, Any] = {
        "total_false_merges": int(total_fp_pairs),
        "total_missed_matches": int(total_fn_pairs),
        "false_merge_causes": (
            fm_df["likely_cause"].value_counts().to_dict() if not fm_df.empty else {}
        ),
        "missed_match_causes": (
            mm_df["likely_cause"].value_counts().to_dict() if not mm_df.empty else {}
        ),
    }

    if norm_emb is not None:
        if not fm_df.empty:
            summary["false_merge_similarity_mean"] = float(fm_df["cosine_similarity"].mean())
            summary["false_merge_similarity_median"] = float(fm_df["cosine_similarity"].median())
        if not mm_df.empty:
            summary["missed_match_similarity_mean"] = float(mm_df["cosine_similarity"].mean())
            summary["missed_match_similarity_median"] = float(mm_df["cosine_similarity"].median())

    if not fm_df.empty:
        fm_df = fm_df.sort_values(
            by=["cosine_similarity", "topic_id", "mention_id_1"],
            ascending=[False, True, True],
            na_position="last",
        ).head(max_examples_per_category).reset_index(drop=True)

    if not mm_df.empty:
        mm_df = mm_df.sort_values(
            by=["cosine_similarity", "topic_id", "mention_id_1"],
            ascending=[False, True, True],
            na_position="last",
        ).head(max_examples_per_category).reset_index(drop=True)

    logger.info(
        "Error analysis complete: %d false merges, %d missed matches (exporting up to %d each).",
        total_fp_pairs,
        total_fn_pairs,
        max_examples_per_category,
    )
    return fm_df, mm_df, summary
