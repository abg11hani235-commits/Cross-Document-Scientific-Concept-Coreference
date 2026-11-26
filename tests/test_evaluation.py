"""Unit and regression tests for pairwise coreference Precision, Recall, and F1."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from src.evaluation import compute_pairwise_counts_for_scope, compute_pairwise_metrics


def _bruteforce_counts(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    n = len(gold)
    for i in range(n):
        for j in range(i + 1, n):
            sg = gold[i] == gold[j]
            sp = pred[i] == pred[j]
            if sg and sp:
                tp += 1
            elif sp and not sg:
                fp += 1
            elif sg and not sp:
                fn += 1
    return tp, fp, fn


def test_pairwise_counts_match_bruteforce_manual_example() -> None:
    # Gold: {0, 1, 2}, {3, 4} -> 4 gold pairs
    # Pred: {0, 1}, {2, 3}, {4} -> 2 pred pairs
    # TP=1 (0,1), FP=1 (2,3), FN=3 (0,2),(1,2),(3,4)
    gold = ["g0", "g0", "g0", "g1", "g1"]
    pred = ["p0", "p0", "p1", "p1", "p2"]

    tp, fp, fn = compute_pairwise_counts_for_scope(gold, pred)
    assert (tp, fp, fn) == (1, 1, 3)
    assert (tp, fp, fn) == _bruteforce_counts(gold, pred)

    df = pd.DataFrame(
        {
            "topic_id": [1] * 5,
            "gold_cluster_id": gold,
            "pred_cluster_id": pred,
        }
    )
    metrics = compute_pairwise_metrics(df)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 3
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.25)
    assert metrics["f1"] == pytest.approx(2 * 0.5 * 0.25 / 0.75)


def test_pairwise_counts_match_bruteforce_randomized() -> None:
    rng = random.Random(123)
    for _ in range(25):
        n = rng.randint(2, 20)
        gold = [f"g{rng.randint(0, 4)}" for _ in range(n)]
        pred = [f"p{rng.randint(0, 4)}" for _ in range(n)]
        assert compute_pairwise_counts_for_scope(gold, pred) == _bruteforce_counts(gold, pred)


def test_pairwise_metrics_respects_topic_scope_and_duplicate_cluster_ids() -> None:
    # Identical cluster IDs across two topics must NOT form cross-topic pairs
    df = pd.DataFrame(
        {
            "topic_id": [1, 1, 2, 2],
            "gold_cluster_id": ["shared", "other", "shared", "shared"],
            "pred_cluster_id": ["pred", "pred2", "pred", "pred"],
        }
    )
    # Topic 1: gold=(shared,other), pred=(pred,pred2) -> 0 pairs
    # Topic 2: gold=(shared,shared), pred=(pred,pred) -> 1 TP pair
    metrics = compute_pairwise_metrics(df)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 0, 0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_pairwise_metrics_singletons_empty_and_zero_denominators() -> None:
    # All singletons -> zero gold pairs and zero pred pairs -> safe 0.0 metrics
    singleton_df = pd.DataFrame(
        {
            "topic_id": [1, 1, 1],
            "gold_cluster_id": ["g0", "g1", "g2"],
            "pred_cluster_id": ["p0", "p1", "p2"],
        }
    )
    s_metrics = compute_pairwise_metrics(singleton_df)
    assert (s_metrics["tp"], s_metrics["fp"], s_metrics["fn"]) == (0, 0, 0)
    assert s_metrics["precision"] == 0.0
    assert s_metrics["recall"] == 0.0
    assert s_metrics["f1"] == 0.0

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["topic_id", "gold_cluster_id", "pred_cluster_id"])
    e_metrics = compute_pairwise_metrics(empty_df)
    assert e_metrics["num_mentions"] == 0
    assert e_metrics["f1"] == 0.0

    # TP = 0 with FP > 0 and FN > 0 -> precision=0, recall=0, f1=0 (zero F1 denominator)
    disjoint_df = pd.DataFrame(
        {
            "topic_id": [1, 1, 1, 1],
            "gold_cluster_id": ["g0", "g0", "g1", "g1"],
            "pred_cluster_id": ["p0", "p1", "p0", "p1"],
        }
    )
    d_metrics = compute_pairwise_metrics(disjoint_df)
    assert (d_metrics["tp"], d_metrics["fp"], d_metrics["fn"]) == (0, 2, 2)
    assert d_metrics["f1"] == 0.0


def test_pairwise_metrics_rejects_missing_cluster_or_scope_values() -> None:
    df_null_gold = pd.DataFrame(
        {
            "topic_id": [1, 1],
            "gold_cluster_id": ["g0", None],
            "pred_cluster_id": ["p0", "p0"],
        }
    )
    with pytest.raises(ValueError, match="missing/null"):
        compute_pairwise_metrics(df_null_gold)

    df_null_topic = pd.DataFrame(
        {
            "topic_id": [1, None],
            "gold_cluster_id": ["g0", "g1"],
            "pred_cluster_id": ["p0", "p1"],
        }
    )
    with pytest.raises(ValueError, match="missing/null"):
        compute_pairwise_metrics(df_null_topic)
