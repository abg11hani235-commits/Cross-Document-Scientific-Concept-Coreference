"""Unit tests for pairwise coreference Precision, Recall, and F1."""

from __future__ import annotations

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
    # Topic with 5 mentions:
    # Gold: {0, 1, 2}, {3, 4} -> 3 + 1 = 4 gold pairs
    # Pred: {0, 1}, {2, 3}, {4} -> 1 + 1 = 2 pred pairs
    # Pair (0,1) is TP; (2,3) is FP; (0,2), (1,2), (3,4) are FN -> TP=1, FP=1, FN=3
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


def test_pairwise_metrics_respects_topic_scope() -> None:
    # Identical cluster IDs across two topics must NOT form cross-topic pairs
    df = pd.DataFrame(
        {
            "topic_id": [1, 2],
            "gold_cluster_id": ["shared_name", "shared_name"],
            "pred_cluster_id": ["shared_pred", "shared_pred"],
        }
    )
    metrics = compute_pairwise_metrics(df)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 0, 0)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
