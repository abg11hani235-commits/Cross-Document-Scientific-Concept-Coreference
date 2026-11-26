"""Unit tests for exact matching, embeddings, clustering, and threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    benchmark_clustering,
    cluster_mentions_by_similarity,
    cluster_topic_agglomerative,
    cluster_topic_connected_components,
)
from src.embedding_model import MentionEmbedder
from src.exact_matching import cluster_exact_matching
from src.threshold_tuning import select_best_threshold


def _sample_mentions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mention_id": "m0",
                "split": "validation",
                "topic_id": 1,
                "paper_id": "p1",
                "paper_title": "Paper 1",
                "mention_text": "BERT",
                "normalized_text": "bert",
                "start": 0,
                "end": 0,
                "token_length": 1,
                "cluster_id": "validation::t1::c0",
                "parent_cluster_id": None,
            },
            {
                "mention_id": "m1",
                "split": "validation",
                "topic_id": 1,
                "paper_id": "p2",
                "paper_title": "Paper 2",
                "mention_text": " bert ",
                "normalized_text": "bert",
                "start": 5,
                "end": 5,
                "token_length": 1,
                "cluster_id": "validation::t1::c0",
                "parent_cluster_id": None,
            },
            {
                "mention_id": "m2",
                "split": "validation",
                "topic_id": 2,
                "paper_id": "p3",
                "paper_title": "Paper 3",
                "mention_text": "BERT",
                "normalized_text": "bert",
                "start": 2,
                "end": 2,
                "token_length": 1,
                "cluster_id": "validation::t2::c0",
                "parent_cluster_id": None,
            },
            {
                "mention_id": "m3",
                "split": "validation",
                "topic_id": 1,
                "paper_id": "p4",
                "paper_title": "Paper 4",
                "mention_text": "",
                "normalized_text": "",
                "start": 10,
                "end": 10,
                "token_length": 1,
                "cluster_id": "validation::t1::c1",
                "parent_cluster_id": None,
            },
            {
                "mention_id": "m4",
                "split": "validation",
                "topic_id": 1,
                "paper_id": "p5",
                "paper_title": "Paper 5",
                "mention_text": "",
                "normalized_text": "",
                "start": 12,
                "end": 12,
                "token_length": 1,
                "cluster_id": "validation::t1::c2",
                "parent_cluster_id": None,
            },
        ]
    )


def test_exact_matching_respects_topic_isolation_and_empty_singletons() -> None:
    df = _sample_mentions_df()
    preds = cluster_exact_matching(df)
    # m0 and m1 share topic 1 and "bert" -> same cluster
    assert preds.loc[0, "pred_cluster_id"] == preds.loc[1, "pred_cluster_id"]
    # m2 is in topic 2 -> must NOT merge with m0
    assert preds.loc[2, "pred_cluster_id"] != preds.loc[0, "pred_cluster_id"]
    # m3 and m4 are empty -> must stay separate singletons
    assert preds.loc[3, "pred_cluster_id"] != preds.loc[4, "pred_cluster_id"]


def test_similarity_clustering_topic_isolation_and_transitivity() -> None:
    # Construct 3 embeddings in topic 1: A=(1,0), B=(0.8,0.6), C=(0.28,0.96)
    # cos(A, B) = 0.80, cos(B, C) = 0.80, cos(A, C) = 0.28
    emb = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.28, 0.96],
            [1.0, 0.0],  # topic 2 duplicate of A -> must NOT merge across topics
        ],
        dtype=np.float32,
    )
    cc_labels = cluster_topic_connected_components(emb[:3], threshold=0.75)
    # Connected components chains A-B-C into 1 cluster
    assert len(set(cc_labels)) == 1

    agg_labels = cluster_topic_agglomerative(emb[:3], threshold=0.75, linkage="average")
    # Average linkage blocks merging {A,B} with C since avg sim < 0.75
    assert len(set(agg_labels)) == 2

    df = pd.DataFrame(
        [
            {
                "mention_id": f"m{i}",
                "split": "validation",
                "topic_id": 1 if i < 3 else 2,
                "paper_id": f"p{i}",
                "paper_title": f"P{i}",
                "mention_text": f"t{i}",
                "normalized_text": f"t{i}",
                "start": 0,
                "end": 0,
                "token_length": 1,
                "cluster_id": f"c{i}",
                "parent_cluster_id": None,
            }
            for i in range(4)
        ]
    )
    preds = cluster_mentions_by_similarity(df, emb, threshold=0.75, algorithm="connected_components")
    assert preds.loc[3, "pred_cluster_id"] != preds.loc[0, "pred_cluster_id"]


def test_threshold_selection_deterministic_tie_breaking() -> None:
    results = pd.DataFrame(
        [
            {"threshold": 0.60, "f1": 0.70, "precision": 0.65, "recall": 0.76},
            {"threshold": 0.70, "f1": 0.70, "precision": 0.72, "recall": 0.68},
            {"threshold": 0.80, "f1": 0.68, "precision": 0.80, "recall": 0.59},
        ]
    )
    best = select_best_threshold(results, tie_breaker="higher_precision")
    assert best["threshold"] == 0.70


def test_embedding_output_dimensions_and_empty_handling() -> None:
    embedder = MentionEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=None)
    vecs = embedder.encode_texts(["BERT", "", "BERT", "transformer"], show_progress_bar=False)
    assert vecs.shape == (4, 384)
    # Duplicate texts receive identical embeddings
    assert np.allclose(vecs[0], vecs[2])
    # Empty string receives zero vector safely
    assert np.allclose(vecs[1], 0.0)


def test_clustering_zero_vectors_and_parameter_validation() -> None:
    emb = np.array(
        [
            [0.0, 0.0],  # zero vector -> must stay singleton
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],  # second zero vector -> must NOT merge with first zero vector
        ],
        dtype=np.float32,
    )
    cc_labels = cluster_topic_connected_components(emb, threshold=0.8)
    agg_labels = cluster_topic_agglomerative(emb, threshold=0.8, linkage="average")
    for lbls in (cc_labels, agg_labels):
        assert list(lbls) == [0, 1, 1, 2]

    with pytest.raises(ValueError, match="Unsupported cosine linkage"):
        cluster_topic_agglomerative(emb, threshold=0.8, linkage="ward")

    with pytest.raises(ValueError, match="Cosine threshold"):
        cluster_topic_connected_components(emb, threshold=1.5)


def test_provenance_columns_preserved_and_benchmark_helper() -> None:
    df = _sample_mentions_df().copy()
    df["flat_start"] = df["start"]
    df["flat_end"] = df["end"]
    df["doc_start"] = df["start"]
    df["doc_end"] = df["end"]
    df["snippet_index"] = 0
    df["coordinate_source"] = "flatten_tokens"

    exact_out = cluster_exact_matching(df)
    for col in ("flat_start", "flat_end", "doc_start", "doc_end", "snippet_index", "coordinate_source"):
        assert col in exact_out.columns

    emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    emb_out = cluster_mentions_by_similarity(df, emb, threshold=0.8, algorithm="agglomerative")
    for col in ("flat_start", "flat_end", "doc_start", "doc_end", "snippet_index", "coordinate_source"):
        assert col in emb_out.columns

    bench = benchmark_clustering(df, emb, split_name="validation", threshold=0.8)
    assert len(bench) == 2
    assert {r["algorithm"] for r in bench} == {"connected_components", "agglomerative"}
    assert all(r["runtime_seconds"] >= 0.0 and r["peak_memory_mb"] >= 0.0 for r in bench)
