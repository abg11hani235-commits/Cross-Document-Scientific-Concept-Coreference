"""Unit and regression tests for SciCo mention extraction, span boundaries, and fallback integrity."""

from __future__ import annotations

import logging

import pandas as pd

from src.mention_extractor import extract_mentions_from_dataframe, make_scoped_cluster_id


def _make_sample_df(
    flatten_tokens: list[str],
    flatten_mentions: list[list[int]],
    tokens: list[list[str]],
    mentions: list[list[int]],
    topic_id: int = 10,
    relations: list[list[int]] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": topic_id,
                "source": "curated",
                "flatten_tokens": flatten_tokens,
                "flatten_mentions": flatten_mentions,
                "tokens": tokens,
                "doc_ids": list(range(len(tokens))),
                "metadata": {
                    "paper_sha": [f"sha_{i}" for i in range(len(tokens))],
                    "title": [f"Title {i}" for i in range(len(tokens))],
                },
                "sentences": [[[0, len(t)]] for t in tokens],
                "mentions": mentions,
                "relations": [[0, 1]] if relations is None else relations,
            }
        ]
    )


def test_inclusive_span_extraction_single_and_multi_token() -> None:
    # Span [1, 1] is 1 token ("BERT"); span [3, 5] is 3 tokens ("masked language model")
    toks = ["Using", "BERT", "for", "masked", "language", "model", "."]
    df = _make_sample_df(
        flatten_tokens=toks,
        flatten_mentions=[[1, 1, 0], [3, 5, 1]],
        tokens=[toks],
        mentions=[[0, 1, 1, 0], [0, 3, 5, 1]],
        topic_id=42,
    )
    mdf, stats = extract_mentions_from_dataframe(df, "validation")
    assert stats.extracted_mentions == 2
    assert stats.malformed_mentions == 0
    assert stats.cluster_id_mismatch_mentions == 0
    assert mdf.iloc[0]["mention_text"] == "BERT"
    assert mdf.iloc[0]["token_length"] == 1
    assert mdf.iloc[0]["coordinate_source"] == "flatten_tokens"
    assert mdf.iloc[1]["mention_text"] == "masked language model"
    assert mdf.iloc[1]["token_length"] == 3
    assert mdf.iloc[1]["parent_cluster_id"] == make_scoped_cluster_id("validation", 42, 0)


def test_cluster_id_scoping_prevents_cross_topic_collisions() -> None:
    c1 = make_scoped_cluster_id("train", 100, 0)
    c2 = make_scoped_cluster_id("train", 101, 0)
    c3 = make_scoped_cluster_id("test", 100, 0)
    assert c1 != c2
    assert c1 != c3


def test_out_of_bounds_flatten_mention_recovers_without_overwriting_flat_coords() -> None:
    # Mimics Test topic id=1034 where flatten_mentions is out-of-bounds but doc mentions is valid
    toks = ["Text", "classification", "task", "."]
    df = _make_sample_df(
        flatten_tokens=toks,
        flatten_mentions=[[99, 101, 2]],
        tokens=[toks],
        mentions=[[0, 0, 2, 2]],
        topic_id=1034,
    )
    mdf, stats = extract_mentions_from_dataframe(df, "test")
    assert stats.extracted_mentions == 1
    assert stats.doc_fallback_mentions == 1
    row = mdf.iloc[0]
    assert row["mention_text"] == "Text classification task"
    assert row["coordinate_source"] == "doc_tokens_fallback"
    assert row["extraction_source"] == "doc_tokens_fallback"
    # Original flattened coordinates are preserved and never overwritten by doc coordinates
    assert row["flat_start"] == 99 and row["flat_end"] == 101
    assert row["start"] == 99 and row["end"] == 101
    assert row["doc_start"] == 0 and row["doc_end"] == 2
    assert row["token_length"] == 3


def test_flatten_and_doc_text_disagreement_uses_doc_fallback_explicitly() -> None:
    flat_toks = ["wrong", "span", "here", "today"]
    doc_toks = ["neural", "machine", "translation"]
    df = _make_sample_df(
        flatten_tokens=flat_toks,
        flatten_mentions=[[0, 1, 5]],
        tokens=[doc_toks],
        mentions=[[0, 0, 2, 5]],
        topic_id=88,
    )
    mdf, stats = extract_mentions_from_dataframe(df, "test")
    assert stats.doc_fallback_mentions == 1
    row = mdf.iloc[0]
    assert row["mention_text"] == "neural machine translation"
    assert row["coordinate_source"] == "doc_tokens_fallback"
    assert row["flat_start"] == 0 and row["flat_end"] == 1
    assert row["doc_start"] == 0 and row["doc_end"] == 2
    assert row["token_length"] == 3


def test_cluster_id_mismatch_is_logged_and_gold_annotation_preserved(caplog) -> None:
    toks = ["Using", "BERT", "models"]
    df = _make_sample_df(
        flatten_tokens=toks,
        flatten_mentions=[[1, 1, 3]],
        tokens=[toks],
        mentions=[[0, 1, 1, 7]],
        topic_id=55,
    )
    with caplog.at_level(logging.WARNING):
        mdf, stats = extract_mentions_from_dataframe(df, "validation")

    assert stats.cluster_id_mismatch_mentions == 1
    assert any("Cluster ID mismatch" in rec.message for rec in caplog.records)
    row = mdf.iloc[0]
    assert row["raw_cluster_id"] == 3
    assert row["doc_cluster_id"] == 7
    assert bool(row["cluster_id_consistent"]) is False
    assert row["cluster_id"] == make_scoped_cluster_id("validation", 55, 3)


def test_missing_doc_fallback_handles_valid_and_invalid_flatten_safely() -> None:
    toks = ["Deep", "learning", "model"]
    # Case 1: valid flatten span when doc mentions are missing -> succeeds via flatten_tokens
    df_valid = _make_sample_df(
        flatten_tokens=toks,
        flatten_mentions=[[0, 1, 0]],
        tokens=[toks],
        mentions=[],
        topic_id=7,
    )
    mdf_valid, stats_valid = extract_mentions_from_dataframe(df_valid, "train")
    assert len(mdf_valid) == 1
    assert mdf_valid.iloc[0]["coordinate_source"] == "flatten_tokens"
    assert pd.isna(mdf_valid.iloc[0]["doc_cluster_id"])
    assert pd.isna(mdf_valid.iloc[0]["cluster_id_consistent"])
    assert mdf_valid.iloc[0]["doc_start"] == -1
    assert stats_valid.flatten_aligned_mentions == 1

    # Case 2: out-of-bounds flatten span when doc mentions are missing -> logged and skipped
    df_bad = _make_sample_df(
        flatten_tokens=toks,
        flatten_mentions=[[5, 10, 0]],
        tokens=[toks],
        mentions=[],
        topic_id=8,
    )
    mdf_bad, stats_bad = extract_mentions_from_dataframe(df_bad, "train")
    assert len(mdf_bad) == 0
    assert stats_bad.malformed_mentions == 1
    assert len(stats_bad.anomalies) == 1
