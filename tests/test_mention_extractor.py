"""Unit tests for SciCo mention extraction, span boundaries, and anomaly handling."""

from __future__ import annotations

import pandas as pd

from src.mention_extractor import extract_mentions_from_dataframe, make_scoped_cluster_id


def _make_sample_df(
    flatten_tokens: list[str],
    flatten_mentions: list[list[int]],
    tokens: list[list[str]],
    mentions: list[list[int]],
    topic_id: int = 10,
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
                "relations": [[0, 1]],
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
    assert mdf.iloc[0]["mention_text"] == "BERT"
    assert mdf.iloc[0]["token_length"] == 1
    assert mdf.iloc[1]["mention_text"] == "masked language model"
    assert mdf.iloc[1]["token_length"] == 3
    assert mdf.iloc[1]["parent_cluster_id"] == make_scoped_cluster_id("validation", 42, 0)


def test_cluster_id_scoping_prevents_cross_topic_collisions() -> None:
    c1 = make_scoped_cluster_id("train", 100, 0)
    c2 = make_scoped_cluster_id("train", 101, 0)
    c3 = make_scoped_cluster_id("test", 100, 0)
    assert c1 != c2
    assert c1 != c3


def test_out_of_bounds_flatten_mention_recovers_via_doc_tokens() -> None:
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
    assert mdf.iloc[0]["mention_text"] == "Text classification task"
    assert mdf.iloc[0]["extraction_source"] == "doc_tokens_fallback"


def test_malformed_mention_without_fallback_is_logged_and_skipped() -> None:
    toks = ["Deep", "learning"]
    df = _make_sample_df(
        flatten_tokens=toks,
        flatten_mentions=[[5, 10, 0]],
        tokens=[toks],
        mentions=[],
        topic_id=7,
    )
    mdf, stats = extract_mentions_from_dataframe(df, "train")
    assert len(mdf) == 0
    assert stats.malformed_mentions == 1
    assert len(stats.anomalies) == 1
