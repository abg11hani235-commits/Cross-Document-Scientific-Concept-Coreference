"""Unit tests for scientific mention text normalization."""

from __future__ import annotations

from src.preprocessing import normalize_mention_text


def test_normalize_bert_variants() -> None:
    assert normalize_mention_text("BERT") == "bert"
    assert normalize_mention_text("bert") == "bert"
    assert normalize_mention_text(" BERT ") == "bert"
    assert (
        normalize_mention_text("Bidirectional Encoder Representations from Transformers")
        == "bidirectional encoder representations from transformers"
    )
    assert normalize_mention_text("BERT") != normalize_mention_text(
        "Bidirectional Encoder Representations from Transformers"
    )


def test_normalize_preserves_technical_tokens_and_symbols() -> None:
    assert normalize_mention_text("ResNet-50") == "resnet-50"
    assert normalize_mention_text("C++") == "c++"
    assert normalize_mention_text("C#") == "c#"
    assert normalize_mention_text("Client/Server") == "client/server"
    assert normalize_mention_text("F1-score") == "f1-score"


def test_normalize_whitespace_unicode_and_outer_punctuation() -> None:
    assert normalize_mention_text("  Self\u2013Attention , ") == "self-attention"
    assert normalize_mention_text("“Transformer\u00a0network”") == "transformer network"
    assert normalize_mention_text("Ｆ１") == "f1"
    assert normalize_mention_text("Rectified Linear Units ( ReLU )") == "rectified linear units ( relu )"
    assert normalize_mention_text("(BERT)") == "bert"
    assert normalize_mention_text("FPN )") == "fpn"


def test_normalize_empty_and_none_values() -> None:
    assert normalize_mention_text("") == ""
    assert normalize_mention_text("   ") == ""
    assert normalize_mention_text(None) == ""
    assert normalize_mention_text(float("nan")) == ""
