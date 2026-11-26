"""Mention extraction and span validation for the SciCo dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.preprocessing import normalize_mention_text

logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    """Summary statistics and anomaly log for mention extraction."""

    split: str
    total_topics: int = 0
    total_raw_mentions: int = 0
    extracted_mentions: int = 0
    flatten_aligned_mentions: int = 0
    doc_fallback_mentions: int = 0
    cluster_id_mismatch_mentions: int = 0
    malformed_mentions: int = 0
    empty_normalized_mentions: int = 0
    anomalies: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to a JSON-compatible dict."""
        return {
            "split": self.split,
            "total_topics": self.total_topics,
            "total_raw_mentions": self.total_raw_mentions,
            "extracted_mentions": self.extracted_mentions,
            "flatten_aligned_mentions": self.flatten_aligned_mentions,
            "doc_fallback_mentions": self.doc_fallback_mentions,
            "cluster_id_mismatch_mentions": self.cluster_id_mismatch_mentions,
            "malformed_mentions": self.malformed_mentions,
            "empty_normalized_mentions": self.empty_normalized_mentions,
            "anomaly_count": len(self.anomalies),
            "anomalies_sample": self.anomalies[:20],
        }


def make_scoped_cluster_id(split: str, topic_id: int, raw_cluster_id: int) -> str:
    """Create a globally unique, topic-scoped cluster identifier.

    Raw cluster IDs in SciCo are 0-based integers local to each topic (e.g., 0, 1, 2).
    Scoping by ``(split, topic_id, raw_cluster_id)`` prevents cross-topic collisions.
    """
    return f"{split}::t{int(topic_id)}::c{int(raw_cluster_id)}"


def _build_parent_lookup(
    relations: Any,
    split: str,
    topic_id: int,
) -> dict[int, str]:
    """Map child raw cluster ID to its scoped parent cluster ID."""
    parent_map: dict[int, str] = {}
    if relations is None:
        return parent_map
    for rel in relations:
        if len(rel) != 2:
            continue
        parent_raw, child_raw = int(rel[0]), int(rel[1])
        parent_map[child_raw] = make_scoped_cluster_id(split, topic_id, parent_raw)
    return parent_map


def _safe_metadata_value(metadata: dict[str, Any] | None, key: str, idx: int) -> str:
    """Safely index a metadata list field by paper index."""
    if not isinstance(metadata, dict):
        return ""
    arr = metadata.get(key)
    if arr is None or idx < 0 or idx >= len(arr):
        return ""
    val = arr[idx]
    return "" if val is None else str(val)


def extract_mentions_from_dataframe(
    df: pd.DataFrame,
    split_name: str,
    *,
    span_end_inclusive: bool = True,
    prefer_doc_tokens_on_mismatch: bool = True,
    preprocessing_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, ExtractionStats]:
    """Extract and validate all mentions from a SciCo split DataFrame.

    SciCo represents spans in ``flatten_mentions`` as ``[start, end, cluster_id]``
    and in ``mentions`` as ``[snippet_idx, doc_start, doc_end, cluster_id]``,
    where both start and end are 0-based inclusive token indices.

    When ``flatten_mentions`` and ``mentions`` disagree or ``flatten_mentions`` is
    out of bounds (observed in Test topic ``id=1034``), this extractor logs the
    anomaly and recovers the exact mention from ``tokens[snippet_idx]`` rather than
    silently dropping or corrupting mentions.

    Args:
        df: Split DataFrame loaded by :func:`src.data_loader.load_split_dataframe`.
        split_name: Split label (``train``, ``validation``, or ``test``).
        span_end_inclusive: Whether ``end`` is inclusive (True for SciCo).
        prefer_doc_tokens_on_mismatch: Whether to fall back to doc-level ``mentions``
            when ``flatten_mentions`` disagrees with ``tokens``.
        preprocessing_cfg: Keyword arguments forwarded to :func:`normalize_mention_text`.

    Returns:
        Tuple of ``(mentions_df, extraction_stats)``.
    """
    prep_kwargs = dict(preprocessing_cfg or {})
    stats = ExtractionStats(split=split_name, total_topics=len(df))
    records: list[dict[str, Any]] = []
    offset_add = 1 if span_end_inclusive else 0

    for row in df.itertuples(index=False):
        topic_id = int(row.id)
        source = str(row.source)
        flat_tokens = list(row.flatten_tokens) if row.flatten_tokens is not None else []
        flat_mentions = list(row.flatten_mentions) if row.flatten_mentions is not None else []
        doc_tokens_list = [list(t) for t in row.tokens] if row.tokens is not None else []
        doc_ids = [int(d) for d in row.doc_ids] if row.doc_ids is not None else []
        doc_mentions = list(row.mentions) if row.mentions is not None else []
        metadata = row.metadata if isinstance(row.metadata, dict) else {}

        parent_lookup = _build_parent_lookup(row.relations, split_name, topic_id)
        has_doc_alignment = len(doc_mentions) == len(flat_mentions) and len(doc_mentions) > 0

        n_flat = len(flat_tokens)
        stats.total_raw_mentions += len(flat_mentions)

        for m_idx, fm in enumerate(flat_mentions):
            if len(fm) != 3:
                stats.malformed_mentions += 1
                stats.anomalies.append(
                    {
                        "topic_id": topic_id,
                        "mention_idx": m_idx,
                        "reason": f"Malformed flatten_mention tuple length {len(fm)}",
                    }
                )
                continue

            f_start, f_end, raw_cid = int(fm[0]), int(fm[1]), int(fm[2])
            f_slice_end = f_end + offset_add

            # Extract document-level coordinates when available
            snippet_idx = -1
            paper_idx = -1
            d_start = -1
            d_end = -1
            d_cid: int | None = None
            cluster_id_consistent: bool | None = None
            doc_text: str | None = None
            doc_valid = False

            if has_doc_alignment and len(doc_mentions[m_idx]) == 4:
                snippet_idx = int(doc_mentions[m_idx][0])
                d_start = int(doc_mentions[m_idx][1])
                d_end = int(doc_mentions[m_idx][2])
                d_cid = int(doc_mentions[m_idx][3])
                d_slice_end = d_end + offset_add
                if 0 <= snippet_idx < len(doc_tokens_list):
                    snip_toks = doc_tokens_list[snippet_idx]
                    if 0 <= d_start and d_slice_end <= len(snip_toks) and d_start < d_slice_end:
                        doc_text = " ".join(snip_toks[d_start:d_slice_end])
                        doc_valid = True
                    if 0 <= snippet_idx < len(doc_ids):
                        paper_idx = doc_ids[snippet_idx]
                cluster_id_consistent = d_cid == raw_cid
                if not cluster_id_consistent:
                    stats.cluster_id_mismatch_mentions += 1
                    logger.warning(
                        "Cluster ID mismatch in split '%s' topic=%d mention=%d: flatten=%d vs doc=%d (preserving flatten gold cluster).",
                        split_name,
                        topic_id,
                        m_idx,
                        raw_cid,
                        d_cid,
                    )
                    stats.anomalies.append(
                        {
                            "topic_id": topic_id,
                            "mention_idx": m_idx,
                            "reason": f"Cluster ID mismatch flatten={raw_cid} doc={d_cid}",
                        }
                    )

            flat_valid = 0 <= f_start and f_slice_end <= n_flat and f_start < f_slice_end
            flat_text = " ".join(flat_tokens[f_start:f_slice_end]) if flat_valid else ""

            coordinate_source = "flatten_tokens"
            mention_text = flat_text

            if not flat_valid:
                if prefer_doc_tokens_on_mismatch and doc_valid and doc_text is not None:
                    coordinate_source = "doc_tokens_fallback"
                    mention_text = doc_text
                    stats.doc_fallback_mentions += 1
                    stats.anomalies.append(
                        {
                            "topic_id": topic_id,
                            "mention_idx": m_idx,
                            "reason": (
                                f"Out-of-bounds flatten_mention ({f_start}, {f_end}) "
                                f"for len={n_flat}; recovered via doc_tokens."
                            ),
                        }
                    )
                else:
                    stats.malformed_mentions += 1
                    stats.anomalies.append(
                        {
                            "topic_id": topic_id,
                            "mention_idx": m_idx,
                            "reason": f"Invalid flatten span ({f_start}, {f_end}) for len={n_flat}",
                        }
                    )
                    continue
            elif doc_valid and doc_text is not None and flat_text != doc_text:
                if prefer_doc_tokens_on_mismatch:
                    coordinate_source = "doc_tokens_fallback"
                    mention_text = doc_text
                    stats.doc_fallback_mentions += 1
                    stats.anomalies.append(
                        {
                            "topic_id": topic_id,
                            "mention_idx": m_idx,
                            "reason": (
                                f"Span mismatch flatten='{flat_text}' vs doc='{doc_text}'; "
                                "using doc_tokens."
                            ),
                        }
                    )
                else:
                    stats.flatten_aligned_mentions += 1
            else:
                stats.flatten_aligned_mentions += 1

            if not mention_text.strip():
                stats.malformed_mentions += 1
                stats.anomalies.append(
                    {
                        "topic_id": topic_id,
                        "mention_idx": m_idx,
                        "reason": "Extracted mention text is empty whitespace.",
                    }
                )
                continue

            normalized_text = normalize_mention_text(mention_text, **prep_kwargs)
            if not normalized_text:
                stats.empty_normalized_mentions += 1

            paper_sha = _safe_metadata_value(metadata, "paper_sha", paper_idx)
            paper_title = _safe_metadata_value(metadata, "title", paper_idx)
            paper_id = paper_sha if paper_sha else f"topic_{topic_id}_paper_{paper_idx}"

            scoped_cid = make_scoped_cluster_id(split_name, topic_id, raw_cid)
            parent_cid = parent_lookup.get(raw_cid)
            token_len = (
                (d_end - d_start + 1)
                if (coordinate_source == "doc_tokens_fallback" and doc_valid)
                else (f_end - f_start + 1)
            )

            records.append(
                {
                    "split": split_name,
                    "topic_id": topic_id,
                    "source": source,
                    "paper_id": paper_id,
                    "paper_index": paper_idx,
                    "snippet_index": snippet_idx,
                    "paper_title": paper_title,
                    "mention_id": f"{split_name}_t{topic_id}_m{m_idx}",
                    "mention_idx": m_idx,
                    "mention_text": mention_text,
                    "normalized_text": normalized_text,
                    "flat_start": f_start,
                    "flat_end": f_end,
                    "start": f_start,
                    "end": f_end,
                    "doc_start": d_start,
                    "doc_end": d_end,
                    "token_length": int(token_len),
                    "raw_cluster_id": raw_cid,
                    "doc_cluster_id": d_cid,
                    "cluster_id_consistent": cluster_id_consistent,
                    "cluster_id": scoped_cid,
                    "parent_cluster_id": parent_cid,
                    "coordinate_source": coordinate_source,
                    "extraction_source": coordinate_source,
                    "is_valid": True,
                }
            )
            stats.extracted_mentions += 1

    if stats.doc_fallback_mentions > 0 or stats.malformed_mentions > 0:
        logger.warning(
            "Split '%s' extraction completed: %d extracted, %d doc_tokens_fallback, %d malformed.",
            split_name,
            stats.extracted_mentions,
            stats.doc_fallback_mentions,
            stats.malformed_mentions,
        )
    else:
        logger.info(
            "Split '%s' extraction completed cleanly: %d/%d mentions aligned.",
            split_name,
            stats.extracted_mentions,
            stats.total_raw_mentions,
        )

    mentions_df = pd.DataFrame.from_records(records)
    return mentions_df, stats
