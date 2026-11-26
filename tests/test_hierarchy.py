"""Unit and regression tests for SciCo hierarchical relation validation."""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_config, load_split_dataframe
from src.hierarchy import validate_split_hierarchy
from src.mention_extractor import extract_mentions_from_dataframe, make_scoped_cluster_id


def _make_hierarchy_inputs(
    topic_id: int,
    cluster_ids: list[int],
    relations: list[list[int]],
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame([{"id": topic_id, "relations": relations}])
    mentions_df = pd.DataFrame(
        [
            {
                "topic_id": topic_id,
                "cluster_id": make_scoped_cluster_id(split, topic_id, cid),
                "mention_text": f"concept_{cid}",
            }
            for cid in cluster_ids
        ]
    )
    return raw_df, mentions_df


def test_validate_hierarchy_detects_cycles_multi_parent_missing_dup_and_self() -> None:
    # Topic 1: cycle 0->1->0, self-relation 2->2, duplicate 0->1, missing parent 99->1, multi-parent 0->1 & 3->1
    raw_df, mentions_df = _make_hierarchy_inputs(
        topic_id=1,
        cluster_ids=[0, 1, 2, 3, 4],  # cluster 4 is isolated
        relations=[[0, 1], [1, 0], [0, 1], [2, 2], [3, 1], [99, 50]],
    )
    report = validate_split_hierarchy(raw_df, mentions_df, "test")
    assert report["directed_cycle_topics"] == 1
    assert report["directed_cycle_edges"] >= 1
    assert report["self_relations"] == 1
    assert report["duplicate_relationships"] == 1
    assert report["missing_parent_clusters"] == 1
    assert report["missing_child_clusters"] == 1
    assert report["multi_parent_nodes"] == 1
    assert report["total_isolated_clusters"] == 1
    assert report["is_acyclic"] is False
    assert report["is_valid_forest"] is False


def test_validate_hierarchy_tracks_connectivity_on_clean_forest() -> None:
    # Two disconnected trees in one topic: (0 -> 1 -> 2) and (3 -> 4)
    raw_df, mentions_df = _make_hierarchy_inputs(
        topic_id=10,
        cluster_ids=[0, 1, 2, 3, 4, 5],
        relations=[[0, 1], [1, 2], [3, 4]],
    )
    report = validate_split_hierarchy(raw_df, mentions_df, "test")
    assert report["is_valid_forest"] is True
    assert report["is_acyclic"] is True
    assert report["total_connected_components"] == 2
    assert report["multi_component_topics"] == 1
    assert report["total_roots"] == 2
    assert report["total_isolated_clusters"] == 1


def test_real_scico_validation_split_hierarchy_is_valid_forest() -> None:
    cfg = load_config("config.yaml")
    val_raw = load_split_dataframe(cfg["paths"]["val_parquet"], "validation")
    val_mentions, _ = extract_mentions_from_dataframe(val_raw, "validation")
    report = validate_split_hierarchy(val_raw, val_mentions, "validation")
    assert report["is_valid_forest"] is True
    assert report["directed_cycle_edges"] == 0
    assert report["multi_parent_nodes"] == 0
    assert report["self_relations"] == 0
    assert report["missing_parent_clusters"] == 0
    assert report["missing_child_clusters"] == 0
