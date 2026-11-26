"""Configuration alignment and pipeline output schema verification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data_loader import load_config
from src.pipeline import run_pipeline


def test_config_clustering_fields_are_valid() -> None:
    cfg = load_config("config.yaml")
    clust = cfg["clustering"]
    assert clust["algorithm"] in {"auto", "connected_components", "agglomerative"}
    assert set(clust["candidate_algorithms"]).issubset({"connected_components", "agglomerative"})
    assert clust["agglomerative_linkage"] in {"average", "complete", "single"}
    assert clust["batch_size"] > 0


def test_pipeline_rejects_invalid_configured_algorithm(tmp_path: Path) -> None:
    cfg = load_config("config.yaml")
    cfg["clustering"]["algorithm"] = "invalid_algo"
    bad_cfg = tmp_path / "bad_config.yaml"
    bad_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported clustering.algorithm"):
        run_pipeline(bad_cfg)


def test_generated_pipeline_outputs_and_schemas() -> None:
    root = Path(".")
    # Processed Parquet files
    expected_counts = {"train": 10922, "validation": 4873, "test": 10423}
    for split, n_rows in expected_counts.items():
        df = pd.read_parquet(root / "data" / "processed" / f"{split}_mentions.parquet")
        assert len(df) == n_rows
        for col in (
            "flat_start",
            "flat_end",
            "doc_start",
            "doc_end",
            "coordinate_source",
            "cluster_id",
            "doc_cluster_id",
            "cluster_id_consistent",
        ):
            assert col in df.columns

    # Cluster CSVs
    for csv_name in ("exact_clusters.csv", "embedding_clusters.csv"):
        cdf = pd.read_csv(root / "outputs" / "clusters" / csv_name)
        assert len(cdf) == 10423
        assert {"mention_id", "topic_id", "gold_cluster_id", "pred_cluster_id", "coordinate_source"}.issubset(
            cdf.columns
        )

    # Metrics JSON & CSVs
    metrics_json = json.loads((root / "outputs" / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics_json["split_isolation"]["is_disjoint"] is True
    assert metrics_json["validation_selection"]["frozen_choice"]["algorithm"] == "agglomerative"
    assert metrics_json["validation_selection"]["frozen_choice"]["threshold"] == pytest.approx(0.74)
    assert "runtime_memory_benchmarks" in metrics_json

    hier_json = json.loads((root / "outputs" / "metrics" / "hierarchical_analysis.json").read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        assert hier_json["hierarchy_validation"][split]["is_valid_forest"] is True

    comp_df = pd.read_csv(root / "outputs" / "metrics" / "method_comparison.csv")
    assert len(comp_df) == 4

    bench_df = pd.read_csv(root / "outputs" / "metrics" / "runtime_memory_benchmarks.csv")
    assert len(bench_df) == 4

    # Error CSVs
    for err_name in ("false_merges.csv", "missed_matches.csv"):
        edf = pd.read_csv(root / "outputs" / "errors" / err_name)
        assert not edf.empty
        assert {"topic_id", "mention_text_1", "mention_text_2", "cosine_similarity", "likely_cause"}.issubset(
            edf.columns
        )

    # Figures
    for i in range(1, 10):
        matches = list((root / "outputs" / "figures").glob(f"0{i}_*.png"))
        assert len(matches) == 1, f"Missing figure 0{i}_*.png"
