"""Unit tests for dataset loading, Parquet schema inspection, and split isolation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    inspect_parquet_schema,
    load_config,
    load_split_dataframe,
    verify_split_isolation,
)


def test_load_config_has_expected_sections() -> None:
    cfg = load_config("config.yaml")
    for section in ("paths", "extraction", "preprocessing", "embedding", "clustering", "threshold_tuning"):
        assert section in cfg


def test_inspect_and_load_real_parquet_splits() -> None:
    cfg = load_config("config.yaml")
    val_path = cfg["paths"]["val_parquet"]
    schema_info = inspect_parquet_schema(val_path)
    assert schema_info["num_rows"] == 100
    col_names = {c["name"] for c in schema_info["columns"]}
    assert {"flatten_tokens", "flatten_mentions", "tokens", "mentions", "relations", "id"}.issubset(col_names)

    df = load_split_dataframe(val_path, "validation")
    assert len(df) == 100
    assert (df["split"] == "validation").all()


def test_verify_split_isolation_detects_overlap() -> None:
    a = pd.DataFrame({"id": [1, 2, 3]})
    b = pd.DataFrame({"id": [3, 4, 5]})
    res = verify_split_isolation({"train": a, "validation": b})
    assert res["is_disjoint"] is False
    assert res["overlaps"]["train__validation"] == [3]


def test_load_split_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_split_dataframe("data/nonexistent_split.parquet", "train")
