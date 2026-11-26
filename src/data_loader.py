"""Dataset loading and PyArrow Parquet schema inspection utilities for SciCo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "flatten_tokens",
    "flatten_mentions",
    "tokens",
    "doc_ids",
    "metadata",
    "sentences",
    "mentions",
    "relations",
    "id",
    "source",
)


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load project YAML configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML configuration in {path}: expected mapping.")
    return cfg


def inspect_parquet_schema(parquet_path: str | Path) -> dict[str, Any]:
    """Inspect a Parquet file's metadata and Arrow schema without loading all rows.

    Args:
        parquet_path: Path to the Parquet file.

    Returns:
        Dictionary summarizing row count, file size, columns, and Arrow types.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    columns = [
        {"name": field.name, "arrow_type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "num_rows": pf.metadata.num_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "columns": columns,
    }


def load_split_dataframe(parquet_path: str | Path, split_name: str) -> pd.DataFrame:
    """Load a SciCo Parquet split using PyArrow engine and validate required columns.

    Args:
        parquet_path: Path to the split Parquet file.
        split_name: Canonical split name (e.g., ``train``, ``validation``, ``test``).

    Returns:
        Loaded Pandas DataFrame with a ``split`` column attached.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Split '{split_name}' Parquet file not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Split '{split_name}' at {path} is missing required columns: {missing}"
        )

    df = df.copy()
    df["split"] = split_name
    logger.info("Loaded split '%s': %d topics from %s", split_name, len(df), path)
    return df


def load_all_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test DataFrames configured in ``config.yaml``.

    Args:
        config: Project configuration dictionary.

    Returns:
        Mapping from ``train``, ``validation``, ``test`` to DataFrames.
    """
    paths = config["paths"]
    splits = {
        "train": load_split_dataframe(paths["train_parquet"], "train"),
        "validation": load_split_dataframe(paths["val_parquet"], "validation"),
        "test": load_split_dataframe(paths["test_parquet"], "test"),
    }
    isolation = verify_split_isolation(splits)
    if not isolation["is_disjoint"]:
        raise ValueError(
            f"Topic leakage detected across splits: {isolation['overlaps']}"
        )
    return splits


def verify_split_isolation(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Verify that topic IDs are strictly disjoint across splits.

    Args:
        splits: Mapping from split name to DataFrame containing ``id``.

    Returns:
        Dictionary with topic counts per split, pairwise overlaps, and ``is_disjoint``.
    """
    topic_sets = {name: set(df["id"].astype(int).tolist()) for name, df in splits.items()}
    names = list(topic_sets.keys())
    overlaps: dict[str, list[int]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = sorted(topic_sets[a] & topic_sets[b])
            if shared:
                overlaps[f"{a}__{b}"] = shared

    return {
        "counts": {k: len(v) for k, v in topic_sets.items()},
        "overlaps": overlaps,
        "is_disjoint": len(overlaps) == 0,
    }
