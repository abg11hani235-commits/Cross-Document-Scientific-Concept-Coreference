"""Visualization utilities saving publication-quality figures to outputs/figures/."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")


def _save_fig(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


def plot_mentions_per_split(
    mentions_all_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Plot number of mentions, clusters, and topics across train/validation/test."""
    order = ["train", "validation", "test"]
    summary = (
        mentions_all_df.groupby("split")
        .agg(
            mentions=("mention_id", "count"),
            clusters=("cluster_id", "nunique"),
            topics=("topic_id", "nunique"),
        )
        .reindex(order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(order))
    width = 0.36

    bars1 = axes[0].bar(x - width / 2, summary["mentions"], width, label="Mentions", color="#2b5c8f")
    bars2 = axes[0].bar(x + width / 2, summary["clusters"], width, label="Gold Clusters", color="#4c9f70")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.title() for s in order])
    axes[0].set_ylabel("Count")
    axes[0].set_title("SciCo Mentions and Gold Clusters per Split")
    axes[0].legend()
    axes[0].bar_label(bars1, padding=3, fontsize=9)
    axes[0].bar_label(bars2, padding=3, fontsize=9)

    bars3 = axes[1].bar(x, summary["topics"], width=0.5, color="#d97724")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.title() for s in order])
    axes[1].set_ylabel("Number of Topics")
    axes[1].set_title("SciCo Topics per Split")
    axes[1].bar_label(bars3, padding=3, fontsize=9)

    return _save_fig(fig, out_dir / "01_mentions_per_split.png")


def plot_cluster_size_distribution(
    mentions_all_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Plot gold coreference cluster size distribution across splits."""
    sizes = (
        mentions_all_df.groupby(["split", "cluster_id"])
        .size()
        .reset_index(name="cluster_size")
    )
    sizes["capped_size"] = sizes["cluster_size"].clip(upper=15)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.histplot(
        data=sizes,
        x="capped_size",
        hue="split",
        hue_order=["train", "validation", "test"],
        multiple="dodge",
        discrete=True,
        stat="percent",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlabel("Gold Cluster Size (mentions per cluster, capped at 15+)")
    ax.set_ylabel("Percentage of Clusters (%)")
    ax.set_title("Gold Cluster Size Distribution Across Splits")
    ax.set_xticks(range(1, 16))
    ax.set_xticklabels([str(i) if i < 15 else "15+" for i in range(1, 16)])

    return _save_fig(fig, out_dir / "02_cluster_size_distribution.png")


def plot_mention_length_distribution(
    mentions_all_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Plot mention token length distribution across splits."""
    df = mentions_all_df.copy()
    df["capped_len"] = df["token_length"].clip(upper=10)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.histplot(
        data=df,
        x="capped_len",
        hue="split",
        hue_order=["train", "validation", "test"],
        multiple="dodge",
        discrete=True,
        stat="percent",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlabel("Mention Length (tokens, capped at 10+)")
    ax.set_ylabel("Percentage of Mentions (%)")
    ax.set_title("Mention Token Length Distribution Across Splits")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(i) if i < 10 else "10+" for i in range(1, 11)])

    return _save_fig(fig, out_dir / "03_mention_length_distribution.png")


def plot_validation_threshold_curve(
    threshold_df: pd.DataFrame,
    best_threshold: float,
    out_dir: Path,
) -> Path:
    """Plot Validation Precision, Recall, and F1 against cosine similarity threshold."""
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Pairwise Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="s", label="Pairwise Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="^", linewidth=2.4, label="Pairwise F1")

    best_row = threshold_df.loc[(threshold_df["threshold"] - best_threshold).abs().idxmin()]
    ax.axvline(best_threshold, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Selected Threshold = {best_threshold:.2f}")
    ax.scatter([best_threshold], [best_row["f1"]], color="#c0392b", s=80, zorder=5)

    ax.set_xlabel("Cosine Similarity Threshold")
    ax.set_ylabel("Validation Score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Validation Pairwise Metrics vs. Cosine Similarity Threshold")
    ax.legend(loc="best")

    return _save_fig(fig, out_dir / "04_validation_f1_vs_threshold.png")


def plot_method_comparison(
    comparison_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Plot Test Precision, Recall, and F1 comparison between methods."""
    test_rows = comparison_df[comparison_df["split"] == "test"].copy()
    melted = test_rows.melt(
        id_vars=["method"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    melted["metric"] = melted["metric"].str.upper()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=melted, x="metric", y="score", hue="method", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Pairwise Evaluation Metric")
    ax.set_ylabel("Score (Test Split)")
    ax.set_title("Test Set Performance: Exact Matching Baseline vs. Embedding Clustering")
    ax.legend(title="Method")

    return _save_fig(fig, out_dir / "05_method_comparison.png")


def plot_gold_vs_pred_clusters(
    exact_preds_df: pd.DataFrame,
    emb_preds_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Compare gold vs predicted cluster counts and cluster size distributions on Test."""
    gold_sizes = exact_preds_df.groupby("gold_cluster_id").size().rename("size").to_frame()
    gold_sizes["source"] = "Gold"

    exact_sizes = exact_preds_df.groupby("pred_cluster_id").size().rename("size").to_frame()
    exact_sizes["source"] = "Exact Matching"

    emb_sizes = emb_preds_df.groupby("pred_cluster_id").size().rename("size").to_frame()
    emb_sizes["source"] = "Embedding Clustering"

    combined = pd.concat([gold_sizes, exact_sizes, emb_sizes], ignore_index=True)
    combined["capped_size"] = combined["size"].clip(upper=12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    counts = combined["source"].value_counts().reindex(["Gold", "Exact Matching", "Embedding Clustering"])
    bars = axes[0].bar(counts.index, counts.values, color=["#2b5c8f", "#d97724", "#4c9f70"])
    axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set_ylabel("Number of Clusters (Test Split)")
    axes[0].set_title("Total Clusters: Gold vs. Predicted")

    sns.histplot(
        data=combined,
        x="capped_size",
        hue="source",
        hue_order=["Gold", "Exact Matching", "Embedding Clustering"],
        multiple="dodge",
        discrete=True,
        stat="percent",
        common_norm=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("Cluster Size (capped at 12+)")
    axes[1].set_ylabel("Percentage of Clusters (%)")
    axes[1].set_title("Cluster Size Distribution on Test")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels([str(i) if i < 12 else "12+" for i in range(1, 13)])

    return _save_fig(fig, out_dir / "06_gold_vs_pred_cluster_stats.png")


def plot_error_distribution(
    emb_metrics: dict[str, Any],
    false_merges_df: pd.DataFrame,
    missed_matches_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Plot TP/FP/FN pair counts and similarity distributions of exported errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    pair_labels = ["True Positives\n(Correct Merges)", "False Positives\n(False Merges)", "False Negatives\n(Missed Matches)"]
    pair_vals = [emb_metrics["tp"], emb_metrics["fp"], emb_metrics["fn"]]
    colors = ["#4c9f70", "#c0392b", "#e67e22"]
    bars = axes[0].bar(pair_labels, pair_vals, color=colors)
    axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set_ylabel("Number of Mention Pairs (Test)")
    axes[0].set_title("Test Pairwise Outcomes (Embedding Clustering)")

    sim_frames = []
    if not false_merges_df.empty and "cosine_similarity" in false_merges_df.columns:
        fm = false_merges_df[["cosine_similarity"]].dropna().copy()
        fm["category"] = "False Merges (FP)"
        sim_frames.append(fm)
    if not missed_matches_df.empty and "cosine_similarity" in missed_matches_df.columns:
        mm = missed_matches_df[["cosine_similarity"]].dropna().copy()
        mm["category"] = "Missed Matches (FN)"
        sim_frames.append(mm)

    if sim_frames:
        sim_df = pd.concat(sim_frames, ignore_index=True)
        sns.histplot(
            data=sim_df,
            x="cosine_similarity",
            hue="category",
            bins=25,
            kde=True,
            ax=axes[1],
        )
        axes[1].set_xlabel("Cosine Similarity")
        axes[1].set_ylabel("Pair Count (Sampled Errors)")
        axes[1].set_title("Cosine Similarity of Error Pairs")
    else:
        axes[1].text(0.5, 0.5, "No similarity error pairs available", ha="center", va="center")

    return _save_fig(fig, out_dir / "07_error_category_distribution.png")


def plot_hierarchy_structure(
    hierarchy_report: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Plot SciCo hierarchy tree components, isolated clusters, and depth/branching stats."""
    splits = ["train", "validation", "test"]
    val_stats = hierarchy_report["hierarchy_validation"]
    split_stats = hierarchy_report["split_statistics"]

    tree_comps = [val_stats[s]["total_connected_components"] for s in splits]
    isolated = [val_stats[s]["total_isolated_clusters"] for s in splits]
    mean_depth = [split_stats[s]["mean_tree_depth"] for s in splits]
    max_depth = [split_stats[s]["max_tree_depth"] for s in splits]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.arange(len(splits))
    width = 0.36

    b1 = axes[0].bar(x - width / 2, tree_comps, width, label="Tree Components", color="#2b5c8f")
    b2 = axes[0].bar(x + width / 2, isolated, width, label="Isolated Clusters", color="#95a5a6")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.title() for s in splits])
    axes[0].set_ylabel("Count")
    axes[0].set_title("SciCo Hierarchy Components vs. Isolated Clusters")
    axes[0].legend()
    axes[0].bar_label(b1, padding=3, fontsize=9)
    axes[0].bar_label(b2, padding=3, fontsize=9)

    b3 = axes[1].bar(x - width / 2, mean_depth, width, label="Mean Tree Depth", color="#4c9f70")
    b4 = axes[1].bar(x + width / 2, max_depth, width, label="Max Tree Depth", color="#d97724")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.title() for s in splits])
    axes[1].set_ylabel("Tree Depth (nodes)")
    axes[1].set_title("SciCo Hierarchy Depth Across Splits")
    axes[1].legend()
    axes[1].bar_label(b3, fmt="%.2f", padding=3, fontsize=9)
    axes[1].bar_label(b4, padding=3, fontsize=9)

    return _save_fig(fig, out_dir / "08_hierarchy_structure.png")


def plot_runtime_memory_comparison(
    benchmark_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Plot measured wall-clock runtime and Python peak memory (tracemalloc) by algorithm."""
    df = benchmark_df.copy()
    df["split_label"] = df["split"].str.title()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sns.barplot(data=df, x="split_label", y="runtime_seconds", hue="algorithm", ax=axes[0])
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3fs", padding=3, fontsize=9)
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Wall-Clock Runtime (s)")
    axes[0].set_title("Clustering Runtime Comparison")
    axes[0].legend(title="Algorithm")

    sns.barplot(data=df, x="split_label", y="peak_memory_mb", hue="algorithm", ax=axes[1])
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f MB", padding=3, fontsize=9)
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Peak Traced Memory (MB)")
    axes[1].set_title("Clustering Peak Memory (tracemalloc)")
    axes[1].legend(title="Algorithm")

    return _save_fig(fig, out_dir / "09_runtime_memory_comparison.png")
