"""End-to-end reproducible pipeline for Cross-Document Scientific Concept Coreference."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.clustering import benchmark_clustering, cluster_mentions_by_similarity
from src.data_loader import load_all_splits, load_config, verify_split_isolation
from src.embedding_model import MentionEmbedder
from src.error_analysis import analyze_coreference_errors
from src.evaluation import compute_pairwise_metrics
from src.exact_matching import cluster_exact_matching
from src.hierarchy import analyze_cluster_hierarchy
from src.mention_extractor import extract_mentions_from_dataframe
from src.threshold_tuning import (
    build_threshold_grid,
    select_best_threshold,
    tune_threshold_on_validation,
)
from src.visualization import (
    plot_cluster_size_distribution,
    plot_error_distribution,
    plot_gold_vs_pred_clusters,
    plot_hierarchy_structure,
    plot_mention_length_distribution,
    plot_mentions_per_split,
    plot_method_comparison,
    plot_runtime_memory_comparison,
    plot_validation_threshold_curve,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_pipeline(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Execute the full SciCo coreference pipeline with strict validation/test separation."""
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    paths = cfg["paths"]
    processed_dir = Path(paths["processed_dir"])
    clusters_dir = Path(paths["clusters_dir"])
    metrics_dir = Path(paths["metrics_dir"])
    errors_dir = Path(paths["errors_dir"])
    figures_dir = Path(paths["figures_dir"])
    for d in (processed_dir, clusters_dir, metrics_dir, errors_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets and verify topic-level split isolation
    splits = load_all_splits(cfg)
    isolation = verify_split_isolation(splits)

    # 2. Extract and validate mentions for all splits
    mentions_by_split: dict[str, pd.DataFrame] = {}
    extraction_reports: dict[str, Any] = {}
    ext_cfg = cfg["extraction"]
    prep_cfg = cfg["preprocessing"]

    for split_name, raw_df in splits.items():
        mdf, stats = extract_mentions_from_dataframe(
            raw_df,
            split_name,
            span_end_inclusive=bool(ext_cfg.get("span_end_inclusive", True)),
            prefer_doc_tokens_on_mismatch=bool(ext_cfg.get("prefer_doc_tokens_on_mismatch", True)),
            preprocessing_cfg=prep_cfg,
        )
        mentions_by_split[split_name] = mdf
        extraction_reports[split_name] = stats.to_dict()
        mdf.to_parquet(processed_dir / f"{split_name}_mentions.parquet", index=False)

    mentions_all_df = pd.concat(
        [mentions_by_split["train"], mentions_by_split["validation"], mentions_by_split["test"]],
        ignore_index=True,
    )

    # 3. Baseline: Exact normalized string matching on Validation and Test
    val_exact_preds = cluster_exact_matching(mentions_by_split["validation"])
    test_exact_preds = cluster_exact_matching(mentions_by_split["test"])
    val_exact_metrics = compute_pairwise_metrics(val_exact_preds)
    test_exact_metrics = compute_pairwise_metrics(test_exact_preds)
    test_exact_preds.to_csv(clusters_dir / "exact_clusters.csv", index=False)

    # 4. Generate/load cached Sentence-Transformer embeddings
    emb_cfg = cfg["embedding"]
    embedder = MentionEmbedder(
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device", "auto"),
        batch_size=int(emb_cfg.get("batch_size", 128)),
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
        cache_dir=paths["cache_dir"],
    )
    use_cache = bool(emb_cfg.get("cache_enabled", True))
    val_embeddings = embedder.encode_mentions(
        mentions_by_split["validation"], "validation", use_cache=use_cache
    )
    test_embeddings = embedder.encode_mentions(
        mentions_by_split["test"], "test", use_cache=use_cache
    )

    # 5. Validation-only threshold & algorithm selection
    clust_cfg = cfg["clustering"]
    tune_cfg = cfg["threshold_tuning"]
    grid = build_threshold_grid(
        min_threshold=float(tune_cfg["min_threshold"]),
        max_threshold=float(tune_cfg["max_threshold"]),
        step=float(tune_cfg["step"]),
    )
    tie_breaker = str(tune_cfg.get("tie_breaker", "higher_precision"))
    linkage = str(clust_cfg.get("agglomerative_linkage", "average"))
    clust_batch = int(clust_cfg.get("batch_size", 512))
    configured_algo = str(clust_cfg.get("algorithm", "auto"))

    if configured_algo == "auto":
        candidates = [
            str(a)
            for a in clust_cfg.get("candidate_algorithms", ["connected_components", "agglomerative"])
        ]
    elif configured_algo in {"connected_components", "agglomerative"}:
        candidates = [configured_algo]
    else:
        raise ValueError(f"Unsupported clustering.algorithm in config: {configured_algo}")

    tuning_frames: list[pd.DataFrame] = []
    best_by_algo: dict[str, dict[str, Any]] = {}
    for algo_name in candidates:
        algo_df, algo_best = tune_threshold_on_validation(
            mentions_by_split["validation"],
            val_embeddings,
            thresholds=grid,
            algorithm=algo_name,  # type: ignore[arg-type]
            agglomerative_linkage=linkage,
            batch_size=clust_batch,
            tie_breaker=tie_breaker,
        )
        tuning_frames.append(algo_df)
        best_by_algo[algo_name] = algo_best

    val_all_tuning_df = pd.concat(tuning_frames, ignore_index=True)
    val_all_tuning_df.to_csv(metrics_dir / "validation_threshold_results.csv", index=False)

    # Freeze best (algorithm, threshold) strictly from validation results
    frozen_choice = select_best_threshold(val_all_tuning_df, tie_breaker=tie_breaker)
    frozen_algo = str(frozen_choice["algorithm"])
    frozen_thr = float(frozen_choice["threshold"])
    logger.info(
        "Frozen configuration from validation: algorithm=%s, threshold=%.4f (Val F1=%.4f)",
        frozen_algo,
        frozen_thr,
        frozen_choice["f1"],
    )

    # 6. Single frozen evaluation on Test split (no tuning on test!)
    val_emb_preds = cluster_mentions_by_similarity(
        mentions_by_split["validation"],
        val_embeddings,
        threshold=frozen_thr,
        algorithm=frozen_algo,  # type: ignore[arg-type]
        agglomerative_linkage=linkage,
        batch_size=clust_batch,
    )
    val_emb_metrics = compute_pairwise_metrics(val_emb_preds)

    test_emb_preds = cluster_mentions_by_similarity(
        mentions_by_split["test"],
        test_embeddings,
        threshold=frozen_thr,
        algorithm=frozen_algo,  # type: ignore[arg-type]
        agglomerative_linkage=linkage,
        batch_size=clust_batch,
    )
    test_emb_metrics = compute_pairwise_metrics(test_emb_preds)
    test_emb_preds.to_csv(clusters_dir / "embedding_clusters.csv", index=False)

    comparison_rows = [
        {"split": "validation", "method": "Exact Matching Baseline", **val_exact_metrics},
        {
            "split": "validation",
            "method": f"Embedding ({frozen_algo}, thr={frozen_thr:.2f})",
            **val_emb_metrics,
        },
        {"split": "test", "method": "Exact Matching Baseline", **test_exact_metrics},
        {
            "split": "test",
            "method": f"Embedding ({frozen_algo}, thr={frozen_thr:.2f})",
            **test_emb_metrics,
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(metrics_dir / "method_comparison.csv", index=False)

    # 7. Error analysis on Test embedding predictions
    err_cfg = cfg.get("error_analysis", {})
    max_err = int(err_cfg.get("max_examples_per_category", 500))
    fm_df, mm_df, error_summary = analyze_coreference_errors(
        test_emb_preds,
        embeddings=test_embeddings,
        threshold=frozen_thr,
        max_examples_per_category=max_err,
    )
    fm_df.to_csv(errors_dir / "false_merges.csv", index=False)
    mm_df.to_csv(errors_dir / "missed_matches.csv", index=False)

    # 8. Bonus Phase 10: Hierarchical coreference analysis (kept separate from flat metrics)
    hierarchy_report = analyze_cluster_hierarchy(
        splits,
        mentions_by_split,
        test_embeddings=test_embeddings,
        test_predictions_df=test_emb_preds,
    )
    with (metrics_dir / "hierarchical_analysis.json").open("w", encoding="utf-8") as fh:
        json.dump(hierarchy_report, fh, indent=2)

    # 9. Benchmark runtime & peak memory for both clustering algorithms at frozen threshold
    bench_records = benchmark_clustering(
        mentions_by_split["validation"],
        val_embeddings,
        split_name="validation",
        threshold=frozen_thr,
        agglomerative_linkage=linkage,
        batch_size=clust_batch,
    ) + benchmark_clustering(
        mentions_by_split["test"],
        test_embeddings,
        split_name="test",
        threshold=frozen_thr,
        agglomerative_linkage=linkage,
        batch_size=clust_batch,
    )
    benchmark_df = pd.DataFrame(bench_records)
    benchmark_df.to_csv(metrics_dir / "runtime_memory_benchmarks.csv", index=False)

    # 10. Save comprehensive test metrics JSON
    final_metrics_payload = {
        "model_name": emb_cfg["model_name"],
        "split_isolation": isolation,
        "extraction_reports": extraction_reports,
        "validation_selection": {
            "configured_algorithm": configured_algo,
            "best_by_algorithm": best_by_algo,
            "best_connected_components": best_by_algo.get("connected_components"),
            "best_agglomerative": best_by_algo.get("agglomerative"),
            "frozen_choice": frozen_choice,
        },
        "validation_metrics": {
            "exact_matching": val_exact_metrics,
            "embedding_clustering": val_emb_metrics,
        },
        "test_metrics": {
            "exact_matching": test_exact_metrics,
            "embedding_clustering": test_emb_metrics,
        },
        "error_analysis_summary": error_summary,
        "runtime_memory_benchmarks": bench_records,
    }
    with (metrics_dir / "test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(final_metrics_payload, fh, indent=2)

    # 11. Generate all 9 visualizations
    plot_mentions_per_split(mentions_all_df, figures_dir)
    plot_cluster_size_distribution(mentions_all_df, figures_dir)
    plot_mention_length_distribution(mentions_all_df, figures_dir)
    winning_curve_df = val_all_tuning_df[val_all_tuning_df["algorithm"] == frozen_algo]
    plot_validation_threshold_curve(winning_curve_df, frozen_thr, figures_dir)
    plot_method_comparison(comparison_df, figures_dir)
    plot_gold_vs_pred_clusters(test_exact_preds, test_emb_preds, figures_dir)
    plot_error_distribution(test_emb_metrics, fm_df, mm_df, figures_dir)
    plot_hierarchy_structure(hierarchy_report, figures_dir)
    plot_runtime_memory_comparison(benchmark_df, figures_dir)

    logger.info("Pipeline completed successfully.")
    return final_metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SciCo Cross-Document Coreference Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    payload = run_pipeline(args.config)
    print(json.dumps(payload["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
