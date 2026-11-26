"""Bonus Phase 10: Hierarchical parent-child cluster relation analysis."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.clustering import l2_normalize

logger = logging.getLogger(__name__)


def _tree_depth(node: str, children_map: dict[str, list[str]], visited: set[str]) -> int:
    """Compute subtree depth safely (cycle-guarded)."""
    if node in visited:
        return 0
    children = children_map.get(node, [])
    if not children:
        return 1
    visited.add(node)
    depth = 1 + max(_tree_depth(ch, children_map, visited) for ch in children)
    visited.remove(node)
    return depth


def analyze_cluster_hierarchy(
    splits: dict[str, pd.DataFrame],
    mentions_by_split: dict[str, pd.DataFrame],
    *,
    test_embeddings: np.ndarray | None = None,
    test_predictions_df: pd.DataFrame | None = None,
    max_examples: int = 12,
) -> dict[str, Any]:
    """Analyze SciCo parent-child cluster relations separately from flat evaluation.

    Inspects the directed hypernym/hyponym ``relations`` ([parent, child]) per topic,
    computes depth and branching statistics, checks embedding similarity between
    parent-child pairs vs unrelated cluster pairs, and quantifies how often flat
    clustering merges parent and child clusters on Test.
    """
    split_stats: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []

    for split_name, raw_df in splits.items():
        mdf = mentions_by_split[split_name]
        cluster_texts: dict[str, list[str]] = (
            mdf.groupby("cluster_id")["mention_text"].apply(list).to_dict()
            if not mdf.empty
            else {}
        )

        topics_with_rels = 0
        total_edges = 0
        parent_nodes: set[str] = set()
        child_nodes: set[str] = set()
        depths: list[int] = []
        branching_factors: list[int] = []

        for row in raw_df.itertuples(index=False):
            topic_id = int(row.id)
            rels = list(row.relations) if row.relations is not None else []
            if not rels:
                continue
            topics_with_rels += 1
            children_map: dict[str, list[str]] = defaultdict(list)
            local_parents: set[str] = set()
            local_children: set[str] = set()

            for pair in rels:
                if len(pair) != 2:
                    continue
                p_cid = f"{split_name}::t{topic_id}::c{int(pair[0])}"
                c_cid = f"{split_name}::t{topic_id}::c{int(pair[1])}"
                children_map[p_cid].append(c_cid)
                local_parents.add(p_cid)
                local_children.add(c_cid)
                parent_nodes.add(p_cid)
                child_nodes.add(c_cid)
                total_edges += 1

            for p_cid, ch_list in children_map.items():
                branching_factors.append(len(ch_list))

            roots = local_parents - local_children
            for rt in roots:
                depths.append(_tree_depth(rt, children_map, set()))

            if split_name == "test" and len(examples) < max_examples:
                for p_cid, ch_list in list(children_map.items())[:2]:
                    p_texts = cluster_texts.get(p_cid, [])[:3]
                    c_sample = [
                        {
                            "child_cluster_id": ch,
                            "mentions": cluster_texts.get(ch, [])[:3],
                        }
                        for ch in ch_list[:4]
                    ]
                    if p_texts and c_sample:
                        examples.append(
                            {
                                "topic_id": topic_id,
                                "source": str(row.source),
                                "parent_cluster_id": p_cid,
                                "parent_mentions": p_texts,
                                "children": c_sample,
                            }
                        )
                    if len(examples) >= max_examples:
                        break

        total_clusters = int(mdf["cluster_id"].nunique()) if not mdf.empty else 0
        split_stats[split_name] = {
            "total_topics": len(raw_df),
            "topics_with_relations": topics_with_rels,
            "total_parent_child_edges": total_edges,
            "total_clusters": total_clusters,
            "clusters_acting_as_parent": len(parent_nodes),
            "clusters_with_parent": len(child_nodes),
            "mean_branching_factor": (
                float(np.mean(branching_factors)) if branching_factors else 0.0
            ),
            "max_branching_factor": int(max(branching_factors)) if branching_factors else 0,
            "mean_tree_depth": float(np.mean(depths)) if depths else 0.0,
            "max_tree_depth": int(max(depths)) if depths else 0,
        }

    test_hierarchy_interaction: dict[str, Any] = {}
    if test_embeddings is not None and "test" in mentions_by_split:
        test_mdf = mentions_by_split["test"].reset_index(drop=True)
        norm_emb = l2_normalize(test_embeddings)

        # Compute cluster centroids in embedding space
        centroids: dict[str, np.ndarray] = {}
        for cid, grp in test_mdf.groupby("cluster_id", sort=False):
            vec = norm_emb[grp.index.to_numpy()].mean(axis=0)
            nrm = np.linalg.norm(vec)
            if nrm > 1e-12:
                centroids[str(cid)] = vec / nrm

        pred_lookup: dict[str, set[str]] = {}
        if test_predictions_df is not None:
            for row in test_predictions_df.itertuples(index=False):
                pred_lookup.setdefault(str(row.gold_cluster_id), set()).add(str(row.pred_cluster_id))

        pc_sims: list[float] = []
        unrelated_sims: list[float] = []
        merged_pc_edges = 0
        total_checked_edges = 0

        for row in splits["test"].itertuples(index=False):
            topic_id = int(row.id)
            rels = list(row.relations) if row.relations is not None else []
            rel_set = {
                (f"test::t{topic_id}::c{int(p)}", f"test::t{topic_id}::c{int(c)}")
                for p, c in rels
                if len((p, c)) == 2
            }
            topic_cids = [
                c for c in test_mdf[test_mdf["topic_id"] == topic_id]["cluster_id"].unique()
                if c in centroids
            ]
            for p_cid, c_cid in rel_set:
                if p_cid in centroids and c_cid in centroids:
                    pc_sims.append(float(np.dot(centroids[p_cid], centroids[c_cid])))
                    total_checked_edges += 1
                    if pred_lookup and (pred_lookup.get(p_cid, set()) & pred_lookup.get(c_cid, set())):
                        merged_pc_edges += 1

            # Sample unrelated pairs within the same topic
            for i in range(len(topic_cids)):
                for j in range(i + 1, len(topic_cids)):
                    a, b = topic_cids[i], topic_cids[j]
                    if (a, b) not in rel_set and (b, a) not in rel_set:
                        unrelated_sims.append(float(np.dot(centroids[a], centroids[b])))

        test_hierarchy_interaction = {
            "num_parent_child_pairs_evaluated": len(pc_sims),
            "mean_parent_child_centroid_cosine": float(np.mean(pc_sims)) if pc_sims else 0.0,
            "mean_unrelated_within_topic_centroid_cosine": (
                float(np.mean(unrelated_sims)) if unrelated_sims else 0.0
            ),
            "parent_child_pairs_merged_by_flat_clustering": merged_pc_edges,
            "parent_child_merge_rate": (
                float(merged_pc_edges / total_checked_edges) if total_checked_edges > 0 else 0.0
            ),
        }

    return {
        "split_statistics": split_stats,
        "test_hierarchy_vs_embeddings": test_hierarchy_interaction,
        "representative_examples": examples,
    }
