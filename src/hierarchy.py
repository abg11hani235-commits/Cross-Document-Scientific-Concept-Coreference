"""Bonus Phase 10: Hierarchical parent-child cluster relation validation and analysis."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from src.clustering import l2_normalize

logger = logging.getLogger(__name__)


def _count_directed_cycles(nodes: set[str], children_map: dict[str, list[str]]) -> int:
    """Count cycle-closing back-edges using 3-color DFS (0=unvisited, 1=visiting, 2=done)."""
    state: dict[str, int] = {}
    cycle_edges = 0

    def dfs(node: str) -> None:
        nonlocal cycle_edges
        state[node] = 1
        for nxt in children_map.get(node, []):
            nxt_state = state.get(nxt, 0)
            if nxt_state == 1:
                cycle_edges += 1
            elif nxt_state == 0:
                dfs(nxt)
        state[node] = 2

    for n in sorted(nodes):
        if state.get(n, 0) == 0:
            dfs(n)
    return cycle_edges


def _count_weakly_connected_components(
    nodes: set[str],
    undirected_adj: dict[str, set[str]],
) -> int:
    """Count connected components in the undirected projection of a topic's relations."""
    visited: set[str] = set()
    components = 0
    for start in sorted(nodes):
        if start in visited:
            continue
        components += 1
        queue: deque[str] = deque([start])
        visited.add(start)
        while queue:
            cur = queue.popleft()
            for nb in undirected_adj.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return components


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


def validate_split_hierarchy(
    raw_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    """Validate structural integrity of parent-child cluster relations for a split.

    Checks for directed cycles, multi-parent nodes, missing parent/child cluster IDs,
    duplicate edges, self-relations, and graph connectivity per topic.
    """
    topic_clusters: dict[int, set[str]] = (
        mentions_df.groupby("topic_id")["cluster_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
        if not mentions_df.empty
        else {}
    )

    directed_cycle_topics = 0
    directed_cycle_edges = 0
    multi_parent_nodes = 0
    missing_parent_clusters = 0
    missing_child_clusters = 0
    duplicate_relationships = 0
    self_relations = 0
    malformed_relation_entries = 0

    topics_with_relations = 0
    single_component_topics = 0
    multi_component_topics = 0
    total_connected_components = 0
    total_roots = 0
    total_isolated_clusters = 0
    anomalies: list[dict[str, Any]] = []

    for row in raw_df.itertuples(index=False):
        topic_id = int(row.id)
        known_clusters = topic_clusters.get(topic_id, set())
        rels = list(row.relations) if row.relations is not None else []

        children_map: dict[str, list[str]] = defaultdict(list)
        parents_map: dict[str, set[str]] = defaultdict(set)
        undirected_adj: dict[str, set[str]] = defaultdict(set)
        seen_edges: set[tuple[str, str]] = set()
        rel_nodes: set[str] = set()

        for pair in rels:
            if len(pair) != 2:
                malformed_relation_entries += 1
                anomalies.append(
                    {
                        "topic_id": topic_id,
                        "issue": "malformed_relation_length",
                        "detail": f"Expected length 2, got {len(pair)}",
                    }
                )
                continue

            p_cid = f"{split_name}::t{topic_id}::c{int(pair[0])}"
            c_cid = f"{split_name}::t{topic_id}::c{int(pair[1])}"
            rel_nodes.update((p_cid, c_cid))

            if p_cid == c_cid:
                self_relations += 1
                anomalies.append(
                    {
                        "topic_id": topic_id,
                        "issue": "self_relation",
                        "detail": p_cid,
                    }
                )

            if p_cid not in known_clusters:
                missing_parent_clusters += 1
                anomalies.append(
                    {
                        "topic_id": topic_id,
                        "issue": "missing_parent_cluster",
                        "detail": p_cid,
                    }
                )

            if c_cid not in known_clusters:
                missing_child_clusters += 1
                anomalies.append(
                    {
                        "topic_id": topic_id,
                        "issue": "missing_child_cluster",
                        "detail": c_cid,
                    }
                )

            edge = (p_cid, c_cid)
            if edge in seen_edges:
                duplicate_relationships += 1
                anomalies.append(
                    {
                        "topic_id": topic_id,
                        "issue": "duplicate_relationship",
                        "detail": f"{p_cid} -> {c_cid}",
                    }
                )
                continue

            seen_edges.add(edge)
            children_map[p_cid].append(c_cid)
            parents_map[c_cid].add(p_cid)
            undirected_adj[p_cid].add(c_cid)
            undirected_adj[c_cid].add(p_cid)

        total_isolated_clusters += len(known_clusters - rel_nodes)

        if not seen_edges:
            continue

        topics_with_relations += 1
        for c_cid, p_set in parents_map.items():
            if len(p_set) > 1:
                multi_parent_nodes += 1
                anomalies.append(
                    {
                        "topic_id": topic_id,
                        "issue": "multiple_parents",
                        "detail": f"{c_cid} has parents {sorted(p_set)}",
                    }
                )

        topic_cycles = _count_directed_cycles(rel_nodes, children_map)
        if topic_cycles > 0:
            directed_cycle_topics += 1
            directed_cycle_edges += topic_cycles
            anomalies.append(
                {
                    "topic_id": topic_id,
                    "issue": "directed_cycle",
                    "detail": f"{topic_cycles} back-edge(s)",
                }
            )

        n_comp = _count_weakly_connected_components(rel_nodes, undirected_adj)
        total_connected_components += n_comp
        if n_comp == 1:
            single_component_topics += 1
        elif n_comp > 1:
            multi_component_topics += 1

        roots = set(children_map.keys()) - set(parents_map.keys())
        total_roots += len(roots)

    is_acyclic = directed_cycle_edges == 0 and self_relations == 0
    is_valid_forest = (
        is_acyclic
        and multi_parent_nodes == 0
        and missing_parent_clusters == 0
        and missing_child_clusters == 0
        and duplicate_relationships == 0
        and malformed_relation_entries == 0
    )

    return {
        "split": split_name,
        "total_topics": len(raw_df),
        "topics_with_relations": topics_with_relations,
        "directed_cycle_topics": directed_cycle_topics,
        "directed_cycle_edges": directed_cycle_edges,
        "multi_parent_nodes": multi_parent_nodes,
        "missing_parent_clusters": missing_parent_clusters,
        "missing_child_clusters": missing_child_clusters,
        "duplicate_relationships": duplicate_relationships,
        "self_relations": self_relations,
        "malformed_relation_entries": malformed_relation_entries,
        "total_connected_components": total_connected_components,
        "single_component_topics": single_component_topics,
        "multi_component_topics": multi_component_topics,
        "total_roots": total_roots,
        "total_isolated_clusters": total_isolated_clusters,
        "is_acyclic": is_acyclic,
        "is_valid_forest": is_valid_forest,
        "anomaly_count": len(anomalies),
        "anomalies_sample": anomalies[:20],
    }


def analyze_cluster_hierarchy(
    splits: dict[str, pd.DataFrame],
    mentions_by_split: dict[str, pd.DataFrame],
    *,
    test_embeddings: np.ndarray | None = None,
    test_predictions_df: pd.DataFrame | None = None,
    max_examples: int = 12,
) -> dict[str, Any]:
    """Validate and analyze SciCo parent-child relations separately from flat evaluation.

    Inspects directed hypernym/hyponym ``relations`` ([parent, child]) per topic,
    validates graph integrity per split, computes depth and branching statistics,
    compares parent-child centroid cosine similarity against unrelated within-topic
    pairs, and quantifies how often flat clustering merges parent and child clusters.
    """
    split_stats: dict[str, Any] = {}
    hierarchy_validation: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []

    for split_name, raw_df in splits.items():
        mdf = mentions_by_split[split_name]
        hierarchy_validation[split_name] = validate_split_hierarchy(raw_df, mdf, split_name)

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
    if test_embeddings is not None and "test" in mentions_by_split and "test" in splits:
        test_mdf = mentions_by_split["test"].reset_index(drop=True)
        norm_emb = l2_normalize(test_embeddings)

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
            rel_set: set[tuple[str, str]] = set()
            for pair in rels:
                if len(pair) != 2:
                    continue
                rel_set.add((f"test::t{topic_id}::c{int(pair[0])}", f"test::t{topic_id}::c{int(pair[1])}"))

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
        "hierarchy_validation": hierarchy_validation,
        "test_hierarchy_vs_embeddings": test_hierarchy_interaction,
        "representative_examples": examples,
    }
