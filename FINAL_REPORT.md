# Final Research & Engineering Report: Cross-Document Scientific Concept Coreference

**Repository:** [Cross-Document-Scientific-Concept-Coreference](https://github.com/abg11hani235-commits/Cross-Document-Scientific-Concept-Coreference)  
**Dataset:** SciCo ([allenai/scico](https://huggingface.co/datasets/allenai/scico) | [ariecattan/SciCo](https://github.com/ariecattan/SciCo))

---

## A. Introduction

### Research Problem
Scientific papers frequently refer to identical methods, models, datasets, and tasks using diverse surface forms (e.g., `"learning word representations"` vs. `"learning word vector embeddings"`, `"handwritten digit recognition"` vs. `"handwriting digit recognition"`), while identical acronyms or overlapping phrases can refer to completely distinct concepts across papers (e.g., `"FPN"` referring to *FacePoseNet*, *Fuzzy Petri Nets*, or *Fixed Pattern Noise*). Cross-document scientific concept coreference groups coreferent textual mentions across paper abstracts/snippets belonging to the same scientific topic.

### Motivation & Objectives
Exact lexical matching misses paraphrase and syntactic variation across papers, whereas unconstrained semantic similarity risks false merges across related but distinct concepts or hypernym/hyponym pairs. This project builds a reproducible, leak-free pipeline that:
1. Recovers and validates mentions from the official SciCo Parquet representation with full coordinate provenance.
2. Compares an exact normalized string baseline against pretrained Sentence-Transformer embedding clustering (`sentence-transformers/all-MiniLM-L6-v2`).
3. Performs algorithm and cosine-threshold selection strictly on the Validation split before running a single frozen evaluation on Test.
4. Diagnoses false merges and missed matches and quantifies how flat coreference clustering interacts with SciCo's hierarchical parent-child relations.

---

## B. Dataset

SciCo organizes computer science paper snippets into topic-level collections (`Training.parquet`, `Validation.parquet`, `Test.parquet`). Topic IDs are strictly disjoint across splits (`verify_split_isolation` confirms `0` overlap):

| Split | Topics | Snippets | Unique Papers | Extracted Mentions | Gold Clusters | Parent-Child Edges | Fallback Recoveries | Malformed Drops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 221 | 10,660 | 10,316 | 10,922 | 4,079 | 2,514 | 0 | 0 |
| Validation | 100 | 4,781 | 4,609 | 4,873 | 1,747 | 1,088 | 0 | 0 |
| Test | 200 | 10,203 | 9,842 | 10,423 | 3,711 | 2,379 | 62 | 0 |

### Annotation Semantics & Coordinate Provenance
- **Span Indexing:** Verified directly on annotated rows (including `483` train, `220` validation, and `427` test single-token spans where `start == end`): `[start, end, cluster_id]` uses **0-based inclusive** token indices (`tokens[start : end + 1]`).
- **Cluster Scoping:** Local cluster integers (`0, 1, 2, ...`) repeat across topics and are scoped as `{split}::t{topic_id}::c{raw_cluster_id}` to prevent cross-topic collisions.
- **Fallback Recovery (`Test` topic `id=1034`):** Across `520 / 521` topics, `flatten_mentions` on `flatten_tokens` matches document-level `mentions` (`[snippet_idx, doc_start, doc_end, cluster_id]`) on `tokens[snippet_idx]` identically. On Test topic `id=1034`, `flatten_mentions` is misaligned (including one out-of-bounds span `[3972, 3975, 6]` for `len(flatten_tokens)=3963`). `src/mention_extractor.py` preserves original `flat_start`/`flat_end` separately from `doc_start`/`doc_end`, logs the anomaly, checks cluster ID consistency (`0` mismatches across all splits), sets `coordinate_source = "doc_tokens_fallback"`, and recovers all `62` affected mentions without dropping records.

---

## C. Methodology

1. **Exact Normalized String Matching (`src/exact_matching.py` & `src/preprocessing.py`):**
   Applies `NFKC` Unicode normalization, dash/quote standardization, whitespace collapsing, bracket-aware outer punctuation stripping (preserving balanced parentheses like `Rectified Linear Units ( ReLU )` and technical suffixes like `C++`, `C#`), and lowercasing. Mentions with identical non-empty `normalized_text` are grouped strictly within each topic; empty strings remain singletons.
2. **Pretrained Mention Embeddings (`src/embedding_model.py`):**
   Uses off-the-shelf `sentence-transformers/all-MiniLM-L6-v2` (384-d L2-normalized vectors, **no fine-tuning**) with normalized-text deduplication and SHA-256 validated disk caching. Mentions are encoded solely from their normalized surface strings (context-free).
3. **Clustering Algorithms (`src/clustering.py`):**
   - *Connected Components:* Builds a topic-local cosine graph where edge $(i, j)$ exists iff $\cos(e_i, e_j) \ge \tau$. Susceptible to transitive chaining ($A \sim B, B \sim C \Rightarrow A, C$ merged even when $\cos(A, C) < \tau$).
   - *Agglomerative Clustering (`average` linkage):* Uses precomputed cosine distance $1 - \cos(e_i, e_j)$ with `distance_threshold = 1 - \tau`, requiring mean cross-cluster similarity $\ge \tau$ and curbing transitive chaining.
4. **Validation-Only Selection (`src/threshold_tuning.py`):**
   With `clustering.algorithm: "auto"` in `config.yaml`, sweeps $\tau \in [0.50, 0.95]$ (step `0.02`, plus `0.95`) on **Validation only** across candidate algorithms, selecting the highest Validation Pairwise F1 with deterministic tie-breaking (`higher_precision`).

---

## D. Experimental Setup

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (`normalize_embeddings: true`, `batch_size: 128`)
- **Seed & Determinism:** `seed: 42` (`random`, `numpy`, `torch`); cluster labels canonicalized deterministically by first appearance.
- **Evaluation (`src/evaluation.py`):** Micro-averaged within-topic pairwise Precision, Recall, and F1 computed in $O(N)$ per topic via cluster contingency counts, verified against brute-force pair enumeration across 25 randomized seeds. Zero denominators safely return `0.0`; null/missing scope or cluster IDs raise `ValueError`.
- **Split Isolation:** Test labels were never consulted during threshold, linkage, or algorithm selection.

---

## E. Results

### Validation Selection Results
- **Exact Matching (Validation):** Precision `0.9149`, Recall `0.0632`, **F1 `0.1183`** (`TP=1,108`, `FP=103`, `FN=16,410`, `3,931` predicted vs. `1,747` gold clusters).
- **Connected Components (Validation best at $\tau = 0.84$):** Precision `0.5941`, Recall `0.6454`, **F1 `0.6187`** (`TP=11,306`, `FP=7,724`, `FN=6,212`, `1,797` predicted clusters).
- **Agglomerative Average (Validation best at $\tau = 0.74$):** Precision `0.6138`, Recall `0.7075`, **F1 `0.6573`** (`TP=12,394`, `FP=7,799`, `FN=5,124`, `1,404` predicted clusters).
- **Frozen Choice:** `algorithm = agglomerative`, `linkage = average`, `threshold = 0.74`.

### Final Independent Test Results

| Method | Precision | Recall | Pairwise F1 | TP | FP | FN | Pred Clusters (Gold = 3,711) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact Matching Baseline | **0.9175** | 0.0581 | 0.1092 | 2,290 | 206 | 37,142 | 8,486 |
| Embedding Clustering (`agglomerative`, $\tau = 0.74$) | 0.5932 | **0.6170** | **0.6048** | 24,328 | 16,685 | 15,104 | 3,252 |

Exact matching achieves high precision (`0.9175`) but misses `94.2%` of coreferent pairs (`FN = 37,142`). Frozen agglomerative clustering improves Test Pairwise F1 by **+0.4956** (`0.1092` → `0.6048`) and reduces cluster fragmentation from `8,486` to `3,252` clusters.

---

## F. Error Analysis

Analyzed on Test predictions (`outputs/errors/false_merges.csv` and `outputs/errors/missed_matches.csv`):

- **False Merges (`16,685` FP pairs; mean cosine `0.8095`, median `0.8056`):**
  - *Lexical overlap across distinct concepts (`13,143`)* and *ambiguous surface forms*: e.g., in Topic `843`, `"FPN"` represents *FacePoseNet*, *Fuzzy Petri Nets*, and *Fixed Pattern Noise* across papers (`cosine = 1.0`); in Topic `842`, `"channel normalization"` refers to distinct gold clusters (`cosine = 1.0`).
  - *Linkage/chaining below threshold (`2,639`)* and *high semantic similarity (`898`)*: distinct variants or parent-child concepts pulled into the same cluster when cluster-level average similarity reaches `0.74`.
- **Missed Matches (`15,104` FN pairs; mean cosine `0.5941`, median `0.6327`):**
  - *Partial lexical overlap below threshold (`9,897`)* and *cross-paper terminology variation (`4,191`)*: e.g., `"affine transformation"` vs. `"affine transformation mapping"` (`0.9127` pairwise cosine, separated because full cluster-average similarity fell below `0.74`), `"transposed convolution"` vs. `"convolution transpose"` (`0.8773`), `"node embedding"` vs. `"graph node embedding"` (`0.8767`).
  - *Acronym vs. expanded form (`1,011`)*: context-free encoders assign low similarity between short acronyms and full expansions (e.g., `"BERT"` vs. `"Bidirectional Encoder Representations from Transformers"` = `0.1560`).

---

## G. Hierarchical Analysis

Evaluated separately from flat pairwise metrics (`src/hierarchy.py`, `outputs/metrics/hierarchical_analysis.json`):

| Split | Topics w/ Relations | Edges | Tree Components | Isolated Clusters | Mean / Max Depth | Mean / Max Branching | Valid Acyclic Forest |
|---|---:|---:|---:|---:|---:|---:|---|
| Train | 220 / 221 | 2,514 | 450 | 1,115 | 2.72 / 6 | 2.79 / 23 | `true` (`0` anomalies) |
| Validation | 100 / 100 | 1,088 | 195 | 464 | 2.83 / 5 | 2.73 / 14 | `true` (`0` anomalies) |
| Test | 199 / 200 | 2,379 | 406 | 926 | 2.83 / 7 | 2.72 / 24 | `true` (`0` anomalies) |

- **Hierarchy vs. Flat Clustering:** On Test, gold parent–child cluster centroids have mean cosine similarity **`0.6694`** compared to **`0.3776`** for unrelated within-topic cluster pairs. Because parent–child concepts share strong topical and lexical similarity without being coreferential, flat clustering merges **`816 / 2,379` (`34.30%`)** of true parent–child pairs.

---

## H. Performance (Runtime & Memory)

Measured via `src/clustering.py:benchmark_clustering` (CPU, `time.perf_counter` + Python `tracemalloc` on precomputed embeddings at $\tau = 0.74$):

| Split | Mentions | Topics | Max Topic Size | Algorithm | Wall-Clock Runtime | Peak Traced Memory |
|---|---:|---:|---:|---|---:|---:|
| Validation | 4,873 | 100 | 92 | Connected Components | `0.1805 s` | `3.735 MB` |
| Validation | 4,873 | 100 | 92 | Agglomerative (`average`) | `0.3564 s` | `3.708 MB` |
| Test | 10,423 | 200 | 82 | Connected Components | `0.4364 s` | `7.799 MB` |
| Test | 10,423 | 200 | 82 | Agglomerative (`average`) | `0.6186 s` | `7.734 MB` |

*Measurement Caveat:* `tracemalloc` tracks Python/NumPy allocations triggered during the clustering call on already-loaded embeddings and excludes PyTorch/transformer model weights (~90 MB) and C-extension internal buffers outside Python's allocator.

---

## I. Limitations

1. **Context-Free Mention Embeddings:** Mentions are embedded in isolation without surrounding sentence or paper context.
2. **Acronym Ambiguity:** Identical abbreviations standing for different concepts across papers (e.g., `"FPN"`) receive `1.0` cosine similarity and are inevitably merged.
3. **Acronym / Full-Name Gap:** General-purpose sentence embeddings do not reliably map abbreviations to expanded names (`"BERT"` vs. full name = `0.1560`).
4. **Flat Clustering vs. Hierarchy:** Symmetric cosine similarity conflates hypernym/hyponym relatedness (`0.6694` mean centroid similarity) with coreference, merging `34.30%` of Test parent–child pairs.
5. **Quadratic Agglomerative Memory:** `cluster_topic_agglomerative` builds an $N \times N$ precomputed cosine distance matrix per topic. While negligible for SciCo ($N \le 92$), scaling to topics with thousands of mentions would require sparse approximate nearest-neighbor blocking (an explicit warning is logged if $N > 2,000$).

---

## J. Conclusion and Future Work

A simple, leak-free validation-tuned agglomerative clustering pipeline over `all-MiniLM-L6-v2` embeddings achieves **0.6048** Test Pairwise F1 (+0.4956 over exact string matching at **0.1092**), while preserving full coordinate provenance and strictly separating flat evaluation from hierarchical structure.

**Promising Future Directions (Not Yet Implemented):**
1. **Context-Aware Cross-Encoders / Domain Encoders:** Fine-tune or prompt SciBERT / SPECTER2 models with surrounding sentence context to disambiguate polysemous acronyms.
2. **In-Document Acronym Resolution:** Expand `Full Name (ACRONYM)` patterns per paper before clustering to bridge abbreviation missed matches.
3. **Joint Coreference & Hypernym Classification:** Add an asymmetric entailment/parent–child classifier to prevent flat merging of hierarchical concept pairs.
4. **Scalable Sparse Agglomerative / Approximate Blocking:** Use k-NN candidate generation for large-topic deployments beyond SciCo's $N \le 92$ scale.
