# Cross-Document Scientific Concept Coreference (SciCo)

A reproducible NLP pipeline for linking textual mentions of the same scientific method, model, technique, or research task across computer science papers using the **SciCo** dataset ([allenai/scico](https://huggingface.co/datasets/allenai/scico)).

---

## 1. Project Overview & Problem Definition

In scientific literature, the same concept is frequently referred to using diverse surface expressions across papers (e.g., `"handwritten digit recognition"` vs. `"handwriting digit recognition"`, `"learning word vector embeddings"` vs. `"learning word representations"`), while superficially similar phrases or acronyms can denote distinct concepts (e.g., `"FPN"` standing for *FacePoseNet*, *Fuzzy Petri Nets*, or *Fixed Pattern Noise*).

**Goal:** Given spans of scientific concept mentions distributed across multiple paper abstracts/snippets within a topic, group coreferent mentions into flat coreference clusters within each topic boundary, while strictly preventing cross-topic leakage and test-set tuning.

---

## 2. Dataset Description & Schema Verification

- **Source:** [allenai/scico on Hugging Face](https://huggingface.co/datasets/allenai/scico) (loaded locally from `data/Training.parquet`, `data/Validation.parquet`, and `data/Test.parquet`).
- **Key Verified Annotation Semantics:**
  - **Evaluation Scope:** Topic-level cross-document coreference. Each Parquet row corresponds to one topic (`id`) containing multiple paper snippets (`tokens`, `doc_ids`, `metadata`). Topic IDs are strictly disjoint across `train` (221), `validation` (100), and `test` (200).
  - **Cluster ID Scoping:** Raw `cluster_id` integers (`0, 1, 2, ...`) repeat across topics. All pipelines scope clusters as `{split}::t{topic_id}::c{raw_cluster_id}` so unrelated clusters never collide.
  - **Span Indexing (`[start, end, cluster_id]`):** Verified directly on annotated examples (including hundreds of single-token spans where `start == end`): SciCo spans are **0-based and inclusive on `end`** (`tokens[start : end + 1]`).
  - **Dataset Anomaly Recovery:** On 520/521 topics, `flatten_mentions` on `flatten_tokens` matches document-level `mentions` (`[snippet_idx, doc_start, doc_end, cluster_id]`) on `tokens[snippet_idx]` identically. On Test topic `id=1034`, `flatten_mentions` is misaligned (including one out-of-bounds span `[3972, 3975, 6]` vs. `len(flatten_tokens)=3963`). `src/mention_extractor.py` detects this mismatch, logs it, and recovers all 62 affected mentions cleanly via `tokens[snippet_idx]` with **0 dropped mentions**.

| Split | Topics | Extracted Mentions | Gold Clusters | Parent-Child Edges | Fallback Recoveries | Malformed Drops |
|---|---:|---:|---:|---:|---:|---:|
| Train | 221 | 10,922 | 4,079 | 2,514 | 0 | 0 |
| Validation | 100 | 4,873 | 1,747 | 1,088 | 0 | 0 |
| Test | 200 | 10,423 | 3,711 | 2,379 | 62 | 0 |

---

## 3. Project Architecture

```text
.
├── config.yaml                         # Central experiment configuration & fixed seed
├── requirements.txt                    # Python dependencies
├── data/
│   ├── Training.parquet                # Raw SciCo train split (untouched)
│   ├── Validation.parquet              # Raw SciCo validation split (untouched)
│   ├── Test.parquet                    # Raw SciCo test split (untouched)
│   └── processed/                      # Extracted & normalized mention Parquet tables
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Schema inspection, span verification, EDA
│   ├── 02_mention_extraction.ipynb     # Mention extraction, scoping, normalization checks
│   ├── 03_exact_matching.ipynb         # Exact normalized string baseline & failure analysis
│   ├── 04_embedding_clustering.ipynb   # MiniLM embeddings, validation-only tuning, clusters
│   └── 05_evaluation_analysis.ipynb    # Frozen test results, error analysis, hierarchy bonus
├── src/
│   ├── data_loader.py                  # PyArrow inspection, Parquet loading, split isolation
│   ├── preprocessing.py                # Unicode/dash/whitespace/bracket-aware normalization
│   ├── mention_extractor.py            # Span extraction, doc_tokens fallback, cluster scoping
│   ├── exact_matching.py               # Topic-scoped exact normalized string clustering
│   ├── embedding_model.py              # Deduplicated Sentence-Transformer encoder + cache
│   ├── clustering.py                   # Chunked connected-components & agglomerative clustering
│   ├── threshold_tuning.py             # Validation-only grid search & deterministic tie-breaking
│   ├── evaluation.py                   # O(N) topic-scoped pairwise Precision, Recall, and F1
│   ├── error_analysis.py               # False merge (FP) & missed match (FN) extraction
│   ├── hierarchy.py                    # Bonus: parent-child hypernym hierarchy analysis
│   ├── visualization.py                # Generates all 7 figures in outputs/figures/
│   └── pipeline.py                     # End-to-end reproducible pipeline runner
├── outputs/
│   ├── clusters/                       # exact_clusters.csv, embedding_clusters.csv
│   ├── metrics/                        # validation_threshold_results.csv, test_metrics.json, etc.
│   ├── errors/                         # false_merges.csv, missed_matches.csv
│   └── figures/                        # 7 PNG figures
└── tests/                              # Automated pytest suite (18 tests)
```

---

## 4. Installation & Reproduction

### Setup
```bash
pip install -r requirements.txt
```

### Run the Full Reproducible Pipeline
```bash
python -m src.pipeline --config config.yaml
```
This runs data loading, split isolation checks, mention extraction, exact matching baseline, cached `all-MiniLM-L6-v2` embedding generation, validation-only threshold tuning, a single frozen evaluation on the test split, error analysis, hierarchical bonus analysis, and all 7 visualizations.

### Run Automated Tests
```bash
pytest -v tests/
```

---

## 5. Methodology

1. **Text Normalization (`src/preprocessing.py`):** Applies `NFKC` Unicode normalization, standardizes typographic dashes/quotes, collapses whitespace, strips stray outer punctuation while preserving balanced parentheses (e.g., `Rectified Linear Units ( ReLU )`) and technical suffixes (`C++`, `C#`, `ResNet-50`), and lowercases.
2. **Exact Matching Baseline (`src/exact_matching.py`):** Groups mentions sharing identical non-empty `normalized_text` strictly within `(split, topic_id)`. Empty strings are isolated as singletons.
3. **Embedding Model (`src/embedding_model.py`):** Encodes deduplicated normalized mentions using `sentence-transformers/all-MiniLM-L6-v2` (384-d L2-normalized embeddings) with automatic CPU/CUDA selection and SHA-256 validated on-disk caching.
4. **Similarity Clustering (`src/clustering.py`):** Evaluates both:
   - **Connected Components** on the thresholded cosine similarity graph (computed in memory-bounded blocks). Susceptible to transitive chaining ($A \sim B$ and $B \sim C$ merges $A$ and $C$ even if $\cos(A, C) < \tau$).
   - **Agglomerative Clustering** with cosine distance (`distance_threshold = 1 - threshold`, `linkage = "average"`), which mitigates transitive chaining.
5. **Validation-Only Selection (`src/threshold_tuning.py`):** Searches cosine thresholds `[0.50, 0.52, ..., 0.94, 0.95]` on the **Validation** split only, selecting the highest Validation Pairwise F1 with deterministic tie-breaking (`higher_precision`).
   - Best Connected Components (Validation): `threshold = 0.84` → `P = 0.5941`, `R = 0.6454`, **`F1 = 0.6187`**
   - Best Agglomerative Average (Validation): `threshold = 0.74` → `P = 0.6138`, `R = 0.7075`, **`F1 = 0.6573`**
   - **Frozen Configuration:** `algorithm = agglomerative`, `threshold = 0.74` (never tuned on Test).
6. **Pairwise Evaluation (`src/evaluation.py`):** Within each topic:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 6. Results

| Split | Method | Precision | Recall | Pairwise F1 | TP | FP | FN | Pred Clusters (Gold) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | Exact Matching Baseline | 0.9149 | 0.0632 | 0.1183 | 1,108 | 103 | 16,410 | 3,931 (1,747) |
| Validation | Embedding (`agglomerative`, $\tau=0.74$) | 0.6138 | 0.7075 | **0.6573** | 12,394 | 7,799 | 5,124 | 1,404 (1,747) |
| Test | Exact Matching Baseline | **0.9175** | 0.0581 | 0.1092 | 2,290 | 206 | 37,142 | 8,486 (3,711) |
| Test | Embedding (`agglomerative`, $\tau=0.74$) | 0.5932 | **0.6170** | **0.6048** | 24,328 | 16,685 | 15,104 | 3,252 (3,711) |

- Exact matching achieves `0.9175` precision on Test but misses `94.2%` of coreferent pairs (`Recall = 0.0581`, `F1 = 0.1092`).
- Frozen embedding-based agglomerative clustering raises Test Pairwise F1 by **+0.4956** (`0.1092` → `0.6048`) and consolidates `8,486` fragmented clusters into `3,252` clusters (compared to `3,711` gold clusters).

---

## 7. Error Analysis Summary (Test Split)

Exported to `outputs/errors/false_merges.csv` and `outputs/errors/missed_matches.csv`:

- **False Merges (`16,685` FP pairs; mean cosine `0.8095`, median `0.8056`):**
  - *Shared lexical overlap / distinct concepts (`13,143`)* and *ambiguous acronyms*: e.g., in Topic `843`, `"FPN"` refers to *FacePoseNet*, *Fuzzy Petri Nets*, and *Fixed Pattern Noise* across papers (`cosine = 1.0`), and in Topic `842`, `"channel normalization"` refers to distinct proposed methods (`cosine = 1.0`). Without surrounding document context, surface-only embeddings cannot disambiguate identical strings.
  - *Average-linkage chaining below threshold (`2,639`)* and *high semantic similarity (`898`)*: closely related sub-methods or hypernym/hyponym variants get pulled into the same cluster.
- **Missed Matches (`15,104` FN pairs; mean cosine `0.5941`, median `0.6327`):**
  - *Partial lexical overlap below threshold (`9,897`)* and *cross-paper variation (`4,191`)*: e.g., `"affine transformation"` vs. `"affine transformation mapping"` (`0.9127`, separated because their cluster averages fell below `0.74`), `"transposed convolution"` vs. `"convolution transpose"`, `"node embedding"` vs. `"graph node embedding"`.
  - *Acronym vs. expanded form (`1,011`)*: general-purpose sentence encoders assign low similarity between short acronyms and their full expansions (e.g., `"BERT"` vs. `"Bidirectional Encoder Representations from Transformers"` scores `0.1560`).

---

## 8. Bonus: Hierarchical Coreference Analysis

Analyzed separately in `src/hierarchy.py` and `outputs/metrics/hierarchical_analysis.json` (without modifying flat evaluation):

- SciCo `relations` encodes directed `[parent_cluster_id, child_cluster_id]` hypernym/hyponym trees (0 cycles, 0 multi-parent nodes) across nearly all topics (220/221 Train, 100/100 Validation, 199/200 Test), with max depth up to 7 and mean branching factor ~2.72.
- Example (Test Topic `845`): `"bidirectional transformers"` → `"XLNet and BERT architectures"` → {`"BERT"` / `"BERT networks"`, `"XLNet"` / `"XLNet models"`}.
- **Key Finding:** On Test, gold parent–child cluster centroids have a mean cosine similarity of **`0.6694`**, compared to **`0.3776`** for unrelated cluster pairs in the same topic. Consequently, flat similarity clustering merges **`816 / 2,379` (`34.30%`)** of true parent–child cluster pairs into the same flat cluster—highlighting why hierarchy-aware or asymmetrical entailment modeling is a natural next step.

---

## 9. Limitations & Future Improvements

1. **Context-Free Mention Encoding:** Current embeddings encode only the mention surface text; incorporating surrounding sentence context (e.g., via SciBERT or SPECTER2 cross-encoders) would disambiguate acronyms like `"FPN"` and `"MoCo"`.
2. **Acronym Expansion:** Lightweight in-document acronym resolution (`FacePoseNet (FPN)`) before embedding would directly recover many of the `1,011` acronym missed matches.
3. **Joint Coreference & Hierarchy Prediction:** Adding an asymmetric parent–child classifier could prevent flat clustering from merging `34.3%` of hypernym/hyponym pairs.
