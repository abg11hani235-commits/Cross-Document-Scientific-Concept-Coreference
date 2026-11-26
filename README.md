# Cross-Document Scientific Concept Coreference

A reproducible NLP pipeline for linking textual mentions of the same scientific method, model, technique, or research task across computer science papers using the **SciCo** dataset ([allenai/scico](https://huggingface.co/datasets/allenai/scico) | [ariecattan/SciCo](https://github.com/ariecattan/SciCo)).

---

## 1. Project Overview

Scientific literature frequently refers to the same concept using varied surface forms across papers while reusing identical acronyms or similar terminology for distinct methods. This project implements and compares an **Exact Normalized String Matching** baseline against **Sentence-Transformer Embedding Clustering**, enforces strict topic-level split isolation and validation-only hyperparameter selection, conducts false-merge and missed-match error analysis, and investigates SciCo's parent–child concept hierarchy separately from flat evaluation.

---

## 2. Problem Definition

Given a set of scientific papers grouped into a topic and annotated with concept mention spans, partition the mentions within each topic into clusters such that two mentions belong to the same cluster if and only if they refer to the same underlying scientific concept. Cross-topic merging is prohibited by task scope.

---

## 3. Scientific Motivation

- **Lexical mismatch:** Identical concepts appear under morphological variants, word-order changes, or abbreviations (`"handwritten digit recognition"` vs. `"handwriting digit recognition"`, `"BERT"` vs. `"Bidirectional Encoder Representations from Transformers"`).
- **Surface ambiguity:** Identical abbreviations or overlapping head nouns across papers can denote distinct concepts (`"FPN"` as *FacePoseNet*, *Fuzzy Petri Nets*, or *Fixed Pattern Noise*).
- **Taxonomic relatedness vs. coreference:** Broader/narrower scientific concepts (e.g., `"policy gradient methods"` → `"TRPO"`) are semantically close without being coreferential.

---

## 4. Dataset Description

- **Source:** Official SciCo Parquet splits (`data/Training.parquet`, `data/Validation.parquet`, `data/Test.parquet`), preserved read-only.
- **Topic-Level Split Isolation:** `src/data_loader.py:verify_split_isolation` verifies zero overlapping topic IDs across splits (`is_disjoint = true`).
- **Span Representation:** Each mention in `flatten_mentions` (`[start, end, cluster_id]`) and document-level `mentions` (`[snippet_idx, doc_start, doc_end, cluster_id]`) uses **0-based inclusive** token indices (`tokens[start : end + 1]`), confirmed across single-token spans (`start == end`: `483` train, `220` validation, `427` test).
- **Scoped Cluster IDs:** Raw local cluster integers (`0, 1, ...`) repeat across topics and are scoped as `{split}::t{topic_id}::c{raw_cluster_id}`.

---

## 5. Dataset Split Statistics

| Split | Topics | Snippets | Unique Papers | Extracted Mentions | Gold Clusters | Parent-Child Edges | Fallback Recoveries | Malformed Drops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 221 | 10,660 | 10,316 | 10,922 | 4,079 | 2,514 | 0 | 0 |
| Validation | 100 | 4,781 | 4,609 | 4,873 | 1,747 | 1,088 | 0 | 0 |
| Test | 200 | 10,203 | 9,842 | 10,423 | 3,711 | 2,379 | 62 | 0 |

---

## 6. Mention Extraction Methodology

Implemented in [`src/mention_extractor.py`](src/mention_extractor.py):

1. Reconstructs mention text from `flatten_tokens[flat_start : flat_end + 1]`.
2. Cross-checks against document-level `tokens[snippet_idx][doc_start : doc_end + 1]` and compares `raw_cluster_id` with `doc_cluster_id` (`0` cluster ID mismatches observed across all three splits).
3. On Test topic `id=1034`, `flatten_mentions` is misaligned (including out-of-bounds `[3972, 3975, 6]` for `len=3963`). The extractor logs the anomaly, preserves original `flat_start`/`flat_end` (`start`/`end`) separately from `doc_start`/`doc_end`/`snippet_index`, marks `coordinate_source = "doc_tokens_fallback"`, and recovers all `62` affected mentions without mutating gold cluster annotations or dropping records.

---

## 7. Data Preprocessing

Implemented in [`src/preprocessing.py`](src/preprocessing.py):
- `NFKC` Unicode normalization and standardization of typographic dashes/quotes.
- Whitespace collapsing and lowercasing.
- Bracket-aware outer punctuation stripping: removes outer wrappers like `(BERT)` → `bert` and stray punctuation like `FPN )` → `fpn`, while preserving internal balanced parentheses (`Rectified Linear Units ( ReLU )` → `rectified linear units ( relu )`) and technical suffixes (`C++`, `C#`, `ResNet-50`).

---

## 8. Exact Matching Baseline

Implemented in [`src/exact_matching.py`](src/exact_matching.py):
- Groups mentions with identical non-empty `normalized_text` strictly within `(split, topic_id)`.
- Isolates empty/invalid normalized strings as unique singletons.
- Produces deterministic cluster IDs (`{split}::t{topic_id}::exact_{idx}`).

---

## 9. Embedding Model

Implemented in [`src/embedding_model.py`](src/embedding_model.py):
- Uses pretrained **`sentence-transformers/all-MiniLM-L6-v2`** (384-d L2-normalized embeddings, `batch_size=128`) without fine-tuning.
- **Context-Free Encoding:** Encodes normalized mention surface strings only (no surrounding sentence context).
- Deduplicates identical normalized mentions before encoding, handles empty strings safely as zero vectors, auto-selects CUDA/CPU, and caches embeddings with SHA-256 content-hash validation in `outputs/cache/`.

---

## 10. Clustering Algorithms

Implemented in [`src/clustering.py`](src/clustering.py):
- **Connected Components:** Thresholds cosine similarity ($\cos(e_i, e_j) \ge \tau$) in chunked blocks (`batch_size=512`) and extracts connected components. Susceptible to transitive chaining.
- **Agglomerative Clustering (`average` linkage):** Computes per-topic cosine distance ($1.0 - \cos(e_i, e_j)$) with `distance_threshold = 1.0 - threshold` (validating threshold $\in [-1, 1]$ and linkage $\in \{\text{average}, \text{complete}, \text{single}\}$). Requires mean inter-cluster similarity $\ge \tau$, mitigating transitive chaining.
- **Topic Isolation & Zero Vectors:** All clustering runs strictly inside `(split, topic_id)` boundaries; zero vectors remain isolated singletons; labels are canonicalized deterministically.

---

## 11. Validation Threshold Selection

Implemented in [`src/threshold_tuning.py`](src/threshold_tuning.py) and controlled via [`config.yaml`](config.yaml):
- With `clustering.algorithm: "auto"`, [`src/pipeline.py`](src/pipeline.py) sweeps $\tau \in [0.50, 0.95]$ (step `0.02`, plus `0.95`) on **Validation only** for every algorithm in `candidate_algorithms` (`connected_components`, `agglomerative`). Setting `clustering.algorithm` to a specific algorithm pins tuning to that algorithm.
- Ties in Validation Pairwise F1 are broken deterministically via `higher_precision`.
- **Validation Winners:**
  - `connected_components` best ($\tau = 0.84$): `P = 0.5941`, `R = 0.6454`, **`F1 = 0.6187`** (`1,797` clusters)
  - `agglomerative` (`average`) best ($\tau = 0.74$): `P = 0.6138`, `R = 0.7075`, **`F1 = 0.6573`** (`1,404` clusters)
- **Frozen Configuration:** `agglomerative` (`average`), `threshold = 0.74` (applied once to Test).

---

## 12. Evaluation Methodology

Implemented in [`src/evaluation.py`](src/evaluation.py):
- Micro-averaged within-topic pairwise Precision, Recall, and F1 computed in $O(N)$ per topic via contingency counts ($\text{TP}, \text{FP}, \text{FN}$), verified against brute-force pair enumeration across 25 randomized seeds.
- Zero denominators safely return `0.0`; missing/null `topic_id`, `gold_cluster_id`, or `pred_cluster_id` values raise `ValueError`.
- Exact matching and embedding clustering share the identical evaluation function.

---

## 13. Experimental Results

| Split | Method | Precision | Recall | Pairwise F1 | TP | FP | FN | Pred Clusters (Gold) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | Exact Matching Baseline | 0.9149 | 0.0632 | 0.1183 | 1,108 | 103 | 16,410 | 3,931 (1,747) |
| Validation | Embedding (`agglomerative`, $\tau=0.74$) | 0.6138 | 0.7075 | **0.6573** | 12,394 | 7,799 | 5,124 | 1,404 (1,747) |
| Test | Exact Matching Baseline | **0.9175** | 0.0581 | 0.1092 | 2,290 | 206 | 37,142 | 8,486 (3,711) |
| Test | Embedding (`agglomerative`, $\tau=0.74$) | 0.5932 | **0.6170** | **0.6048** | 24,328 | 16,685 | 15,104 | 3,252 (3,711) |

- Frozen embedding clustering improves Test Pairwise F1 by **+0.4956** over exact matching (`0.1092` → `0.6048`) and reduces cluster count from `8,486` to `3,252` (gold = `3,711`).

---

## 14. False Merge Analysis

Exported in [`outputs/errors/false_merges.csv`](outputs/errors/false_merges.csv) (`16,685` total Test FP pairs; mean cosine `0.8095`, median `0.8056`):
- **Shared lexical overlap / distinct concepts (`13,143`):** e.g., `"channel normalization"` in Topic `842` (`cosine = 1.0`) and ambiguous acronyms like `"FPN"` in Topic `843` (*FacePoseNet*, *Fuzzy Petri Nets*, *Fixed Pattern Noise*, `cosine = 1.0`).
- **Linkage bridging below threshold (`2,639`) & high semantic similarity (`898`):** Average linkage permits merging clusters whose inter-cluster average exceeds `0.74` even when individual cross-cluster pairs fall below `0.74`.

---

## 15. Missed Match Analysis

Exported in [`outputs/errors/missed_matches.csv`](outputs/errors/missed_matches.csv) (`15,104` total Test FN pairs; mean cosine `0.5941`, median `0.6327`):
- **Partial lexical overlap below threshold (`9,897`) & cross-paper terminology variation (`4,191`):** e.g., `"affine transformation"` vs. `"affine transformation mapping"` (`0.9127`, separated by cluster-average linkage), `"transposed convolution"` vs. `"convolution transpose"` (`0.8773`), `"node embedding"` vs. `"graph node embedding"` (`0.8767`).
- **Acronym vs. expanded form (`1,011`):** e.g., `"BERT"` vs. `"Bidirectional Encoder Representations from Transformers"` (`0.1560`).

---

## 16. Hierarchical Analysis

Implemented separately in [`src/hierarchy.py`](src/hierarchy.py) and saved to [`outputs/metrics/hierarchical_analysis.json`](outputs/metrics/hierarchical_analysis.json) without altering flat coreference metrics:

| Split | Tree Components | Isolated Clusters | Mean / Max Depth | Mean / Max Branching | Valid Acyclic Forest |
|---|---:|---:|---:|---:|---|
| Train | 450 | 1,115 | 2.72 / 6 | 2.79 / 23 | `true` (`0` anomalies) |
| Validation | 195 | 464 | 2.83 / 5 | 2.73 / 14 | `true` (`0` anomalies) |
| Test | 406 | 926 | 2.83 / 7 | 2.72 / 24 | `true` (`0` anomalies) |

- All three splits exhibit `0` directed cycles, `0` multi-parent nodes, `0` missing cluster references, `0` duplicate edges, and `0` self-relations.
- **Flat Clustering Impact:** On Test, gold parent–child centroids have mean cosine similarity **`0.6694`** vs. **`0.3776`** for unrelated pairs, causing flat clustering to merge **`816 / 2,379` (`34.30%`)** of parent–child pairs.

---

## 17. Runtime and Memory Analysis

Measured live via `benchmark_clustering` (CPU, Python `tracemalloc` over cached embeddings at $\tau = 0.74$):

| Split | Mentions | Max Topic Size | Connected Components | Agglomerative (`average`) |
|---|---:|---:|---|---|
| Validation | 4,873 | 92 | `0.1805 s` (`3.735 MB` peak) | `0.3564 s` (`3.708 MB` peak) |
| Test | 10,423 | 82 | `0.4364 s` (`7.799 MB` peak) | `0.6186 s` (`7.734 MB` peak) |

`cluster_topic_agglomerative` builds an $N \times N$ distance matrix per topic ($\le 34\text{ KB}$ for SciCo's $N \le 92$), and logs an explicit warning if $N > 2,000$.

---

## 18. Limitations

1. **Context-free embeddings:** Mention strings are encoded without surrounding document context.
2. **Acronym ambiguity & expansion gap:** Polysemous acronyms (`"FPN"`) falsely merge (`1.0`), while acronym–expansion pairs (`"BERT"` vs. full name) rarely merge (`0.1560`).
3. **Flat vs. hierarchical confusion:** Flat similarity clustering conflates hypernym relatedness with coreference (`34.30%` Test parent–child merge rate).
4. **Quadratic agglomerative matrix:** Not suited for single topics containing many thousands of mentions without candidate blocking.

---

## 19. Installation

```bash
python -m pip install -r requirements.txt
```

---

## 20. Execution Instructions

Run the complete end-to-end reproducible pipeline:
```bash
python -m src.pipeline --config config.yaml
```
This validates split isolation, extracts mentions to `data/processed/`, runs exact matching, computes/loads cached embeddings, tunes thresholds on Validation only, evaluates once on Test, runs error and hierarchy analyses, benchmarks runtime/memory, and saves all 9 figures to `outputs/figures/`.

---

## 21. Testing Instructions

```bash
pytest -v tests/
python -m compileall -q src tests
```
Current suite: **31 passed, 0 failed** across `tests/`.

---

## 22. Project Structure

```text
.
├── config.yaml
├── requirements.txt
├── README.md
├── FINAL_REPORT.md
├── docs/
│   └── REPRODUCIBILITY.md
├── data/
│   ├── Training.parquet
│   ├── Validation.parquet
│   ├── Test.parquet
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_mention_extraction.ipynb
│   ├── 03_exact_matching.ipynb
│   ├── 04_embedding_clustering.ipynb
│   └── 05_evaluation_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── mention_extractor.py
│   ├── exact_matching.py
│   ├── embedding_model.py
│   ├── clustering.py
│   ├── threshold_tuning.py
│   ├── evaluation.py
│   ├── error_analysis.py
│   ├── hierarchy.py
│   ├── visualization.py
│   └── pipeline.py
├── outputs/
│   ├── clusters/
│   ├── metrics/
│   ├── errors/
│   └── figures/
└── tests/
    ├── test_clustering.py
    ├── test_data_loader.py
    ├── test_evaluation.py
    ├── test_hierarchy.py
    ├── test_mention_extractor.py
    ├── test_pipeline_config.py
    └── test_preprocessing.py
```

---

## 23. Reproducibility

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for environment specifications, seed control, cache integrity hashes, and output schema inspection commands, and [`FINAL_REPORT.md`](FINAL_REPORT.md) for the complete technical report.

---

## 24. References and Citations

- **SciCo Dataset & Paper:** Cattan, A., Johnson, S., Weld, D., Dagan, I., Downey, D., & Hope, T. (2021). *SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts*. ([allenai/scico](https://huggingface.co/datasets/allenai/scico) | [ariecattan/SciCo](https://github.com/ariecattan/SciCo))
- **Sentence-Transformers (`all-MiniLM-L6-v2`):** Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
