# Reproducibility Guide

This document provides exact instructions and verification checks to reproduce the **Cross-Document Scientific Concept Coreference** experiments on SciCo.

---

## 1. Environment Requirements

- **Python:** 3.12+ (tested on Python `3.12.7`)
- **Hardware:** Runs cleanly on CPU (`torch 2.15.0+cpu`); automatically selects CUDA (`device: "auto"`) if a compatible GPU is available.
- **Core Dependencies:** Listed in [`requirements.txt`](../requirements.txt) (`pandas>=2.2.0`, `pyarrow>=16.0.0`, `numpy>=1.26.0`, `scipy>=1.13.0`, `scikit-learn>=1.5.0`, `torch>=2.2.0`, `transformers>=4.40.0`, `sentence-transformers>=3.0.0`, `pyyaml>=6.0.1`, `tqdm>=4.66.0`, `matplotlib>=3.9.0`, `seaborn>=0.13.0`, `pytest>=7.4.0`, `nbformat>=5.10.0`, `nbconvert>=7.16.0`).

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

---

## 2. Dataset Integrity & Split Isolation

The raw SciCo Parquet files reside in `data/` and are never modified by the pipeline:
- `data/Training.parquet` (`221` topics, `10,922` mentions, `4,079` gold clusters)
- `data/Validation.parquet` (`100` topics, `4,873` mentions, `1,747` gold clusters)
- `data/Test.parquet` (`200` topics, `10,423` mentions, `3,711` gold clusters)

`src/data_loader.py:verify_split_isolation` asserts zero topic ID overlap across splits on every run.

---

## 3. Centralized Configuration (`config.yaml`)

All paths, seeds, extraction flags, normalization options, embedding model identity, clustering behavior, and threshold-search grids are controlled via [`config.yaml`](../config.yaml):

- `seed: 42` (`random`, `numpy`, and `torch` seeded in `src/pipeline.py:set_seed`)
- `embedding.model_name: "sentence-transformers/all-MiniLM-L6-v2"`
- `embedding.normalize_embeddings: true`
- `clustering.algorithm: "auto"` (evaluates `candidate_algorithms: ["connected_components", "agglomerative"]` on Validation only and freezes the winner)
- `clustering.agglomerative_linkage: "average"`
- `threshold_tuning`: grid `[0.50, 0.52, ..., 0.94, 0.95]`, `tie_breaker: "higher_precision"`

---

## 4. Exact Execution Commands

### A. Run Automated Tests
```bash
pytest -v tests/
```

### B. Run Python Compilation Check
```bash
python -m compileall -q src tests
```

### C. Run the End-to-End Pipeline
```bash
python -m src.pipeline --config config.yaml
```
If cached embeddings in `outputs/cache/` are absent, `MentionEmbedder` downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, encodes deduplicated normalized mentions, and validates cache integrity via SHA-256 content hashes.

### D. Inspect Generated Outputs
- **Processed Mentions:** `data/processed/{train,validation,test}_mentions.parquet`
- **Cluster Predictions:** `outputs/clusters/exact_clusters.csv`, `outputs/clusters/embedding_clusters.csv`
- **Metrics & Tuning:** `outputs/metrics/validation_threshold_results.csv`, `outputs/metrics/method_comparison.csv`, `outputs/metrics/test_metrics.json`, `outputs/metrics/runtime_memory_benchmarks.csv`, `outputs/metrics/hierarchical_analysis.json`
- **Errors:** `outputs/errors/false_merges.csv`, `outputs/errors/missed_matches.csv`
- **Figures:** `outputs/figures/01_*.png` through `09_*.png`

---

## 5. Reproducibility Notes & Hardware Caveats

- Floating-point rounding in dense cosine similarity (`float32`) and multi-threaded BLAS reductions can introduce tiny (`~1e-7`) differences across CPU/GPU architectures, though tie-breaking and discrete clustering at `threshold = 0.74` are deterministic on the tested CPU environment.
- `MentionEmbedder` requires initial network access to fetch `sentence-transformers/all-MiniLM-L6-v2` if not already present in the local Hugging Face cache.
