"""Sentence-Transformer embedding generation with deduplication and disk caching."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve inference device automatically when set to ``auto``."""
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


class MentionEmbedder:
    """Reusable Sentence-Transformer encoder with normalized-text deduplication and caching."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "auto",
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        cache_dir: str | Path | None = "outputs/cache",
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily load and reuse the SentenceTransformer model instance."""
        if self._model is None:
            logger.info("Loading SentenceTransformer '%s' on device='%s'", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _cache_paths(self, split_name: str, text_col: str) -> tuple[Path, Path]:
        assert self.cache_dir is not None
        safe_model = self.model_name.replace("/", "__").replace(":", "_")
        tag = hashlib.sha1(
            f"{self.model_name}|{text_col}|{self.normalize_embeddings}".encode("utf-8")
        ).hexdigest()[:10]
        stem = f"{split_name}_{safe_model}_{tag}"
        return (
            self.cache_dir / f"{stem}_embeddings.npy",
            self.cache_dir / f"{stem}_meta.json",
        )

    def encode_texts(self, texts: list[str], show_progress_bar: bool = True) -> np.ndarray:
        """Encode a list of strings with deduplication and safe handling of empty strings."""
        if not texts:
            dim = self.model.get_sentence_embedding_dimension() or 384
            return np.zeros((0, dim), dtype=np.float32)

        # Deduplicate strings to avoid repeated inference on identical mentions
        unique_nonempty: list[str] = []
        seen: dict[str, int] = {}
        indices: list[int] = []

        for t in texts:
            cleaned = "" if t is None else str(t).strip()
            if not cleaned:
                indices.append(-1)
                continue
            if cleaned not in seen:
                seen[cleaned] = len(unique_nonempty)
                unique_nonempty.append(cleaned)
            indices.append(seen[cleaned])

        if unique_nonempty:
            unique_emb = self.model.encode(
                unique_nonempty,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            ).astype(np.float32)
        else:
            dim = self.model.get_sentence_embedding_dimension() or 384
            unique_emb = np.zeros((0, dim), dtype=np.float32)

        dim = unique_emb.shape[1]
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for row_i, u_i in enumerate(indices):
            if u_i >= 0:
                out[row_i] = unique_emb[u_i]
        return out

    def encode_mentions(
        self,
        mentions_df: pd.DataFrame,
        split_name: str,
        *,
        text_col: str = "normalized_text",
        use_cache: bool = True,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode all mentions in a split DataFrame, using on-disk cache if valid."""
        if text_col not in mentions_df.columns:
            raise ValueError(f"Column '{text_col}' not found in mentions_df.")

        texts = ["" if v is None else str(v) for v in mentions_df[text_col].tolist()]
        content_hash = hashlib.sha256(
            "\n".join(texts).encode("utf-8", errors="replace")
        ).hexdigest()

        if use_cache and self.cache_dir is not None:
            npy_path, meta_path = self._cache_paths(split_name, text_col)
            if npy_path.exists() and meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as fh:
                        meta: dict[str, Any] = json.load(fh)
                    if (
                        meta.get("content_hash") == content_hash
                        and meta.get("num_rows") == len(texts)
                        and meta.get("model_name") == self.model_name
                    ):
                        arr = np.load(npy_path)
                        if arr.shape[0] == len(texts):
                            logger.info(
                                "Loaded cached embeddings for '%s' (%s) from %s",
                                split_name,
                                arr.shape,
                                npy_path,
                            )
                            return arr.astype(np.float32, copy=False)
                except Exception as exc:
                    logger.warning("Ignoring invalid embedding cache (%s): %s", npy_path, exc)

        embeddings = self.encode_texts(texts, show_progress_bar=show_progress_bar)

        if use_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            npy_path, meta_path = self._cache_paths(split_name, text_col)
            np.save(npy_path, embeddings)
            meta_payload = {
                "split": split_name,
                "model_name": self.model_name,
                "text_col": text_col,
                "normalize_embeddings": self.normalize_embeddings,
                "num_rows": len(texts),
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "unique_nonempty_texts": len({t.strip() for t in texts if t.strip()}),
                "content_hash": content_hash,
            }
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(meta_payload, fh, indent=2)
            logger.info("Saved embedding cache for '%s' to %s", split_name, npy_path)

        return embeddings
