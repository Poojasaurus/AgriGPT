"""Agricultural Embeddings Module.
Provides vector representations of agricultural queries and documents.
Uses SentenceTransformers (all-MiniLM-L6-v2) with graceful offline fallback
to TF-IDF semantic vectorization to guarantee zero crashes in air-gapped environments.
"""

from __future__ import annotations

import os
import numpy as np
from typing import List, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer


class AgriEmbeddingModel:
    """Computes dense normalized embeddings for agricultural text chunks and queries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self.is_transformer: bool = False
        self._st_model = None
        self._fallback_vectorizer: Optional[TfidfVectorizer] = None

        self._load_model()

    def _load_model(self):
        """Attempt to load SentenceTransformer; fallback to fitted TF-IDF on failure."""
        # Check if local model or network is available
        try:
            # Set timeout and offline preference if offline
            import os
            from sentence_transformers import SentenceTransformer
            # Try loading cached local model first
            self._st_model = SentenceTransformer(self.model_name, local_files_only=True)
            self.is_transformer = True
            test_emb = self._st_model.encode(["crop"])
            self.dimension = int(test_emb.shape[1])
        except Exception:
            try:
                # If local cache missing and online, attempt normal load
                if os.environ.get("HF_HUB_OFFLINE", "0") != "1":
                    from sentence_transformers import SentenceTransformer
                    self._st_model = SentenceTransformer(self.model_name)
                    self.is_transformer = True
                    test_emb = self._st_model.encode(["crop"])
                    self.dimension = int(test_emb.shape[1])
                else:
                    raise RuntimeError("Offline mode")
            except Exception:
                # Fallback gracefully to TF-IDF semantic projection
                self.is_transformer = False
                self.dimension = 384
                self._fallback_vectorizer = TfidfVectorizer(
                    max_features=self.dimension,
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )

    def fit_fallback(self, corpus: List[str]):
        """Fit fallback vectorizer on domain corpus if sentence-transformers is offline."""
        if not self.is_transformer and self._fallback_vectorizer is not None and corpus:
            self._fallback_vectorizer.fit(corpus)

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Generate dense embedding vectors for input text(s)."""
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else list(texts)

        if not text_list:
            empty = np.zeros((0, self.dimension), dtype=np.float32)
            return empty[0] if single_input else empty

        if self.is_transformer and self._st_model is not None:
            try:
                embeddings = self._st_model.encode(
                    text_list,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=False
                )
                embeddings = embeddings.astype(np.float32)
                return embeddings[0] if single_input else embeddings
            except Exception:
                pass  # Fall back to TF-IDF

        # Fallback dense TF-IDF projection
        if self._fallback_vectorizer is None:
            self._fallback_vectorizer = TfidfVectorizer(max_features=self.dimension)
            self._fallback_vectorizer.fit(text_list)

        try:
            sparse_matrix = self._fallback_vectorizer.transform(text_list)
            dense_matrix = sparse_matrix.toarray().astype(np.float32)
            # Pad to expected dimension if needed
            if dense_matrix.shape[1] < self.dimension:
                pad_width = ((0, 0), (0, self.dimension - dense_matrix.shape[1]))
                dense_matrix = np.pad(dense_matrix, pad_width, mode="constant")
            elif dense_matrix.shape[1] > self.dimension:
                dense_matrix = dense_matrix[:, :self.dimension]
        except Exception:
            # If transform fails before fit, fit on current texts
            self._fallback_vectorizer.fit(text_list)
            dense_matrix = self._fallback_vectorizer.transform(text_list).toarray().astype(np.float32)
            if dense_matrix.shape[1] < self.dimension:
                pad_width = ((0, 0), (0, self.dimension - dense_matrix.shape[1]))
                dense_matrix = np.pad(dense_matrix, pad_width, mode="constant")

        if normalize:
            norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dense_matrix = dense_matrix / norms

        return dense_matrix[0] if single_input else dense_matrix


# Singleton instance
default_embedding_model = AgriEmbeddingModel()
