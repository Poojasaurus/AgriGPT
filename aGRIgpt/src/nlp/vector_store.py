"""Agricultural Vector Store & Semantic Search.
Provides FAISS vector indexing with dense embedding search and hybrid keyword filtering
over the agricultural domain knowledge base.
"""

from __future__ import annotations

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

from src.nlp.embeddings import AgriEmbeddingModel, default_embedding_model


class AgriVectorStore:
    """FAISS and NumPy-powered vector store for agricultural knowledge retrieval."""

    DEFAULT_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "agri_knowledge_base.json")

    def __init__(self, embedding_model: Optional[AgriEmbeddingModel] = None, kb_path: Optional[str] = None):
        self.embedding_model = embedding_model or default_embedding_model
        self.kb_path = os.path.abspath(kb_path or self.DEFAULT_KB_PATH)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.use_faiss = False

        self._initialize()

    def _initialize(self):
        """Load knowledge base documents and build index."""
        self.load_documents()
        self.build_index()

    def load_documents(self):
        """Load domain knowledge documents from JSON."""
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # If fallback vectorizer is used in embedding model, fit on document corpus
        if not self.embedding_model.is_transformer:
            corpus = [
                f"{doc.get('title', '')} {doc.get('content', '')} {' '.join(doc.get('keywords', []))}"
                for doc in self.documents
            ]
            self.embedding_model.fit_fallback(corpus)

    def build_index(self):
        """Encode all documents and construct vector search index."""
        if not self.documents:
            return

        texts_to_embed = [
            f"{doc.get('title', '')}. {doc.get('content', '')}"
            for doc in self.documents
        ]

        self.embeddings = self.embedding_model.encode(texts_to_embed, normalize=True)

        # Try building FAISS index
        try:
            import faiss
            dim = self.embeddings.shape[1]
            # Inner product index for normalized vectors = Cosine similarity
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings.astype(np.float32))
            self.use_faiss = True
        except Exception as e:
            # Fallback to NumPy cosine similarity
            self.use_faiss = False
            self.faiss_index = None

    def search(
        self,
        query: str,
        top_k: int = 3,
        crop_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        keywords_boost: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search with optional hybrid filtering and keyword boosts."""
        if not self.documents or self.embeddings is None:
            return []

        query_vec = self.embedding_model.encode(query, normalize=True).reshape(1, -1)

        # 1. Base semantic scores
        if self.use_faiss and self.faiss_index is not None:
            # Search larger candidate set if filters apply
            candidate_k = min(len(self.documents), top_k * 3 if (crop_filter or category_filter) else top_k)
            scores, indices = self.faiss_index.search(query_vec.astype(np.float32), candidate_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # NumPy dot product for cosine similarity
            scores = np.dot(self.embeddings, query_vec.T).squeeze()
            indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(indices):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]
            base_score = float(scores[rank]) if self.use_faiss else float(scores[idx])

            # Filter by crop if specified and document isn't "General"
            if crop_filter:
                doc_crop = doc.get("crop", "").lower()
                target_crop = crop_filter.lower()
                if doc_crop != "general" and doc_crop != target_crop:
                    continue

            # Filter by category if specified
            if category_filter and doc.get("category", "").lower() != category_filter.lower():
                continue

            # Hybrid keyword boost
            boost = 0.0
            if keywords_boost:
                doc_text = (doc.get("title", "") + " " + doc.get("content", "")).lower()
                for kw in keywords_boost:
                    if kw.lower() in doc_text:
                        boost += 0.05

            final_score = min(1.0, max(0.0, base_score + boost))

            results.append({
                "id": doc.get("id"),
                "crop": doc.get("crop"),
                "category": doc.get("category"),
                "title": doc.get("title"),
                "content": doc.get("content"),
                "score": round(final_score, 4),
                "keywords": doc.get("keywords", [])
            })

            if len(results) >= top_k:
                break

        # Re-sort by final combined score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# Singleton instance
default_vector_store = AgriVectorStore()
