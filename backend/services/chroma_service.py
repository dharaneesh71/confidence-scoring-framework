"""
ChromaDB vector database service for semantic search.

PRODUCTION VERSION — TF-IDF embedding (no PyTorch):
─────────────────────────────────────────────────────────────────────────────
1. TFIDF EMBEDDING: Replaced SentenceTransformer (PyTorch) with sklearn
   TfidfVectorizer. No native ML libraries — runs on any Linux node.

2. VECTORIZER PERSISTENCE: Fitted vectorizer saved to /app/data/ (persistent
   volume) so vocabulary survives container restarts.

3. COSINE SIMILARITY: 1.0 - (distance / 2.0) — correct ChromaDB formula.

4. DOMAIN IN METADATA: Domain stored per-chunk for direct filtering.

5. SAFE BATCH SIZES: db_batch_size=50 prevents OOM on large PDFs.

6. DUAL-RETRIEVAL + DEDUPLICATION: Normalised + original query merged.
"""

import gc
import joblib
import logging
import math
import os
import re
from typing import Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sklearn.feature_extraction.text import TfidfVectorizer

from core.config import settings

logger = logging.getLogger(__name__)

VECTORIZER_PATH = "/app/data/tfidf_vectorizer.pkl"


class ChromaService:
    """Manages ChromaDB vector database operations for semantic search."""

    def __init__(self) -> None:
        self.client      = None
        self.collection  = None
        self.vectorizer  = None
        self._is_fitted  = False
        self.reranker    = None
        self._initialize()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        try:
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # ── Load or create TF-IDF vectorizer ──────────────────────────
            if os.path.exists(VECTORIZER_PATH):
                logger.info("Loading TF-IDF vectorizer from disk: %s", VECTORIZER_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self._is_fitted = True
                logger.info("✅ TF-IDF vectorizer loaded (vocabulary ready)")
            else:
                logger.info("Creating new TF-IDF vectorizer (will fit on first upload)...")
                self.vectorizer = TfidfVectorizer(
                    max_features=384,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                    analyzer="word",
                    min_df=1,
                )
                self._is_fitted = False
                logger.info("✅ TF-IDF vectorizer created — upload documents to fit")

            # ── ChromaDB collection ────────────────────────────────────────
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={
                    "description": "Ground truth documents for confidence scoring",
                    "hnsw:space":  "cosine",
                },
            )

            count = self.collection.count()
            logger.info(
                "✅ ChromaDB ready — collection='%s', chunks=%d",
                settings.CHROMA_COLLECTION_NAME, count,
            )
            if count == 0:
                logger.warning(
                    "⚠️  ChromaDB collection is EMPTY. "
                    "Upload documents via the Admin panel before querying."
                )

        except Exception as exc:
            logger.error("❌ Failed to initialise ChromaDB: %s", exc)
            raise

        # Reranker disabled — no PyTorch available
        self.reranker = None
        logger.info("Reranker disabled (TF-IDF mode — no CrossEncoder)")

    # ── Embedding helpers ──────────────────────────────────────────────────────

    def _embed_text(self, text: str) -> list:
        """Convert single text to TF-IDF vector."""
        if not self._is_fitted:
            return [0.0] * 384
        return self.vectorizer.transform([text]).toarray()[0].tolist()

    def _embed_texts(self, texts: List[str]) -> List[list]:
        """Convert list of texts to TF-IDF vectors."""
        if not self._is_fitted:
            return [[0.0] * 384] * len(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    # ── Query normalisation ────────────────────────────────────────────────────

    def _normalize_query(self, query: str) -> str:
        q = query.strip().lower()
        q = re.sub(r"(\w+)\s+(\d+)", r"\1\2", q)
        q = re.sub(r"(\d+)\s+([a-z]{1,4})\b", r"\1\2", q)
        return q

    # ── Low-level single-embedding search ─────────────────────────────────────

    def _raw_search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Embed query and return top_k results from ChromaDB."""
        count = self.collection.count()
        if count == 0:
            logger.warning("[ChromaDB] Empty collection — ingest docs first.")
            return []

        if not self._is_fitted:
            logger.warning("[ChromaDB] Vectorizer not fitted — upload docs first.")
            return []

        query_embedding = self._embed_text(query)

        query_kwargs: Dict = {
            "query_embeddings": [query_embedding],
            "n_results":        min(top_k, count),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        results  = self.collection.query(**query_kwargs)
        passages = []

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance   = results["distances"][0][i]
                similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                passages.append({
                    "text":             doc,
                    "metadata":         results["metadatas"][0][i],
                    "similarity_score": round(similarity, 4),
                    "source":           results["metadatas"][0][i].get("source",  "unknown"),
                    "page":             results["metadatas"][0][i].get("page",    0),
                    "domain":           results["metadatas"][0][i].get("domain",  "general"),
                })

        logger.debug(
            "[ChromaDB] query=%r → %d results (top sim=%.3f)",
            query[:60], len(passages),
            passages[0]["similarity_score"] if passages else 0.0,
        )
        return passages

    # ── Public search ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int            = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Semantic search with dual-retrieval and deduplication."""
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL

        try:
            normalised   = self._normalize_query(query)
            original_low = query.strip().lower()

            passages = self._raw_search(normalised, top_k, where)

            if normalised != original_low:
                orig_passages = self._raw_search(original_low, top_k, where)
                passages = passages + orig_passages

            best: Dict[str, Dict] = {}
            for p in passages:
                txt = p["text"]
                if (
                    txt not in best
                    or p["similarity_score"] > best[txt]["similarity_score"]
                ):
                    best[txt] = p

            return sorted(
                best.values(),
                key=lambda p: p["similarity_score"],
                reverse=True,
            )[:top_k]

        except Exception as exc:
            logger.error("Error searching ChromaDB: %s", exc)
            raise

    # ── Two-stage retrieval ────────────────────────────────────────────────────

    def search_with_rerank(
        self,
        query: str,
        top_k: int            = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Reranker disabled in TF-IDF mode — returns top_k by similarity."""
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL
        candidates = self.search(query, top_k=20, where=where)
        return candidates[:top_k]

    # ── Document ingestion ────────────────────────────────────────────────────

    def add_documents(
        self,
        chunks: List[Dict],
        domain: str        = "general",
        db_batch_size: int = 50,
    ) -> int:
        """
        Add document chunks to ChromaDB.
        Fits TF-IDF vectorizer on all texts and saves to disk.
        """
        if not chunks:
            return 0

        total_added  = 0
        total_chunks = len(chunks)
        logger.info("[ChromaDB] Ingesting %d chunks (domain=%s)…", total_chunks, domain)

        try:
            # ── Fit vectorizer on ALL texts at once ────────────────────────
            all_texts = [c["text"] for c in chunks]
            logger.info("Fitting TF-IDF on %d texts...", len(all_texts))
            self.vectorizer.fit(all_texts)
            self._is_fitted = True

            # Save fitted vectorizer to persistent volume
            os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)
            logger.info("✅ Vectorizer fitted and saved to %s", VECTORIZER_PATH)

            # ── Add in batches ─────────────────────────────────────────────
            for i in range(0, total_chunks, db_batch_size):
                batch     = chunks[i: i + db_batch_size]
                texts     = [c["text"] for c in batch]
                ids       = [
                    f"{c.get('document_id', 'doc')}_{c['chunk_id']}"
                    for c in batch
                ]
                metadatas = [
                    {
                        "source":      str(c.get("source",      "unknown")),
                        "document_id": str(c.get("document_id", "unknown")),
                        "chunk_id":    int(c.get("chunk_id",    0)),
                        "page":        int(c.get("page",        0)),
                        "domain":      domain.lower(),
                    }
                    for c in batch
                ]

                logger.info(
                    "[ChromaDB] Embedding batch %d/%d (%d chunks)…",
                    i // db_batch_size + 1,
                    math.ceil(total_chunks / db_batch_size),
                    len(texts),
                )

                embeddings = self.vectorizer.transform(texts).toarray().tolist()

                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )

                total_added += len(batch)
                gc.collect()

                logger.info(
                    "[ChromaDB] Batch done — %d/%d ingested.",
                    total_added, total_chunks,
                )

            logger.info(
                "✅ [ChromaDB] Ingested %d chunks (domain=%s)",
                total_added, domain,
            )
            return total_added

        except Exception as exc:
            logger.error("❌ Error adding documents: %s", exc)
            raise

    # ── Utility ────────────────────────────────────────────────────────────────

    def delete_document(self, filename: str) -> bool:
        try:
            self.collection.delete(where={"source": filename})
            logger.info("[ChromaDB] Deleted vectors for '%s'", filename)
            return True
        except Exception as exc:
            logger.error("[ChromaDB] Error deleting '%s': %s", filename, exc)
            return False

    def get_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def is_ready(self) -> bool:
        try:
            return self.collection is not None and self.vectorizer is not None
        except Exception:
            return False