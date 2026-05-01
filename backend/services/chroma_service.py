"""
ChromaDB vector database service for semantic search.

Additions over the original:
  - _normalize_query()      : lowercase + collapse word-digit compounds
  - search()                : dual-query (normalised + original), dedup, top_k
  - search_with_rerank()    : broad retrieval (top-20) → cross-encoder rerank → top_k
  - CrossEncoder            : loaded in __init__ with graceful fallback to None
"""
import re
import logging
import time
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger(__name__)


class ChromaService:
    """Manages vector database operations using ChromaDB."""

    def __init__(self) -> None:
        self.client         = None
        self.collection     = None
        self.embedding_model = None
        self.reranker       = None
        self._initialize()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        try:
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
            for attempt in range(3):
                try:
                    self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                    logger.info("Embedding model loaded successfully")
                    break
                except Exception as exc:
                    logger.warning(
                        "Embedding model load attempt %d/3 failed: %s", attempt + 1, exc
                    )
                    if attempt < 2:
                        logger.info("Retrying in 10 seconds…")
                        time.sleep(10)
                    else:
                        logger.error("All 3 attempts failed — raising exception")
                        raise

            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={
                    "description": "Ground truth documents for confidence scoring",
                    "hnsw:space":  "cosine",
                },
            )
            logger.info(
                "ChromaDB initialised. Collection: %s, doc count: %d",
                settings.CHROMA_COLLECTION_NAME,
                self.collection.count(),
            )

        except Exception as exc:
            logger.error("Failed to initialise ChromaDB: %s", exc)
            raise

        # Cross-encoder reranker — optional; falls back to None on failure
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder reranker loaded successfully")
        except Exception as exc:
            logger.warning(
                "Failed to load cross-encoder reranker: %s. Reranking disabled.", exc
            )
            self.reranker = None

    # ── Query normalisation ────────────────────────────────────────────────────

    def _normalize_query(self, query: str) -> str:
        """
        Normalise a search query for better embedding recall.

        Transformations applied (in order):
          1. Strip leading/trailing whitespace and lowercase.
          2. Collapse word + digit pairs:
               "GPT 4"     → "gpt4"
               "word 2"    → "word2"
          3. Collapse digit + short-word (≤ 4 chars) pairs — handles
             compound terms like word2vec without touching longer
             descriptor words like "model":
               "2 vec"     → "2vec"   (→ overall: "word2vec")
               "4 model"   → unchanged (model = 5 chars)
        """
        q = query.strip().lower()
        # Step 1 — word + digit
        q = re.sub(r"(\w+)\s+(\d+)", r"\1\2", q)
        # Step 2 — digit + short word (≤ 4 alpha chars at word boundary)
        q = re.sub(r"(\d+)\s+([a-z]{1,4})\b", r"\1\2", q)
        return q

    # ── Raw ChromaDB query (single embedding) ─────────────────────────────────

    def _raw_search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Embed `query` and retrieve `top_k` results from ChromaDB.
        This is the low-level method; callers should use `search()` instead.
        """
        query_embedding = self.embedding_model.encode([query])[0]

        query_kwargs: Dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results":        top_k,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        results  = self.collection.query(**query_kwargs)
        passages = []

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance   = results["distances"][0][i]
                similarity = max(0.0, min(1.0, 1.0 - distance))  # cosine [0,2] → [0,1]
                passages.append({
                    "text":             doc,
                    "metadata":         results["metadatas"][0][i],
                    "similarity_score": similarity,
                    "source":           results["metadatas"][0][i].get("source", "unknown"),
                    "page":             results["metadatas"][0][i].get("page", 0),
                })

        return passages

    # ── Public search (normalised + original, merged & deduped) ───────────────

    def search(
        self,
        query: str,
        top_k: int = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search with query normalisation and dual-retrieval.

        Steps:
          1. Normalise the query (lowercase + digit-compound collapse).
          2. Search with the normalised query.
          3. If the normalised form differs from the lowercased original,
             also search with the original and merge both result lists.
          4. Deduplicate by text content (keep the higher similarity_score).
          5. Sort descending by similarity_score and return top_k.

        Args:
            query : User question string.
            top_k : Number of results (defaults to settings.TOP_K_RETRIEVAL).
            where : Optional ChromaDB metadata filter dict.
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL

        try:
            normalised    = self._normalize_query(query)
            original_low  = query.strip().lower()

            passages = self._raw_search(normalised, top_k, where)

            # If normalisation changed anything, also retrieve with original
            if normalised != original_low:
                logger.debug(
                    "[ChromaDB] Dual-search: normalised=%r, original=%r",
                    normalised, original_low,
                )
                orig_passages = self._raw_search(original_low, top_k, where)
                passages = passages + orig_passages

            # Deduplicate by text content, keeping the higher score
            best: Dict[str, Dict] = {}
            for p in passages:
                txt = p["text"]
                if txt not in best or p["similarity_score"] > best[txt]["similarity_score"]:
                    best[txt] = p

            deduped = sorted(best.values(), key=lambda p: p["similarity_score"], reverse=True)
            return deduped[:top_k]

        except Exception as exc:
            logger.error("Error searching ChromaDB: %s", exc)
            raise

    # ── Cross-encoder reranking ────────────────────────────────────────────────

    def search_with_rerank(
        self,
        query: str,
        top_k: int = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Two-stage retrieval with optional cross-encoder reranking.

        Step 1: Retrieve up to 20 broad candidates via search().
        Step 2: If the cross-encoder is available, score every (query, passage)
                pair and return the top_k results sorted by reranker score.
                If the cross-encoder is unavailable, fall back to the
                embedding-similarity ranking from step 1.

        Args:
            query : User question string.
            top_k : Number of results to return (defaults to settings.TOP_K_RETRIEVAL).
            where : Optional ChromaDB metadata filter.
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL

        candidates = self.search(query, top_k=20, where=where)

        if not candidates:
            return candidates

        if self.reranker is not None:
            try:
                pairs  = [[query, p["text"]] for p in candidates]
                scores = self.reranker.predict(pairs)
                ranked = sorted(
                    zip(candidates, scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                return [c for c, _ in ranked[:top_k]]
            except Exception as exc:
                logger.warning(
                    "Reranker prediction failed (%s) — falling back to embedding rank.", exc
                )

        return candidates[:top_k]

    # ── Document management ───────────────────────────────────────────────────

    def add_documents(self, chunks: List[Dict], db_batch_size: int = 250) -> int:
        """Add document chunks to the vector database in safe batches."""
        if not chunks:
            return 0

        import gc

        try:
            total_added  = 0
            total_chunks = len(chunks)
            logger.info("Starting vector ingestion for %d chunks…", total_chunks)

            for i in range(0, total_chunks, db_batch_size):
                batch_chunks = chunks[i : i + db_batch_size]

                texts = [c["text"] for c in batch_chunks]
                ids   = [
                    f"{c.get('document_id', 'doc')}_{c['chunk_id']}"
                    for c in batch_chunks
                ]
                metadatas = [
                    {
                        "source":      str(c.get("source",      "unknown")),
                        "document_id": str(c.get("document_id", "unknown")),
                        "chunk_id":    int(c.get("chunk_id",    0)),
                        "page":        int(c.get("page",        0)),
                    }
                    for c in batch_chunks
                ]

                logger.info(
                    "Generating embeddings for batch %d (%d chunks)…",
                    i // db_batch_size + 1,
                    len(texts),
                )
                embeddings = self.embedding_model.encode(
                    texts, batch_size=32, show_progress_bar=False
                )
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                total_added += len(batch_chunks)
                del embeddings, texts
                gc.collect()

            logger.info("Successfully added %d chunks to ChromaDB", total_added)
            return total_added

        except Exception as exc:
            logger.error("Error adding documents to ChromaDB: %s", exc)
            raise

    def delete_document(self, filename: str) -> bool:
        """Delete all vector chunks associated with a specific filename."""
        try:
            logger.info("Deleting vectors for source: %s", filename)
            self.collection.delete(where={"source": filename})
            logger.info("Successfully deleted vectors for %s", filename)
            return True
        except Exception as exc:
            logger.error("Error deleting document %s: %s", filename, exc)
            return False

    def get_count(self) -> int:
        """Return total number of chunks in collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def is_ready(self) -> bool:
        """Return True if ChromaDB and the embedding model are available."""
        try:
            return self.collection is not None and self.embedding_model is not None
        except Exception:
            return False
