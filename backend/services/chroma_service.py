"""
ChromaDB vector database service for semantic search.

PRODUCTION VERSION — All fixes applied:
─────────────────────────────────────────────────────────────────────────────
1. COSINE SIMILARITY FORMULA:
   ChromaDB cosine space: distance = 1 − cos_sim, range [0, 2].
   CORRECT: similarity = 1.0 - (distance / 2.0) → maps [0,2] → [1.0, 0.0]

2. DOMAIN IN CHUNK METADATA:
   Domain stored per-chunk for direct ChromaDB filtering:
   where={"domain": "nlp"}

3. COLLECTION-EMPTY GUARD:
   Clear warning at startup and on every query if collection is empty.

4. SAFE BATCH SIZES:
   db_batch_size=50  (was 250 — caused OOM crash on large PDFs)
   encode batch_size=16 (was 32 — reduces peak memory per batch)

5. DUAL-RETRIEVAL + DEDUPLICATION:
   Searches with both normalised and original query form, merges and
   deduplicates by text, keeping the higher similarity score.

6. GRACEFUL RERANKER FALLBACK:
   Cross-encoder reranker loaded optionally — falls back cleanly if
   unavailable without crashing the service.
"""

import gc
import logging
import math
import re
import time
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger(__name__)


class ChromaService:
    """Manages ChromaDB vector database operations for semantic search."""

    def __init__(self, embedding_model=None) -> None:
        self.client          = None
        self.collection      = None
        self.embedding_model = embedding_model
        self.reranker        = None
        self._initialize()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """
        Initialise ChromaDB client, embedding model, collection, and
        optional cross-encoder reranker. Retries model load up to 3 times.
        """
        try:
            # ── ChromaDB persistent client ─────────────────────────────────
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # ── Embedding model (SentenceTransformer) ──────────────────────
            if self.embedding_model is None:
                logger.info("Loading embedding model...")
                for attempt in range(1, 4):
                    try:
                        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
                        logger.info("✅ Embedding model loaded successfully")
                        break
                    except Exception as exc:
                        logger.warning("Embedding model load attempt %d/3 failed: %s",attempt, exc,)
                        if attempt < 3:
                            logger.info("Retrying in 10 seconds…")
                            time.sleep(10)
                        else:
                            logger.error("All 3 attempts failed — raising")
                            raise
            else:
                logger.info("✅ Embedding model provided externally — skipping load")
                    
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
                settings.CHROMA_COLLECTION_NAME,
                count,
            )
            if count == 0:
                logger.warning(
                    "⚠️  ChromaDB collection is EMPTY. "
                    "Upload documents via the Admin panel before querying."
                )

        except Exception as exc:
            logger.error("❌ Failed to initialise ChromaDB: %s", exc)
            raise

        # ── Optional cross-encoder reranker ────────────────────────────────
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info("✅ Cross-encoder reranker loaded")
        except Exception as exc:
            logger.warning(
                "⚠️  Cross-encoder unavailable (%s). Reranking disabled.", exc
            )
            self.reranker = None

    # ── Query normalisation ────────────────────────────────────────────────────

    def _normalize_query(self, query: str) -> str:
        """
        Normalise a search query for better embedding recall.
        1. Strip + lowercase.
        2. Collapse word + digit:              "GPT 4"  → "gpt4"
        3. Collapse digit + short word (≤4 c): "2 vec"  → "2vec"
        """
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
        """Embed `query` and return top_k results from ChromaDB."""
        count = self.collection.count()
        if count == 0:
            logger.warning(
                "[ChromaDB] _raw_search on empty collection — ingest docs first."
            )
            return []

        query_embedding = self.embedding_model.encode(
            [query], convert_to_tensor=False
        )[0]

        query_kwargs: Dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results":        min(top_k, count),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        results  = self.collection.query(**query_kwargs)
        passages = []

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]

                # FIX: ChromaDB cosine space → distance = 1 − cos_sim ∈ [0,2]
                # CORRECT mapping: similarity = 1.0 - (distance / 2.0)
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
            query[:60],
            len(passages),
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
        """
        Semantic search with dual-retrieval and deduplication.

        1. Normalise query (lowercase + digit-compound collapse).
        2. Search with normalised form.
        3. If normalised ≠ original, search with original and merge.
        4. Deduplicate by text (keep higher similarity score).
        5. Sort descending and return top_k.
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL

        try:
            normalised   = self._normalize_query(query)
            original_low = query.strip().lower()

            passages = self._raw_search(normalised, top_k, where)

            if normalised != original_low:
                logger.debug(
                    "[ChromaDB] Dual-search: normalised=%r original=%r",
                    normalised, original_low,
                )
                orig_passages = self._raw_search(original_low, top_k, where)
                passages = passages + orig_passages

            # Deduplicate — keep higher similarity score per unique text
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

    # ── Two-stage retrieval with optional reranking ───────────────────────────

    def search_with_rerank(
        self,
        query: str,
        top_k: int            = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Stage 1: Retrieve up to 20 broad candidates via search().
        Stage 2: Cross-encoder rerank → return top_k.
        Falls back to embedding-similarity ranking if reranker unavailable.
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
                    "Reranker failed (%s) — falling back to similarity rank.",
                    exc,
                )

        return candidates[:top_k]

    # ── Document ingestion ────────────────────────────────────────────────────

    def add_documents(
        self,
        chunks: List[Dict],
        domain: str        = "general",
        db_batch_size: int = 50,           # ✅ was 250 — reduced to prevent OOM
    ) -> int:
        """
        Add document chunks to ChromaDB in safe batches.

        Stores `domain` in every chunk's metadata for direct filtering:
            where={"domain": "nlp"}

        Args:
            chunks        : chunk dicts from PDFProcessor.process_pdf()
            domain        : domain tag stored in metadata
            db_batch_size : max chunks per ChromaDB upsert (keep ≤ 50)

        Returns:
            Total number of chunks added.
        """
        if not chunks:
            return 0

        total_added  = 0
        total_chunks = len(chunks)
        logger.info(
            "[ChromaDB] Ingesting %d chunks (domain=%s)…",
            total_chunks, domain,
        )

        try:
            for i in range(0, total_chunks, db_batch_size):
                batch = chunks[i: i + db_batch_size]

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

                # ✅ encode batch_size=16 (was 32) — reduces peak memory
                embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=16,
                    show_progress_bar=False,
                    convert_to_tensor=False,
                )

                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )

                total_added += len(batch)
                del embeddings, texts
                gc.collect()

                logger.info(
                    "[ChromaDB] Batch done — %d/%d chunks ingested.",
                    total_added, total_chunks,
                )

            logger.info(
                "✅ [ChromaDB] Successfully ingested %d chunks (domain=%s)",
                total_added, domain,
            )
            return total_added

        except Exception as exc:
            logger.error(
                "❌ Error adding documents to ChromaDB: %s", exc
            )
            raise

    # ── Utility ────────────────────────────────────────────────────────────────

    def delete_document(self, filename: str) -> bool:
        """Delete all vector chunks for a given source filename."""
        try:
            self.collection.delete(where={"source": filename})
            logger.info("[ChromaDB] Deleted vectors for '%s'", filename)
            return True
        except Exception as exc:
            logger.error(
                "[ChromaDB] Error deleting '%s': %s", filename, exc
            )
            return False

    def get_count(self) -> int:
        """Return total number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def is_ready(self) -> bool:
        """Return True if ChromaDB and embedding model are both initialised."""
        try:
            return (
                self.collection      is not None
                and self.embedding_model is not None
            )
        except Exception:
            return False