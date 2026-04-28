"""
ChromaDB vector database service for semantic search
Updated with Delete Functionality + Domain Filtering
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from core.config import settings
import time

logger = logging.getLogger(__name__)

class ChromaService:
    """Manages vector database operations using ChromaDB"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB and load embedding model"""
        try:
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            for attempt in range(3):
                try:
                    self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                    logger.info("Embedding model loaded successfully")
                    break
                except Exception as e:
                    logger.warning(f"Embedding model load attempt {attempt + 1}/3 failed: {e}")
                    if attempt < 2:
                        logger.info("Retrying in 10 seconds...")
                        time.sleep(10)
                    else:
                        logger.error("All 3 attempts failed — raising exception")
                        raise
            
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={
                    "description": "Ground truth documents for confidence scoring",
                    "hnsw:space": "cosine"
                }
            )
            
            logger.info(f"ChromaDB initialized. Collection: {settings.CHROMA_COLLECTION_NAME}")
            logger.info(f"Current document count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict], db_batch_size: int = 250) -> int:
        """Add document chunks to the vector database in safe batches"""
        if not chunks:
            return 0
        
        import gc
        
        try:
            total_added  = 0
            total_chunks = len(chunks)
            logger.info(f"Starting vector ingestion for {total_chunks} chunks...")
            
            for i in range(0, total_chunks, db_batch_size):
                batch_chunks = chunks[i:i + db_batch_size]
                
                texts = [chunk["text"] for chunk in batch_chunks]
                ids   = [
                    f"{chunk.get('document_id', 'doc')}_{chunk['chunk_id']}"
                    for chunk in batch_chunks
                ]
                
                metadatas = []
                for chunk in batch_chunks:
                    metadatas.append({
                        "source":      str(chunk.get("source",      "unknown")),
                        "document_id": str(chunk.get("document_id", "unknown")),
                        "chunk_id":    int(chunk.get("chunk_id",    0)),
                        "page":        int(chunk.get("page",        0)),
                    })
                
                logger.info(
                    f"Generating embeddings for batch {i//db_batch_size + 1} "
                    f"({len(texts)} chunks)..."
                )
                embeddings = self.embedding_model.encode(
                    texts, batch_size=32, show_progress_bar=False
                )
                
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_added += len(batch_chunks)
                del embeddings, texts
                gc.collect()
                
            logger.info(f"Successfully added {total_added} chunks to ChromaDB")
            return total_added
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def delete_document(self, filename: str) -> bool:
        """Delete all vector chunks associated with a specific filename."""
        try:
            logger.info(f"Deleting vectors for source: {filename}")
            self.collection.delete(where={"source": filename})
            logger.info(f"Successfully deleted vectors for {filename}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return False

    def search(
        self,
        query:   str,
        top_k:   int             = None,
        where:   Optional[Dict]  = None,   # ← NEW: optional ChromaDB metadata filter
    ) -> List[Dict]:
        """
        Semantic search for relevant passages.
        
        Args:
            query:  The user's question string.
            top_k:  How many results to return (defaults to settings.TOP_K_RETRIEVAL).
            where:  Optional ChromaDB metadata filter dict, e.g.
                    {"document_id": {"$in": ["file1.pdf", "file2.pdf"]}}
                    Pass None (default) to search across ALL documents.
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]

            # Build query kwargs — only include 'where' if a filter was provided.
            # Passing where=None to ChromaDB raises an error, so we exclude it entirely.
            query_kwargs = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results":        top_k,
                "include":          ["documents", "metadatas", "distances"],
            }
            if where is not None:
                query_kwargs["where"] = where   # ← only added when domain filter is active
                logger.info(f"[ChromaDB] Applying domain filter: {where}")

            results = self.collection.query(**query_kwargs)
            
            passages = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    distance   = results['distances'][0][i]
                    similarity = max(0.0, min(1.0, 1.0 - distance))  # cosine [0,2] → [0,1]
                    
                    passages.append({
                        "text":             doc,
                        "metadata":         results['metadatas'][0][i],
                        "similarity_score": similarity,
                        "source":           results['metadatas'][0][i].get('source', 'unknown'),
                        "page":             results['metadatas'][0][i].get('page', 0)
                    })
            
            return passages
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise
    
    def get_count(self) -> int:
        """Get total number of chunks in collection"""
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def is_ready(self) -> bool:
        """Check if ChromaDB service is ready"""
        try:
            return self.collection is not None and self.embedding_model is not None
        except Exception:
            return False
