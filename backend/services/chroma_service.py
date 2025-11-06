"""
ChromaDB vector database service for semantic search
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from core.config import settings

logger = logging.getLogger(__name__)


class ChromaService:
    """Manages vector database operations using ChromaDB"""
    
    def __init__(self):
        """Initialize ChromaDB client and embedding model"""
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB and load embedding model"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Load embedding model
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "Ground truth documents for confidence scoring"}
            )
            
            logger.info(f"ChromaDB initialized. Collection: {settings.CHROMA_COLLECTION_NAME}")
            logger.info(f"Current document count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict]) -> int:
        """
        Add document chunks to the vector database
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        try:
            # Extract texts and prepare metadata
            texts = [chunk["text"] for chunk in chunks]
            ids = [f"{chunk.get('document_id', 'doc')}_{chunk['chunk_id']}" for chunk in chunks]
            
            # Prepare metadata (ChromaDB requires dict values to be str, int, float, or bool)
            metadatas = []
            for chunk in chunks:
                metadata = {
                    "source": str(chunk.get("source", "unknown")),
                    "document_id": str(chunk.get("document_id", "unknown")),
                    "chunk_id": int(chunk.get("chunk_id", 0)),
                    "page": int(chunk.get("page", 0)),
                }
                metadatas.append(metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Semantic search for relevant passages
        
        Args:
            query: Search query text
            top_k: Number of results to return (default from settings)
            
        Returns:
            List of relevant passages with metadata and scores
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            passages = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    # ChromaDB returns squared L2 distances
                    # Convert to similarity: smaller distance = higher similarity
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity score (0 to 1)
                    # Using exponential decay: similarity = exp(-distance)
                    # This ensures: distance=0 → similarity=1, large distance → similarity≈0
                    import math
                    similarity = math.exp(-distance)
                    
                    # Clamp to [0, 1] range
                    similarity = max(0.0, min(1.0, similarity))
                    
                    passage = {
                        "text": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity_score": similarity,
                        "source": results['metadatas'][0][i].get('source', 'unknown'),
                        "page": results['metadatas'][0][i].get('page', 0)
                    }
                    passages.append(passage)
            
            logger.info(f"Found {len(passages)} relevant passages for query")
            
            # Log similarity scores for debugging
            if passages:
                scores = [p['similarity_score'] for p in passages]
                logger.info(f"Similarity scores: {scores}")
            
            return passages
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise
    
    def get_count(self) -> int:
        """Get total number of documents in collection"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the entire collection (use with caution!)"""
        try:
            self.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            logger.info(f"Deleted collection: {settings.CHROMA_COLLECTION_NAME}")
            self._initialize()  # Reinitialize
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if ChromaDB service is ready"""
        try:
            return self.collection is not None and self.embedding_model is not None
        except:
            return False