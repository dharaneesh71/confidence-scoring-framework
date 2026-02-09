"""
ChromaDB vector database service for semantic search
Updated with Delete Functionality
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import math
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
        """Add document chunks to the vector database"""
        if not chunks:
            return 0
        
        try:
            texts = [chunk["text"] for chunk in chunks]
            # Create unique IDs for chunks: filename_chunkIndex
            ids = [f"{chunk.get('document_id', 'doc')}_{chunk['chunk_id']}" for chunk in chunks]
            
            metadatas = []
            for chunk in chunks:
                metadata = {
                    "source": str(chunk.get("source", "unknown")),
                    "document_id": str(chunk.get("document_id", "unknown")),
                    "chunk_id": int(chunk.get("chunk_id", 0)),
                    "page": int(chunk.get("page", 0)),
                }
                metadatas.append(metadata)
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
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

    # --- NEW METHOD: DELETE DOCUMENT ---
    def delete_document(self, filename: str) -> bool:
        """
        Delete all vector chunks associated with a specific filename.
        This removes the document's knowledge from the AI.
        """
        try:
            logger.info(f"Deleting vectors for source: {filename}")
            # Delete where metadata 'source' matches filename
            self.collection.delete(
                where={"source": filename}
            )
            logger.info(f"Successfully deleted vectors for {filename}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return False
    # -----------------------------------

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Semantic search for relevant passages"""
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            passages = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]
                    
                    # Convert L2 distance to similarity score [0, 1]
                    similarity = math.exp(-distance)
                    similarity = max(0.0, min(1.0, similarity))
                    
                    passages.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity_score": similarity,
                        "source": results['metadatas'][0][i].get('source', 'unknown'),
                        "page": results['metadatas'][0][i].get('page', 0)
                    })
            
            return passages
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise
    
    def get_count(self) -> int:
        """Get total number of documents/chunks in collection"""
        try:
            return self.collection.count()
        except Exception as e:
            return 0
    
    def is_ready(self) -> bool:
        """Check if ChromaDB service is ready"""
        try:
            return self.collection is not None and self.embedding_model is not None
        except:
            return False