"""
Simplified Single-File Backend for Confidence Scoring Framework
This is a working, standalone version for Sprint 1 demo

Save this as: backend/simple_backend.py
Run with: python simple_backend.py
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uvicorn
import secrets
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import re
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_HOST = "localhost"
BACKEND_PORT = 8000
CHROMA_PERSIST_DIRECTORY = "../data/chroma_db"
CHROMA_COLLECTION_NAME = "confidence_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
UPLOAD_DIRECTORY = "../data/uploads"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Create required directories
Path(CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Confidence Scoring Framework", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()

# Global services
chroma_client = None
collection = None
embedding_model = None

# ==================== Pydantic Models ====================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    excerpt: str
    similarity_score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    confidence_label: str
    citations: List[Citation]
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    document_id: Optional[str] = None
    chunks_created: Optional[int] = None

class StatusResponse(BaseModel):
    status: str
    knowledge_base_ready: bool
    llm_ready: bool
    documents_count: int

# ==================== Helper Functions ====================

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
    """Verify admin credentials"""
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

def initialize_services():
    """Initialize ChromaDB and embedding model"""
    global chroma_client, collection, embedding_model
    
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        
        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        logger.info(f"Services initialized. Documents: {collection.count()}")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

def extract_text_from_pdf(pdf_path: str) -> tuple:
    """Extract text from PDF"""
    try:
        text = ""
        page_texts = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    page_texts[page_num + 1] = page_text
                    text += f"\n{page_text}\n"
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        return text, page_texts
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """Split text into chunks"""
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": end,
            })
            chunk_id += 1
        
        start = end - overlap
    
    return chunks

def generate_mock_answer(question: str, context: str = None) -> str:
    """Generate mock answers for demo"""
    question_lower = question.lower()
    
    if "confidence" in question_lower and "score" in question_lower:
        return ("The confidence scoring framework evaluates AI-generated answers by comparing them "
               "to verified ground truth documents. It uses semantic similarity, factual consistency, "
               "and completeness metrics to generate a score between 0.0 and 1.0.")
    
    elif "architecture" in question_lower or "component" in question_lower:
        return ("The system consists of six core components: a React.js frontend, "
               "a FastAPI backend, a Llama 3.1 AI model, a ChromaDB vector database, "
               "a confidence scoring pipeline, and curated ground truth datasets.")
    
    elif "how" in question_lower and "work" in question_lower:
        return ("When a user submits a question, the system generates an answer using the AI model, "
               "retrieves relevant passages from the knowledge base, compares them using semantic "
               "similarity, and computes a confidence score based on multiple dimensions.")
    
    elif context:
        sentences = context.split('.')[:2]
        return ' '.join(sentences).strip() + '.'
    
    else:
        return (f"Based on the available documentation, I can provide information about {question}. "
               "The system uses advanced AI techniques to ensure accuracy and reliability.")

def compute_confidence_score(answer: str, passages: List[dict]) -> tuple:
    """Compute confidence score"""
    if not answer or not passages:
        return 0.0, "Insufficient information", []
    
    try:
        # Encode answer and passages
        answer_emb = embedding_model.encode([answer])[0]
        passage_texts = [p["text"] for p in passages]
        passage_embs = embedding_model.encode(passage_texts)
        
        # Compute similarities
        from numpy import dot
        from numpy.linalg import norm
        
        similarities = []
        for p_emb in passage_embs:
            sim = dot(answer_emb, p_emb) / (norm(answer_emb) * norm(p_emb))
            similarities.append(float(sim))
        
        # Final score (simple average)
        score = sum(similarities) / len(similarities)
        score = max(0.0, min(1.0, score))
        
        # Create citations
        citations = []
        for i, passage in enumerate(passages):
            citations.append({
                "source": passage.get("source", "unknown"),
                "page": passage.get("page"),
                "excerpt": passage["text"][:200] + "..." if len(passage["text"]) > 200 else passage["text"],
                "similarity_score": round(similarities[i], 2)
            })
        
        # Generate label
        if score >= 0.8:
            label = "High"
        elif score >= 0.5:
            label = "Medium"
        else:
            label = "Low"
        
        explanation = f"Confidence level: {label}. Based on semantic similarity with source documents."
        
        return round(score, 2), explanation, citations
        
    except Exception as e:
        logger.error(f"Error computing score: {e}")
        return 0.5, "Error occurred during scoring", []

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Confidence Scoring Framework...")
    initialize_services()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Confidence Scoring Framework API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "upload": "/api/upload",
            "status": "/api/status",
        }
    }

@app.post("/api/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Submit a question and get AI answer with confidence score"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Search knowledge base
        query_embedding = embedding_model.encode([request.question])[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found. Please upload documents first."
            )
        
        # Format passages
        passages = []
        for i, doc in enumerate(results['documents'][0]):
            passages.append({
                "text": doc,
                "metadata": results['metadatas'][0][i],
                "similarity_score": 1 - results['distances'][0][i],
                "source": results['metadatas'][0][i].get('source', 'unknown'),
                "page": results['metadatas'][0][i].get('page', 0)
            })
        
        # Generate answer
        context = "\n\n".join([p["text"] for p in passages])
        answer = generate_mock_answer(request.question, context)
        
        # Compute confidence score
        confidence_score, explanation, citations = compute_confidence_score(answer, passages)
        
        # Determine label
        if confidence_score >= 0.8:
            label = "High"
        elif confidence_score >= 0.5:
            label = "Medium"
        else:
            label = "Low"
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            confidence_score=confidence_score,
            confidence_label=label,
            citations=citations,
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    is_admin: bool = Depends(verify_admin)
):
    """Upload PDF document to knowledge base"""
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        # Save file
        file_path = Path(UPLOAD_DIRECTORY) / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text
        text, page_texts = extract_text_from_pdf(str(file_path))
        
        if not text:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        
        # Create chunks
        chunks = chunk_text(text)
        
        # Add metadata
        for chunk in chunks:
            chunk["source"] = file.filename
            chunk["document_id"] = file.filename
            chunk["total_pages"] = len(page_texts)
            
            # Estimate page
            chars_per_page = len(text) / len(page_texts) if page_texts else len(text)
            estimated_page = int(chunk["start_char"] / chars_per_page) + 1
            chunk["page"] = min(estimated_page, len(page_texts))
        
        # Generate embeddings and add to ChromaDB
        texts = [chunk["text"] for chunk in chunks]
        ids = [f"{file.filename}_{chunk['chunk_id']}" for chunk in chunks]
        metadatas = [{
            "source": chunk["source"],
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"]
        } for chunk in chunks]
        
        embeddings = embedding_model.encode(texts)
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(chunks)} chunks to knowledge base")
        
        return UploadResponse(
            success=True,
            message=f"Document '{file.filename}' uploaded successfully",
            filename=file.filename,
            document_id=file.filename,
            chunks_created=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    try:
        doc_count = collection.count() if collection else 0
        kb_ready = collection is not None and embedding_model is not None
        
        return StatusResponse(
            status="healthy" if kb_ready else "degraded",
            knowledge_base_ready=kb_ready,
            llm_ready=True,  # Mock mode always ready
            documents_count=doc_count
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return StatusResponse(
            status="error",
            knowledge_base_ready=False,
            llm_ready=False,
            documents_count=0
        )

@app.get("/api/health")
async def health_check():
    """Health check"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ==================== Run Server ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Confidence Scoring Framework - Starting Backend")
    print("="*60)
    print(f"📍 URL: http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(f"📚 Docs: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    print(f"🔑 Admin: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        log_level="info"
    )