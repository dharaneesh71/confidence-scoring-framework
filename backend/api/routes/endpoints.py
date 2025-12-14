"""
API endpoint definitions
Modified for Retrieval-Augmented Verification (Blind Generation + Evidence Grading)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from api.models.schemas import (
    QueryRequest, QueryResponse, UploadResponse, 
    StatusResponse, ErrorResponse, Citation
)
from services.pdf_processor import PDFProcessor
from services.chroma_service import ChromaService
from services.llama_service import LlamaService
from services.scoring_service import ScoringService
from core.security import verify_admin
from core.config import settings
from pathlib import Path
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services (singleton pattern)
pdf_processor = PDFProcessor()
chroma_service = ChromaService()
llama_service = LlamaService()
scoring_service = ScoringService()

# Minimum similarity threshold (kept for logging, but we no longer exit early)
MIN_SIMILARITY_THRESHOLD = 0.3


@router.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """
    Submit a question for 'Blind Generation' and 'Evidence-Based Verification'
    
    Process:
    1. AI generates answer blindly (No context provided).
    2. System retrieves relevant documents (Answer Key).
    3. Scorer compares AI answer vs Documents.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received query: {request.question[:100]}...")
        
        # --- PHASE 1: GENERATION (Blind) ---
        # Generate answer using Llama model *without* context.
        # We explicitly pass context=None to force the AI to use its own training.
        logger.info("Generating 'Blind' answer (No RAG context provided to LLM)...")
        answer = llama_service.generate_answer(request.question, context=None)
        
        # --- PHASE 2: RETRIEVAL (For Verification Only) ---
        # Retrieve relevant passages from knowledge base to serve as the "Answer Key"
        logger.info("Retrieving ground truth documents for grading...")
        retrieved_passages = chroma_service.search(request.question, top_k=3)
        
        # Handle case where no documents exist in DB
        if not retrieved_passages:
            logger.warning("No relevant information found in knowledge base for verification.")
            # We return the AI's answer, but the score is forced to 0.0 because we can't verify it.
            return QueryResponse(
                question=request.question,
                answer=answer,
                confidence_score=0.0,
                confidence_label="Unverified - No Data",
                explanation="The AI generated an answer, but no documents were found in the database to verify it against.",
                citations=[],
                score_breakdown={"consistency": 0, "semantic": 0, "completeness": 0, "precision": 0},
                timestamp=datetime.now(),
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        # --- PHASE 3: SCORING (The Grading) ---
        # Compute confidence score by comparing the Blind Answer vs Retrieved Passages
        logger.info("Computing confidence score (Verification)...")
        confidence_score, explanation, citations, score_breakdown = scoring_service.compute_confidence_score(
            answer=answer,
            question=request.question,
            retrieved_passages=retrieved_passages
        )
        
        # Determine confidence label
        if confidence_score >= settings.HIGH_CONFIDENCE_THRESHOLD:
            confidence_label = "High - Verified"
        elif confidence_score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_label = "Medium - Partially Verified"
        else:
            confidence_label = "Low - Unverified"
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create response
        response = QueryResponse(
            question=request.question,
            answer=answer,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            explanation=explanation,
            citations=citations,
            score_breakdown=score_breakdown,
            timestamp=datetime.now(),
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"Query processed successfully. Score: {confidence_score:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    is_admin: bool = Depends(verify_admin)
):
    """
    Upload a PDF document to the knowledge base (Admin only)
    """
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Check file size (Read content to check size)
        try:
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)
            await file.seek(0)  # Reset file pointer
            
            if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum of {settings.MAX_UPLOAD_SIZE_MB}MB"
                )
        except Exception as e:
            logger.error(f"Error checking file size: {e}")
        
        # Save file temporarily
        upload_dir = Path(settings.UPLOAD_DIRECTORY)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved to: {file_path}")
        
        # Process PDF
        logger.info("Processing PDF...")
        chunks = pdf_processor.process_pdf(str(file_path), document_id=file.filename)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF"
            )
        
        # Add to ChromaDB
        logger.info(f"Adding {len(chunks)} chunks to knowledge base...")
        num_added = chroma_service.add_documents(chunks)
        
        response = UploadResponse(
            success=True,
            message=f"Document '{file.filename}' uploaded and processed successfully",
            filename=file.filename,
            document_id=file.filename,
            chunks_created=num_added
        )
        
        logger.info(f"Upload complete: {num_added} chunks added")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get system status and health check
    """
    try:
        kb_ready = chroma_service.is_ready()
        llm_ready = llama_service.is_ready()
        doc_count = chroma_service.get_count()
        
        status = "healthy" if (kb_ready and llm_ready) else "degraded"
        
        return StatusResponse(
            status=status,
            knowledge_base_ready=kb_ready,
            llm_ready=llm_ready,
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


@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}