"""
API endpoint definitions
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

# Minimum similarity threshold to consider document relevant
MIN_SIMILARITY_THRESHOLD = 0.3


@router.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """
    Submit a question and receive AI-generated answer with confidence score
    
    This is the main endpoint for the Q&A functionality. It:
    1. Generates an answer using the Llama model
    2. Retrieves relevant passages from the knowledge base
    3. Computes a confidence score
    4. Returns answer, score, and citations
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received query: {request.question[:100]}...")
        
        # Step 1: Retrieve relevant passages from knowledge base
        logger.info("Searching knowledge base...")
        retrieved_passages = chroma_service.search(request.question, top_k=3)
        
        if not retrieved_passages:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in knowledge base. Please upload documents first."
            )
        
        # Check if retrieved passages are actually relevant
        max_similarity = max(p["similarity_score"] for p in retrieved_passages)
        
        logger.info(f"Top similarity score: {max_similarity:.2f}")
        
        # If no passage is sufficiently similar, the question is likely not in ground truth
        if max_similarity < MIN_SIMILARITY_THRESHOLD:
            logger.warning(f"Low similarity ({max_similarity:.2f}) - Question not in ground truth")
            
            # Generate answer anyway but mark as low confidence
            answer = llama_service.generate_answer(request.question, context=None)
            
            response = QueryResponse(
                question=request.question,
                answer=answer + "\n\n⚠️ Note: This question may not be covered in the uploaded documents.",
                confidence_score=0.0,
                confidence_label="Low - Not in Ground Truth",
                citations=[],
                timestamp=datetime.now(),
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
            
            return response
        
        # Step 2: Generate answer using Llama model with context
        logger.info("Generating answer...")
        context = "\n\n".join([p["text"] for p in retrieved_passages])
        answer = llama_service.generate_answer(request.question, context)
        
        # Step 3: Compute confidence score
        logger.info("Computing confidence score...")
        confidence_score, explanation, citations = scoring_service.compute_confidence_score(
            answer, retrieved_passages
        )
        
        # Determine confidence label
        if confidence_score >= settings.HIGH_CONFIDENCE_THRESHOLD:
            confidence_label = "High"
        elif confidence_score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"
        
        # Add warning if confidence is low but question matches documents
        if confidence_score < 0.4 and max_similarity > MIN_SIMILARITY_THRESHOLD:
            answer += "\n\n⚠️ Note: Low confidence - AI answer may not fully align with source documents."
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create response
        response = QueryResponse(
            question=request.question,
            answer=answer,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            citations=citations,
            timestamp=datetime.now(),
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"Query processed successfully in {processing_time:.2f}ms")
        logger.info(f"Confidence: {confidence_score:.2f} ({confidence_label})")
        
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
    
    This endpoint:
    1. Validates the uploaded file
    2. Extracts text from the PDF
    3. Chunks the text
    4. Stores embeddings in ChromaDB
    """
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Check file size
        file_size_mb = 0
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
    
    Returns information about:
    - Overall system status
    - Knowledge base availability
    - LLM availability
    - Number of documents in knowledge base
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