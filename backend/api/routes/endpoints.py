"""
API endpoint definitions
Modified for Retrieval-Augmented Verification (Blind Generation + Evidence Grading)
Updated with Auth, History, Role-Based Access Control, and Document Management
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func
from pathlib import Path
import time
import logging
from datetime import datetime

# --- LOCAL IMPORTS ---
from api.models.schemas import (
    QueryRequest, QueryResponse, UploadResponse, 
    StatusResponse, UserCreate, UserResponse, Token,
    FeedbackRequest, FeedbackResponse, Citation
)
from services.pdf_processor import PDFProcessor
from services.chroma_service import ChromaService
from services.llama_service import LlamaService
from services.scoring_service import ScoringService

# --- SECURITY & DATABASE IMPORTS ---
from core.security import (
    create_access_token, get_current_user, 
    get_current_active_admin, get_password_hash, verify_password
)
from core.database import get_db, User, ChatHistory, Feedback, Document # <--- Added Document
from core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services (singleton pattern)
pdf_processor = PDFProcessor()
chroma_service = ChromaService()
llama_service = LlamaService()
scoring_service = ScoringService()


# ==========================================
# 1. AUTHENTICATION ROUTES
# ==========================================

@router.post("/auth/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user account"""
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pwd = get_password_hash(user.password)
    
    # First user created is automatically an Admin, others are Users
    is_first_user = db.query(User).count() == 0
    role = "admin" if is_first_user else "user"
    
    new_user = User(email=user.email, hashed_password=hashed_pwd, role=role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login to get an access token"""
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(subject=user.email, role=user.role)
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}


# ==========================================
# 2. CHAT & QUERY ROUTES (Authenticated)
# ==========================================

@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest, 
    current_user: User = Depends(get_current_user), # <-- Requires Login
    db: Session = Depends(get_db)
):
    """
    Submit a question for 'Blind Generation' and 'Evidence-Based Verification'
    NOW SAVES TO HISTORY!
    """
    start_time = time.time()
    
    try:
        logger.info(f"User {current_user.email} query: {request.question[:100]}...")
        
        # --- PHASE 1: GENERATION (Blind) ---
        logger.info("Generating 'Blind' answer...")
        answer = llama_service.generate_answer(request.question, context=None)
        
        # --- PHASE 2: RETRIEVAL ---
        logger.info("Retrieving ground truth documents...")
        retrieved_passages = chroma_service.search(request.question, top_k=3)
        
        # Handle case where no documents exist
        if not retrieved_passages:
            logger.warning("No relevant information found.")
            
            # Save unverified attempt to DB
            history_entry = ChatHistory(
                user_id=current_user.id,
                question=request.question,
                answer=answer,
                confidence_score=0.0
            )
            db.add(history_entry)
            db.commit()
            
            return QueryResponse(
                history_id=history_entry.id,
                question=request.question,
                answer=answer,
                confidence_score=0.0,
                confidence_label="Unverified - No Data",
                explanation="The AI generated an answer, but no documents were found to verify it.",
                citations=[],
                score_breakdown={"consistency": 0, "semantic": 0, "completeness": 0, "precision": 0},
                timestamp=datetime.now(),
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        # --- PHASE 3: SCORING ---
        logger.info("Computing confidence score...")
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
        
        # --- SAVE TO DATABASE ---
        history_entry = ChatHistory(
            user_id=current_user.id,
            question=request.question,
            answer=answer,
            confidence_score=confidence_score
        )
        db.add(history_entry)
        db.commit()
        db.refresh(history_entry) # Get ID for frontend
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            history_id=history_entry.id,
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
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    feedback: FeedbackRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """User rates an answer (1-5 stars)"""
    # Verify the history item belongs to the user
    history_item = db.query(ChatHistory).filter(ChatHistory.id == feedback.history_id).first()
    
    if not history_item:
        raise HTTPException(status_code=404, detail="Chat entry not found")
    if history_item.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to rate this chat")

    new_feedback = Feedback(
        chat_history_id=feedback.history_id,
        rating=feedback.rating,
        comment=feedback.comment
    )
    db.add(new_feedback)
    db.commit()
    return {"message": "Feedback received successfully"}


@router.get("/history")
def get_my_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get past conversations for the sidebar"""
    history = db.query(ChatHistory).filter(ChatHistory.user_id == current_user.id).order_by(ChatHistory.timestamp.desc()).all()
    return history


# ==========================================
# 3. ADMIN ROUTES (Admin Only)
# ==========================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_admin), # <-- ADMIN ONLY CHECK
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document to the knowledge base (Admin only)
    """
    try:
        logger.info(f"Received file upload from admin {current_user.email}: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size
        try:
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)
            await file.seek(0)
            
            if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
                raise HTTPException(status_code=400, detail=f"File size exceeds maximum of {settings.MAX_UPLOAD_SIZE_MB}MB")
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
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Add to ChromaDB
        logger.info(f"Adding {len(chunks)} chunks to knowledge base...")
        num_added = chroma_service.add_documents(chunks)
        
        # --- NEW: Save to SQL for Admin Dashboard ---
        # Check if doc exists (overwrite metadata if so)
        existing_doc = db.query(Document).filter(Document.filename == file.filename).first()
        if existing_doc:
            existing_doc.upload_date = datetime.utcnow()
            existing_doc.chunk_count = num_added
        else:
            new_doc = Document(filename=file.filename, chunk_count=num_added)
            db.add(new_doc)
        
        db.commit()
        # ---------------------------------------------

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
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@router.get("/admin/analytics")
def get_analytics(
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    """Get feedback statistics for dashboard charts"""
    # 1. Average Rating
    avg_rating = db.query(func.avg(Feedback.rating)).scalar() or 0
    
    # 2. Rating Distribution
    rating_counts = db.query(Feedback.rating, func.count(Feedback.rating))\
                      .group_by(Feedback.rating).all()
    
    # Format for Frontend Charts
    distribution = [{"name": f"{i} Stars", "value": 0} for i in range(1, 6)]
    for r, count in rating_counts:
        if 1 <= r <= 5:
            distribution[r-1]["value"] = count

    return {
        "average_rating": round(avg_rating, 1),
        "total_feedback": db.query(Feedback).count(),
        "distribution": distribution
    }


@router.get("/admin/documents")
def list_documents(
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    """List all managed documents"""
    return db.query(Document).order_by(Document.upload_date.desc()).all()


@router.delete("/admin/documents/{doc_id}")
def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    """Delete a document from SQL and Vector DB"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 1. Delete from Chroma (Vectors)
    chroma_service.delete_document(doc.filename)
    
    # 2. Delete from SQL (Metadata)
    db.delete(doc)
    db.commit()
    
    return {"message": f"Deleted {doc.filename}"}


# ==========================================
# 4. SYSTEM ROUTES
# ==========================================

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status and health check"""
    try:
        kb_ready = chroma_service.is_ready()
        llm_ready = llama_service.is_ready()
        doc_count = chroma_service.get_count()
        
        status_code = "healthy" if (kb_ready and llm_ready) else "degraded"
        
        return StatusResponse(
            status=status_code,
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