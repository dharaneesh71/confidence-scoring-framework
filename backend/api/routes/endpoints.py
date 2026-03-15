"""
API endpoint definitions
Modified for Retrieval-Augmented Verification (Blind Generation + Evidence Grading)
Updated with Auth, History (Session Support), Role-Based Access Control, and Document Management
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func
from pathlib import Path
import time
import logging
from datetime import datetime
from typing import List # Added List import

# --- LOCAL IMPORTS ---
from api.models.schemas import (
    QueryRequest, QueryResponse, UploadResponse, 
    StatusResponse, UserCreate, UserResponse, Token,
    FeedbackRequest, FeedbackResponse, Citation, SessionResponse
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
# Note: Imported Session as ChatSession to avoid conflicts with SQLAlchemy's Session
from core.database import get_db, User, ChatHistory, Feedback, Document, Session as ChatSession
from core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
pdf_processor = PDFProcessor()
chroma_service = ChromaService()
llama_service = LlamaService()
scoring_service = ScoringService()


# ==========================================
# 1. AUTHENTICATION ROUTES
# ==========================================

@router.post("/auth/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pwd = get_password_hash(user.password)
    is_first_user = db.query(User).count() == 0
    role = "admin" if is_first_user else "user"
    
    new_user = User(email=user.email, hashed_password=hashed_pwd, role=role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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
# 2. CHAT & QUERY ROUTES
# ==========================================

@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    try:
        # --- 1. SESSION MANAGEMENT & AUTO-TITLING ---
        session_id = request.session_id
        
        if not session_id:
            # Create a new session with an auto-generated title (First 5 words of prompt)
            words = request.question.split()
            title = " ".join(words[:5]) + ("..." if len(words) > 5 else "")
            
            new_session = ChatSession(user_id=current_user.id, title=title)
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            session_id = new_session.id
        else:
            # Verify the session belongs to the user
            session_db = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == current_user.id
            ).first()
            if not session_db:
                raise HTTPException(status_code=404, detail="Session not found")

        # --- 2. CONTEXT RESTORATION ---
        # Fetch the last 5 messages from this session to give the LLM memory
        past_messages = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.timestamp.desc()).limit(5).all()
        
        context_string = ""
        if past_messages:
            # Reverse to chronological order so it reads naturally to the AI
            for msg in reversed(past_messages):
                context_string += f"User: {msg.question}\nAI: {msg.answer}\n\n"
        
        # --- 3. GENERATE ANSWER ---
        # Pass the context string to the Llama service
        answer = llama_service.generate_answer(request.question, context=context_string if context_string else None)
        retrieved_passages = chroma_service.search(request.question, top_k=3)
        
        # --- 4. SCORING & SAVING ---
        if not retrieved_passages:
            confidence_score = 0.0
            confidence_label = "Unverified - No Data"
            explanation = "No documents found to verify."
            citations = []
            score_breakdown = {"consistency": 0, "semantic": 0, "completeness": 0, "precision": 0}
        else:
            confidence_score, explanation, citations, score_breakdown = scoring_service.compute_confidence_score(
                answer=answer, question=request.question, retrieved_passages=retrieved_passages
            )
            if confidence_score >= settings.HIGH_CONFIDENCE_THRESHOLD:
                confidence_label = "High - Verified"
            elif confidence_score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_label = "Medium - Partially Verified"
            else:
                confidence_label = "Low - Unverified"
        
        history_entry = ChatHistory(
            user_id=current_user.id,
            session_id=session_id,  # Link to the session
            question=request.question,
            answer=answer,
            confidence_score=confidence_score
        )
        db.add(history_entry)
        db.commit()
        db.refresh(history_entry)
        
        return QueryResponse(
            history_id=history_entry.id,
            session_id=session_id,
            question=request.question,
            answer=answer,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            explanation=explanation,
            citations=citations,
            score_breakdown=score_breakdown,
            timestamp=datetime.now(),
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
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

@router.get("/history", response_model=List[SessionResponse])
def get_my_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Fetch list of sessions for the sidebar"""
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id
    ).order_by(ChatSession.created_at.desc()).all()
    return sessions

@router.get("/session/{session_id}")
def get_session_details(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Fetch all messages inside a specific session"""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    messages = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp.asc()).all()
    
    return {
        "session_id": session.id,
        "title": session.title,
        "messages": messages
    }

# ==========================================
# 3. ADMIN ROUTES 
# ==========================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_admin), 
    db: Session = Depends(get_db)
):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        upload_dir = Path(settings.UPLOAD_DIRECTORY)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        chunks = pdf_processor.process_pdf(str(file_path), document_id=file.filename)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        num_added = chroma_service.add_documents(chunks)
        
        existing_doc = db.query(Document).filter(Document.filename == file.filename).first()
        if existing_doc:
            existing_doc.upload_date = datetime.utcnow()
            existing_doc.chunk_count = num_added
        else:
            new_doc = Document(filename=file.filename, chunk_count=num_added)
            db.add(new_doc)
        
        db.commit()

        return UploadResponse(
            success=True,
            message=f"Document '{file.filename}' uploaded and processed successfully",
            filename=file.filename,
            document_id=file.filename,
            chunks_created=num_added
        )
        
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
    avg_rating = db.query(func.avg(Feedback.rating)).scalar() or 0
    rating_counts = db.query(Feedback.rating, func.count(Feedback.rating)).group_by(Feedback.rating).all()
    
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
    return db.query(Document).order_by(Document.upload_date.desc()).all()

@router.delete("/admin/documents/{doc_id}")
def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chroma_service.delete_document(doc.filename)
    db.delete(doc)
    db.commit()
    
    return {"message": f"Deleted {doc.filename}"}

@router.get("/admin/feedback")
def get_admin_feedback_logs(
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    feedbacks = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    
    formatted_feedback = []
    for fb in feedbacks:
        user_email = "Unknown User"
        if fb.chat_history and fb.chat_history.user:
            user_email = fb.chat_history.user.email
            
        formatted_feedback.append({
            "timestamp": fb.created_at,
            "user_email": user_email,
            "rating": fb.rating,
            "comment": fb.comment
        })
        
    return formatted_feedback

# ==========================================
# 4. SYSTEM ROUTES
# ==========================================

@router.get("/status", response_model=StatusResponse)
async def get_status():
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
        return StatusResponse(
            status="error",
            knowledge_base_ready=False,
            llm_ready=False,
            documents_count=0
        )

@router.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}