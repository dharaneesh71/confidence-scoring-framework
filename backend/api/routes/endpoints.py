import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends,
    File, Form, HTTPException, UploadFile, status
)
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.models.schemas import (
    Citation, FeedbackRequest, FeedbackResponse,
    QueryRequest, QueryResponse, SessionResponse,
    StatusResponse, Token, UploadResponse,
    UserCreate, UserResponse
)
from core.config import settings
from core.database import (
    ChatHistory, Document, Feedback,
    Session as ChatSession, User, get_db
)
from core.security import (
    create_access_token,
    get_current_active_admin,
    get_current_user,
    get_password_hash,
    verify_password
)
from services.chroma_service import ChromaService
from services.llama_service import LlamaService
from services.pdf_processor import PDFProcessor
from services.scoring_service import ScoringService

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Service singletons ─────────────────────────────────────────────────────────
pdf_processor   = PDFProcessor()
chroma_service  = ChromaService()
llama_service   = LlamaService()
scoring_service = ScoringService()

# ── Sprint 5: Global training state ───────────────────────────────────────────
training_status = {
    "status":       "idle",
    "progress":     0,
    "message":      "No training job has been run yet.",
    "started_at":   None,
    "completed_at": None,
}


# ==========================================
# 1. AUTH ROUTES
# ==========================================

@router.post("/auth/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user. First user registered is automatically admin."""
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pwd    = get_password_hash(user.password)
    is_first_user = db.query(User).count() == 0
    role          = "admin" if is_first_user else "user"

    new_user = User(email=user.email, hashed_password=hashed_pwd, role=role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/auth/login", response_model=Token)
def login(
    username: str     = Form(...),
    password: str     = Form(...),
    db:       Session = Depends(get_db)
):
    """Login and receive a JWT access token."""
    user = db.query(User).filter(User.email == username).first()
    if not user or not verify_password(password, user.hashed_password):
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
    request:      QueryRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db)
):
    """Submit a question and receive an AI answer with confidence scoring."""
    start_time = time.time()
    try:
        session_id = request.session_id

        # ── Create or validate session ─────────────────────────────────────
        if not session_id:
            words       = request.question.split()
            title       = " ".join(words[:5]) + ("..." if len(words) > 5 else "")
            new_session = ChatSession(user_id=current_user.id, title=title)
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            session_id = new_session.id
        else:
            session_db = db.query(ChatSession).filter(
                ChatSession.id      == session_id,
                ChatSession.user_id == current_user.id
            ).first()
            if not session_db:
                raise HTTPException(status_code=404, detail="Session not found")

        # ── Build conversation context ─────────────────────────────────────
        past_messages = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.timestamp.desc()).limit(5).all()

        context_string = ""
        if past_messages:
            for msg in reversed(past_messages):
                context_string += f"User: {msg.question}\nAI: {msg.answer}\n\n"

        # ── Domain-filtered RAG retrieval ──────────────────────────────────
        # NEW: if user selected a specific domain, filter ChromaDB to
        # only retrieve chunks that belong to that domain's documents.
        domain_filter = None
        if request.domain and request.domain.lower() != "all":
            # Get all filenames that belong to the selected domain
            domain_docs = db.query(Document.filename).filter(
                Document.domain == request.domain
            ).all()
            domain_filenames = [d[0] for d in domain_docs]

            if domain_filenames:
                # ChromaDB where filter: only match chunks from those files
                domain_filter = {"document_id": {"$in": domain_filenames}}
                logger.info(
                    f"[Domain Filter] Restricting RAG to domain='{request.domain}' "
                    f"({len(domain_filenames)} docs)"
                )
            else:
                logger.warning(
                    f"[Domain Filter] No documents found for domain='{request.domain}'. "
                    f"Falling back to full search."
                )

        # ── Generate answer + score ────────────────────────────────────────
        answer = llama_service.generate_answer(
            request.question,
            context=context_string or None
        )

        # Pass domain_filter into ChromaDB search
        retrieved_passages = chroma_service.search(
            request.question,
            top_k=3,
            where=domain_filter         # ← NEW: None means search all domains
        )

        if not retrieved_passages:
            confidence_score = 0.0
            confidence_label = "Unverified - No Data"
            explanation      = "No documents found to verify."
            citations        = []
            score_breakdown  = {
                "consistency":  0,
                "semantic":     0,
                "completeness": 0,
                "precision":    0
            }
        else:
            confidence_score, explanation, citations, score_breakdown = (
                scoring_service.compute_confidence_score(
                    answer=answer,
                    question=request.question,
                    retrieved_passages=retrieved_passages
                )
            )
            if confidence_score >= settings.HIGH_CONFIDENCE_THRESHOLD:
                confidence_label = "High - Verified"
            elif confidence_score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_label = "Medium - Partially Verified"
            else:
                confidence_label = "Low - Unverified"

        # ── Persist to DB ──────────────────────────────────────────────────
        history_entry = ChatHistory(
            user_id          = current_user.id,
            session_id       = session_id,
            question         = request.question,
            answer           = answer,
            confidence_score = confidence_score,
            explanation      = explanation,
            citations        = json.dumps(citations)
        )
        db.add(history_entry)
        db.commit()
        db.refresh(history_entry)

        return QueryResponse(
            history_id         = history_entry.id,
            session_id         = session_id,
            question           = request.question,
            answer             = answer,
            confidence_score   = confidence_score,
            confidence_label   = confidence_label,
            explanation        = explanation,
            citations          = citations,
            score_breakdown    = score_breakdown,
            timestamp          = datetime.now(),
            processing_time_ms = round((time.time() - start_time) * 1000, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    feedback:     FeedbackRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db)
):
    """Submit a star rating + optional comment for a chat response."""
    history_item = db.query(ChatHistory).filter(
        ChatHistory.id == feedback.history_id
    ).first()
    if not history_item:
        raise HTTPException(status_code=404, detail="Chat entry not found")
    if history_item.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to rate this chat")

    new_feedback = Feedback(
        chat_history_id = feedback.history_id,
        rating          = feedback.rating,
        comment         = feedback.comment
    )
    db.add(new_feedback)
    db.commit()
    return {"message": "Feedback received successfully"}


@router.get("/history", response_model=List[SessionResponse])
def get_my_sessions(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db)
):
    """Returns all chat sessions for the current user (reverse chronological)."""
    return db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id
    ).order_by(ChatSession.created_at.desc()).all()


# ── Serialization helper ───────────────────────────────────────────────────────
def _serialize_message(msg) -> dict:
    return {
        "history_id":       msg.id,
        "question":         msg.question,
        "answer":           msg.answer,
        "confidence_score": msg.confidence_score,
        "timestamp":        msg.timestamp,
        "explanation":      msg.explanation,
        "citations":        json.loads(msg.citations or "[]"),
    }


@router.get("/session/{session_id}")
def get_session_details(
    session_id:   int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db)
):
    """Returns full message history for a specific session."""
    session = db.query(ChatSession).filter(
        ChatSession.id      == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.timestamp.asc()).all()

    return {
        "session_id": session.id,
        "title":      session.title,
        "messages":   [_serialize_message(m) for m in messages],
    }


@router.get("/session/{session_id}/analytics")
def get_session_analytics(
    session_id:   int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db)
):
    """Returns confidence score trend data for a session."""
    session = db.query(ChatSession).filter(
        ChatSession.id      == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.timestamp.asc()).all()

    scores    = [m.confidence_score for m in messages if m.confidence_score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "session_id":         session_id,
        "average_confidence": round(avg_score, 2),
        "total_interactions": len(messages),
        "trend": [
            {"turn": i + 1, "score": round(s * 100, 1)}
            for i, s in enumerate(scores)
        ],
    }


# ==========================================
# 3. ADMIN ROUTES
# ==========================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file:         UploadFile = File(...),
    domain:       str        = Form(default="General"),   # ← NEW: admin assigns domain
    current_user: User       = Depends(get_current_active_admin),
    db:           Session    = Depends(get_db)
):
    """Upload a PDF to the knowledge base with a domain tag. Admin only."""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        upload_dir = Path(settings.UPLOAD_DIRECTORY)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename

        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = pdf_processor.process_pdf(str(file_path), document_id=file.filename)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF"
            )

        num_added    = chroma_service.add_documents(chunks)
        existing_doc = db.query(Document).filter(
            Document.filename == file.filename
        ).first()

        if existing_doc:
            existing_doc.upload_date = datetime.utcnow()
            existing_doc.chunk_count = num_added
            existing_doc.domain      = domain       # ← NEW: update domain if re-uploaded
        else:
            db.add(Document(
                filename    = file.filename,
                chunk_count = num_added,
                domain      = domain                # ← NEW: save domain on first upload
            ))
        db.commit()

        logger.info(
            f"[Upload] '{file.filename}' uploaded by {current_user.email} "
            f"→ domain='{domain}', chunks={num_added}"
        )

        return UploadResponse(
            success        = True,
            message        = f"Document '{file.filename}' uploaded to domain '{domain}' successfully",
            filename       = file.filename,
            document_id    = file.filename,
            chunks_created = num_added
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


# ── NEW: Domain endpoints ──────────────────────────────────────────────────────

@router.get("/domains")
def get_available_domains(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db)
):
    """
    NEW — Returns all unique domains that have at least one uploaded document.
    Used by the user chat UI to populate the domain selector chips.
    """
    domains = (
        db.query(Document.domain)
        .filter(Document.domain != None)
        .distinct()
        .order_by(Document.domain)
        .all()
    )
    domain_list = [d[0] for d in domains if d[0]]
    logger.info(f"[Domains] Available domains: {domain_list}")
    return domain_list   # e.g. ["Finance", "General", "Legal", "Medical"]


@router.patch("/admin/documents/{doc_id}/domain")
def update_document_domain(
    doc_id:       int,
    domain:       str     = Form(...),
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    """
    NEW — Lets admin update the domain of an already-uploaded document.
    Useful if a file was uploaded with the wrong domain tag.
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    old_domain  = doc.domain
    doc.domain  = domain
    db.commit()

    logger.info(
        f"[Domain Update] '{doc.filename}' domain changed: "
        f"'{old_domain}' → '{domain}' by {current_user.email}"
    )
    return {"message": f"Domain updated to '{domain}' for '{doc.filename}'"}


# ── End domain endpoints ───────────────────────────────────────────────────────

@router.get("/admin/analytics")
def get_analytics(
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    """Returns aggregate feedback stats for the admin dashboard."""
    avg_rating    = db.query(func.avg(Feedback.rating)).scalar() or 0
    rating_counts = db.query(
        Feedback.rating,
        func.count(Feedback.rating)
    ).group_by(Feedback.rating).all()

    distribution = [{"name": f"{i} Stars", "value": 0} for i in range(1, 6)]
    for r, count in rating_counts:
        if 1 <= r <= 5:
            distribution[r - 1]["value"] = count

    return {
        "average_rating": round(avg_rating, 1),
        "total_feedback": db.query(Feedback).count(),
        "distribution":   distribution,
    }


@router.get("/admin/documents")
def list_documents(
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    """Lists all uploaded documents with their domain tags. Admin only."""
    docs = db.query(Document).order_by(Document.upload_date.desc()).all()
    return [
        {
            "id":          doc.id,
            "filename":    doc.filename,
            "domain":      doc.domain or "General",   # ← NEW: include domain
            "chunk_count": doc.chunk_count,
            "upload_date": doc.upload_date,
        }
        for doc in docs
    ]


@router.delete("/admin/documents/{doc_id}")
def delete_document(
    doc_id:       int,
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    """Deletes a document from DB and ChromaDB. Admin only."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chroma_service.delete_document(doc.filename)
    db.delete(doc)
    db.commit()
    return {"message": f"Deleted {doc.filename}"}


@router.get("/admin/feedback")
def get_admin_feedback_logs(
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    """Returns all user feedback entries for admin review."""
    feedbacks          = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    formatted_feedback = []

    for fb in feedbacks:
        user_email = "Unknown User"
        if fb.chat_history and fb.chat_history.user:
            user_email = fb.chat_history.user.email

        formatted_feedback.append({
            "timestamp":  fb.created_at,
            "user_email": user_email,
            "rating":     fb.rating,
            "comment":    fb.comment,
        })
    return formatted_feedback


@router.get("/admin/low-confidence")
def get_low_confidence_sessions(
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    """Returns chat entries with low confidence scores for admin review."""
    low_conf_entries = db.query(ChatHistory).filter(
        ChatHistory.confidence_score < 0.5,
        ChatHistory.confidence_score > 0.05
    ).order_by(ChatHistory.timestamp.desc()).limit(100).all()

    results = []
    for entry in low_conf_entries:
        user_email = "Unknown User"
        if entry.user:
            user_email = entry.user.email

        results.append({
            "history_id":       entry.id,
            "user_email":       user_email,
            "question":         entry.question,
            "answer":           (entry.answer[:200] + "...") if len(entry.answer or "") > 200 else entry.answer,
            "confidence_score": entry.confidence_score,
            "timestamp":        entry.timestamp,
        })
    return results


# ==========================================
# 4. SPRINT 5 — RETRAINING ROUTES
# ==========================================

@router.post("/admin/trigger-retrain")
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    current_user:     User    = Depends(get_current_active_admin),
    db:               Session = Depends(get_db)
):
    """Starts the full retraining pipeline as a background job. Admin only."""
    global training_status

    if training_status["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="A training job is already running. Please wait for it to finish."
        )

    gold_rows = (
        db.query(ChatHistory)
        .join(Feedback, Feedback.chat_history_id == ChatHistory.id)
        .filter(ChatHistory.confidence_score >= settings.HIGH_CONFIDENCE_THRESHOLD)
        .filter(Feedback.rating >= 4)
        .limit(500)
        .all()
    )

    hard_rows = (
        db.query(ChatHistory)
        .join(Feedback, Feedback.chat_history_id == ChatHistory.id)
        .filter(Feedback.rating <= 2)
        .limit(500)
        .all()
    )

    if not gold_rows and not hard_rows:
        raise HTTPException(
            status_code=400,
            detail="Not enough labelled data to start training. Need more user feedback first."
        )

    def row_to_dict(r) -> dict:
        return {
            "question":         r.question,
            "answer":           r.answer,
            "confidence_score": r.confidence_score,
        }

    gold_data = [row_to_dict(r) for r in gold_rows]
    hard_data = [row_to_dict(r) for r in hard_rows]

    training_status.update({
        "status":       "running",
        "progress":     0,
        "message":      f"Starting — {len(gold_data)} gold samples, {len(hard_data)} hard negatives.",
        "started_at":   datetime.now().isoformat(),
        "completed_at": None,
    })

    background_tasks.add_task(_run_retraining_job, gold_data, hard_data)

    return {
        "message":      "Retraining job started successfully.",
        "triggered_by": current_user.email,
        "gold_count":   len(gold_data),
        "hard_count":   len(hard_data),
    }


def _run_retraining_job(gold_list: list, hard_list: list):
    """Background worker — updates global training_status at each step."""
    global training_status
    try:
        training_status.update({"progress": 10, "message": "Formatting training data..."})

        llama_service.retrain(
            gold_data=gold_list,
            hard_data=hard_list,
            status_callback=_update_training_progress
        )

        training_status.update({
            "status":       "completed",
            "progress":     100,
            "message":      "Training complete! New fine-tuned model is now active.",
            "completed_at": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Retraining job failed: {e}", exc_info=True)
        training_status.update({
            "status":  "failed",
            "message": f"Training failed: {str(e)}",
        })


def _update_training_progress(progress: int, message: str):
    """Callback passed into llama_service.retrain() to report progress."""
    global training_status
    training_status.update({"progress": progress, "message": message})


@router.get("/admin/training-status")
def get_training_status(
    current_user: User = Depends(get_current_active_admin)
):
    """Polled by frontend every 2s to update the progress bar. Admin only."""
    return training_status


# ==========================================
# 5. TASK 18 — MODEL DEPLOYMENT ROUTES
# ==========================================

class DeployModelRequest(BaseModel):
    new_model_path:  str
    backup_path: Optional[str] = None


@router.post("/admin/deploy-model")
def deploy_model(
    request:      DeployModelRequest,
    current_user: User = Depends(get_current_active_admin)
):
    """Hot-swap to a new model path. Auto-rolls back on failure. Admin only."""
    backup  = request.backup_path or settings.LLAMA_MODEL_NAME
    success = llama_service.hot_swap_model(request.new_model_path)

    if not success:
        rolled_back = llama_service.rollback_model(backup)
        return {
            "status":      "rolled_back",
            "message":     f"Swap to '{request.new_model_path}' failed. Reverted to '{backup}'.",
            "rollback_ok": rolled_back,
        }
    return {
        "status":  "success",
        "message": f"Model hot-swapped to '{request.new_model_path}' successfully.",
    }


# ==========================================
# 6. TASK 19 — MODEL VERSION HISTORY ROUTE
# ==========================================

@router.get("/admin/model-history")
def get_model_history(
    current_user: User = Depends(get_current_active_admin)
):
    """Returns model version history with validation scores. Admin only."""
    history_path = Path("data/model_history.json")
    if not history_path.exists():
        return {"history": []}
    try:
        return {"history": json.loads(history_path.read_text())}
    except Exception:
        return {"history": []}


# ==========================================
# 7. SYSTEM ROUTES
# ==========================================

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Public health check — returns model and knowledge base readiness."""
    try:
        kb_ready   = chroma_service.is_ready()
        llm_ready  = llama_service.is_ready()
        doc_count  = chroma_service.get_count()
        status_str = "healthy" if (kb_ready and llm_ready) else "degraded"

        return StatusResponse(
            status               = status_str,
            knowledge_base_ready = kb_ready,
            llm_ready            = llm_ready,
            documents_count      = doc_count
        )
    except Exception:
        return StatusResponse(
            status               = "error",
            knowledge_base_ready = False,
            llm_ready            = False,
            documents_count      = 0
        )


@router.get("/health")
async def health_check():
    """Simple ping endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}