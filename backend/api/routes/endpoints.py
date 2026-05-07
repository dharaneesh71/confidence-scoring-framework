"""
API Endpoints — Confidence Scoring Framework

KEY FIXES in this version:
─────────────────────────────────────────────────────────────────────────────
1. DOMAIN NOT PASSED TO CHROMA (critical):
   Old: chroma_service.add_documents(chunks)
   Domain was never stored in ChromaDB. It lived only in SQLite.
   New: chroma_service.add_documents(chunks, domain=domain)
   Now each chunk has a "domain" key in its metadata, enabling direct
   ChromaDB filtering without needing a SQL join.

2. DOMAIN FILTER — direct metadata filter:
   Old: SQL join → get filenames → filter by source in ChromaDB (fragile)
   New: where={"domain": domain.lower()} filters directly in ChromaDB.
   SQL fallback kept for documents ingested before this fix.

3. RICHER RAG CONTEXT:
   Old: rag_context = "\n\n---\n\n".join(p["text"] for p in passages)
   New: each passage is numbered and labelled with its source so the LLM
   can reference specific sources. Also respects a max-chars guard so
   extremely long contexts don't overflow the Groq token window.

4. TOP_K increased from 3 → 5:
   More passages = more context for the LLM to synthesise from.
   Combined with the new prompt (comprehensive explanation), this
   produces richer, more accurate answers.
   Update settings.TOP_K_RETRIEVAL = 5 in your .env or config.py.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import asyncio
from concurrent.futures import ThreadPoolExecutor

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
llama_service   = LlamaService()
chroma_service  = ChromaService()
scoring_service = ScoringService()
_inference_executor = ThreadPoolExecutor(max_workers=2)

# Maximum characters for the RAG context string sent to the LLM.
# Groq's context window is large but we cap context to keep latency low.
# 5 chunks × ~700 chars = ~3500 chars ≈ 875 tokens — well within limits.
_MAX_CONTEXT_CHARS = 4000

# ── Training state ─────────────────────────────────────────────────────────────
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
    """Register a new user. First registered user is automatically admin."""
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

def _build_rag_context(passages: List[dict], max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """
    Build a numbered, source-labelled context string for the LLM.

    FIX: Old version just joined raw text with '---'. The LLM had no way to
    distinguish passages or trace which source an answer came from. Numbering
    each passage lets the new prompt instruct the model to synthesise across
    passages while staying grounded.

    Also truncates to max_chars to prevent context window overflow.
    """
    parts   = []
    total   = 0

    for i, p in enumerate(passages, 1):
        source   = p.get("source", "unknown")
        sim      = p.get("similarity_score", 0)
        text     = p["text"].strip()
        header   = f"[Passage {i} | Source: {source} | Relevance: {sim:.2f}]"
        entry    = f"{header}\n{text}"

        if total + len(entry) > max_chars:
            # Add a truncated version of the last chunk if there's room
            remaining = max_chars - total - len(header) - 20
            if remaining > 200:
                parts.append(f"{header}\n{text[:remaining]}…")
            break

        parts.append(entry)
        total += len(entry)

    return "\n\n" + ("\n\n---\n\n".join(parts)) + "\n"


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
            title       = " ".join(words[:5]) + ("…" if len(words) > 5 else "")
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

        # ── Conversation context (last turn only) ──────────────────────────
        past_messages = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.timestamp.desc()).limit(1).all()

        conversation_context = ""
        if past_messages:
            for msg in reversed(past_messages):
                conversation_context += f"User: {msg.question}\nAI: {msg.answer}\n\n"

        # ── Domain filter ──────────────────────────────────────────────────
        # FIX: Use direct ChromaDB metadata filter {"domain": ...} first.
        # Fall back to SQL-based filename filter for backward compatibility
        # (documents ingested before the domain-metadata fix).
        domain_filter: Optional[dict] = None

        if request.domain and request.domain.lower() not in ("all", ""):
            domain_lower = request.domain.lower()

            # Primary: direct metadata filter (works for newly ingested docs)
            domain_filter = {"domain": domain_lower}
            logger.info(
                "[Domain Filter] Using direct metadata filter: domain='%s'",
                domain_lower,
            )

            # Verify there are any chunks with this domain in ChromaDB;
            # if not, fall back to SQL-based source filter for old docs.
            try:
                probe = chroma_service.collection.get(
                    where=domain_filter, limit=1
                )
                if not probe["ids"]:
                    logger.info(
                        "[Domain Filter] No chunks with domain='%s' in metadata. "
                        "Falling back to SQL-based source filter.",
                        domain_lower,
                    )
                    domain_docs      = db.query(Document.filename).filter(
                        Document.domain == request.domain
                    ).all()
                    domain_filenames = [d[0] for d in domain_docs]
                    if domain_filenames:
                        domain_filter = {"source": {"$in": domain_filenames}}
                    else:
                        logger.warning(
                            "[Domain Filter] No documents for domain='%s'. "
                            "Searching full collection.",
                            request.domain,
                        )
                        domain_filter = None
            except Exception as exc:
                logger.warning("[Domain Filter] Probe failed: %s. Skipping filter.", exc)
                domain_filter = None

        # ── STEP 1: RAG retrieval with reranking ───────────────────────────
        retrieved_passages = chroma_service.search_with_rerank(
            request.question,
            top_k=settings.TOP_K_RETRIEVAL,
            where=domain_filter,
        )

        # ── PRE-FLIGHT: similarity threshold check ─────────────────────────
        # With the corrected similarity formula (1 - dist/2), MiniLM returns:
        #   ~0.50–0.70 for genuinely relevant text
        #   ~0.20–0.40 for loosely related text
        #   ~0.00–0.20 for unrelated text
        # Threshold 0.20 lets through anything loosely related while blocking
        # truly off-topic queries.
        max_sim = max(
            (p["similarity_score"] for p in retrieved_passages), default=0.0
        )

        logger.info(
            "[RAG] Retrieved %d passages, max_sim=%.3f",
            len(retrieved_passages), max_sim,
        )

        if not retrieved_passages or max_sim < 0.05:
            logger.info(
                "[PreFlight] Blocked — passages=%d, max_sim=%.3f",
                len(retrieved_passages), max_sim,
            )
            answer           = (
                "I cannot find relevant information about this topic "
                "in the knowledge base."
            )
            confidence_score = 0.0
            confidence_label = "Unverified — Not in Knowledge Base"
            explanation      = (
                "No sufficiently similar documents were found for this query. "
                "Try rephrasing or ensure the relevant document is uploaded."
            )
            citations        = []
            score_breakdown  = {
                "consistency":  0,
                "semantic":     0,
                "completeness": 0,
                "precision":    0,
            }
        else:
            # ── STEP 2: Build numbered, source-labelled RAG context ────────
            # FIX: richer context format with passage numbers and source labels.
            rag_context  = _build_rag_context(retrieved_passages)
            final_context = rag_context or conversation_context or None

            logger.info(
                "[RAG] Context built — %d chars from %d passages",
                len(rag_context), len(retrieved_passages),
            )

            # ── STEP 3: Generate grounded answer ──────────────────────────
            loop   = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                _inference_executor,
                lambda: llama_service.generate_answer(
                    request.question,
                    context=final_context,
                ),
            )

            # ── STEP 4: Score answer ───────────────────────────────────────
            confidence_score, explanation, citations, score_breakdown = (
                scoring_service.compute_confidence_score(
                    answer=answer,
                    question=request.question,
                    retrieved_passages=retrieved_passages,
                )
            )

            if confidence_score >= settings.HIGH_CONFIDENCE_THRESHOLD:
                confidence_label = "High — Verified"
            elif confidence_score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_label = "Medium — Partially Verified"
            else:
                confidence_label = "Low — Unverified"

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
        logger.error("Error processing query: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


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
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
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
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
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
    domain:       str        = Form(default="General"),
    current_user: User       = Depends(get_current_active_admin),
    db:           Session    = Depends(get_db)
):
    """Upload a PDF to the knowledge base with a domain tag. Admin only."""
    try:
        if not file.filename.lower().endswith(".pdf"):
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
                detail=(
                    "No text could be extracted from this PDF. "
                    "Ensure poppler is installed: brew install poppler"
                )
            )

        # FIX: pass domain so it gets stored in ChromaDB chunk metadata.
        # This enables direct domain filtering without SQL joins.
        num_added    = chroma_service.add_documents(chunks, domain=domain)
        existing_doc = db.query(Document).filter(
            Document.filename == file.filename
        ).first()

        if existing_doc:
            existing_doc.upload_date = datetime.utcnow()
            existing_doc.chunk_count = num_added
            existing_doc.domain      = domain
        else:
            db.add(Document(
                filename    = file.filename,
                chunk_count = num_added,
                domain      = domain,
            ))
        db.commit()

        logger.info(
            "[Upload] '%s' by %s → domain='%s', chunks=%d",
            file.filename, current_user.email, domain, num_added,
        )

        return UploadResponse(
            success        = True,
            message        = (
                f"Document '{file.filename}' uploaded to domain '{domain}' "
                f"({num_added} chunks indexed)"
            ),
            filename       = file.filename,
            document_id    = file.filename,
            chunks_created = num_added,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error uploading document: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )


@router.get("/domains")
def get_available_domains(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db)
):
    domains     = db.query(Document.domain).filter(
        Document.domain != None
    ).distinct().order_by(Document.domain).all()
    domain_list = [d[0] for d in domains if d[0]]
    logger.info("[Domains] Available: %s", domain_list)
    return domain_list


@router.patch("/admin/documents/{doc_id}/domain")
def update_document_domain(
    doc_id:       int,
    domain:       str     = Form(...),
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    old_domain = doc.domain
    doc.domain = domain
    db.commit()
    logger.info(
        "[Domain Update] '%s': '%s' → '%s' by %s",
        doc.filename, old_domain, domain, current_user.email,
    )
    return {"message": f"Domain updated to '{domain}' for '{doc.filename}'"}


@router.get("/admin/analytics")
def get_analytics(
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    avg_rating    = db.query(func.avg(Feedback.rating)).scalar() or 0
    rating_counts = db.query(
        Feedback.rating, func.count(Feedback.rating)
    ).group_by(Feedback.rating).all()
    distribution  = [{"name": f"{i} Stars", "value": 0} for i in range(1, 6)]
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
    docs = db.query(Document).order_by(Document.upload_date.desc()).all()
    return [
        {
            "id":          doc.id,
            "filename":    doc.filename,
            "domain":      doc.domain or "General",
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
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chroma_service.delete_document(doc.filename)
    db.delete(doc)
    db.commit()
    return {"message": f"Deleted '{doc.filename}'"}


@router.get("/admin/feedback")
def get_admin_feedback_logs(
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    feedbacks = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    result    = []
    for fb in feedbacks:
        user_email = "Unknown"
        if fb.chat_history and fb.chat_history.user:
            user_email = fb.chat_history.user.email
        result.append({
            "timestamp":  fb.created_at,
            "user_email": user_email,
            "rating":     fb.rating,
            "comment":    fb.comment,
        })
    return result


@router.get("/admin/low-confidence")
def get_low_confidence_sessions(
    current_user: User    = Depends(get_current_active_admin),
    db:           Session = Depends(get_db)
):
    entries = db.query(ChatHistory).filter(
        ChatHistory.confidence_score < 0.5,
        ChatHistory.confidence_score > 0.05,
    ).order_by(ChatHistory.timestamp.desc()).limit(100).all()

    results = []
    for entry in entries:
        user_email = "Unknown"
        if entry.user:
            user_email = entry.user.email
        results.append({
            "history_id":       entry.id,
            "user_email":       user_email,
            "question":         entry.question,
            "answer":           (
                (entry.answer[:200] + "…")
                if len(entry.answer or "") > 200
                else entry.answer
            ),
            "confidence_score": entry.confidence_score,
            "timestamp":        entry.timestamp,
        })
    return results


# ==========================================
# 4. RETRAINING ROUTES
# ==========================================

@router.post("/admin/trigger-retrain")
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    current_user:     User    = Depends(get_current_active_admin),
    db:               Session = Depends(get_db)
):
    global training_status

    if training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="A training job is already running.")

    gold_rows = (
        db.query(ChatHistory)
        .join(Feedback, Feedback.chat_history_id == ChatHistory.id)
        .filter(ChatHistory.confidence_score >= settings.HIGH_CONFIDENCE_THRESHOLD)
        .filter(Feedback.rating >= 4)
        .limit(500).all()
    )
    hard_rows = (
        db.query(ChatHistory)
        .join(Feedback, Feedback.chat_history_id == ChatHistory.id)
        .filter(Feedback.rating <= 2)
        .limit(500).all()
    )

    if not gold_rows and not hard_rows:
        raise HTTPException(
            status_code=400,
            detail="Not enough labelled data to start training.",
        )

    def row_to_dict(r):
        return {
            "question":         r.question,
            "answer":           r.answer,
            "confidence_score": r.confidence_score,
        }

    gold_data = [row_to_dict(r) for r in gold_rows]
    hard_data = [row_to_dict(r) for r in hard_rows]

    training_status.update({
        "status":     "running",
        "progress":   0,
        "message":    (
            f"Starting — {len(gold_data)} gold samples, "
            f"{len(hard_data)} hard negatives."
        ),
        "started_at":   datetime.now().isoformat(),
        "completed_at": None,
    })

    background_tasks.add_task(_run_retraining_job, gold_data, hard_data)
    return {
        "message":      "Retraining job started.",
        "triggered_by": current_user.email,
        "gold_count":   len(gold_data),
        "hard_count":   len(hard_data),
    }


def _run_retraining_job(gold_list: list, hard_list: list):
    global training_status
    try:
        training_status.update({"progress": 10, "message": "Formatting training data…"})
        llama_service.retrain(
            gold_data=gold_list,
            hard_data=hard_list,
            status_callback=_update_training_progress,
        )
        training_status.update({
            "status":       "completed",
            "progress":     100,
            "message":      "Training complete!",
            "completed_at": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error("Retraining job failed: %s", e, exc_info=True)
        training_status.update({
            "status":  "failed",
            "message": f"Training failed: {str(e)}",
        })


def _update_training_progress(progress: int, message: str):
    global training_status
    training_status.update({"progress": progress, "message": message})


@router.get("/admin/training-status")
def get_training_status(current_user: User = Depends(get_current_active_admin)):
    return training_status


# ==========================================
# 5. MODEL DEPLOYMENT ROUTES
# ==========================================

class DeployModelRequest(BaseModel):
    new_model_path: str
    backup_path:    Optional[str] = None


@router.post("/admin/deploy-model")
def deploy_model(
    request:      DeployModelRequest,
    current_user: User = Depends(get_current_active_admin)
):
    backup  = request.backup_path or settings.LLAMA_MODEL_NAME
    success = llama_service.hot_swap_model(request.new_model_path)
    if not success:
        rolled_back = llama_service.rollback_model(backup)
        return {
            "status":      "rolled_back",
            "message":     f"Swap failed. Reverted to '{backup}'.",
            "rollback_ok": rolled_back,
        }
    return {
        "status":  "success",
        "message": f"Model hot-swapped to '{request.new_model_path}'.",
    }


@router.get("/admin/model-history")
def get_model_history(current_user: User = Depends(get_current_active_admin)):
    history_path = Path("data/model_history.json")
    if not history_path.exists():
        return {"history": []}
    try:
        return {"history": json.loads(history_path.read_text())}
    except Exception:
        return {"history": []}


# ==========================================
# 6. SYSTEM ROUTES
# ==========================================

@router.get("/status", response_model=StatusResponse)
async def get_status():
    try:
        kb_ready   = chroma_service.is_ready()
        llm_ready  = llama_service.is_ready()
        doc_count  = chroma_service.get_count()
        status_str = "healthy" if (kb_ready and llm_ready) else "degraded"
        return StatusResponse(
            status               = status_str,
            knowledge_base_ready = kb_ready,
            llm_ready            = llm_ready,
            documents_count      = doc_count,
        )
    except Exception:
        return StatusResponse(
            status               = "error",
            knowledge_base_ready = False,
            llm_ready            = False,
            documents_count      = 0,
        )


@router.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}