"""
Pydantic models for API request/response validation
Updated for Auth, History, and Feedback
"""
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

# ==========================================
# AUTHENTICATION SCHEMAS (NEW)
# ==========================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    role: str
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

# ==========================================
# FEEDBACK SCHEMAS (NEW)
# ==========================================

class FeedbackRequest(BaseModel):
    history_id: int
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Reason if rating is low")

class FeedbackResponse(BaseModel):
    message: str

# ==========================================
# CHAT & QUERY SCHEMAS
# ==========================================

class QueryRequest(BaseModel):
    """Request model for user question submission"""
    question:   str          = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: Optional[int] = Field(None, description="Optional ID of existing chat session")
    domain:     Optional[str] = Field(None, description="Domain to filter RAG search (e.g. 'Medical', 'Legal')")  # ← ADD THIS

    class Config:
        json_schema_extra = {
            "example": {
                "question":  "What is the confidence scoring framework?",
                "domain":    "General"   # ← ADD THIS to example too
            }
        }


class SessionResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class Citation(BaseModel):
    """Citation information for supporting evidence"""
    source: str = Field(..., description="Source document name")
    page: Optional[int] = Field(None, description="Page number in source document")
    excerpt: str = Field(..., description="Relevant text excerpt from source")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score with answer")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score (alias for similarity)")

class QueryResponse(BaseModel):
    """Response model for AI-generated answer with confidence score"""
    # NEW: ID to link feedback later
    history_id: Optional[int] = Field(None, description="Database ID of this chat interaction")
    session_id: Optional[int] = Field(None, description="Current Chat Session ID")
    
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="AI-generated answer")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 - 1.0)")
    confidence_label: str = Field(..., description="Human-readable confidence label")
    explanation: Optional[str] = Field(None, description="Detailed explanation of the confidence score")
    citations: List[Citation] = Field(..., description="Supporting evidence citations")
    score_breakdown: Optional[Dict[str, float]] = Field(
        None, 
        description="Breakdown of scoring dimensions (consistency, semantic, completeness, precision)"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

# ==========================================
# SYSTEM & ADMIN SCHEMAS
# ==========================================

class UploadResponse(BaseModel):
    """Response model for file upload"""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    document_id: Optional[str] = Field(None, description="Document ID in knowledge base")
    chunks_created: Optional[int] = Field(None, description="Number of text chunks created")

class StatusResponse(BaseModel):
    """Response model for system status"""
    status: str = Field(..., description="System status")
    knowledge_base_ready: bool = Field(..., description="Knowledge base availability")
    llm_ready: bool = Field(..., description="LLM model availability")
    documents_count: int = Field(..., description="Number of documents in knowledge base")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")