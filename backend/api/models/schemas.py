"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for user question submission"""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the confidence scoring framework?"
            }
        }


class Citation(BaseModel):
    """Citation information for supporting evidence"""
    source: str = Field(..., description="Source document name")
    page: Optional[int] = Field(None, description="Page number in source document")
    excerpt: str = Field(..., description="Relevant text excerpt from source")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score with answer")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score (alias for similarity)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "Design_Document.pdf",
                "page": 2,
                "excerpt": "The confidence scoring framework consists of six core components...",
                "similarity_score": 0.92,
                "relevance_score": 0.92
            }
        }


class QueryResponse(BaseModel):
    """Response model for AI-generated answer with confidence score"""
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the properties of a square?",
                "answer": "A square has four equal sides and four right angles (90 degrees each).",
                "confidence_score": 0.87,
                "confidence_label": "High",
                "explanation": "Confidence Level: High (0.87). Based on: strong factual support (2/2 claims verified); strong semantic alignment; comprehensive coverage of key concepts; no hallucination detected.",
                "citations": [
                    {
                        "source": "geometry_basics.pdf",
                        "page": 5,
                        "excerpt": "A square is a quadrilateral with four equal sides and four right angles...",
                        "similarity_score": 0.92,
                        "relevance_score": 0.92
                    }
                ],
                "score_breakdown": {
                    "consistency": 0.89,
                    "semantic": 0.91,
                    "completeness": 0.85,
                    "precision": 0.84
                },
                "timestamp": "2024-03-15T10:30:00",
                "processing_time_ms": 1250.5
            }
        }


class UploadResponse(BaseModel):
    """Response model for file upload"""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    document_id: Optional[str] = Field(None, description="Document ID in knowledge base")
    chunks_created: Optional[int] = Field(None, description="Number of text chunks created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document uploaded and processed successfully",
                "filename": "textbook_chapter1.pdf",
                "document_id": "doc_12345",
                "chunks_created": 42
            }
        }


class StatusResponse(BaseModel):
    """Response model for system status"""
    status: str = Field(..., description="System status")
    knowledge_base_ready: bool = Field(..., description="Knowledge base availability")
    llm_ready: bool = Field(..., description="LLM model availability")
    documents_count: int = Field(..., description="Number of documents in knowledge base")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "knowledge_base_ready": True,
                "llm_ready": True,
                "documents_count": 5
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")