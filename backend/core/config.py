"""
Configuration settings for the Confidence Scoring Framework
"""
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME:    str  = "Confidence Scoring Framework"
    APP_VERSION: str  = "1.0.0"

    # ── Backend ────────────────────────────────────────────────────────────────
    BACKEND_HOST: str  = "0.0.0.0"
    BACKEND_PORT: int  = 8000
    DEBUG_MODE:   bool = False
    LOG_LEVEL:    str  = "INFO"

    # ── CORS ───────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # ── Security ───────────────────────────────────────────────────────────────
    SECRET_KEY: str = ""

    # ── External APIs ──────────────────────────────────────────────────────────
    HUGGINGFACE_TOKEN: str = ""
    GROQ_API_KEY:      str = ""

    # ── Models ─────────────────────────────────────────────────────────────────
    LLAMA_MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # ── Embedding ──────────────────────────────────────────────────────────────
    # ✅ Using hash embedding — no model path needed
    EMBEDDING_MODEL:      str = "hash"
    EMBEDDING_MODEL_PATH: str = ""

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME:   str = "confidence_documents"

    # ── Retrieval ──────────────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 5

    # ── Confidence thresholds ──────────────────────────────────────────────────
    HIGH_CONFIDENCE_THRESHOLD:   float = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.5

    # ── File upload ────────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 10
    UPLOAD_DIRECTORY:   str = "./data/uploads"

    # ── Admin ──────────────────────────────────────────────────────────────────
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = ""

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        case_sensitive    = True


settings = Settings()

# Ensure data directories exist
Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)