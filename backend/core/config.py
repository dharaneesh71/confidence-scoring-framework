"""
Configuration settings for the Confidence Scoring Framework
"""
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application Info
    APP_NAME: str    = "Confidence Scoring Framework"
    APP_VERSION: str = "1.0.0"

    # Backend
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    DEBUG_MODE: bool  = True
    LOG_LEVEL: str    = "INFO"

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]

    # Security — must be set in .env, no default
    SECRET_KEY: str = ""

    # HuggingFace — must be set in .env, no default
    HUGGINGFACE_TOKEN: str = ""

    # Model
    LLAMA_MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str   = "confidence_documents"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    TOP_K_RETRIEVAL: int = 3

    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD: float   = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.5

    # File Upload
    MAX_UPLOAD_SIZE_MB: int  = 10
    UPLOAD_DIRECTORY: str    = "./data/uploads"

    # Admin
    ADMIN_USERNAME: str  = "admin"
    ADMIN_PASSWORD: str  = ""     # Must be set in .env

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        case_sensitive    = True


settings = Settings()

Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)
