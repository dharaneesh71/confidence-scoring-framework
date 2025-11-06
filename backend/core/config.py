"""
Configuration settings for the Confidence Scoring Framework
"""
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application Info
    APP_NAME: str = "Confidence Scoring Framework"
    APP_VERSION: str = "1.0.0"
    
    # Backend Configuration
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    DEBUG_MODE: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    
    # Model Configuration
    LLAMA_MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    HUGGINGFACE_TOKEN: str = ""
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "confidence_documents"
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Retrieval Settings
    TOP_K_RETRIEVAL: int = 3
    
    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.5
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 10
    UPLOAD_DIRECTORY: str = "./data/uploads"
    
    # Admin Credentials
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure directories exist
Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)