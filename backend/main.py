"""
FastAPI application entry point for Confidence Scoring Framework
"""
import logging
import os
from contextlib import asynccontextmanager

# 1. FIX: Prevent Hugging Face Tokenizer Multiprocessing Leak
# Must be set before importing any ML libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse
from joblib.externals.loky import get_reusable_executor

from api.routes import endpoints
from core.config import settings

# --- Logging Configuration ---
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- App Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Events
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {'Development' if settings.DEBUG_MODE else 'Production'}")
    logger.info(f"Backend running on {settings.BACKEND_HOST}:{settings.BACKEND_PORT}")
    
    yield  # Application runs while paused here
    
    # Shutdown Events
    logger.info("Shutting down application...")
    try:
        # 2. FIX: Gracefully shut down background workers (e.g., ChromaDB or local tokenizers)
        get_reusable_executor().shutdown(wait=True)
        logger.info("Successfully cleaned up multiprocessing semaphores.")
    except Exception as e:
        logger.warning(f"Could not cleanly shutdown multiprocessing: {e}")


# --- FastAPI Instance Initialization ---
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Confidence Scoring Framework for Decision Support",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# --- Middleware Pipeline ---

# Optimization: Compress large JSON responses (over 500 bytes). 
# Highly recommended for RAG systems returning long context windows or confidence matrices.
app.add_middleware(GZipMiddleware, minimum_size=500)

# Security: CORS Configuration 
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# --- Router Registration ---
app.include_router(
    endpoints.router,
    prefix="/api",
    tags=["main"]
)


# --- Base Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    """Redirects the base URL straight to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/api", tags=["main"])
async def api_root():
    """Health check and API directory mapping."""
    return {
        "message": "Confidence Scoring Framework API",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "query":  "/api/query",
            "upload": "/api/upload",
            "status": "/api/status",
            "health": "/api/health"
        }
    }


# --- Execution ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG_MODE,
        log_level=settings.LOG_LEVEL.lower()
    )