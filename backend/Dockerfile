# ============================================================
# CONFID.AI — Backend Dockerfile (Groq API — no local model)
# Base: python:3.11-slim
# Entry: uvicorn main:app
# Port:  8000

FROM python:3.11-slim
WORKDIR /app
# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
# ── PyTorch CPU-only (for embeddings only — no LLM loading) ──────────────────
RUN pip install --no-cache-dir \
    torch==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu
# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ── Application code ──────────────────────────────────────────────────────────
COPY . .
# ── Persistent-data directories ───────────────────────────────────────────────
RUN mkdir -p data/chroma_db data/uploads data/finetuned_model

# ── Runtime environment defaults ──────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]