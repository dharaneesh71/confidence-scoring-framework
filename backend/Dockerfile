# ============================================================
# CONFID.AI — Backend Dockerfile
# Base: python:3.11-slim
# Entry: uvicorn main:app  (runs from /app)
# Port:  8000
# ============================================================

FROM python:3.11-slim
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# gcc/g++ required by chromadb + sentence-transformers native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch CPU-only (separate layer so it caches independently) ──────────────
# CPU build keeps the image ~3 GB smaller than the CUDA build.
# The LlamaService._resolve_device() falls back to CPU automatically.
RUN pip install --no-cache-dir \
    torch==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Persistent-data directories (overridden by volume mounts at runtime) ──────
RUN mkdir -p data/chroma_db data/uploads data/finetuned_model

# ── Runtime environment defaults (override via .env or docker-compose) ────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
