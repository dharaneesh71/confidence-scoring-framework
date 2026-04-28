# ============================================================
# CONFID.AI — Backend Dockerfile
# Base: python:3.11-slim (Debian/glibc)
# Entry: uvicorn main:app  (runs from /app)
# Port:  8000
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl cmake libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ── llama-cpp-python (compiled from source for glibc/Debian) ─────────────────
# Uses OpenBLAS for faster CPU matrix math
# NOTE: This step takes 15-20 minutes — Docker caches it after first build
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install llama-cpp-python==0.2.90 --no-cache-dir

# ── PyTorch CPU-only ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Persistent-data directories (overridden by volume mounts at runtime) ──────
RUN mkdir -p data/chroma_db data/uploads data/finetuned_model data/gguf_models

# ── Runtime environment defaults ──────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]