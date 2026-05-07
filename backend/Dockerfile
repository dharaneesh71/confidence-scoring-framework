FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --- THE AVX/SEGFAULT FIXES ---
# 1. Force ChromaDB's C++ library (hnswlib) to use standard CPU instructions
ENV HNSWLIB_NO_NATIVE=1

# 2. Tell the C-compiler to avoid ultra-modern CPU instructions during build
ENV CFLAGS="-mno-avx512f -mno-avx512vl"
ENV CXXFLAGS="-mno-avx512f -mno-avx512vl"

# 3. Force pip to download pre-compiled safe binaries for the usual suspects
RUN pip install --no-cache-dir --only-binary :all: scipy scikit-learn numpy || true
# ------------------------------

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY models/all-MiniLM-L6-v2 /app/models/all-MiniLM-L6-v2

RUN mkdir -p data/chroma_db data/uploads data/finetuned_model data/model_cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]