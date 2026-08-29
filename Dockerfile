# GreenMind AI — containerized Streamlit app
FROM python:3.11-slim

WORKDIR /app

# System deps needed by chromadb's onnxruntime/sqlite backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Persisted ChromaDB store — mount a volume here in production so the
# knowledge base survives container restarts instead of rebuilding
# (re-fetching + re-embedding all 17 SDGs) on every cold start.
VOLUME ["/app/data"]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
