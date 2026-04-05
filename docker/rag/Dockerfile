# Container for RAG (Retrieval-Augmented Generation) service
# This service handles vector search and document retrieval
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=2 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
# Additional packages for document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    libpq-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy RAG system requirements
COPY rag_system/requirements.txt /app/requirements.txt

# Install Python dependencies
# Includes Qdrant client, sentence-transformers, etc.
RUN pip install --no-cache-dir -r requirements.txt

# Copy RAG system code
COPY rag_system/ /app/rag_system/

# Copy document sources
# These contain medical literature, guidelines, and protocols
COPY documents/ /app/documents/

# Create directories for vector storage and indexes
RUN mkdir -p /app/vector_store /app/indexes /app/temp

# Set environment for vector database connection
# These will be overridden in docker-compose or Kubernetes
ENV QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION=medical_knowledge \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Create non-root user
RUN groupadd -r medintel && useradd -r -g medintel medintel \
    && chown -R medintel:medintel /app

USER medintel

# Expose RAG service port
EXPOSE 8002

# Health check ensures Qdrant connection is available
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health')" || exit 1

# Run RAG service
# This service provides document retrieval and embedding generation
CMD ["python", "-m", "rag_system.service"]