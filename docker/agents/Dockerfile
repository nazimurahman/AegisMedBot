# Specialized container for AI agents that may require GPU support
FROM python:3.11-slim as base

# Set environment for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=2 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies including CUDA toolkit for GPU support
# Note: For GPU support, use nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy agent-specific requirements
COPY agents/requirements.txt /app/requirements.txt
COPY ml_training/requirements.txt /app/ml_requirements.txt

# Install PyTorch with CUDA support if GPU is available
# CPU-only version is used for non-GPU deployments
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r ml_requirements.txt

# Copy agent code and supporting modules
COPY agents/ /app/agents/
COPY rag_system/ /app/rag_system/
COPY ml_training/ /app/ml_training/
COPY inference/ /app/inference/

# Copy model weights and configurations
# Models are large, so we download at runtime or mount as volume
RUN mkdir -p /app/models /app/cache

# Set environment for model caching
ENV TRANSFORMERS_CACHE=/app/cache/transformers \
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch

# Create non-root user
RUN groupadd -r medintel && useradd -r -g medintel medintel \
    && chown -R medintel:medintel /app

USER medintel

# Expose agent service port
EXPOSE 8001

# Health check for agent service
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail --silent --output /dev/null http://localhost:8001/health || exit 1

# Run agent service
# Note: In production, agents run as separate microservices
CMD ["python", "-m", "agents.orchestrator.agent_service"]