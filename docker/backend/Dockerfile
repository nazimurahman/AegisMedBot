# Multi-stage build for backend FastAPI service
# Stage 1: Builder stage for compiling dependencies
FROM python:3.11-slim as builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements for installation
COPY backend/requirements.txt /build/requirements.txt
COPY ml_training/requirements.txt /build/ml_requirements.txt

# Install Python packages to a temporary directory for later copying
# --target specifies installation directory
# --no-deps prevents installing dependencies separately (they're in requirements)
RUN pip install --target=/install --no-cache-dir -r requirements.txt \
    && pip install --target=/install --no-cache-dir -r ml_requirements.txt

# Stage 2: Final production image
FROM python:3.11-slim

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=2 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    # Python path includes installed packages
    PYTHONPATH="/app:/install"

WORKDIR /app

# Install runtime system dependencies only
# These are minimal compared to build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Copy application code
# Each copy is separated for better layer caching
COPY backend/ /app/backend/
COPY agents/ /app/agents/
COPY rag_system/ /app/rag_system/
COPY ml_training/ /app/ml_training/
COPY inference/ /app/inference/

# Copy configuration files
COPY .env.example /app/.env
COPY pyproject.toml /app/
COPY Makefile /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/models \
    && chmod 755 /app/logs /app/data /app/models

# Create non-root user for security
RUN groupadd -r medintel && useradd -r -g medintel medintel \
    && chown -R medintel:medintel /app

# Switch to non-root user
USER medintel

# Expose FastAPI default port
EXPOSE 8000

# Health check endpoint
# FastAPI provides /health endpoint for monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail --silent --output /dev/null http://localhost:8000/health || exit 1

# Run with uvicorn for production
# --host 0.0.0.0 binds to all interfaces
# --port 8000 matches exposed port
# --workers 4 enables multiple processes (adjust based on CPU cores)
# --log-level info provides appropriate logging detail
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]