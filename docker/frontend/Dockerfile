# Multi-stage build for frontend Gradio application
# Stage 1: Base image with Python runtime
FROM python:3.11-slim as base

# Set environment variables for Python optimization
# PYTHONUNBUFFERED ensures logs are output immediately without buffering
# PYTHONDONTWRITEBYTECODE prevents creation of .pyc files reducing container size
# PYTHONOPTIMIZE enables basic Python optimizations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory inside container
# All subsequent commands will run from this location
WORKDIR /app

# Install system dependencies required for building Python packages
# gcc and g++ are needed for compiling some Python packages
# curl and wget for health checks and downloading resources
# libpq-dev for PostgreSQL client libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    wget \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only requirements first to leverage Docker cache
# This prevents re-installing dependencies when code changes
COPY frontend/requirements.txt /app/requirements.txt

# Install Python dependencies with pip
# --no-cache-dir prevents storing pip cache reducing image size
# --default-timeout increases timeout for slow network connections
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the entire frontend application code
# This includes app.py, components, static files, etc.
COPY frontend/ /app/

# Create non-root user for security
# Running containers as root is a security risk
RUN groupadd -r medintel && useradd -r -g medintel medintel \
    && chown -R medintel:medintel /app

# Switch to non-root user for better security
USER medintel

# Expose port 7860 which is Gradio's default port
EXPOSE 7860

# Health check to ensure the service is running properly
# curl tests if the application responds to HTTP requests
# --fail returns error if HTTP status is 4xx or 5xx
# --silent suppresses output
# --output sends response to /dev/null to discard
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail --silent --output /dev/null http://localhost:7860 || exit 1

# Define the command to run the application
# Using exec form ensures signals are properly forwarded
# python -u enables unbuffered output for real-time logging
CMD ["python", "-u", "app.py"]