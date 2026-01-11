# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/hf_cache

WORKDIR /app

# Install System Dependencies
# 'git' is required for your git_history_check tool
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (appuser)
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Create Cache Directory & Fix Permissions
# We make the folder and give 'appuser' ownership so it can download models
RUN mkdir -p /app/hf_cache /workspace && \
    chown -R appuser:appuser /app /workspace /home/appuser

# Install Dependencies
# Copy requirements first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
# We use '--chown' so the appuser owns the files (needed for write_to_file tool)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Setup Workspace
# We switch directory to /workspace so the Agent scans THIS folder by default
WORKDIR /workspace

# Run the Agent
# We use the absolute path to main.py because we are in /workspace
ENTRYPOINT ["python", "/app/main.py"]