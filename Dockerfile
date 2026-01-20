# Dockerfile for ELSPETH - Auditable Sense/Decide/Act Pipelines
#
# Multi-stage build for minimal runtime image.
# All plugins (LLM, Azure) are bundled - activation is via configuration.
#
# Usage:
#   docker build -t elspeth .
#   docker run -v ./config:/app/config:ro elspeth --version
#   docker run -v ./config:/app/config:ro elspeth run --settings /app/config/pipeline.yaml

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

# Install uv for fast, deterministic dependency resolution
# Using official installer (https://docs.astral.sh/uv/getting-started/installation/)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up working directory
WORKDIR /build

# Copy only dependency specification first (layer caching)
COPY pyproject.toml ./

# Create virtual environment and install all dependencies
# We install [all] to bundle all plugins (LLM, Azure)
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install hatchling

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Build wheel and install with all optional dependencies (LLM, Azure)
RUN . /opt/venv/bin/activate && \
    uv pip install ".[all]"

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

# Labels for container registry
LABEL org.opencontainers.image.title="ELSPETH"
LABEL org.opencontainers.image.description="Auditable Sense/Decide/Act Pipelines"
LABEL org.opencontainers.image.source="https://github.com/your-org/elspeth"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd --gid 1000 elspeth && \
    useradd --uid 1000 --gid elspeth --shell /bin/bash --create-home elspeth

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up PATH to use venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Create standard mount point directories
# These will typically be mounted from host
RUN mkdir -p /app/config /app/input /app/output /app/state /app/secrets && \
    chown -R elspeth:elspeth /app

# Switch to non-root user
USER elspeth

# Health check - verify CLI is functional
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["elspeth", "--version"]

# Entry point is the elspeth CLI
# Arguments after image name are passed directly to elspeth
ENTRYPOINT ["elspeth"]

# Default command shows help if no arguments provided
CMD ["--help"]
