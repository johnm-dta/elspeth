# Dockerfile for ELSPETH - Auditable Sense/Decide/Act Pipelines
#
# Multi-stage build for minimal runtime image.
# All plugins (LLM, Azure) are bundled - activation is via configuration.
#
# No default command - explicit command required (web, run, etc.).
# Container orchestrators should configure appropriate health checks per deployment.
#
# Usage:
#   docker build -t elspeth .
#   docker run elspeth --help                                                # Show available commands
#   docker run elspeth --version                                             # Show version
#   docker run elspeth run --settings /app/config/pipeline.yaml              # Run batch pipeline
#   docker run -p 8451:8451 -e ELSPETH_WEB__SECRET_KEY=<key> elspeth web     # Start web server

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim AS builder

# Install uv for fast, deterministic dependency resolution
# Using official installer (https://docs.astral.sh/uv/getting-started/installation/)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up working directory
WORKDIR /build

# Copy only dependency specification first (layer caching)
COPY pyproject.toml uv.lock ./

# Create virtual environment and sync locked dependencies
# We install the "all" extra to bundle all plugins (LLM, Azure)
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --frozen --extra all --no-install-project --active

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Install project from lockfile (non-editable) with all optional dependencies
RUN . /opt/venv/bin/activate && \
    uv sync --frozen --extra all --no-editable --active

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim AS runtime

# Labels for container registry
LABEL org.opencontainers.image.title="ELSPETH"
LABEL org.opencontainers.image.description="Auditable Sense/Decide/Act Pipelines"
LABEL org.opencontainers.image.source="https://github.com/johnm-dta/elspeth"
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
# /app/state is for the default audit.db location (sqlite:///./state/audit.db)
RUN mkdir -p /app/config /app/input /app/output /app/state /app/secrets && \
    chown -R elspeth:elspeth /app

# Switch to non-root user
USER elspeth

# Expose web interface port (used when running `elspeth web`)
EXPOSE 8451

# No image-level HEALTHCHECK - container orchestrators should configure
# appropriate health checks per deployment type:
#
#   Web containers:     elspeth health --port 8451
#   Batch pipelines:    process exit code (no persistent health endpoint)
#
# Using `elspeth health` at image level would mark batch containers as
# unhealthy (no web server) even when the pipeline is working correctly.

# Entry point is the elspeth CLI
# Arguments after image name are passed directly to elspeth
ENTRYPOINT ["elspeth"]

# Default command shows help - explicit command required for all operations.
# The web server requires ELSPETH_WEB__SECRET_KEY for non-loopback hosts,
# so we don't default to `web` which would fail without configuration.
CMD ["--help"]
