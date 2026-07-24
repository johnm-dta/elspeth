# Dockerfile for ELSPETH - Auditable Sense/Decide/Act Pipelines
#
# Multi-stage build for minimal runtime image.
# Default builds bundle all plugins; INSTALL_EXTRAS selects a lean plugin set.
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

# One canonical build selection is threaded through every stage. The runtime
# label makes the selected extras inspectable on the final artifact; official
# generic GHCR/ACR builds set this explicitly to "all".
ARG INSTALL_EXTRAS="all"

# =============================================================================
# Stage 1: Frontend Builder
# =============================================================================
FROM node:24.13.0-bookworm-slim@sha256:4660b1ca8b28d6d1906fd644abe34b2ed81d15434d26d845ef0aced307cf4b6f AS frontend-builder

WORKDIR /frontend

# Install frontend dependencies from the lockfile first (layer caching)
COPY src/elspeth/web/frontend/package.json src/elspeth/web/frontend/package-lock.json ./
RUN npm ci

# Build the React SPA. The resulting dist/ is ignored by git and .dockerignore,
# so the release image must build it inside Docker.
COPY src/elspeth/web/frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Python Builder
# =============================================================================
FROM python:3.13-slim@sha256:b04b5d7233d2ad9c379e22ea8927cd1378cd15c60d4ef876c065b25ea8fb3bf3 AS builder

# Install uv for fast, deterministic dependency resolution
# Using official installer (https://docs.astral.sh/uv/getting-started/installation/)
COPY --from=ghcr.io/astral-sh/uv@sha256:e590846f4776907b254ac0f44b5b380347af5d90d668138ca7938d1b0c2f98d3 /uv /usr/local/bin/uv

# Set up working directory
WORKDIR /build

# Copy only dependency specification first (layer caching)
COPY pyproject.toml uv.lock ./
COPY elspeth-lints/ ./elspeth-lints/

# Create virtual environment and sync the selected locked dependencies.
# The default "all" preserves the shared GHCR/ACR image behavior.
ARG INSTALL_EXTRAS
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    test -n "$INSTALL_EXTRAS" && \
    set -f && \
    set -- && \
    for e in $INSTALL_EXTRAS; do \
        case "$e" in [a-z0-9]*) ;; *) exit 2 ;; esac; \
        case "$e" in *[!a-z0-9-]*) exit 2 ;; esac; \
        set -- "$@" --extra "$e"; \
    done && \
    test "$#" -gt 0 && \
    uv sync --frozen "$@" --no-install-project --active

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Install the project from the lockfile (non-editable) with the same selected extras.
RUN . /opt/venv/bin/activate && \
    test -n "$INSTALL_EXTRAS" && \
    set -f && \
    set -- && \
    for e in $INSTALL_EXTRAS; do \
        case "$e" in [a-z0-9]*) ;; *) exit 2 ;; esac; \
        case "$e" in *[!a-z0-9-]*) exit 2 ;; esac; \
        set -- "$@" --extra "$e"; \
    done && \
    test "$#" -gt 0 && \
    uv sync --frozen "$@" --no-editable --active

# Copy built SPA assets into the installed package, where app.py looks for
# elspeth/web/frontend/dist at runtime.
COPY --from=frontend-builder /frontend/dist /tmp/frontend-dist/
RUN . /opt/venv/bin/activate && \
    python -c 'from pathlib import Path; import shutil; import elspeth.web; target = Path(elspeth.web.__file__).parent / "frontend" / "dist"; shutil.rmtree(target, ignore_errors=True); target.parent.mkdir(parents=True, exist_ok=True); shutil.copytree("/tmp/frontend-dist", target)' && \
    rm -rf /tmp/frontend-dist

# =============================================================================
# Stage 3: Runtime
# =============================================================================
FROM python:3.13-slim@sha256:b04b5d7233d2ad9c379e22ea8927cd1378cd15c60d4ef876c065b25ea8fb3bf3 AS runtime

ARG INSTALL_EXTRAS

# Labels for container registry
LABEL org.opencontainers.image.title="ELSPETH"
LABEL org.opencontainers.image.description="Auditable Sense/Decide/Act Pipelines"
LABEL org.opencontainers.image.source="https://github.com/johnm-dta/elspeth"
LABEL org.opencontainers.image.licenses="MIT"
LABEL io.elspeth.install-extras="$INSTALL_EXTRAS"

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
RUN mkdir -p /app/config /app/input /app/ops /app/output /app/state /app/secrets && \
    chown -R elspeth:elspeth /app

# Switch to non-root user
USER elspeth

# Expose web interface port (used when running `elspeth web`)
EXPOSE 8451

# No image-level HEALTHCHECK - container orchestrators should configure
# appropriate health checks per deployment type:
#
#   Web task definitions: loopback GET /api/health
#   ALB target groups:     GET /api/ready
#   Batch tasks:           process exit code (no persistent health endpoint)
#
# An image-level probe would mark batch containers unhealthy even when their
# process-exit contract is working correctly.

# Entry point is the elspeth CLI
# Arguments after image name are passed directly to elspeth
ENTRYPOINT ["elspeth"]

# Default command shows help - explicit command required for all operations.
# The web server requires ELSPETH_WEB__SECRET_KEY for non-loopback hosts,
# so we don't default to `web` which would fail without configuration.
CMD ["--help"]
