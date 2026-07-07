# Docker Deployment Guide

This guide covers running ELSPETH in Docker containers for development and production deployments.

## Table of Contents

- [Quick Start](#quick-start)
- [Volume Mounts](#volume-mounts)
- [Environment Variables](#environment-variables)
- [Common Commands](#common-commands)
- [Using docker-compose](#using-docker-compose)
- [Health Checks](#health-checks)
- [Image Tags](#image-tags)
- [Container Registries](#container-registries)
- [Pipeline Configuration](#pipeline-configuration)
- [Building Locally](#building-locally)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

ELSPETH containers follow a **CLI-first design** - arguments are passed directly to the `elspeth` CLI:

```bash
IMAGE_TAG=v0.7.0

# Show help
docker run ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} --help

# Check version
docker run ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} --version

# List available plugins
docker run ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} plugins list
```

Replace `v0.7.0` with the exact release or immutable `sha-<commit>` tag that
matches the deployment you are operating.

---

## Volume Mounts

Mount your configuration and data directories to standard container paths:

| Host Path | Container Path | Mode | Purpose |
|-----------|----------------|------|---------|
| `./config` | `/app/config` | `ro` | Pipeline YAML, settings |
| `./input` | `/app/input` | `ro` | Source data files (CSV, JSON, etc.) |
| `./output` | `/app/output` | `rw` | Sink output files |
| `./data` | `/app/data` | `rw` | SQLite audit DB, checkpoints, payloads |
| `./secrets` | `/app/secrets` | `ro` | Sensitive config files (optional) |

**Example:**

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data:/app/data \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  run --settings /app/config/pipeline.yaml --execute
```

---

## Environment Variables

Pass secrets and configuration via environment variables. See the [Environment Variables Reference](../reference/environment-variables.md) for the complete list.

```bash
docker run --rm \
  -e DATABASE_URL="sqlite:////app/data/audit.db" \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  -e ELSPETH_FINGERPRINT_KEY="${ELSPETH_FINGERPRINT_KEY}" \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data:/app/data \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  run --settings /app/config/pipeline.yaml --execute
```

**Key variables for Docker:**

| Variable | Purpose |
|----------|---------|
| `ELSPETH_FINGERPRINT_KEY` | Secret fingerprinting (required if config contains API keys) |
| `OPENROUTER_API_KEY` | LLM provider API key |
| `DATABASE_URL` | Audit database (default: SQLite) |

---

## Common Commands

For complete CLI reference including all options and flags, see [User Manual - CLI Commands](user-manual.md#cli-commands).

### Run a Pipeline

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data:/app/data \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  run --settings /app/config/pipeline.yaml --execute
```

### Validate Configuration

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  validate --settings /app/config/pipeline.yaml
```

### Explain a Row

For interactive exploration, mount the state and use the TUI (requires `-it`):

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data:ro \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  explain --run latest --row 42 --database /app/data/audit.db
```

For non-interactive environments (CI/CD), use text or JSON explain output:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  explain --run latest --row 42 --no-tui --database /app/data/audit.db

docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  explain --run latest --row 42 --json --database /app/data/audit.db
```

### Resume an Interrupted Run

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data:/app/data \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  resume abc123 --execute
```

---

## Using docker-compose

For easier management, use docker-compose:

```yaml
# docker-compose.yaml
services:
  elspeth:
    image: ghcr.io/johnm-dta/elspeth:${IMAGE_TAG:?set IMAGE_TAG to sha-<commit> or v*}
    environment:
      - DATABASE_URL=${DATABASE_URL:-sqlite:////app/data/audit.db}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
      - ELSPETH_FINGERPRINT_KEY=${ELSPETH_FINGERPRINT_KEY:-}
    volumes:
      - ./config:/app/config:ro
      - ./input:/app/input:ro
      - ./output:/app/output
      - ./data:/app/data
    command: ["--help"]
```

### docker-compose Commands

```bash
# Run a pipeline
docker compose run --rm elspeth run --settings /app/config/pipeline.yaml --execute

# Validate config
docker compose run --rm elspeth validate --settings /app/config/pipeline.yaml

# Check health
docker compose run --rm elspeth health --verbose

# Explain a decision (interactive TUI)
docker compose run -it --rm elspeth explain --run latest --row 42 --database /app/data/audit.db
```

### Production docker-compose

```yaml
# docker-compose.prod.yaml
services:
  elspeth:
    image: ghcr.io/johnm-dta/elspeth:${IMAGE_TAG:?set IMAGE_TAG to sha-<commit> or v*}
    environment:
      - DATABASE_URL=postgresql://<user>:<password>@db:5432/elspeth  # secret-scan: allow-this-line
      - OPENROUTER_API_KEY
      - ELSPETH_FINGERPRINT_KEY
      - ELSPETH_SIGNING_KEY
    volumes:
      - ./config:/app/config:ro
      - ./input:/app/input:ro
      - ./output:/app/output
      - elspeth_data:/app/data
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=elspeth
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  elspeth_state:
  postgres_data:
```

---

## Health Checks

The `health` command verifies system readiness:

```bash
# Basic health check
docker run --rm ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} health

# Verbose output
docker run --rm ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} health --verbose

# JSON output (for automation)
docker run --rm ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} health --json
```

### Example JSON Output

```json
{
  "status": "healthy",
  "version": "0.7.0",
  "commit": "abc123f",
  "checks": {
    "version": {"status": "ok", "value": "0.7.0"},
    "python": {"status": "ok", "value": "3.13.1"},
    "database": {"status": "ok", "value": "connected"},
    "plugins": {"status": "ok", "value": "4 sources, 11 transforms, 4 sinks"}
  }
}
```

### Kubernetes Liveness Probe

```yaml
livenessProbe:
  exec:
    command: ["elspeth", "health", "--json"]
  initialDelaySeconds: 5
  periodSeconds: 30
```

---

## Image Tags

| Tag Pattern | Example | Use Case |
|-------------|---------|----------|
| `sha-<commit>` | `sha-abc123f` | CI/CD deployments (immutable, recommended) |
| `v<version>` | `v0.7.0` | Release versions |

Use `sha-<commit>` tags for immutable deployments. The build workflow does not
publish `latest`.

---

## Container Registries

Images are published to:

- **GitHub Container Registry**: `ghcr.io/johnm-dta/elspeth`
- **Azure Container Registry**: `<your-acr>.azurecr.io/elspeth` (if configured)

### Pulling from Private Registry

```bash
# GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker pull ghcr.io/johnm-dta/elspeth:${IMAGE_TAG}

# Azure Container Registry
az acr login --name your-acr
docker pull your-acr.azurecr.io/elspeth:${IMAGE_TAG}
```

---

## Pipeline Configuration

Pipeline configurations in containers should use **absolute container paths**:

```yaml
# config/pipeline.yaml
source:
  plugin: csv
  on_success: output              # Route rows directly to sink
  options:
    path: /app/input/data.csv     # Container path, not host path
    schema:
      mode: observed

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: /app/output/results.csv  # Container path

landscape:
  url: ${DATABASE_URL:-sqlite:////app/data/audit.db}

payload_store:
  base_path: /app/data/payloads
```

**Common mistake:** Using host paths like `./input/data.csv` instead of container paths `/app/input/data.csv`.

---

## Building Locally

```bash
# Build the image
docker build -t elspeth:local .

# Run locally built image
docker run --rm elspeth:local --version

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.12 -t elspeth:local-py312 .
```

### Multi-stage Build

The Dockerfile uses multi-stage builds for smaller images:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
# ... install build deps, compile wheels

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime
# ... copy only runtime requirements
```

---

## Troubleshooting

For general ELSPETH troubleshooting (API errors, configuration issues, etc.), see the [Troubleshooting Guide](troubleshooting.md). Below are Docker-specific issues.

### Common Docker Errors

- **"File not found"** - See [File Not Found Errors](troubleshooting.md#file-not-found-errors) (verify volume mounts and container paths)
- **"Permission denied"** - See [Permission Denied on Output](troubleshooting.md#permission-denied-on-output) (create output dir with `mkdir -p ./output && chmod 777 ./output`)

### Database connection refused

**Symptom:** `OperationalError: could not connect to server`

**Cause:** PostgreSQL not accessible from container.

**Fix:**
- In docker-compose: Use service name as host (`db` not `localhost`)
- Standalone: Use `--network host` or ensure container can reach database

### Secrets not fingerprinted

**Symptom:** `SecretFingerprintError: ELSPETH_FINGERPRINT_KEY not set`

**Cause:** Missing required environment variable.

**Fix:**
```bash
docker run --rm \
  -e ELSPETH_FINGERPRINT_KEY="your-key" \
  ...
```

### Health check fails in Kubernetes

**Symptom:** Pod keeps restarting due to failed liveness probe.

**Cause:** Health check requires database connection that's not ready.

**Fix:** Increase `initialDelaySeconds` or use readiness probe:
```yaml
readinessProbe:
  exec:
    command: ["elspeth", "health", "--json"]
  initialDelaySeconds: 10
  periodSeconds: 5
livenessProbe:
  exec:
    command: ["elspeth", "health", "--json"]
  initialDelaySeconds: 30
  periodSeconds: 30
```

---

## See Also

- [Your First Pipeline](your-first-pipeline.md) - Getting started guide
- [Configuration Reference](../reference/configuration.md) - Complete config options
- [Runbooks](../runbooks/) - Operational procedures
