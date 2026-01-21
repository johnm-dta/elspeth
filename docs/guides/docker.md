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
# Show help
docker run ghcr.io/johnm-dta/elspeth --help

# Check version
docker run ghcr.io/johnm-dta/elspeth --version

# List available plugins
docker run ghcr.io/johnm-dta/elspeth plugins list
```

---

## Volume Mounts

Mount your configuration and data directories to standard container paths:

| Host Path | Container Path | Mode | Purpose |
|-----------|----------------|------|---------|
| `./config` | `/app/config` | `ro` | Pipeline YAML, settings |
| `./input` | `/app/input` | `ro` | Source data files (CSV, JSON, etc.) |
| `./output` | `/app/output` | `rw` | Sink output files |
| `./state` | `/app/state` | `rw` | SQLite landscape DB, checkpoints, payloads |
| `./secrets` | `/app/secrets` | `ro` | Sensitive config files (optional) |

**Example:**

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  run --settings /app/config/pipeline.yaml --execute
```

---

## Environment Variables

Pass secrets and configuration via environment variables:

```bash
docker run --rm \
  -e DATABASE_URL="sqlite:////app/state/landscape.db" \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  -e ELSPETH_FINGERPRINT_KEY="${ELSPETH_FINGERPRINT_KEY}" \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  run --settings /app/config/pipeline.yaml --execute
```

### Required Variables

| Variable | Purpose | Required When |
|----------|---------|---------------|
| `ELSPETH_FINGERPRINT_KEY` | Secret fingerprinting | Config contains API keys or passwords |
| `OPENROUTER_API_KEY` | LLM provider | Using LLM plugins |
| `DATABASE_URL` | Audit database | Using PostgreSQL (default: SQLite) |

### Optional Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ELSPETH_SIGNING_KEY` | Signed audit exports | None (unsigned) |
| `ELSPETH_ALLOW_RAW_SECRETS` | Dev mode (redact secrets) | false |

---

## Common Commands

### Run a Pipeline

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  run --settings /app/config/pipeline.yaml --execute
```

### Validate Configuration

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  validate --settings /app/config/pipeline.yaml
```

### Explain a Row

For interactive exploration, mount the state and use the TUI (requires `-it`):

```bash
docker run -it --rm \
  -v $(pwd)/state:/app/state:ro \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  explain --run latest --row 42
```

For non-interactive environments (CI/CD), query the audit database directly:

```bash
docker run --rm \
  -v $(pwd)/state:/app/state:ro \
  --entrypoint sqlite3 \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  /app/state/landscape.db \
  "SELECT ns.node_id, ns.status FROM node_states ns
   JOIN tokens t ON ns.token_id = t.token_id
   JOIN rows r ON t.row_id = r.row_id
   WHERE r.row_index = 42 ORDER BY ns.step_index;"
```

### Resume an Interrupted Run

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  resume abc123
```

---

## Using docker-compose

For easier management, use docker-compose:

```yaml
# docker-compose.yaml
services:
  elspeth:
    image: ghcr.io/johnm-dta/elspeth:${IMAGE_TAG:-latest}
    environment:
      - DATABASE_URL=${DATABASE_URL:-sqlite:////app/state/landscape.db}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
      - ELSPETH_FINGERPRINT_KEY=${ELSPETH_FINGERPRINT_KEY:-}
    volumes:
      - ./config:/app/config:ro
      - ./input:/app/input:ro
      - ./output:/app/output
      - ./state:/app/state
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
docker compose run -it --rm elspeth explain --run latest --row 42
```

### Production docker-compose

```yaml
# docker-compose.prod.yaml
services:
  elspeth:
    image: ghcr.io/johnm-dta/elspeth:v0.1.0
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/elspeth
      - OPENROUTER_API_KEY
      - ELSPETH_FINGERPRINT_KEY
      - ELSPETH_SIGNING_KEY
    volumes:
      - ./config:/app/config:ro
      - ./input:/app/input:ro
      - ./output:/app/output
      - elspeth_state:/app/state
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
docker run --rm ghcr.io/johnm-dta/elspeth health

# Verbose output
docker run --rm ghcr.io/johnm-dta/elspeth health --verbose

# JSON output (for automation)
docker run --rm ghcr.io/johnm-dta/elspeth health --json
```

### Example JSON Output

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "commit": "abc123f",
  "checks": {
    "version": {"status": "ok", "value": "0.1.0"},
    "python": {"status": "ok", "value": "3.11.9"},
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
| `v<version>` | `v0.1.0` | Release versions |
| `latest` | `latest` | Development only (avoid in production) |

**Production recommendation:** Use `sha-<commit>` tags for immutable deployments.

---

## Container Registries

Images are published to:

- **GitHub Container Registry**: `ghcr.io/johnm-dta/elspeth`
- **Azure Container Registry**: `<your-acr>.azurecr.io/elspeth` (if configured)

### Pulling from Private Registry

```bash
# GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker pull ghcr.io/johnm-dta/elspeth:v0.1.0

# Azure Container Registry
az acr login --name your-acr
docker pull your-acr.azurecr.io/elspeth:v0.1.0
```

---

## Pipeline Configuration

Pipeline configurations in containers should use **absolute container paths**:

```yaml
# config/pipeline.yaml
datasource:
  plugin: csv
  options:
    path: /app/input/data.csv  # Container path, not host path
    schema:
      fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: /app/output/results.csv  # Container path

landscape:
  url: ${DATABASE_URL:-sqlite:////app/state/landscape.db}

payload_store:
  base_path: /app/state/payloads
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

### "File not found" errors

**Symptom:** `FileNotFoundError: /app/input/data.csv`

**Cause:** Volume not mounted or wrong path in config.

**Fix:**
1. Verify volume mount: `docker run --rm -v $(pwd)/input:/app/input:ro elspeth:local ls /app/input`
2. Check pipeline config uses container paths (`/app/input/...`)

### Permission denied on output

**Symptom:** `PermissionError: [Errno 13] Permission denied: '/app/output/results.csv'`

**Cause:** Output directory doesn't exist or wrong permissions.

**Fix:**
```bash
# Create output directory with correct permissions
mkdir -p ./output
chmod 777 ./output  # Or use appropriate UID/GID
```

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
