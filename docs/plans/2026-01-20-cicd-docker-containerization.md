# CI/CD Pipeline & Docker Containerization Plan

**Created:** 2026-01-20
**Status:** Planning
**Target:** RC-1 release infrastructure

## Executive Summary

Containerize Elspeth in a single Docker container with a complete CI/CD pipeline supporting:
- **CLI-first design** - container runs the `elspeth` CLI with arguments passed through
- **External configuration** - pipeline configs mounted from host filesystem
- Automated lint, test, build stages
- Dual registry support (GitHub Container Registry + Azure Container Registry)
- VM deployment initially, Kubernetes-ready for future migration
- Health checks, smoke tests, and rollback capability

---

## 1. Container Strategy

### Single "Batteries-Included" Image

All plugins (including LLM, Azure, future plugins) are bundled in one image. Plugin activation is controlled by **configuration**, not image variants.

```
elspeth:sha-abc123f     # Git commit SHA (immutable, for CI)
elspeth:0.1.0           # Semantic version (releases)
elspeth:latest          # Most recent (avoid in production)
```

**Rationale:**
- One image to build, test, and deploy
- No "which image variant?" confusion
- Avoids combinatorial explosion as plugin count grows
- Simpler CI/CD pipeline

### Multi-Stage Dockerfile

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Builder                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ python:3.11-slim                                        │ │
│ │ • Install uv                                            │ │
│ │ • Copy pyproject.toml + src/                            │ │
│ │ • uv pip install -e ".[all]"                            │ │
│ │ • Build wheel                                           │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Runtime                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ python:3.11-slim                                        │ │
│ │ • Copy .venv from builder                               │ │
│ │ • Copy installed elspeth package                        │ │
│ │ • Non-root user (security)                              │ │
│ │ • WORKDIR /app                                         │ │
│ │ • ENTRYPOINT ["elspeth"]                               │ │
│ │ • CMD ["--help"]  (default if no args)                 │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Benefits of multi-stage:**
- Smaller runtime image (no build tools, no uv)
- Reduced attack surface
- Faster pulls during deployment

### CLI-First Design

The container is designed as a **CLI tool in a box**, not a long-running service:

```bash
# ENTRYPOINT = elspeth (the CLI binary)
# CMD = arguments passed to the CLI

# Run with default (--help)
docker run elspeth

# Run a pipeline with mounted config
docker run -v /path/to/config:/app/config:ro \
           -v /path/to/data:/app/data \
           elspeth run --settings /app/config/pipeline.yaml

# Check version
docker run elspeth --version

# Explain a specific row
docker run -v /path/to/data:/app/data \
           elspeth explain --run latest --row 42
```

**Key principle:** Arguments after the image name are passed directly to the `elspeth` CLI.

### Volume Mounts & External Configuration

The container expects configuration and data to be mounted from the host:

```
┌─────────────────────────────────────────────────────────────────┐
│  HOST FILESYSTEM                    CONTAINER                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  /srv/elspeth/                                                  │
│  ├── config/          ──────────▶  /app/config/  (read-only)    │
│  │   ├── pipeline.yaml            Pipeline definitions           │
│  │   ├── settings.yaml            Runtime settings               │
│  │   └── profiles/                Environment-specific configs   │
│  │                                                               │
│  ├── input/           ──────────▶  /app/input/   (read-only)    │
│  │   ├── customers.csv            Source data files              │
│  │   └── orders.json              (CSV, JSON, etc.)              │
│  │                                                               │
│  ├── output/          ──────────▶  /app/output/  (read-write)   │
│  │   ├── results.csv              Sink output files              │
│  │   └── reports/                 Generated artifacts            │
│  │                                                               │
│  ├── state/           ──────────▶  /app/state/   (read-write)   │
│  │   └── landscape.db             Audit database (if SQLite)     │
│  │                                                               │
│  └── secrets/         ──────────▶  /app/secrets/ (read-only)    │
│      └── .env                     Environment secrets            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Standard mount points:**

| Host Path | Container Path | Mode | Purpose |
|-----------|----------------|------|---------|
| `./config` | `/app/config` | `ro` | Pipeline YAML, settings |
| `./input` | `/app/input` | `ro` | Source data files (CSV, JSON, etc.) |
| `./output` | `/app/output` | `rw` | Sink output files |
| `./state` | `/app/state` | `rw` | SQLite landscape DB, checkpoints |
| `./secrets` | `/app/secrets` | `ro` | Sensitive config (optional) |

**Example usage:**

```bash
# Full pipeline run with all mounts
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  -e DATABASE_URL=${DATABASE_URL} \
  elspeth:0.1.0 run --settings /app/config/pipeline.yaml

# Query audit trail (read-only access)
docker run --rm \
  -v $(pwd)/state:/app/state:ro \
  elspeth:0.1.0 explain --run latest --row 42

# Validate config without running (no data mounts needed)
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  elspeth:0.1.0 validate --settings /app/config/pipeline.yaml
```

**Path resolution in config files:**

Config files should use absolute container paths:

```yaml
# pipeline.yaml
source:
  type: csv
  path: /app/input/customers.csv    # Source reads from /app/input

transforms:
  - type: field_mapper
    # ...

sinks:
  - name: results
    type: csv
    path: /app/output/results.csv   # Sink writes to /app/output

landscape:
  database_url: sqlite:///app/state/landscape.db  # Or use DATABASE_URL env var
```

**Why separate input/output directories?**

| Reason | Benefit |
|--------|---------|
| Clear data flow | Obvious which files are source vs generated |
| Security | Input can be mounted read-only |
| Cleanup | Output can be cleared without affecting sources |
| Audit | Easy to verify what was input vs what was produced |

**Environment variables vs mounted secrets:**

| Method | Use When |
|--------|----------|
| `-e VAR=value` | Non-sensitive config, CI/CD pipelines |
| `--env-file` | Multiple env vars, local development |
| Mounted secrets file | Sensitive credentials, production |

---

## 2. Registry Strategy: Dual Registry Support

### Supported Registries

| Registry | Image Name | Use Case |
|----------|------------|----------|
| **GHCR** | `ghcr.io/<org>/elspeth` | GitHub-native, good for OSS, free |
| **ACR** | `<name>.azurecr.io/elspeth` | Azure-native, faster for Azure VMs, private |

### Why Both?

- **GHCR:** Native GitHub Actions authentication, free for public repos
- **ACR:** Faster pulls within Azure network, integrates with Azure AD/Managed Identity

Build workflow pushes to both registries by default. Can be configured to push to one or the other.

### ACR Authentication

**For GitHub Actions (CI/CD push):** Service Principal

```bash
# One-time setup: Create service principal with push access
az ad sp create-for-rbac \
  --name "github-actions-elspeth" \
  --role acrpush \
  --scopes /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.ContainerRegistry/registries/<acr-name>

# Outputs:
# - appId     → AZURE_CLIENT_ID (GitHub secret)
# - password  → AZURE_CLIENT_SECRET (GitHub secret)
# - tenant    → AZURE_TENANT_ID (GitHub secret)
```

**For Azure VMs (pull):** Managed Identity (recommended)

```bash
# Assign identity to VM, grant acrpull role
az vm identity assign --name myVM --resource-group myRG
az role assignment create \
  --assignee <vm-principal-id> \
  --role acrpull \
  --scope <acr-resource-id>
```

**For non-Azure VMs (pull):** Docker login with service principal

```bash
docker login myacr.azurecr.io -u $AZURE_CLIENT_ID -p $AZURE_CLIENT_SECRET
```

### Required Secrets

| Secret | Purpose | Required For |
|--------|---------|--------------|
| `GHCR_TOKEN` | Push to GitHub Container Registry | GHCR |
| `AZURE_CLIENT_ID` | Service principal app ID | ACR |
| `AZURE_CLIENT_SECRET` | Service principal password | ACR |
| `AZURE_TENANT_ID` | Azure AD tenant | ACR |
| `ACR_REGISTRY` | e.g., `myacr.azurecr.io` | ACR |

---

## 3. Pipeline Architecture

### Complete Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CI/CD PIPELINE                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│  │  LINT   │──▶│  TEST   │──▶│  BUILD  │──▶│ STAGING │──▶│  PROD   │        │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
│       │             │             │             │             │              │
│       ▼             ▼             ▼             ▼             ▼              │
│   • ruff        • pytest     • Docker      • Deploy      • Deploy           │
│   • mypy        • coverage   • Push to     • Health      • Health           │
│   • no-bug-     • hypothesis   GHCR+ACR     check        check             │
│     hiding                   • Tag SHA     • Smoke       • Smoke            │
│                                              tests        tests             │
│                                            • Verify      • Monitor          │
│                                                          • Rollback         │
│                                                            trigger          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Lint (Fast Feedback - ~30 seconds)

**Runs on:** Every push, every PR

| Check | Tool | Fail Condition |
|-------|------|----------------|
| Code style | `ruff check` | Any error |
| Formatting | `ruff format --check` | Any unformatted file |
| Type safety | `mypy --strict` | Any error |
| Bug-hiding | `no_bug_hiding.py` | Pattern detected |

**Rationale:** Fastest checks first. Fail early before expensive test runs.

### Stage 2: Test (Quality Gate - ~3-5 minutes)

**Runs on:** Every push, every PR

```
┌────────────────────────────────────────────────┐
│              TEST MATRIX                        │
├────────────────────────────────────────────────┤
│  Python 3.11 ─┬─ Unit tests (parallel)         │
│               ├─ Integration tests              │
│               └─ Property tests (hypothesis)    │
│                                                 │
│  Python 3.12 ─── Unit tests (compatibility)    │
└────────────────────────────────────────────────┘
```

| Test Type | Marker | Parallelization | Coverage |
|-----------|--------|-----------------|----------|
| Unit | default | `pytest -n auto` | Required ≥80% |
| Integration | `@pytest.mark.integration` | Sequential | Included |
| Slow | `@pytest.mark.slow` | Sequential | Included |
| Property | hypothesis | Parallel | Included |

**Coverage enforcement:** Fail if coverage < 80%

### Stage 3: Build (Artifact Creation - ~2 minutes)

**Runs on:** After tests pass on `main` branch or version tags

**Outputs:**
- Docker image pushed to GHCR: `ghcr.io/<org>/elspeth:sha-<commit>`
- Docker image pushed to ACR: `<acr>.azurecr.io/elspeth:sha-<commit>`
- Python wheel as GitHub artifact

**Registry selection (workflow input):**
- `ghcr` - GitHub Container Registry only
- `acr` - Azure Container Registry only
- `both` - Push to both (default)

### Stage 4: Deploy to Staging

**Runs on:** After successful build on `main`

1. Pull image by SHA
2. Run database migrations (Alembic)
3. Start new container
4. Health check (HTTP 200 from /health)
5. Run smoke tests
6. Keep old container reference for rollback (1 hour)

### Stage 5: Deploy to Production

**Runs on:** Manual approval after staging verification passes

Uses same deployment flow as staging with additional monitoring.

---

## 4. Deployment Strategy

### Phase 1: Linux VM (Current Target)

Since Elspeth is a **CLI tool** (not a daemon), deployment means updating the image that gets invoked by cron jobs, scripts, or manual runs:

```
┌─────────────────────────────────────────────────────────────────┐
│  LINUX VM DEPLOYMENT (CLI Tool)                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  /srv/elspeth/                                             │  │
│  │  ├── docker-compose.yaml   # Defines image + mounts        │  │
│  │  ├── config/               # Pipeline configurations        │  │
│  │  ├── data/                 # Input/output data              │  │
│  │  └── elspeth.sh            # Wrapper script                │  │
│  │                                                            │  │
│  │  Pipelines invoked by:                                     │  │
│  │  • Cron jobs (scheduled processing)                        │  │
│  │  • Manual runs (ad-hoc analysis)                           │  │
│  │  • External triggers (webhooks, file watchers)             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**VM Deployment Flow:**

```bash
#!/bin/bash
# scripts/deploy-vm.sh

set -e

IMAGE_TAG="${1:?Usage: deploy-vm.sh <image-tag>}"
DEPLOY_DIR="/srv/elspeth"

cd "$DEPLOY_DIR"

# 1. Record current image for rollback
PREVIOUS_TAG=$(grep 'IMAGE_TAG=' .env | cut -d= -f2 || echo "none")
echo "Previous image: $PREVIOUS_TAG"

# 2. Update image tag
sed -i "s/IMAGE_TAG=.*/IMAGE_TAG=$IMAGE_TAG/" .env

# 3. Pull new image
docker compose pull

# 4. Run migrations (if any)
docker compose run --rm elspeth alembic upgrade head

# 5. Verify new image works (smoke test)
echo "Running smoke tests..."
docker compose run --rm elspeth --version
docker compose run --rm elspeth validate --settings /app/config/pipeline.yaml
docker compose run --rm elspeth health

echo "=== Deployment successful: $IMAGE_TAG ==="

# 6. If smoke tests fail, the script exits (set -e) and manual rollback is needed:
# sed -i "s/IMAGE_TAG=.*/IMAGE_TAG=$PREVIOUS_TAG/" .env
# docker compose pull
```

**No downtime:** CLI tools aren't "restarted" — the new image is simply used for the next invocation.

### Phase 2: Kubernetes (Future)

```
┌─────────────────────────────────────────────────────────────────┐
│  KUBERNETES DEPLOYMENT                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Same Docker image (no changes needed)                     │  │
│  │  + Deployment manifest                                     │  │
│  │  + Service + Ingress                                       │  │
│  │  + ConfigMap/Secrets                                       │  │
│  │                                                            │  │
│  │  Native: rolling updates, health probes, auto-scaling      │  │
│  │  Zero-downtime deployments                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key principle:** Dockerfile and image are identical. Only orchestration layer changes.

---

## 5. Database Migrations (Alembic)

### Migration Flow in CI/CD

**CI (Test Stage):**
1. Create test database
2. Run: `alembic upgrade head`
3. Run tests against migrated schema
4. Run: `alembic downgrade -1` (test rollback works)
5. Verify rollback succeeded

**Deployment:**
1. Backup database
2. Run: `alembic upgrade head`
3. Start new container
4. Health check
5. If failed: `alembic downgrade` + rollback container

### Backward-Compatible Migrations

For breaking schema changes, use 3-deployment pattern:

1. **Deploy 1:** Add new column (nullable), code handles both old and new
2. **Deploy 2:** Code uses new column exclusively
3. **Deploy 3:** Drop old column (cleanup)

---

## 6. Verification & Health Checks

### CLI-Based Health Checks

Since Elspeth is a **CLI tool** (not a long-running service), health checks verify the container can execute successfully:

```bash
# Basic health check: can the CLI run?
docker run --rm elspeth --version

# Database connectivity check
docker run --rm \
  -v ./data:/app/data:ro \
  -e DATABASE_URL=${DATABASE_URL} \
  elspeth health

# Full health check with output
docker run --rm \
  -v ./config:/app/config:ro \
  -v ./data:/app/data:ro \
  -e DATABASE_URL=${DATABASE_URL} \
  elspeth health --json
```

**Health command output:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "commit": "abc123f",
  "checks": {
    "database": "connected",
    "config_readable": true,
    "data_dir_writable": true
  }
}
```

### Smoke Tests

Post-deployment verification (run as part of CI/CD):

| Test | Command | Fail Action |
|------|---------|-------------|
| CLI runs | `docker run --rm elspeth --version` | Rollback |
| Config validation | `docker run --rm -v ./config:/app/config:ro elspeth validate --settings /app/config/test.yaml` | Rollback |
| DB connectivity | `docker run --rm -e DATABASE_URL=... elspeth health` | Rollback |
| Sample pipeline | `docker run --rm -v ... elspeth run --settings /app/config/smoke-test.yaml` | Rollback |

### Smoke Test Pipeline

Create a minimal pipeline specifically for deployment verification:

```yaml
# config/smoke-test.yaml
# Minimal pipeline that exercises core functionality

source:
  type: inline  # Or a small CSV committed to the repo
  data:
    - { id: 1, value: "test" }

transforms: []  # No transforms needed for smoke test

sinks:
  - name: null_sink
    type: null  # Discards output, just verifies pipeline runs

landscape:
  database_url: ${DATABASE_URL}
```

### Deployment Verification Script

```bash
#!/bin/bash
# scripts/smoke-test.sh

set -e

IMAGE="${1:-elspeth:latest}"

echo "=== Smoke Test: ${IMAGE} ==="

echo "1. Version check..."
docker run --rm "$IMAGE" --version

echo "2. Config validation..."
docker run --rm \
  -v "$(pwd)/config:/app/config:ro" \
  "$IMAGE" validate --settings /app/config/smoke-test.yaml

echo "3. Database connectivity..."
docker run --rm \
  -e DATABASE_URL="${DATABASE_URL}" \
  "$IMAGE" health

echo "4. Sample pipeline run..."
docker run --rm \
  -v "$(pwd)/config:/app/config:ro" \
  -v "$(pwd)/data:/app/data" \
  -e DATABASE_URL="${DATABASE_URL}" \
  "$IMAGE" run --settings /app/config/smoke-test.yaml

echo "=== All smoke tests passed ==="
```

### Rollback Criteria

For CLI tools, rollback is triggered by deployment verification failures:

| Condition | Detection | Action |
|-----------|-----------|--------|
| CLI won't start | `--version` fails | Rollback immediately |
| Config parsing broken | `validate` fails | Rollback immediately |
| DB connection fails | `health` fails | Rollback immediately |
| Pipeline execution fails | smoke test fails | Rollback immediately |

**Note:** Unlike long-running services, there's no "error rate over time" or "response latency" — the container either works or it doesn't.

---

## 7. GitHub Actions Workflow Structure

### Workflow Files

```
.github/workflows/
├── ci.yaml                # Lint + Test (every push/PR)
├── build-push.yaml        # Docker build + push to registries
├── deploy-staging.yaml    # Auto-deploy to staging (future)
├── deploy-prod.yaml       # Manual deploy to production (future)
└── no-bug-hiding.yaml     # (existing) Custom static analysis
```

### Workflow Triggers

| Workflow | Trigger | Condition |
|----------|---------|-----------|
| `ci.yaml` | push, pull_request | Always |
| `build-push.yaml` | workflow_run (ci.yaml success) | `main` branch or tags |
| `deploy-staging.yaml` | workflow_run (build success) | Automatic |
| `deploy-prod.yaml` | workflow_dispatch | Manual approval |

---

## 8. Estimated Pipeline Timing

```
┌────────────────────────────────────────────────────────────────┐
│                    PIPELINE TIMELINE                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LINT         [====]                           ~30 sec          │
│  TEST              [================]          ~3-5 min         │
│  BUILD                              [====]     ~2 min           │
│  PUSH (both)                            [==]   ~1 min           │
│                                                                 │
│  TOTAL (to registry): ~6-8 minutes                              │
│                                                                 │
│  STAGING                                  [===] ~1-2 min        │
│  PROD (manual)                                 [===] ~2 min     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 9. Implementation Checklist

### Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `Dockerfile` | Multi-stage build with all plugins | P0 |
| `.dockerignore` | Exclude dev files, tests, docs, .git | P0 |
| `docker-compose.yaml` | VM deployment orchestration | P0 |
| `.github/workflows/ci.yaml` | Lint + Test pipeline | P0 |
| `.github/workflows/build-push.yaml` | Docker build + push (GHCR+ACR) | P0 |
| `scripts/deploy-vm.sh` | VM deployment script | P1 |
| `scripts/smoke-test.sh` | Post-deployment verification | P1 |
| `.github/workflows/deploy-staging.yaml` | Staging deployment | P2 |
| `.github/workflows/deploy-prod.yaml` | Production deployment | P2 |
| `k8s/` | Kubernetes manifests | P3 (future) |

### Code Changes Required

| Change | Purpose | Priority |
|--------|---------|----------|
| Add health check endpoint/command | Deployment verification | P1 |
| Environment variable config for DB URL | Container portability | P1 |
| Ensure `elspeth --version` includes commit SHA | Traceability | P2 |

---

## 10. Security Considerations

### Container Security

- Non-root user in container
- Read-only filesystem where possible
- No secrets baked into image
- Regular base image updates

### Secret Management

- Store secrets in GitHub Secrets / Azure Key Vault
- Never log secret values
- Different secrets per environment (staging vs prod)
- Rotate credentials quarterly

### Registry Security

- ACR: Private by default, use Managed Identity for pulls
- GHCR: Consider private for production images
- Enable vulnerability scanning on both registries

---

## 11. Open Items

- [ ] Determine ACR name and resource group
- [ ] Create service principal for GitHub Actions → ACR
- [ ] Decide on staging environment (separate VM? namespace?)
- [ ] Implement `/health` endpoint in Elspeth CLI
- [ ] Set up monitoring/alerting for production deployments

---

## Appendix A: docker-compose.yaml Template

```yaml
# docker-compose.yaml
#
# This compose file supports two usage patterns:
# 1. One-shot CLI runs: docker compose run --rm elspeth <command>
# 2. Scheduled/batch runs via cron or external scheduler

services:
  elspeth:
    image: ${REGISTRY:-ghcr.io/your-org}/elspeth:${IMAGE_TAG:-latest}
    # No 'restart' - this is a CLI tool, not a daemon
    environment:
      - DATABASE_URL=${DATABASE_URL:-sqlite:///app/state/landscape.db}
      - OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_ENDPOINT:-}
      - ELSPETH_LOG_LEVEL=${LOG_LEVEL:-INFO}
    working_dir: /app
    volumes:
      # Configuration (read-only)
      - ./config:/app/config:ro
      # Source data files (read-only)
      - ./input:/app/input:ro
      # Sink output files (read-write)
      - ./output:/app/output
      # State: SQLite DB, checkpoints (read-write)
      - ./state:/app/state
      # Optional: secrets file
      # - ./secrets:/app/secrets:ro
    # Default command shows help; override with 'docker compose run'
    command: ["--help"]

  # Optional: PostgreSQL for production landscape database
  postgres:
    image: postgres:16-alpine
    profiles: ["with-postgres"]
    environment:
      POSTGRES_DB: elspeth
      POSTGRES_USER: elspeth
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD required}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U elspeth"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

### CLI Usage Patterns with docker-compose

```bash
# Run a pipeline
docker compose run --rm elspeth run --settings /app/config/pipeline.yaml

# Validate configuration
docker compose run --rm elspeth validate --settings /app/config/pipeline.yaml

# Explain a row from a previous run
docker compose run --rm elspeth explain --run latest --row 42

# Check version
docker compose run --rm elspeth --version

# Run with PostgreSQL backend
docker compose --profile with-postgres up -d postgres
docker compose run --rm elspeth run --settings /app/config/pipeline.yaml

# Interactive shell for debugging
docker compose run --rm --entrypoint /bin/bash elspeth
```

### Wrapper Script (Optional)

For convenience, create a wrapper script `elspeth.sh`:

```bash
#!/bin/bash
# elspeth.sh - Run elspeth CLI via Docker
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec docker compose run --rm elspeth "$@"
```

Usage:
```bash
./elspeth.sh run --settings /app/config/pipeline.yaml
./elspeth.sh explain --run latest --row 42
```

---

## Appendix B: Build Workflow Registry Selection

```yaml
# In build-push.yaml
on:
  workflow_dispatch:
    inputs:
      registry:
        description: 'Which registry to push to'
        required: false
        default: 'both'
        type: choice
        options:
          - ghcr
          - acr
          - both

# Automatic triggers always push to both
```
