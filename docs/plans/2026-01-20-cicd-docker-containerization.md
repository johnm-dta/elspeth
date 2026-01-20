# CI/CD Pipeline & Docker Containerization Plan

**Created:** 2026-01-20
**Status:** Planning
**Target:** RC-1 release infrastructure

## Executive Summary

Containerize Elspeth in a single Docker container with a complete CI/CD pipeline supporting:
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
│ │ • ENTRYPOINT ["elspeth"]                               │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits of multi-stage:**
- Smaller runtime image (no build tools, no uv)
- Reduced attack surface
- Faster pulls during deployment

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

```
┌─────────────────────────────────────────────────────────────────┐
│  LINUX VM DEPLOYMENT                                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Docker Compose                                            │  │
│  │  • elspeth container                                       │  │
│  │  • postgres container (or external DB)                     │  │
│  │  • traefik/nginx for TLS + health routing (optional)       │  │
│  │                                                            │  │
│  │  Deployment: docker compose pull && docker compose up -d   │  │
│  │  Rollback: docker compose up -d (with previous image tag)  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**VM Deployment Flow:**

```bash
# 1. Pull new image
docker compose pull

# 2. Run migrations (if any)
docker compose run --rm elspeth alembic upgrade head

# 3. Recreate container
docker compose up -d --force-recreate

# 4. Health check loop (30 sec timeout)
until curl -f http://localhost:8080/health; do sleep 2; done

# 5. If health fails: rollback
docker compose pull elspeth:previous-sha
docker compose up -d --force-recreate
```

**Downtime:** ~5-10 seconds during container restart. Acceptable for initial deployment.

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

### Health Check Endpoint

**Required:** `/health` endpoint (or CLI command) returning:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "commit": "abc123f",
  "database": "connected",
  "uptime_seconds": 3600
}
```

### Smoke Tests

Post-deployment verification:

| Test | Description | Fail Action |
|------|-------------|-------------|
| Health check | `GET /health` returns 200 | Rollback |
| CLI invocation | `elspeth --version` succeeds | Rollback |
| DB connectivity | Can query Landscape tables | Rollback |
| Sample pipeline | Run minimal Source→Sink | Rollback |

### Auto-Rollback Triggers

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Health check failure | 2 consecutive fails | Auto-rollback |
| Error rate | > 5% for 3 minutes | Auto-rollback |
| Response time | > 2x baseline p95 | Alert + manual review |

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
services:
  elspeth:
    image: ${REGISTRY:-ghcr.io/your-org}/elspeth:${IMAGE_TAG:-latest}
    restart: unless-stopped
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_ENDPOINT:-}
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "elspeth", "health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    volumes:
      - ./data:/app/data  # For file-based sources/sinks
      - ./config:/app/config:ro  # Pipeline configurations

  # Optional: local database for development
  postgres:
    image: postgres:16-alpine
    profiles: ["dev"]
    environment:
      POSTGRES_DB: elspeth
      POSTGRES_USER: elspeth
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-localdev}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
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
