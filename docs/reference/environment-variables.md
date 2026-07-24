# Environment Variables Reference

Reference for ELSPETH environment variables and `.env` configuration.

---

## Table of Contents

- [Automatic .env Loading](#automatic-env-loading)
- [Required Variables](#required-variables)
- [Optional Variables](#optional-variables)
- [LLM Provider Variables](#llm-provider-variables)
- [Azure Service Variables](#azure-service-variables)
- [Telemetry Variables](#telemetry-variables)
- [Secret Field Detection](#secret-field-detection)
- [Example .env File](#example-env-file)
- [Security Best Practices](#security-best-practices)
- [Skipping .env Loading](#skipping-env-loading)
- [Docker and CI/CD](#docker-and-cicd)

---

## Automatic .env Loading

ELSPETH automatically loads environment variables from a `.env` file when you run any command. This eliminates the need to manually `source .env` before running pipelines.

**How it works:**

1. When any `elspeth` command runs, it looks for `.env` in the current directory
2. If not found, it searches parent directories
3. Variables from `.env` are loaded into the environment
4. Existing environment variables are **not** overwritten

---

## Required Variables

| Variable | Purpose | When Required |
|----------|---------|---------------|
| `ELSPETH_FINGERPRINT_KEY` | Secret fingerprinting | Config contains API keys or passwords |
| `ELSPETH_SIGNING_KEY` | Signed audit exports | `landscape.export.sign: true` |

### ELSPETH_FINGERPRINT_KEY

Used to HMAC-hash API keys and passwords before storing them in the audit trail. This ensures secrets are never stored in plain text while still allowing verification of which credentials were used.

Without this key, ELSPETH will refuse to run if your config contains API keys. This prevents accidental secret leakage to audit databases.

```bash
# Generate a secure key
python -c "import secrets; print(secrets.token_hex(32))"

# Set in environment
export ELSPETH_FINGERPRINT_KEY="your-generated-key"
```

### ELSPETH_SIGNING_KEY

Used to HMAC-sign exported audit records for integrity verification. Only required if you enable signed exports in your landscape configuration.

---

## Optional Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ELSPETH_ALLOW_RAW_SECRETS` | Skip fingerprinting (development only) | `false` |
| `ELSPETH_KEYVAULT_ALLOWED_VAULT_URLS` | Pin `secrets.vault_url` to exact vaults | unset (approved-suffix check only) |
| `DATABASE_URL` | CLI/MCP audit database connection | `sqlite:///./data/audit.db` |
| `ELSPETH_WEB__REGISTRATION_MODE` | Local-auth registration mode: `open`, `email_verified`, or `closed` | `open` |
| `ELSPETH_WEB__PUBLIC_BASE_URL` | Public origin used to generate email-verification links on non-local hosts | unset |

### ELSPETH_ALLOW_RAW_SECRETS

**Development only.** When set to `true`, allows running pipelines without `ELSPETH_FINGERPRINT_KEY` even when configs contain secrets. Secrets will be stored in plain text in the audit trail.

**Never use in production.** This is intended only for local development and testing.

### ELSPETH_KEYVAULT_ALLOWED_VAULT_URLS

**Deployment-owned Key Vault allowlist.** A comma- or whitespace-separated list of exact Key Vault URLs. When set, a `secrets.vault_url` that is not an exact match is refused before any Key Vault call — closing the residual risk that a settings file points ELSPETH's Azure credentials at a real but foreign vault (e.g. one in another tenant).

Set this at the deployment/host level, **never** in pipeline YAML. When unset, `vault_url` is still restricted to approved Azure Key Vault host suffixes (`.vault.azure.net` and its sovereign-cloud variants). See the [Key Vault runbook](../runbooks/configure-keyvault-secrets.md) for details.

```bash
export ELSPETH_KEYVAULT_ALLOWED_VAULT_URLS="https://elspeth-prod-vault.vault.azure.net"
```

### ELSPETH_WEB__REGISTRATION_MODE

Controls local Composer registration. `open` creates active local users
immediately, `email_verified` creates pending users and writes one-use
verification links to `data/email-verifications.jsonl`, and `closed` disables
self-registration.

Non-local `email_verified` deployments must also set
`ELSPETH_WEB__PUBLIC_BASE_URL` to a public origin, for example
`https://elspeth.example.gov.au`. The value must be an origin only: no path,
query, or fragment.

---

## LLM Provider Variables

### OpenRouter

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | API key for OpenRouter LLM service |

Used by the `llm` transform (provider: openrouter).

### Azure OpenAI

| Variable | Purpose |
|----------|---------|
| `AZURE_OPENAI_API_KEY` | API key for Azure OpenAI service |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint URL |

Used by `llm` (provider: azure) transforms.

**Endpoint format:** `https://your-resource.openai.azure.com`

---

## Azure Service Variables

### Azure Content Safety

| Variable | Purpose |
|----------|---------|
| `AZURE_CONTENT_SAFETY_KEY` | API key for Azure Content Safety |
| `AZURE_CONTENT_SAFETY_ENDPOINT` | Content Safety resource endpoint URL |

Used by the `azure_content_safety` transform plugin.

### Azure Prompt Shield

| Variable | Purpose |
|----------|---------|
| `AZURE_PROMPT_SHIELD_KEY` | API key for Azure Prompt Shield |
| `AZURE_PROMPT_SHIELD_ENDPOINT` | Prompt Shield resource endpoint URL |

Used by the `azure_prompt_shield` transform plugin.

### Azure Document Intelligence

| Variable | Purpose |
|----------|---------|
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | API key for Azure AI Document Intelligence |

Used by the `azure_document_intelligence` transform plugin when the plugin
option references `${AZURE_DOCUMENT_INTELLIGENCE_KEY}`. Configure the Document
Intelligence endpoint in the transform's `options.endpoint`; there is no
implicit endpoint environment variable.

### Azure Blob Storage

| Variable | Purpose |
|----------|---------|
| `AZURE_STORAGE_CONNECTION_STRING` | Connection string for Azure Blob Storage |

Used by `azure_blob` source and sink plugins.

---

## Telemetry Variables

### OpenTelemetry (OTLP)

| Variable | Purpose |
|----------|---------|
| `OTEL_ENDPOINT` | OTLP endpoint URL (non-sensitive; can be set in YAML instead) |
| `OTEL_TOKEN` | Auth token for OTLP exporter (sensitive) |

### Azure Application Insights

| Variable | Purpose |
|----------|---------|
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Application Insights connection string (sensitive) |

### Datadog

| Variable | Purpose |
|----------|---------|
| `DD_API_KEY` | Datadog API key for Datadog Agent deployment (not read by ELSPETH Datadog exporter options) |

---

## Secret Field Detection

ELSPETH automatically detects and fingerprints fields containing secrets based on naming patterns:

**Exact matches:**
- `api_key`
- `token`
- `password`
- `secret`
- `credential`

**Suffix patterns:**
- `*_secret`
- `*_key`
- `*_token`
- `*_password`
- `*_credential`

Fields matching these patterns in your configuration will be HMAC-hashed using `ELSPETH_FINGERPRINT_KEY` before being stored in the audit trail.

---

## Example .env File

Create a `.env` file in your project root:

```bash
# .env - ELSPETH environment configuration

# =====================================================================
# ELSPETH Security Settings
# =====================================================================

# Secret fingerprinting key (REQUIRED for production)
# Used to hash API keys before storing in audit trail
ELSPETH_FINGERPRINT_KEY=your-stable-secret-key

# Signing key for audit exports (optional)
# Enables HMAC signatures on exported audit records
ELSPETH_SIGNING_KEY=your-signing-key

# =====================================================================
# LLM API Keys
# =====================================================================

# OpenRouter (for llm transform with provider: openrouter)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Azure OpenAI (for llm (provider: azure) transforms)
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# =====================================================================
# Azure Content Safety Services
# =====================================================================

# Azure Content Safety (for azure_content_safety transform)
AZURE_CONTENT_SAFETY_KEY=your-content-safety-key
AZURE_CONTENT_SAFETY_ENDPOINT=https://your-resource.cognitiveservices.azure.com

# Azure Prompt Shield (for azure_prompt_shield transform)
AZURE_PROMPT_SHIELD_KEY=your-prompt-shield-key
AZURE_PROMPT_SHIELD_ENDPOINT=https://your-resource.cognitiveservices.azure.com

# Azure Document Intelligence (for azure_document_intelligence transform)
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-document-intelligence-key

# =====================================================================
# Azure Storage
# =====================================================================

# Azure Blob Storage (for azure_blob source/sink)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# =====================================================================
# Web Composer Local Registration
# =====================================================================

# Set to closed for managed deployments, or email_verified when an operator or
# mailer will deliver links from data/email-verifications.jsonl.
ELSPETH_WEB__REGISTRATION_MODE=closed
# Required for non-local email_verified deployments.
# ELSPETH_WEB__PUBLIC_BASE_URL=https://elspeth.example.gov.au

# =====================================================================
# Telemetry (secrets only)
# =====================================================================

# OTLP auth token (optional; required if your OTLP endpoint enforces auth)
OTEL_TOKEN=your-otel-token

# Azure Application Insights connection string
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...

# Datadog API key (optional if using local agent)
DD_API_KEY=your-datadog-api-key

# =====================================================================
# Development Settings (DO NOT USE IN PRODUCTION)
# =====================================================================

# Skip secret fingerprinting (development only!)
# ELSPETH_ALLOW_RAW_SECRETS=true
```

---

## Security Best Practices

### 1. Never commit .env to version control

Add to `.gitignore`:

```gitignore
.env
.env.local
.env.*.local
```

### 2. Use different keys per environment

```bash
# Production: Real fingerprint key for audit integrity
ELSPETH_FINGERPRINT_KEY=prod-key-that-never-changes

# Development: Can use any value
ELSPETH_FINGERPRINT_KEY=dev-key
```

### 3. Keep production keys stable

The fingerprint key affects how secrets appear in audit trails. Changing it mid-pipeline means you can't correlate which API key was used across runs.

### 4. Rotate API keys, not fingerprint keys

When you rotate an LLM provider API key, the new key gets a new fingerprint automatically. The `ELSPETH_FINGERPRINT_KEY` should remain stable to maintain audit consistency.

---

## Skipping .env Loading

In CI/CD or containerized environments where secrets are injected externally:

```bash
# Skip .env loading entirely
elspeth --no-dotenv run -s settings.yaml --execute
```

This is useful when:
- Secrets are injected via CI/CD environment variables
- Running in Kubernetes with secrets mounted
- Using Docker with `-e` flags

---

## Docker and CI/CD

When running ELSPETH in containers, pass environment variables directly:

```bash
docker run --rm \
  -e ELSPETH_FINGERPRINT_KEY="${ELSPETH_FINGERPRINT_KEY}" \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  -e DATABASE_URL="sqlite:////app/data/audit.db" \
  -v elspeth-data:/app/data \
  -v $(pwd)/config:/app/config:ro \
  ghcr.io/johnm-dta/elspeth:v0.7.2 \
  run --settings /app/config/pipeline.yaml --execute
```

For docker-compose:

```yaml
services:
  elspeth:
    image: ghcr.io/johnm-dta/elspeth:latest
    environment:
      - ELSPETH_FINGERPRINT_KEY=${ELSPETH_FINGERPRINT_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - DATABASE_URL=${DATABASE_URL:-sqlite:////app/data/audit.db}
    volumes:
      - elspeth-data:/app/data
```

See the [Docker Deployment Guide](../guides/docker.md) for complete container usage instructions.

---

## See Also

- [Configuration Reference](configuration.md) - Complete pipeline configuration options
- [Docker Deployment Guide](../guides/docker.md) - Container deployment
- [User Manual](../guides/user-manual.md) - Day-to-day CLI usage
