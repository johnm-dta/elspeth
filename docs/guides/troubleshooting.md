# Troubleshooting Guide

This guide covers common errors and their solutions when running ELSPETH pipelines.

## Table of Contents

- [Common Errors](#common-errors)
  - [Secret Fingerprinting Errors](#secret-fingerprinting-errors)
  - [Plugin Errors](#plugin-errors)
  - [API Authentication Errors](#api-authentication-errors)
- [Pipeline Issues](#pipeline-issues)
  - [Pipeline Hangs or Times Out](#pipeline-hangs-or-times-out)
  - [Row Processing Failures](#row-processing-failures)
- [Database Issues](#database-issues)
  - [Database Connection Refused](#database-connection-refused)
  - [Database Locked](#database-locked)
- [Docker-Specific Issues](#docker-specific-issues)
  - [File Not Found Errors](#file-not-found-errors)
  - [Permission Denied on Output](#permission-denied-on-output)
  - [Health Check Fails in Kubernetes](#health-check-fails-in-kubernetes)
- [Configuration Issues](#configuration-issues)
  - [YAML Parsing Errors](#yaml-parsing-errors)
  - [Schema Validation Failures](#schema-validation-failures)

---

## Common Errors

### Secret Fingerprinting Errors

#### Error: "Secret field found but ELSPETH_FINGERPRINT_KEY is not set"

**Cause:** Your configuration contains API keys or secrets, but ELSPETH cannot fingerprint them without the fingerprint key.

**Solution:**

Option 1 - Set the fingerprint key (recommended for production):
```bash
export ELSPETH_FINGERPRINT_KEY="your-secure-key-here"
```

Or add to your `.env` file:
```bash
ELSPETH_FINGERPRINT_KEY=your-secure-key-here
```

Option 2 - Allow raw secrets (development only):
```bash
export ELSPETH_ALLOW_RAW_SECRETS=true
```

**Docker:**
```bash
docker run --rm \
  -e ELSPETH_FINGERPRINT_KEY="your-key" \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  run --settings /app/config/pipeline.yaml --execute
```

---

### Plugin Errors

#### Error: "Unknown plugin: xyz"

**Cause:** The plugin name in your configuration doesn't match any registered plugin.

**Solution:**

1. Check available plugins:
   ```bash
   elspeth plugins list
   ```

2. Verify the plugin name is spelled correctly (names are case-sensitive)

3. Ensure you have the correct optional dependencies installed:
   ```bash
   # For LLM plugins
   uv pip install -e ".[llm]"

   # For all plugins
   uv pip install -e ".[all]"
   ```

---

### API Authentication Errors

#### Error: HTTP 401 Unauthorized or 403 Forbidden

**Cause:** Invalid or missing API credentials for external services (LLM providers, Azure services, etc.).

**Solution:**

1. Check your `.env` file has the correct API key:
   ```bash
   OPENROUTER_API_KEY=sk-or-...
   AZURE_OPENAI_API_KEY=...
   ```

2. Verify the key is valid and not expired

3. Ensure `.env` is in the current directory or a parent directory

4. Debug environment loading:
   ```bash
   # Check if .env is being loaded
   elspeth run -s settings.yaml --execute --verbose
   ```

5. For Azure services, verify your resource endpoint is correct:
   ```yaml
   row_plugins:
     - plugin: azure_llm
       options:
         endpoint: "https://your-resource.openai.azure.com"
         deployment_name: "your-deployment"
   ```

---

## Pipeline Issues

### Pipeline Hangs or Times Out

**Cause:** Rate limiting, slow external APIs, or misconfigured timeouts.

**Solution:**

1. Check rate limiting configuration:
   ```yaml
   concurrency:
     max_workers: 4
     rate_limit:
       calls_per_minute: 60
   ```

2. Increase timeout for slow operations:
   ```yaml
   row_plugins:
     - plugin: azure_llm
       options:
         timeout: 120  # seconds
   ```

3. Use verbose mode to identify the bottleneck:
   ```bash
   elspeth run -s settings.yaml --execute --verbose
   ```

4. Check external service status (OpenAI status page, Azure status, etc.)

---

### Row Processing Failures

**Cause:** Malformed data, schema mismatches, or transform errors.

**Solution:**

1. Check the audit database for error details:
   ```bash
   elspeth explain --run latest --no-tui --database ./runs/audit.db
   ```

2. Look for quarantined rows:
   ```sql
   SELECT * FROM node_states WHERE status = 'quarantined';
   ```

3. Validate your source data matches the expected schema

4. Use the MCP analysis server for detailed investigation:
   ```bash
   elspeth-mcp --database ./runs/audit.db
   ```

---

## Database Issues

### Database Connection Refused

**Cause:** Database server not running, wrong connection string, or network issues.

**Solution:**

1. For SQLite, verify the path exists and is writable:
   ```bash
   ls -la ./runs/
   ```

2. For PostgreSQL, check the connection string:
   ```bash
   export DATABASE_URL="postgresql://user:pass@localhost:5432/elspeth"
   ```

3. In Docker Compose, use service names not `localhost`:
   ```yaml
   environment:
     - DATABASE_URL=postgresql://user:pass@db:5432/elspeth  # 'db' is the service name
   ```

4. Verify the database server is running:
   ```bash
   # PostgreSQL
   pg_isready -h localhost -p 5432
   ```

---

### Database Locked

**Cause:** Multiple processes trying to write to SQLite simultaneously.

**Solution:**

1. Ensure only one pipeline is running at a time against the same SQLite database

2. For concurrent workloads, switch to PostgreSQL:
   ```yaml
   landscape:
     url: postgresql://user:pass@localhost:5432/elspeth
   ```

3. If a process crashed, check for stale locks:
   ```bash
   # Remove journal files (only if no other process is running)
   rm -f ./runs/audit.db-journal
   rm -f ./runs/audit.db-wal
   ```

---

## Docker-Specific Issues

### File Not Found Errors

**Error:** `FileNotFoundError: /app/input/data.csv`

**Cause:** Volume not mounted or pipeline config uses wrong paths.

**Solution:**

1. Verify the volume is mounted correctly:
   ```bash
   docker run --rm \
     -v $(pwd)/input:/app/input:ro \
     ghcr.io/johnm-dta/elspeth:v0.1.0 \
     ls /app/input
   ```

2. Ensure pipeline config uses container paths, not host paths:
   ```yaml
   # CORRECT - container path
   datasource:
     plugin: csv
     options:
       path: /app/input/data.csv

   # WRONG - host path
   datasource:
     plugin: csv
     options:
       path: ./input/data.csv
   ```

---

### Permission Denied on Output

**Error:** `PermissionError: [Errno 13] Permission denied: '/app/output/results.csv'`

**Cause:** Output directory doesn't exist on host or has wrong permissions.

**Solution:**

1. Create the output directory on the host:
   ```bash
   mkdir -p ./output
   ```

2. Set appropriate permissions:
   ```bash
   chmod 777 ./output  # Or use appropriate UID/GID matching container user
   ```

3. Verify the volume mount mode allows writing:
   ```bash
   # Mount without :ro (read-only)
   -v $(pwd)/output:/app/output  # Default is read-write
   ```

---

### Health Check Fails in Kubernetes

**Error:** Pod keeps restarting due to failed liveness probe.

**Cause:** Health check requires database connection that's not ready at startup.

**Solution:**

Use separate readiness and liveness probes with appropriate delays:

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

The readiness probe prevents traffic before the app is ready. The liveness probe restarts stuck containers but gives more time for initial startup.

---

## Configuration Issues

### YAML Parsing Errors

**Error:** `yaml.scanner.ScannerError: mapping values are not allowed here`

**Cause:** Indentation errors or invalid YAML syntax.

**Solution:**

1. Validate your YAML syntax:
   ```bash
   elspeth validate --settings settings.yaml
   ```

2. Common YAML mistakes:
   - Mixing tabs and spaces (use spaces only)
   - Incorrect indentation levels
   - Missing colons after keys
   - Unquoted special characters

3. Use a YAML linter:
   ```bash
   # Install yamllint
   uv pip install yamllint
   yamllint settings.yaml
   ```

---

### Schema Validation Failures

**Error:** `ValidationError: field required` or `ValidationError: value is not a valid...`

**Cause:** Missing required configuration fields or wrong types.

**Solution:**

1. Run validation to see all errors:
   ```bash
   elspeth validate --settings settings.yaml
   ```

2. Check the configuration reference for required fields

3. Common issues:
   - Missing `datasource` section
   - Missing `sinks` section
   - Plugin options with wrong types
   - Undefined environment variables in `${VAR}` syntax

---

## Getting Help

If you're still stuck:

1. Run with verbose logging:
   ```bash
   elspeth run -s settings.yaml --execute --verbose
   ```

2. Check the audit database for detailed error context:
   ```bash
   elspeth explain --run latest --database ./runs/audit.db
   ```

3. Use the MCP analysis server for investigation:
   ```bash
   elspeth-mcp
   ```

4. For bug reports, include:
   - ELSPETH version (`elspeth --version`)
   - Python version (`python --version`)
   - Full error message and stack trace
   - Sanitized configuration (remove secrets)

---

## See Also

- [User Manual](../USER_MANUAL.md) - CLI commands and usage
- [Docker Deployment Guide](docker.md) - Container deployment
- [Environment Variables Reference](../reference/environment-variables.md) - Configuration options
