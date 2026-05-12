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
- [Web Composer — Guided Mode](#web-composer--guided-mode)
  - [Auto-dropped to freeform — what happened?](#auto-dropped-to-freeform--what-happened)
  - [Wizard disagreed with my source schema](#wizard-disagreed-with-my-source-schema)
  - [Recipe didn't appear for my (CSV, JSONL) pipeline](#recipe-didnt-appear-for-my-csv-jsonl-pipeline)

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
  ghcr.io/johnm-dta/elspeth:v0.5.1 \
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
   transforms:
     - plugin: llm
       options:
         provider: azure
         endpoint: "https://your-resource.openai.azure.com"
         deployment_name: "your-deployment"
   ```

---

### LLM Transform Errors

#### Error: `missing_output_field` in error_json

**Cause:** The LLM returned valid JSON, but a required output field was not present in the response. This commonly happens when the LLM generates a response that doesn't match the expected schema.

**Example error_json:**
```json
{
  "reason": "missing_output_field",
  "query_name": "classify",
  "field": "category",
  "available_fields": ["label", "confidence"]
}
```

**Solution:**

1. Check your prompt template — ensure it instructs the LLM to include the required fields
2. Review the `available_fields` in the error to see what the LLM actually returned
3. Consider adding few-shot examples to your prompt to guide the output format
4. If the field is genuinely optional, remove it from `output_fields` in your query spec

#### Error: `content_filtered` in error_json

**Cause:** The LLM provider returned a null content response, typically because the request or response was flagged by the provider's content safety filters.

**Solution:**

1. Review the prompt being sent — check for content that might trigger safety filters
2. Check the provider's content policy documentation
3. For Azure OpenAI, review the content filtering configuration on your deployment
4. The row will be marked as errored (not quarantined) — investigate via:
   ```bash
   elspeth explain --run <run_id> --database ./runs/audit.db
   ```

#### Error: `RuntimeError: langfuse ... not installed`

**Cause:** Langfuse tracing is configured in your pipeline YAML but the `langfuse` package is not installed. As of RC3.3, this fails fast instead of silently degrading.

**Solution:**
```bash
uv pip install 'elspeth[tracing-langfuse]'
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
   transforms:
     - plugin: llm
       options:
         provider: azure
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
   export DATABASE_URL="postgresql://<user>:<password>@localhost:5432/elspeth"  # secret-scan: allow-this-line
   ```

3. In Docker Compose, use service names not `localhost`:
   ```yaml
   environment:
     - DATABASE_URL=postgresql://<user>:<password>@db:5432/elspeth  # 'db' is the service name; secret-scan: allow-this-line
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
     url: postgresql://<user>:<password>@localhost:5432/elspeth  # secret-scan: allow-this-line
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
     ghcr.io/johnm-dta/elspeth:v0.5.1 \
     ls /app/input
   ```

2. Ensure pipeline config uses container paths, not host paths:
   ```yaml
   # CORRECT - container path
   source:
     plugin: csv
     options:
       path: /app/input/data.csv

   # WRONG - host path
   source:
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
   - Missing `source` section
   - Missing `sinks` section
   - Plugin options with wrong types
   - Undefined environment variables in `${VAR}` syntax

---

## Web Composer — Guided Mode

### Auto-dropped to freeform — what happened?

**Cause:** Guided mode dropped you to the freeform composer because the
wizard could not complete the step you were on. Two paths trigger an
auto-drop:

- **`solver_exhausted`** — at Step 3, the LLM proposed a transform chain
  that failed `preview_pipeline`. The wizard tried one repair attempt
  (feeding the validator's rejection reason back to the LLM) and optionally
  consulted the advisor. Both came back red, so the wizard handed the
  partial pipeline state to freeform mode rather than loop forever.
- **`protocol_violation`** — the LLM emitted a turn type that is not
  legal at the current step. The wizard granted one retry, the LLM
  emitted another illegal turn, and the wizard auto-dropped.

**Solution:**

1. Scroll up in the chat history. The drop is recorded as a system
   message with the `drop_reason` field set. For `solver_exhausted`, the
   validator's rejection reason is also recorded.
2. Inspect the last `propose_chain` turn (if any) to see the chain the
   LLM tried and what the validator said about it. The validator usually
   names the specific edge or required-field constraint that was
   violated.
3. Finish the pipeline by hand in freeform mode. The composer carries
   over the partial pipeline you had at the moment of the drop, so you
   are not starting from scratch.
4. If the LLM keeps producing the same broken chain on similar inputs,
   that is a real bug — open an issue with the chat history attached.

---

### Wizard disagreed with my source schema

**Cause:** Step 1 of guided mode runs `inspect_source` on the blob you
attached and shows you the columns it observed. If the inspector's
opinion of your data does not match what you expected (wrong column
names, missing columns, columns that should not be there), one of the
following is usually true:

- The blob was uploaded with the wrong `mime_type`, so the inspector
  picked the wrong parser (for example, JSONL parsed as plain text).
- The CSV has a non-standard delimiter, encoding, or header row that
  the inspector's default heuristics missed.
- The data actually does have the columns the inspector reported, and
  your expectation was based on a different file.

**Solution:**

1. Read the column list and sample values the `inspect_and_confirm` turn
   is showing you. The truth is in the bytes of the blob, not in your
   memory of the file.
2. If the columns are wrong, edit the column list directly on the
   `inspect_and_confirm` turn before continuing. Your edits are recorded
   as the schema-of-record for the rest of the wizard.
3. If the underlying problem is the parser choice, exit to freeform,
   re-upload the blob with the correct `mime_type`, and use the
   freeform composer's `inspect_source` tool to get an authoritative
   reading before retrying guided mode.

---

### Recipe didn't appear for my (CSV, JSONL) pipeline

**Cause:** The recipe pre-match step (Step 2.5) matches on the pipeline
**topology** (source plugin, sink plugin, and sink count) **plus a
discriminator** — usually a required output field that names the recipe's
purpose. For example, the `classify-rows-llm-jsonl` recipe requires a
`classifier_keyword`-style field in the output. A bare (CSV, JSONL)
shape is not enough; the discriminator has to match too.

**Solution:**

1. Switch to freeform mode briefly and call `list_recipes`. The response
   names every registered recipe, its required slots, and the
   discriminator field(s) it matches on.
2. If a recipe exists but its discriminator did not match your output
   fields, go back to guided Step 2 and add the discriminator field to
   the required-output list. The recipe will then match on the next
   pass through Step 2.5.
3. If no registered recipe matches your (source, sink) shape at all,
   that is expected: the wizard will skip Step 2.5 and let the LLM
   propose a chain in Step 3. You can also pre-apply a recipe manually
   from freeform mode with `list_recipes` + the recipe-application tool,
   then return to guided mode to finish the rest.

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

- [User Manual](user-manual.md) - CLI commands and usage
- [Docker Deployment Guide](../runbooks/docker.md) - Container deployment
- [Environment Variables Reference](../reference/environment-variables.md) - Configuration options
