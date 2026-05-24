# Composer Tool Reference

Complete reference for the composer-facing tools and export surfaces available to the LLM pipeline composer. These tools are used in a tool-use loop where the LLM translates natural-language pipeline descriptions into valid ELSPETH pipeline configurations.

---

## Table of Contents

- [How Tools Work](#how-tools-work)
- [Discovery Tools](#discovery-tools)
- [Export Surface](#export-surface)
- [Mutation Tools](#mutation-tools)
- [Blob Tools](#blob-tools)
- [Secret Tools](#secret-tools)
- [Tool Result Format](#tool-result-format)
- [Expression Syntax Quick Reference](#expression-syntax-quick-reference)

---

## How Tools Work

The composer operates as a **tool-use loop**: the LLM receives the user's message plus the current pipeline state, calls tools to build or modify the pipeline, reviews validation results after each tool call, and continues until the pipeline is valid — then responds with a summary.

Tools fall into two categories:

| Category | Behaviour | Budget |
|----------|-----------|--------|
| **Discovery** | Read-only. Return plugin catalogs, schemas, or grammar references. Cacheable per-composition. | 10 discovery-only turns |
| **Mutation** | Modify the `CompositionState`. Return updated state + validation. | 15 turns with at least one mutation |

Every mutation tool returns a `ToolResult` containing the full validation state, so the LLM can detect and fix errors immediately.

---

## Discovery Tools

### `list_sources`

List available source plugins with name and summary.

**Parameters:** None

**Returns:** Array of `{name, summary}` for each registered source plugin.

**When to use:** At the start of composition to discover what source types are available, or when the user asks what data formats are supported.

---

### `list_transforms`

List available transform plugins with name and summary.

**Parameters:** None

**Returns:** Array of `{name, summary}` for each registered transform plugin.

**When to use:** When exploring what processing steps are available — field mapping, LLM classification, content safety, RAG retrieval, etc.

---

### `list_sinks`

List available sink plugins with name and summary.

**Parameters:** None

**Returns:** Array of `{name, summary}` for each registered sink plugin.

**When to use:** When the user describes where output should go — CSV files, databases, vector stores, cloud storage.

---

### `get_plugin_schema`

Get the full configuration schema for a specific plugin.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plugin_type` | string | **Yes** | `"source"`, `"transform"`, or `"sink"` |
| `name` | string | **Yes** | Plugin name (e.g. `"csv"`, `"llm"`, `"database"`) |

**Returns:** Full JSON Schema for the plugin's `options` configuration — field names, types, defaults, enums, and descriptions.

**When to use:** Before configuring any plugin. The schema is the source of truth for what options a plugin accepts. Do not guess option names — always check the schema first.

---

### `get_expression_grammar`

Get the gate expression syntax reference.

**Parameters:** None

**Returns:** Text reference for valid expression constructs in gate conditions.

**When to use:** Before writing any gate `condition` expression. Expressions are security-validated — only a restricted subset of Python is allowed.

---

### `list_models`

List available LLM model identifiers for use in LLM transform nodes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | string | No | Optional provider prefix filter (e.g. `"openrouter/"`, `"azure/"`) |

**Returns:** Array of model identifiers.

**When to use:** When configuring an `llm` transform and the user hasn't specified a model, or wants to see what's available.

---

### `preview_pipeline`

Preview the current pipeline configuration — validation status, source summary, and node/output overview without executing.

**Parameters:** None

**Returns:** Structured summary of the pipeline's current state including:
- `is_valid`
- structured validation `errors`, `warnings`, and `suggestions`
- `edge_contracts` for declared producer/consumer field contracts
- source, node, and output summary data

Each `edge_contracts` entry reports:
- `from` / `to`
- `producer_guarantees`
- `consumer_requires`
- `satisfied`

**When to use:** After making a series of changes, to confirm the pipeline is set up correctly before responding to the user or calling `generate_yaml`.

**Important:** `edge_contracts: []` is not positive contract evidence. It means no field contracts were declared. Also treat skipped contract-check warnings as unresolved rather than satisfied.

---

### `explain_validation_error`

Get a human-readable explanation of a validation error with suggested fixes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `error_text` | string | **Yes** | The exact validation error message to explain |

**Returns:** Explanation of what the error means and how to fix it.

**When to use:** When a mutation tool or export surface returns validation errors that aren't immediately clear. Pass the exact error text from the validation result.

**Contract-aware behaviour:** Schema contract violations are recognised explicitly. For node contract failures, the guidance directs you to `preview_pipeline` plus `patch_source_options` / `patch_node_options`. For sink contract failures, it can also direct you to `patch_output_options`.

---

## Export Surface

### `generate_yaml`

Export the current composition state as deterministic pipeline YAML.

**Parameters:** None

**Returns:** YAML text for the current composition state.

**Validation backstop:** The LLM-facing export surfaces re-run validation before emitting YAML.
- The MCP `generate_yaml` session tool returns failure if the current composition state is invalid.
- The MCP `generate_yaml` session tool also enforces generated-file controls such as explicit `collision_policy` on CSV/JSON sinks, so loaded older session state cannot bypass the mutation-tool checks.
- The HTTP `GET /api/sessions/{id}/state/yaml` endpoint returns `409` for invalid state.

**Important:** This is an export step, not the primary validator. Use `preview_pipeline` first to inspect validation results and `edge_contracts`. The underlying serializer remains a pure serializer and is not universally validation-aware at every internal call site.

---

## Mutation Tools

### `set_source`

Set or replace the pipeline source.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plugin` | string | **Yes** | Source plugin name (e.g. `"csv"`, `"json"`, `"text"`) |
| `on_success` | string | **Yes** | Connection name for downstream nodes (e.g. `"source_out"`) |
| `options` | object | **Yes** | Plugin-specific configuration (check schema first) |
| `on_validation_failure` | string | **Yes** | `"discard"` or `"quarantine"` |

**Behaviour:** Replaces the current source entirely. A pipeline must have exactly one source.

**Connection model:** The `on_success` value becomes the connection name that downstream nodes reference in their `input` field. Choose a descriptive name like `"raw_data"` or `"source_out"`.

---

### `upsert_node`

Add or update a pipeline node — transforms, gates, aggregations, or coalesces.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | **Yes** | Unique node identifier (e.g. `"classifier"`, `"quality_gate"`) |
| `node_type` | string | **Yes** | `"transform"`, `"gate"`, `"aggregation"`, or `"coalesce"` |
| `input` | string | **Yes** | Input connection name (must match an upstream `on_success`) |
| `plugin` | string | No | Plugin name. Required for transforms and aggregations. Null for gates and coalesces. |
| `on_success` | string | No | Output connection name. Required for transforms. Null for gates (which use routes). |
| `on_error` | string | No | Error output — a sink name or `"discard"` |
| `options` | object | No | Plugin-specific configuration |
| `condition` | string | No | Gate expression (gates only) |
| `routes` | object | No | Gate route mapping to a sink name, downstream connection name, `"fork"`, or virtual `"discard"` target, e.g. `{"true": "sink_name", "false": "discard"}` (gates only) |
| `fork_to` | array | No | Fork destination connection names (fork gates only) |
| `branches` | array | No | Branch input connection names (coalesce only, min 2) |
| `policy` | string | No | Coalesce policy: `"require_all"`, `"quorum"`, `"best_effort"`, `"first"` |
| `merge` | string | No | Coalesce merge strategy: `"union"`, `"nested"`, `"select"` |

**Behaviour:** If a node with the given `id` already exists, it is replaced. The `id` must be unique across all node types.

**Node type specifics:**

| Type | Required fields | Key behaviour |
|------|----------------|---------------|
| `transform` | `plugin`, `on_success` | Processes rows, emits to on_success |
| `gate` | `condition`, `routes` | Evaluates condition, routes by result |
| `aggregation` | `plugin` | Batches rows until trigger fires |
| `coalesce` | `branches` | Merges tokens from parallel fork paths |

---

### `upsert_edge`

Add or update a connection between nodes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | **Yes** | Unique edge identifier (e.g. `"source_to_gate"`) |
| `from_node` | string | **Yes** | Source node ID, or `"source"` for the pipeline source |
| `to_node` | string | **Yes** | Destination node ID or sink name |
| `edge_type` | string | **Yes** | `"on_success"`, `"on_error"`, `"route_true"`, `"route_false"`, or `"fork"` |
| `label` | string | No | Display label (used as route key for gate edges) |

**Behaviour:** Edges define the data flow through the pipeline. Every node must be connected — orphan nodes cause validation errors.

**Edge types:**

| Type | Meaning |
|------|---------|
| `on_success` | Normal data flow from one node to the next |
| `on_error` | Error routing (rows that fail processing) |
| `route_true` | Gate route when condition evaluates to `True` |
| `route_false` | Gate route when condition evaluates to `False` |
| `fork` | Gate fork to parallel paths |

---

### `remove_node`

Remove a node and all its edges.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | **Yes** | Node ID to remove |

**Behaviour:** Removes the node and all edges connected to it (both incoming and outgoing).

---

### `remove_edge`

Remove an edge by ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | **Yes** | Edge ID to remove |

---

### `set_output`

Add or replace a pipeline output (sink).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sink_name` | string | **Yes** | Sink name — this is the connection point used in routes and edges |
| `plugin` | string | **Yes** | Sink plugin name (e.g. `"csv"`, `"json"`, `"database"`) |
| `options` | object | **Yes** | Plugin-specific configuration (check schema first) |
| `on_write_failure` | string | No | `"discard"` (default) or `"quarantine"` |

**Behaviour:** If a sink with the given name exists, it is replaced. A pipeline must have at least one sink.

**File sink collision control:** For generated `csv` and `json` file sinks, include `options.collision_policy` explicitly. Use `fail_if_exists` when the requested filename must be free, `auto_increment` when repeated runs may create `name-1.ext`, or `append_or_create` with `mode: "append"`.

---

### `remove_output`

Remove a pipeline output (sink) by name.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sink_name` | string | **Yes** | Sink name to remove |

---

### `set_metadata`

Update pipeline metadata.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `patch` | object | **Yes** | Partial update — only included fields change. Fields: `name` (string), `description` (string). |

---

### `patch_source_options`

Apply a shallow merge-patch to the current source options.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `patch` | object | **Yes** | Keys in the patch overwrite existing keys. Keys set to `null` are deleted. Missing keys are unchanged. |

**When to use:** To modify specific source options without replacing the entire source. More surgical than `set_source`.

---

### `patch_node_options`

Apply a shallow merge-patch to a node's options.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | **Yes** | ID of the node to patch |
| `patch` | object | **Yes** | Merge-patch (same semantics as `patch_source_options`) |

---

### `patch_output_options`

Apply a shallow merge-patch to an output's options.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sink_name` | string | **Yes** | Name of the output (sink) to patch |
| `patch` | object | **Yes** | Merge-patch (same semantics as `patch_source_options`) |

---

### `set_pipeline`

Atomically replace the entire pipeline in one call.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | object | **Yes** | `{plugin, options, on_success, on_validation_failure?}` |
| `nodes` | array | **Yes** | Array of node specs: `[{id, input, plugin?, node_type, options?, on_success?, on_error?, condition?, routes?, fork_to?, branches?, policy?, merge?}]` |
| `edges` | array | **Yes** | Array of edge specs: `[{id, from_node, to_node, edge_type}]` |
| `outputs` | array | **Yes** | Array of output specs: `[{name, plugin, options, on_write_failure?}]` |
| `metadata` | object | No | `{name?, description?}` |

**When to use:** When the user describes a complete pipeline and you can construct the whole thing at once. More efficient than sequential `set_source` + `upsert_node` + `upsert_edge` + `set_output` calls. Also useful for major restructuring where most components change.

Generated `csv` and `json` outputs must include `options.collision_policy`; the web composer and MCP composer controls reject missing policy choices before export.

---

### `clear_source`

Remove the source from the pipeline composition state.

**Parameters:** None

**When to use:** When the user wants to start over with a different data source, or when replacing a blob-based source with a file-path source.

---

## Blob Tools

These tools work with files uploaded through the web UI.

### `list_blobs`

List uploaded/created files (blobs) in this session with metadata.

**Parameters:** None

**Returns:** Array of blob records with ID, filename, MIME type, size, and creation time.

---

### `list_composer_blobs`

List ready blobs using the inline-content authoring visibility shape.

**Parameters:** None

**Returns:** `{blobs: [...]}` descriptors with `blob_id`, `mime_type`, `size_bytes`, `content_hash`, and `filename`. Content bytes, previews, storage paths, and source descriptions are not returned.

---

### `get_blob_metadata`

Get metadata for a specific blob by ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `blob_id` | string | **Yes** | Blob ID |

---

### `set_source_from_blob`

Wire a blob as the pipeline source. Resolves the blob's storage path internally and infers the source plugin from its MIME type.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `blob_id` | string | **Yes** | Blob ID to use as source |
| `plugin` | string | No | Source plugin override. Inferred from MIME type if omitted. |
| `on_success` | string | **Yes** | Connection name for downstream nodes |
| `on_validation_failure` | string | No | `"quarantine"` (default) or `"discard"` |

**When to use:** When the user has uploaded a file through the web UI and wants to use it as the pipeline input. Prefer this over `set_source` for uploaded files — it handles path resolution and plugin inference.

---

### `wire_blob_inline_ref`

Wire a ready blob into a plugin option as audited inline content.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `field_path` | string | **Yes** | Canonical target path such as `source.options.<field>`, `node:<node_id>.options.<field>`, or `output:<name>.options.<field>` |
| `blob_id` | string | **Yes** | Ready blob ID to wire |
| `encoding` | string | No | One of `utf-8`, `utf-8-sig`, `utf-16`, or `latin-1`; defaults to `utf-8` |

**When to use:** When a long-form config field, such as an LLM prompt template or SQL query, should be supplied from a blob without exposing the bytes to the composer LLM. The tool pins `sha256` from blob metadata.

---

## Secret Tools

These tools manage API keys and credentials without exposing secret values.

### `list_secret_refs`

List available secret references (API keys, credentials). Shows names and scopes, never values.

**Parameters:** None

**Returns:** Array of secret reference names available to the current user.

**When to use:** When the user wants to configure a plugin that requires credentials (LLM API keys, database passwords, Azure credentials). Check what's available before asking the user to provide secrets.

---

### `validate_secret_ref`

Check if a secret reference exists and is accessible to the current user.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | **Yes** | Secret reference name (e.g. `"OPENROUTER_API_KEY"`) |

---

### `wire_secret_ref`

Place a secret reference marker in the pipeline config. The secret is resolved at execution time — never stored in the composition state.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | **Yes** | Secret reference name |
| `target` | string | **Yes** | `"source"`, `"node"`, or `"output"` |
| `target_id` | string | No | Node ID or output name (required for `"node"` and `"output"` targets) |
| `option_key` | string | **Yes** | Config option key to set (e.g. `"api_key"`) |

**When to use:** When a plugin needs a credential. Always use `list_secret_refs` first to check availability, then `validate_secret_ref` to confirm access, then `wire_secret_ref` to connect it.

---

## Tool Result Format

Every tool returns a `ToolResult` with this structure:

```json
{
  "success": true,
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": ["Source schema mode is 'observed' — consider 'fixed' for stricter validation"],
    "suggestions": []
  },
  "affected_nodes": ["classifier", "quality_gate"],
  "version": 3,
  "data": null
}
```

| Field | Description |
|-------|-------------|
| `success` | Whether the tool call succeeded |
| `validation` | Current pipeline validation state after this change |
| `affected_nodes` | Node IDs that were changed or have changed edges |
| `version` | Pipeline state version (increments on each mutation) |
| `data` | Discovery tool payload (plugin lists, schemas, etc.) — null for mutations |

**Validation drives the loop.** After each mutation, check `validation.errors`. If there are errors, fix them before responding to the user. The LLM should not present a pipeline as complete until `is_valid` is `true`.

---

## Expression Syntax Quick Reference

Gate conditions use a restricted expression language validated by `ExpressionParser`.

### Allowed

| Construct | Example |
|-----------|---------|
| Field access | `row['field']`, `row.get('field')` |
| Built-in functions | `len()`, `abs()` |
| Comparisons | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Boolean operators | `and`, `or`, `not` |
| Arithmetic | `+`, `-`, `*`, `/`, `%` |
| Membership | `in`, `not in` |
| Literals | `True`, `False`, `None`, numbers, strings |
| Ternary | `x if condition else y` |

### Forbidden

| Construct | Reason |
|-----------|--------|
| Type coercion (`int()`, `str()`, `float()`, `bool()`) | Not needed — source schema guarantees type safety |
| `row.get('field', default)` | Default values fabricate data the source never provided |
| Imports, lambdas, comprehensions | Security |
| Attribute access (except `row.get()`) | Security |
| F-strings | Security |

### Examples

```python
row['amount'] > 1000
row['status'] == 'approved'
row['category'] in ('A', 'B', 'C')
row.get('optional_field') is not None
row['score'] > 0.5 and row['status'] != 'rejected'
len(row['name']) > 0
```
