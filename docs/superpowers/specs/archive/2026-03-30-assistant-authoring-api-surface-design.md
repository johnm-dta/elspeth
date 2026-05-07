# Assistant Authoring API Surface

**Status:** Draft
**Date:** 2026-03-30
**Branch:** post-RC4 web UX follow-on
**Relates to:** `docs/superpowers/specs/2026-03-28-web-ux-composer-mvp-design.md`

---

## Overview

The current Web UX Composer MVP proves that ELSPETH can author pipelines
through chat, but the assistant-facing API surface is still too thin. The
server owns the real pipeline state, artifact staging, validation, secret
availability, and execution wiring, yet the assistant currently has only a
partial mutation surface. This creates avoidable friction:

- the assistant can describe a source but not fully configure it
- pasted text and uploaded files are not first-class pipeline inputs
- secret handling is awkward and unsafe to expose through chat
- validation exists, but mutation calls do not return validation as a first-class result
- iterative repair is harder than it should be because the assistant lacks full-fidelity read/write symmetry

This spec defines an assistant-first authoring API surface for the Web UX. The
goal is not to expose graph mechanics directly to the user. The goal is to let
the assistant translate natural-language intent into a valid, previewable,
executable ELSPETH pipeline while preserving auditability, trust boundaries, and
safe secret handling.

---

## Goals

- Allow the assistant to translate natural language into a complete pipeline.
- Let the assistant fully configure sources, transforms, and sinks.
- Materialize pasted text, JSON, CSV, and uploaded files into reusable artifacts.
- Validate and repair pipeline state iteratively without requiring graph-level user knowledge.
- Represent secrets as references, not raw secret values.
- Support preview-oriented workflows in a later increment.
- Keep the user interaction natural-language-first: tasks and intent, not node/edge mechanics.

---

## Non-Goals

- Direct graph editing by end users.
- Returning secret values to the assistant.
- Replacing the existing session/chat model with transaction-only authoring.
- Destructive rollback of prior chat history or executed jobs.
- General multi-tenant secret governance in this first spec. The surface should
  allow it later, but the first implementation may remain single-tenant with
  user/admin scoping.

---

## Design Principles

### 1. Read/Write Symmetry

Anything materially visible in pipeline state should be writable through the
authoring surface. The assistant should not need one set of tools for reading
state and a different, lower-fidelity conceptual model for mutating it.

### 2. Natural-Language-First

The assistant absorbs the graph mechanics, path staging, and plugin-specific
configuration details. The user should not need to know what a node is or how
edges are wired.

### 3. Safe Secret Handling

Secrets are never returned as plaintext to the assistant or the web interface.
Mutations accept secret references, not raw values. If a user enters a new
secret through the web UX, the value is treated as write-only input: it is
handed off immediately to profile-backed or admin-backed storage, then removed
from the live web state. The secret reference remains visible as an available
capability; only the value disappears.

### 4. Materialized Input as a First-Class Concept

Text, JSON, CSV, binary uploads, and copied/pasted content become artifacts that
can be bound to sources and jobs cleanly.

### 5. Validation is Part of Mutation

Every mutation should return validation state, warnings, and suggestions. A
separate `validate_pipeline()` tool still exists, but validation must not be
treated as an afterthought.

### 6. Fork, Don’t Roll Back

Conversational revision from a prior message should create a new branch/fork of
the session state. Executed work and prior history are not destructively
rewritten.

### 7. ELSPETH Custody of Boundaries

External content, uploads, secret-backed config, and provider responses all cross
 trust boundaries. The API surface must keep validation close to those
boundaries and preserve auditability of decisions and state transitions.

---

## Resource Model

The assistant-facing API operates over four primary resource types.

### Pipeline

Represents the current authoring state.

```json
{
  "metadata": {},
  "source": {},
  "nodes": [],
  "edges": [],
  "outputs": [],
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": [],
    "suggestions": []
  }
}
```

### Artifact

A staged data object usable by sources, previews, and future blob-backed flows.
Artifacts may represent:

- pasted text
- structured JSON
- tabular CSV
- uploaded binary files
- future derived outputs (for example, generated YAML or preview result blobs)

### Secret Reference

A named handle to a secret already available to the runtime. The handle is safe
to expose to the assistant and web UX; the value is not. The web UX may collect
new secret values for submission, but those values must never become readable
state after submission. After submission, the UI may continue to show that the
secret reference exists and is usable, but not what the value is.

```json
{
  "secret_ref": "OPENROUTER_API_KEY"
}
```

### Preview / Run Result

A sampled, validated, or executed view of pipeline behavior. This is separate
from the authoring state because previews and runs are observational products of
the pipeline, not the pipeline definition itself.

---

## API Surface

## Discovery

These tools let the assistant discover valid building blocks without hardcoding
plugin knowledge into prompts.

### `list_sources()`

Returns available source plugins.

### `list_transforms()`

Returns available transform plugins.

### `list_sinks()`

Returns available sink plugins.

### `get_plugin_schema(plugin_type, name)`

Returns:

- full schema
- required fields
- defaults
- examples
- field descriptions
- conditional requirements

This tool is the schema authority for assistant-guided configuration.

### `get_expression_grammar()`

Returns the grammar and examples for gate expressions / routing conditions.

### `list_models(provider)`

Returns valid model identifiers for provider-backed LLM plugins. This reduces
assistant hallucination around model names and supports provider-aware UX.

### `list_secret_refs()`

Returns secret names and metadata only. Never returns values.

Recommended metadata:

- `scope`: `user | org | server`
- `available_to_runtime`: boolean
- `usable_by`: list of provider or plugin families
- `display_name`: optional human-readable label
- `resolution_scope`: where ELSPETH will resolve the reference at runtime

---

## Pipeline State

### `get_pipeline_state()`

Returns the full current authoring state, including validation and warnings.

### `validate_pipeline()`

Returns a full validation report independent of any mutation.

### `explain_validation_error(error_id | error_text)`

Returns a human-readable diagnosis with likely repair steps. This is especially
useful for DAG contract errors, required field mismatches, secret reference
issues, and provider readiness failures.

---

## Pipeline Authoring

## Metadata

### `set_metadata(patch)`

Sets or patches pipeline metadata such as name and description.

## Source

### `set_source(payload)`

Creates or replaces the source with full-fidelity configuration.

Expected to support:

```json
{
  "plugin": "json",
  "on_success": "seed_rows",
  "on_validation_failure": "discard",
  "options": {
    "path": "/staging/seed.json",
    "format": "json",
    "schema": {
      "mode": "observed",
      "fields": [
        { "name": "url", "type": "str", "required": true, "nullable": false }
      ]
    }
  }
}
```

### `patch_source_options(patch)`

Patch-only convenience for iterative edits.

### `clear_source()`

Optional convenience for clearing the source without replacing the entire
pipeline.

## Nodes

### `upsert_node(payload)`

Creates or replaces a node with full `options`.

Expected to support:

```json
{
  "id": "identify_layers",
  "node_type": "transform",
  "plugin": "llm",
  "input": "scraped_rules",
  "on_success": "llm_results",
  "on_error": "llm_errors",
  "options": {
    "provider": "openrouter",
    "model": "openai/gpt-5.4",
    "api_key": { "secret_ref": "OPENROUTER_API_KEY" },
    "template": "...",
    "response_field": "layer_rules_extraction"
  }
}
```

### `patch_node_options(id, patch)`

Patch-only convenience for targeted iterative edits.

### `clone_node(id, new_id)`

Convenience for experimentation.

### `remove_node(id)`

Removes a node and any dependent edges, with validation returned in the response.

## Edges

### `upsert_edge(payload)`

Creates or replaces an edge / route.

### `remove_edge(id)`

Removes an edge.

### `auto_wire_connections()`

Optional helper that connects matching producer / consumer names where the graph
shape is obvious.

### `find_unconnected_components()`

Optional graph sanity tool for authoring repair loops.

## Outputs

### `set_output(payload)`

Creates or replaces a sink with full configuration and `on_write_failure`.

### `patch_output_options(sink_name, patch)`

Targeted sink edit helper.

### `remove_output(sink_name)`

Removes a sink.

## Whole-Pipeline Convenience

### `set_pipeline(payload)`

Atomically replaces the full pipeline state.

### `apply_pipeline_patch(payload)`

Structured multi-component patch for assistant-driven repair operations.

### Transaction Support

The API may later add:

- `begin_transaction()`
- `commit_transaction()`
- `rollback_transaction()`

These are useful for batch edits that are transiently invalid. However, the
conversational UX should not depend on transactions for history manipulation.

---

## Artifact Management

Artifact staging is the missing bridge between conversational input and
executable pipeline configuration.

## Artifact Creation

### `create_text_artifact(payload)`

```json
{
  "content": "raw text here",
  "filename": "input.txt"
}
```

### `create_json_artifact(payload)`

```json
{
  "data": [
    { "url": "https://example.com/file.txt" }
  ],
  "filename": "seed.json"
}
```

### `create_csv_artifact(payload)`

```json
{
  "rows": [
    { "name": "Alice", "score": 10 }
  ],
  "filename": "input.csv"
}
```

### `create_binary_artifact(payload)`

Supports uploaded files, drag-and-drop, and future blob-backed flows.

## Artifact Management

### `list_artifacts()`

Returns available staged artifacts with metadata.

### `get_artifact_metadata(artifact_id)`

Returns metadata only.

### `read_artifact_text(artifact_id)`

Optional but useful for assistant confirmation and debugging. Only valid for
textual artifacts.

### `delete_artifact(artifact_id)`

Deletes the staged artifact if it is safe to do so.

### `set_source_from_artifact(payload)`

Convenience helper that binds an artifact to an appropriate source plugin and
fills inferred path/format details automatically.

---

## Secret Handling

Secrets should be referenced, never echoed through the assistant surface or
retained in readable web state.

### Secret Reference Shape

All secret-bearing config fields should accept:

```json
{ "secret_ref": "OPENROUTER_API_KEY" }
```

instead of raw values.

### `list_secret_refs()`

Returns names and metadata only.

### `set_secret_ref(payload)`

Accepts a new secret value for immediate storage in user-profile, organization, or
server-owned
secret storage and returns only non-sensitive metadata, for example:

```json
{
  "secret_ref": "OPENROUTER_API_KEY",
  "scope": "user",
  "available_to_runtime": true
}
```

The submitted secret value is write-only. Once accepted, it must not be
returned by any read API. After submission, the web interface should continue
to show the secret reference as available, but the secret value itself must no
longer be present in readable state.

### `validate_secret_ref(name)`

Confirms existence and runtime accessibility without disclosing value.

### Optional: `resolve_secret_capabilities(name)`

Returns metadata such as:

- runtime availability
- scope
- provider families it may be used with

### Runtime Secret Resolution

ELSPETH must resolve secret references through scoped runtime lookup rather than
through a flat secret namespace.

Minimum supported scopes:

- `user`: secrets stored in the current user's profile
- `org`: secrets made available to members of the current organization/workspace
- `server`: secrets provisioned for the whole server/runtime

The runtime should support two related behaviors:

- listing available references and their scopes to the assistant/web UX
- resolving the actual secret value only inside trusted server/runtime code at
  the moment it is needed

Recommended resolution behavior:

1. If a config field specifies an explicit scope, resolve only within that
   scope.
2. If no explicit scope is provided, resolve according to server policy, for
   example `user -> org -> server`.
3. Record which scope satisfied the lookup for audit/debug visibility, but never
   expose the secret value itself.

This implies ELSPETH needs runtime support for fetching profile-scoped,
organization-scoped, and server-scoped secrets during validation and execution.

### Security Rules

- No tool returns the plaintext secret to the assistant.
- No web API returns the plaintext secret to the browser after submission.
- Secret entry UI is write-only and clears immediately after successful
  submission.
- Secret inventory UI may continue to show that a named secret reference is
  available after submission, but never its value.
- Validation errors should identify missing or unusable references without
  leaking values.
- Secret-backed mutations should remain auditable as references and scopes, not
  as raw secret material.
- The assistant may enumerate usable secret references, but it must never see
  secret contents.
- Secret resolution happens server-side at use time; the browser never receives
  resolved secret material.

---

## Preview and Execution

These are not required for the minimal assistant authoring MVP, but they are
high-leverage.

### `preview_node_output(payload)`

Runs a node on sample rows and returns:

- transformed rows
- schema changes
- warnings
- errors

### `preview_pipeline(payload)`

Runs the whole pipeline on sample data or a source-limited subset and returns:

- sample outputs
- warnings
- errors

### `run_pipeline(payload)`

Starts execution.

### `get_run_status(run_id)`

Returns current run status and summary.

---

## Session and Forking Surface

The conversational UX needs explicit branch semantics.

### `fork_session_from_message(message_id)`

Creates a new authoring session that preserves chat history and composition state
through the chosen message boundary, then resumes from there. This is the
correct replacement for “roll back and resubmit.”

### `fork_session_from_version(version_id)`

Alternative fork point when the user’s intent is version-centric rather than
message-centric.

### Why Forking, Not Rollback

- prior history remains auditable
- executed jobs are not destructively invalidated
- user intent is preserved as branching rather than mutation
- the assistant can explain the fork naturally: “I created a new branch from
  before the bad step and continued from there”

---

## Validation Model

Every mutation should return:

```json
{
  "updated_resource": {},
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": [],
    "suggestions": []
  }
}
```

### Error Categories

- structural configuration errors
- DAG contract errors
- provider readiness errors
- secret reference errors
- artifact resolution errors

### Warnings

Warnings are first-class. Examples:

- selected model/provider is unavailable to the runtime
- sink path extension does not match configured format
- `schema: observed` may allow field drift
- route/output exists but is currently unconnected
- preview sample may exceed model context budget

### Suggestions

Suggestions are non-binding but helpful to assistant repair loops:

- recommended source plugin from artifact type
- inferred required input fields from template analysis
- suggested sink names or route repairs

---

## Recommended Minimal V1

The smallest coherent assistant-first surface is:

### Discovery

- `list_sources`
- `list_transforms`
- `list_sinks`
- `get_plugin_schema`
- `get_expression_grammar`

### Pipeline

- `get_pipeline_state`
- `set_metadata`
- `set_source`
- `upsert_node`
- `upsert_edge`
- `set_output`
- `remove_node`
- `remove_edge`
- `remove_output`
- `validate_pipeline`

### Artifacts

- `create_text_artifact`
- `create_json_artifact`
- `create_csv_artifact`
- `list_artifacts`
- `delete_artifact`
- `set_source_from_artifact`

### Secrets

- `list_secret_refs`
- `validate_secret_ref`

This is enough to make the assistant materially useful without requiring users
to handle files, paths, or credentials manually.

---

## Recommended V2

Add:

- `patch_source_options`
- `patch_node_options`
- `patch_output_options`
- `set_pipeline`
- `apply_pipeline_patch`
- `list_models`
- `preview_node_output`
- `preview_pipeline`
- `infer_schema_from_artifact`
- `explain_validation_error`
- `fork_session_from_message`
- `fork_session_from_version`

This turns the surface from “capable authoring API” into a genuinely strong
natural-language pipeline authoring substrate.

---

## Example Workflow

User says:

> Here’s some text. Build a pipeline that finds all rules related to layers and save the result as JSON.

Assistant workflow:

1. `create_text_artifact`
2. optionally `create_json_artifact` if row-wrapping is needed
3. `get_plugin_schema` for `json` source, `llm` transform, and `json` sink
4. `set_source_from_artifact`
5. `upsert_node` for the LLM transform using `api_key: {secret_ref: "OPENROUTER_API_KEY"}`
6. `set_output`
7. `upsert_edge` or `auto_wire_connections`
8. `validate_pipeline`
9. optionally `preview_pipeline`
10. explain the resulting pipeline in user language

The user never needs to manually save a file, stage a blob path, or paste raw
credentials.

---

## Open Questions

- Should artifacts be session-scoped, user-scoped, or both?
- Should `org` and `server` secret creation both be exposed in the web UX in
  v1, or should `server` provisioning remain out-of-band initially?
- Is transaction support still needed if `set_pipeline` and forking exist?
- Should preview use a separate worker/process boundary from execution in the
  first implementation, or can it remain in-process initially?
- What minimum metadata is needed on artifacts to safely support future blob
  manager UX?

---

## Recommendation

Engineering should implement this in three layers:

1. Full-fidelity pipeline CRUD
   - source/node/output `options` included
2. Artifact staging
   - create/list/delete/bind text/json/csv/binary artifacts
3. Safe secret references
   - assistant sees names and capabilities, never raw values

Then add:

4. richer validation responses
5. preview/sample execution
6. session forking primitives

That combination makes the assistant a true natural-language interface for
ELSPETH pipeline authoring instead of a partial planner that still requires
manual operational glue.
