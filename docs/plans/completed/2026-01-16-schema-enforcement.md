# Optional Schema Enforcement (Sources / Transforms / Sinks)

Status: Draft
Owner: TBD
Date: 2026-01-16

## Problem Statement

ELSPETH has plugin schema declarations (`input_schema`/`output_schema`) and
compatibility utilities, but nothing enforces them. Pipelines can compile and
run with incompatible shapes, and failures surface late or silently. We need
an optional schema enforcement system that:
- Validates pipeline compatibility at config time.
- Enforces runtime guards at source/transform/gate/sink boundaries.
- Preserves current behavior when disabled.

## Goals

- Optional schema enforcement via configuration (default off).
- Config-time schema compatibility checks across the execution graph.
- Runtime validation of rows at plugin boundaries with clear error messages.
- Record schema fingerprints in the audit trail for traceability.

## Non-goals

- Automatic schema inference from data.
- Changing plugin configuration validation (this is about row schemas only).
- Full DAG fork/join schema reconciliation (beyond current linear engine).

## Proposed Design

### 1) Configuration + Policy

Add a new settings block in `ElspethSettings`:

- `schema_validation.enabled: bool = False`
- `schema_validation.compatibility_mode: "off" | "warn" | "error"`
- `schema_validation.runtime_mode: "off" | "warn" | "error"`
- `schema_validation.on_error: "fail_run" | "fail_row"` (default `fail_run`)

This keeps enforcement optional while allowing stricter modes for production.

### 2) Schema Registry + Resolution

Unify schema discovery with plugin lookup:
- Use plugin classes (not instances) to read `input_schema`/`output_schema`.
- Centralize plugin resolution (source/transform/gate/sink) so the compiler
  can inspect schemas without instantiation.
- For SinkAdapter, expose the wrapped sink's `input_schema` to validation.

### 3) Schema Compilation (Config-Time)

Introduce a `SchemaCompiler` (new module under `core/`):
- Inputs: `ElspethSettings`, plugin registry, `ExecutionGraph`.
- Output: `SchemaPlan` with node schemas and edge compatibility results.

Checks to perform:
- Source output -> first transform input (or output sink if no transforms).
- Transform output -> next transform input.
- Gate output -> sink input for each routed edge.
- Last transform output -> output sink input.
- Sinks with multiple producers validate against each upstream edge.

Compatibility uses `check_compatibility` and reports missing fields + type
mismatches with node and edge context.

Dynamic/unknown schemas:
- Provide a built-in `AnySchema` (or marker) that treats all fields as
  compatible for config-time checks.
- Allow explicit overrides in config if the source schema is known but the
  plugin is dynamic.

### 4) Runtime Enforcement (Guards)

Add schema checks at runtime boundaries using `validate_row`:
- **Source**: validate each row against `output_schema` before token creation.
- **Transform/Gate**: validate input before execution; validate output on success.
- **Aggregation**: validate input rows on accept; validate outputs on flush.
- **Sink**: validate each row before writing.

Error handling based on `schema_validation.on_error`:
- `fail_run`: raise immediately to fail the run.
- `fail_row`: convert into an error result for transforms (and record a failed
  node state), and skip writing to sinks for invalid rows.

When runtime validation is `warn`, record the error in logs but continue.

### 5) Audit Trail Integration

Record schema hashes when registering nodes:
- Use the existing schema hash function in plugin manager.
- Store `input_schema_hash`/`output_schema_hash` on node registration.
- Include schema mismatch details in node_state errors when validation fails.

This provides evidence of expected shape alongside actual data history.

## Risks / Open Questions

- **Dynamic schemas**: CSV/JSON sources currently accept any fields. We need
  a clear policy for when compatibility checks should pass (e.g., `AnySchema`).
- **Failure policy**: `fail_row` for sinks is ambiguous; do we skip or reroute?
  A future quarantine sink could make this explicit.
- **Performance**: validation adds overhead; should support sampling or only
  validate N rows in `warn` mode.

## Test Plan

- Config-time validation:
  - Compatible pipeline passes.
  - Missing required fields fails with actionable error.
  - Type mismatch errors include edge context.
- Runtime enforcement:
  - Source row mismatch fails/warns per policy.
  - Transform output mismatch produces error node_state.
  - Sink rejects invalid rows and fails/warns per policy.
- Settings parsing for new schema_validation block.

## Implementation Phases

1) Add settings + plumbing (schema_validation config).
2) Implement SchemaCompiler + CLI integration for `validate` and `run`.
3) Runtime guards in executors + SinkAdapter schema passthrough.
4) Schema hash recording in node registration.
5) Tests and doc updates.
