# Bug Report: PluginSpec computes schema hashes from plugin classes, but built-in schemas are instance/config-driven (hashes become `None`)

## Summary

- `PluginSpec.from_plugin(...)` accepts a *plugin class* and computes `input_schema_hash`/`output_schema_hash` via `getattr(plugin_cls, "input_schema", None)` / `getattr(plugin_cls, "output_schema", None)`.
- Built-in plugins (sources/transforms/sinks) set `input_schema`/`output_schema` on the *instance* during `__init__` based on config (`create_schema_from_config(...)`), so the class typically has no schema attributes.
- If/when PluginSpec is used for audit metadata, schema hashes will be `None` (or incomplete), defeating compatibility/change detection.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Instantiate a built-in transform (e.g., `FieldMapper`) with a config-driven schema.
2. Observe that `transform.input_schema` exists on the instance, but `FieldMapper.input_schema` is typically not a concrete class attribute.
3. Call `PluginSpec.from_plugin(FieldMapper, node_type=...)` and observe schema hashes are `None` (because `getattr` returns `None`).

## Expected Behavior

- Schema hashes should reflect the actual schemas used by a plugin instance in a specific run/configuration.

## Actual Behavior

- Schema hashes are computed from class-level attributes and can be missing for config-driven schemas.

## Evidence

- PluginSpec reads schemas from the plugin class via `getattr(...)`: `src/elspeth/plugins/manager.py:77-107`
- Built-in plugins set schema attributes per instance during `__init__`:
  - `CSVSource`: `self.output_schema = self._schema_class`: `src/elspeth/plugins/sources/csv_source.py:71-82`
  - `FieldMapper`: `self.input_schema = schema; self.output_schema = schema`: `src/elspeth/plugins/transforms/field_mapper.py:64-73`
  - `CSVSink`: `self.input_schema = self._schema_class`: `src/elspeth/plugins/sinks/csv_sink.py:69-82`

## Impact

- User-facing impact: low today (PluginManager appears unused), but schema hashes are critical for future audit/compatibility features.
- Data integrity / security impact: missing schema hashes reduce ability to verify compatibility changes and reproduce runs.
- Performance or cost impact: potential future debugging/repro cost.

## Root Cause Hypothesis

- PluginSpec was designed around static class-level schemas, but the system has moved to config-driven schema generation at instance construction time.

## Proposed Fix

- Code changes (modules/files):
  - Change `PluginSpec.from_plugin` to accept a plugin *instance* (or accept schemas directly) so it can hash the real schema classes.
  - Alternatively, compute schema hashes from `schema_config` deterministically (and store `schema_config` hash) rather than introspecting Pydantic models.
- Config or schema changes: none.
- Tests to add/update:
  - Add a unit test that `PluginSpec` schema hashes are non-None when using config-driven schemas.
- Risks or migration steps:
  - Changing API signature is breaking; consider adding a new method `from_instance(...)` and deprecating `from_plugin(...)`.

## Architectural Deviations

- Spec or doc reference: `docs/arch-analysis-2026-01-20-0105/02-subsystem-catalog.md` (PluginSpec intended for Landscape node metadata)
- Observed divergence: schema hashing path doesn’t align with config-driven schemas.
- Reason (if known): refactor from static schemas to config-driven generation.
- Alignment plan or decision needed: decide canonical representation for schema identity (Pydantic introspection vs schema_config hash).

## Acceptance Criteria

- Schema hashes (or a replacement) exist and are stable/deterministic for config-driven plugin instances.

## Tests

- Suggested tests to run: `pytest tests/`
- New tests required: yes

## Notes / Links

- Related ticket: `docs/bugs/open/2026-01-19-plugin-registries-drift-and-unused-plugin-manager.md`
