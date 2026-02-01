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

## Resolution: DEAD CODE DELETED (2026-02-02)

**Status: CLOSED - CODE REMOVED**

### Investigation Findings

The bug report correctly identified a design flaw in `PluginSpec.from_plugin()`, but further investigation revealed a more significant issue: **`PluginSpec` and the `schema_hash` mechanism were completely dead code.**

1. **`PluginSpec` was never used in production:**
   - Only imported in tests (`test_manager.py`, `test_manager_validation.py`)
   - Never imported by engine, landscape, orchestrator, or any runtime code
   - The `nodes.schema_hash` column in the database was never populated

2. **The system had already evolved past this approach:**
   - Instead of hashing Pydantic schema classes, the orchestrator now stores `schema_mode` and `schema_fields_json` from `SchemaConfig` directly
   - This provides MORE detail than a hash and enables full schema reconstruction for audit purposes

3. **Tests didn't catch the bug:**
   - Tests used mock plugins with class-level schemas, hiding the design flaw
   - Real plugins set schemas on instances, which `PluginSpec.from_plugin()` couldn't access

### Resolution

Per CLAUDE.md's NO LEGACY CODE POLICY, the dead code was deleted rather than fixed:

**Deleted:**
- `PluginSpec` class from `src/elspeth/plugins/manager.py`
- `_schema_hash()` helper function from `src/elspeth/plugins/manager.py`
- `PluginSpec` export from `src/elspeth/plugins/__init__.py`
- `tests/plugins/test_manager_validation.py` (entire file - only tested PluginSpec)
- `TestPluginSpec` class from `tests/plugins/test_manager.py`
- `TestPluginSpecSchemaHashes` class from `tests/plugins/test_manager.py`

**Retained (no action needed):**
- `nodes.schema_hash` column - harmless NULL column, can be removed in future migration if desired
- `schema_mode` and `schema_fields_json` columns - the working replacement

### Why Delete Instead of Fix?

| Consideration | Delete | Fix |
|--------------|--------|-----|
| Current usage | Zero - never called in production | Would need to wire into orchestrator |
| Existing alternative | `schema_config` storage works and provides more detail | Would duplicate functionality |
| Audit value | Full schema fields > opaque hash for reconstruction | Hash only tells you "something changed" |
| CLAUDE.md policy | NO LEGACY CODE - delete unused code | N/A |

The current approach of storing `schema_mode` + `schema_fields_json` is strictly superior for audit purposes because it allows full schema reconstruction, not just change detection.
