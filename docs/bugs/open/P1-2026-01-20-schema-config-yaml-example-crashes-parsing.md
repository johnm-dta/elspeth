# Bug Report: SchemaConfig YAML example (list-of-maps) crashes parsing with AttributeError

## Summary

- `SchemaConfig.from_dict()` expects `fields` to be a `list[str]` (e.g., `"id: int"`), but the module docstring example uses YAML list-of-maps (`- id: int`), which parses as `list[dict]` in YAML loaders.
- When `fields` contains dict entries, `FieldDefinition.parse()` is called with a `dict` and crashes with `AttributeError: 'dict' object has no attribute 'strip'`, bypassing the intended `ValueError`/`PluginConfigError` path.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into System 2 (Contracts) and write bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: local code inspection + minimal Python repro

## Steps To Reproduce

1. Use a schema config shaped like common YAML output for `- id: int` list items:
   - `{"mode": "strict", "fields": [{"id": "int"}]}`
2. Call `SchemaConfig.from_dict(...)`.

Minimal repro:

```python
from elspeth.contracts.schema import SchemaConfig

SchemaConfig.from_dict(
    {"mode": "strict", "fields": [{"id": "int"}]}
)
```

## Expected Behavior

- Either:
  - (A) `SchemaConfig.from_dict()` accepts `list[dict[str, str]]` and converts each `{name: type}` mapping into a field spec, OR
  - (B) it rejects the format with a clear `ValueError` explaining the expected shape (and plugin config loading wraps it as `PluginConfigError`).
- Docs and examples show the supported YAML syntax (and won’t lead users into a hard crash).

## Actual Behavior

- Raises `AttributeError: 'dict' object has no attribute 'strip'` during parsing, producing an unhandled exception instead of a user-facing config validation error.

## Evidence

- The module docstring example uses list-of-maps YAML entries:
  - `src/elspeth/contracts/schema.py:10` (example shows `- id: int`)
- `SchemaConfig.from_dict()` assumes `fields_value` is `list[str]` and passes items directly to `FieldDefinition.parse()`:
  - `src/elspeth/contracts/schema.py:196`
- `FieldDefinition.parse()` assumes `spec` is a string and calls `.strip()`:
  - `src/elspeth/contracts/schema.py:67`
- Plugin config parsing only catches `ValidationError` and `ValueError`, not `AttributeError`, so this escapes the normal error surface:
  - `src/elspeth/plugins/config_base.py:59-72`

## Impact

- User-facing impact: schema configuration errors surface as raw crashes with poor messaging; the doc example is likely to produce this immediately.
- Data integrity / security impact: low (configuration-time failure), but it blocks pipelines from running.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `SchemaConfig.from_dict()` does not validate the element type of `fields` before calling `FieldDefinition.parse()`, and the docstring example suggests a YAML syntax that produces dicts rather than strings.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/contracts/schema.py`: In `SchemaConfig.from_dict()`, validate that each `fields` entry is a `str`; if an entry is a `dict`, accept the common YAML map form by converting it to a spec string (e.g., `{"id": "int"}` → `"id: int"`). If neither, raise `ValueError` with a clear message.
  - `src/elspeth/contracts/schema.py`: Update the docstring example to match supported syntax (e.g., `- "id: int"`), even if dict-form support is added.
- Tests to add/update:
  - Add tests covering:
    - list-of-strings form parses successfully
    - list-of-dicts form either parses successfully (if supported) or raises `ValueError` (not `AttributeError`)
    - plugin config parsing wraps the error as `PluginConfigError` with helpful context

## Architectural Deviations

- Spec or doc reference: `src/elspeth/contracts/schema.py` module docstring example
- Observed divergence: documented YAML form does not match accepted config shape
- Alignment plan or decision needed: decide whether to support both list-of-strings and list-of-maps; either way, ensure errors are `ValueError` with actionable message

## Acceptance Criteria

- Configs using `schema: {mode: strict, fields: ["id: int"]}` parse successfully.
- Configs using list-of-maps YAML form (e.g., `- id: int`) either parse successfully or fail with a clear `ValueError` (no `AttributeError` leaks).
- Error surfaced through `PluginConfig.from_dict()` is a `PluginConfigError` with a useful message.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/ -k schema_config`
- New tests required: yes

## Notes / Links

- Related design docs: `docs/contracts/plugin-protocol.md` (schema configuration)
