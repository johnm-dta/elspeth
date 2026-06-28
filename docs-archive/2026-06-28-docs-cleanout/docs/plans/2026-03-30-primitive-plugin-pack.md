# Primitive Plugin Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the remaining universally useful primitive plugins: `null` sink, `console` sink, `text` sink, `sqlite` sink, and `template_text` transform.

**Architecture:** Keep primitives infrastructure-light and composition-friendly. The sinks should only depend on local process resources (`stdout`, filesystem, local SQLite file) or no external system at all (`null`), while the `template_text` transform should render Tier 2 pipeline rows into a single text field and leave persistence to an ordinary sink. Shared behavior should be reused where it already exists in plugin infrastructure, especially typed config validation, schema contracts, display-header handling where applicable, and sink diversion semantics.

**Tech Stack:** Python, Pydantic config models, pluggy discovery, SQLAlchemy Core, Jinja2 template helpers already used elsewhere in ELSPETH, pytest.

**Prerequisites:**
- Work on a clean branch after the other active agent finishes and lands its changes.
- Run in the project virtualenv: `.venv/bin/python`, `.venv/bin/pytest`.
- Preserve current plugin discovery and web catalog patterns; no special registration files should be introduced.

---

## Scope

This plan covers:
- `null` sink: valid terminal sink that intentionally discards rows
- `console` sink: write rows to stdout/stderr in human-friendly or machine-friendly formats
- `text` sink: write one configured field per row as line-oriented plain text
- `sqlite` sink: local embedded database sink for persistent, queryable outputs
- `template_text` transform: render each input row through a validated template into a configured output field

This plan does not cover:
- Vendor-specific sinks
- Rich report-generation sinks
- Templated blob-path / multipart output logic
- A generic “templated sink” abstraction

## Design Principles

1. Primitive means “universally applicable,” not “featureless.”
2. Source/transform/sink responsibilities stay clean:
   - sources ingest external data
   - transforms shape pipeline data
   - sinks persist or emit data
3. Template rendering belongs in a transform, not a sink.
4. Sinks do not coerce pipeline data. Wrong types at sink boundaries are upstream bugs.
5. All probative write behavior must remain audit-first; logging is not a substitute for audit records.

## Plugin Contracts and Runtime Semantics

### `null` sink

- Purpose: satisfy graph terminal requirements while intentionally producing no external artifact of value.
- Determinism: inherit `IO_WRITE` from `BaseSink`.
- Write behavior:
  - accept any batch
  - return a `SinkWriteResult` with a synthetic `ArtifactDescriptor`
  - produce no row diversions
- Validation:
  - schema still required through normal sink config patterns so the sink participates in existing validation and required-field machinery
  - `validate_output_target()` always succeeds
- Resume:
  - supported trivially; resume mode has no behavioral change

### `console` sink

- Purpose: emit rows to stdout or stderr for debugging, shell composition, demos, and ad hoc inspection.
- Formats:
  - `jsonl`: one JSON object per line
  - `text`: one rendered line per row from a configured field
  - `pretty`: compact human-readable row dump intended for debugging only
- Write behavior:
  - emit synchronously in `write()`
  - artifact hash should reflect the bytes actually emitted for the batch
- Validation:
  - for `text` mode require a configured `field`
  - for `jsonl` / `pretty`, field optional
- Resume:
  - supported, but semantically append-only by nature

### `text` sink

- Purpose: write a single field per row to a line-oriented text file.
- Config:
  - `path`
  - `field`
  - `mode: write|append`
  - `encoding`
  - `line_ending`
  - optional `skip_missing_field` is **not** recommended; missing field should fail fast as an upstream bug unless explicitly designed as diversion behavior
- Write behavior:
  - each row contributes one line
  - no header concepts
  - blank strings are allowed and should yield blank lines
- Validation:
  - configured field must be valid identifier
  - sink input schema still applies
- Resume:
  - supported via append mode

### `sqlite` sink

- Purpose: local durable tabular storage without external infrastructure.
- Positioning:
  - separate from existing `database` sink because it is a first-class primitive with tighter defaults and lower operational burden
  - implemented either as a thin dedicated wrapper over current SQLAlchemy table-writing logic or as a factored extraction from `database_sink.py`
- Config:
  - `path`
  - `table`
  - `if_exists: append|replace`
  - schema
  - `validate_input`
- Write behavior:
  - create database file if missing
  - create table on first write
  - infer columns from schema / first row following the same infer-and-lock semantics as the current database sink
- Resume:
  - supported via append mode

### `template_text` transform

- Purpose: render row data into a single text output field using a Jinja2 template.
- Inputs:
  - `template`
  - `output_field`
  - `required_input_fields`
  - schema
- Output:
  - preserves existing row fields
  - adds one declared output field containing rendered text
- Template lifecycle:
  - parse once at init time
  - render per row
  - structural template errors fail pipeline setup
  - operational render failures return `TransformResult.error()` and quarantine the row
- Contract behavior:
  - must declare `declared_output_fields = frozenset({output_field})`
  - must publish `_output_schema_config` with the added guaranteed field for DAG validation

## Shared Plumbing to Reuse or Extend

- Discovery: automatic via `src/elspeth/plugins/sinks/*.py` and `src/elspeth/plugins/transforms/*.py`
- Validator mappings:
  - `src/elspeth/plugins/infrastructure/validation.py`
- Catalog/web schema exposure:
  - `src/elspeth/web/catalog/service.py` relies on validator mappings, so wired config models are mandatory
- Base contracts:
  - `src/elspeth/plugins/infrastructure/base.py`
  - `src/elspeth/plugins/infrastructure/config_base.py`
- Template references:
  - existing LLM/template helpers and `elspeth.core.templates.extract_jinja2_fields()`
- Artifact and diversion contracts:
  - `src/elspeth/contracts/diversion.py`
  - `src/elspeth/contracts/results.py`

## Recommended Delivery Order

1. `null` sink
2. `console` sink
3. `text` sink
4. `template_text` transform
5. `sqlite` sink

Rationale:
- `null` and `console` are tiny and unlock immediate UX wins.
- `text` pairs naturally with the new `text` source and the planned transform.
- `template_text` increases composability without introducing infrastructure.
- `sqlite` is the heaviest primitive and should be informed by any refactoring pressure discovered while implementing the simpler sinks.

## Task 1: Add `null` sink

**Files:**
- Create: `src/elspeth/plugins/sinks/null_sink.py`
- Modify: `src/elspeth/plugins/infrastructure/validation.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/web/catalog/test_routes.py`
- Create: `tests/unit/plugins/sinks/test_null_sink.py`

**Implementation notes:**
- Use `DataPluginConfig` directly or a tiny dedicated config with only `schema` and `validate_input: bool = False`.
- Return a deterministic artifact descriptor representing a no-op write, not `None`.
- Keep the sink explicit and auditable: “discarded N rows intentionally” belongs in artifact metadata / success semantics, not logs.

**Tests:**
- create sink successfully
- write empty and non-empty batches
- no filesystem side effects
- catalog exposes `null`
- resume support behaves as no-op

**Run:**
- `.venv/bin/pytest -q tests/unit/plugins/sinks/test_null_sink.py tests/unit/web/catalog/test_service.py tests/unit/web/catalog/test_routes.py`

**Definition of Done:**
- [ ] `null` sink is discoverable and validated
- [ ] sink can terminate a pipeline without external output
- [ ] tests cover write contract and catalog exposure

## Task 2: Add `console` sink

**Files:**
- Create: `src/elspeth/plugins/sinks/console_sink.py`
- Modify: `src/elspeth/plugins/infrastructure/validation.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/web/catalog/test_routes.py`
- Create: `tests/unit/plugins/sinks/test_console_sink.py`

**Implementation notes:**
- Config model should include:
  - `schema`
  - `format: Literal["jsonl", "text", "pretty"]`
  - `field: str | None`
  - `stream: Literal["stdout", "stderr"] = "stdout"`
  - `encoding: str = "utf-8"` only if bytes conversion is handled explicitly
- `text` mode should require `field`.
- Capture exact emitted content in tests with `capsys`.
- Avoid adding bespoke logging. Console output is the sink artifact itself.

**Tests:**
- `jsonl` writes one JSON object per line
- `text` mode writes only selected field
- `pretty` mode produces stable human-readable output
- `stderr` selection works
- field validation errors surface clearly

**Run:**
- `.venv/bin/pytest -q tests/unit/plugins/sinks/test_console_sink.py tests/unit/web/catalog/test_service.py tests/unit/web/catalog/test_routes.py`

**Definition of Done:**
- [ ] console output is deterministic enough for tests per format
- [ ] sink is usable for debugging and shell pipelines
- [ ] no row-level logging duplicates sink behavior

## Task 3: Add `text` sink

**Files:**
- Create: `src/elspeth/plugins/sinks/text_sink.py`
- Modify: `src/elspeth/plugins/infrastructure/validation.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/web/catalog/test_routes.py`
- Create: `tests/unit/plugins/sinks/test_text_sink.py`

**Implementation notes:**
- Prefer a dedicated config inheriting from `PathConfig`, not `SinkPathConfig`, because header handling is irrelevant.
- Required config:
  - `path`
  - `field`
  - `schema`
- Optional config:
  - `mode: write|append`
  - `encoding`
  - `line_ending`
  - `validate_input`
- Missing configured field in a row should raise or divert consistently; default design should be fail-fast because sinks receive trusted pipeline schema.

**Tests:**
- truncating write mode
- append mode
- blank lines preserved
- UTF-8 output
- missing field behavior
- resume config switches to append

**Run:**
- `.venv/bin/pytest -q tests/unit/plugins/sinks/test_text_sink.py tests/unit/web/catalog/test_service.py tests/unit/web/catalog/test_routes.py`

**Definition of Done:**
- [ ] `text` sink can round-trip simple line-oriented workflows
- [ ] append/resume behavior is explicit and tested
- [ ] no header/display-header infrastructure leaks into the design

## Task 4: Add `template_text` transform

**Files:**
- Create: `src/elspeth/plugins/transforms/template_text.py`
- Modify: `src/elspeth/plugins/infrastructure/validation.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/web/catalog/test_routes.py`
- Create: `tests/unit/plugins/transforms/test_template_text.py`
- Consider reference updates in: `src/elspeth/web/composer/prompts.py` if transform examples are curated there

**Implementation notes:**
- Config should inherit from `TransformDataConfig`.
- Required fields:
  - `template`
  - `output_field`
- Optional fields:
  - `required_input_fields`
  - `validate_input`
- Parse template once in `__init__`.
- Use contract-aware field extraction or existing template helpers to guide `required_input_fields` behavior.
- Set `declared_output_fields` from configured `output_field`.
- Build `_output_schema_config` so downstream DAG validation sees the new field.

**Tests:**
- successful render adds output field
- existing row data preserved
- structural template syntax error fails init
- missing row field during render returns transform error
- declared output field collision prevented by executor-level checks in integration coverage
- required input fields surface in config model and catalog

**Run:**
- `.venv/bin/pytest -q tests/unit/plugins/transforms/test_template_text.py tests/unit/web/catalog/test_service.py tests/unit/web/catalog/test_routes.py`

**Definition of Done:**
- [ ] template rendering behavior matches ELSPETH template error taxonomy
- [ ] transform is composable with `text`, `console`, and `sqlite` sinks
- [ ] DAG/schema propagation for the new field is covered

## Task 5: Add `sqlite` sink

**Files:**
- Create: `src/elspeth/plugins/sinks/sqlite_sink.py`
- Modify: `src/elspeth/plugins/infrastructure/validation.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/web/catalog/test_routes.py`
- Create: `tests/unit/plugins/sinks/test_sqlite_sink.py`
- Consider refactor target: `src/elspeth/plugins/sinks/database_sink.py`

**Implementation notes:**
- Start with a design decision:
  - either thin wrapper over refactored shared table-writing helpers
  - or copy-minimize by extracting reusable internal helpers from `database_sink.py`
- Keep user-facing config primitive:
  - `path`
  - `table`
  - `if_exists`
  - `schema`
  - `validate_input`
- Internally derive SQLAlchemy URL from `path` (`sqlite:///...`), not from a raw generic database URL.
- Preserve infer-and-lock semantics for observed/flexible schemas.
- Ensure complex `any`-typed values follow existing canonical serialization expectations rather than inventing a new encoding.

**Tests:**
- create database file and table
- append and replace modes
- observed schema first-row inference
- flexible/fixed schema handling
- persisted rows query correctly via sqlite
- resume uses append

**Run:**
- `.venv/bin/pytest -q tests/unit/plugins/sinks/test_sqlite_sink.py tests/unit/web/catalog/test_service.py tests/unit/web/catalog/test_routes.py`

**Definition of Done:**
- [ ] `sqlite` sink is materially simpler to use than generic `database`
- [ ] no remote DB configuration is required
- [ ] schema/type behavior matches existing sink conventions

## Task 6: Cross-plugin integration and documentation pass

**Files:**
- Modify: `src/elspeth/web/composer/prompts.py` if curated examples need updating
- Modify: `src/elspeth/web/composer/tools.py` if new source/sink examples or defaults are added
- Modify: user-facing docs where primitive plugins are enumerated
- Create or modify: targeted integration tests under `tests/integration/plugins/` or `tests/e2e/pipelines/`

**Integration scenarios:**
- `text source -> console sink`
- `text source -> template_text transform -> text sink`
- `text source -> template_text transform -> sqlite sink`
- `csv/json source -> null sink`

**Run:**
- `.venv/bin/pytest -q tests/integration/plugins tests/e2e/pipelines`

**Definition of Done:**
- [ ] primitive pack is documented as a coherent set
- [ ] at least one production-path integration test exercises `template_text`
- [ ] web/catalog visibility is complete for all new plugins

## Open Design Decisions

### 1. Should `console` sink support `failsink` diversion?

Recommendation: no initial diversion behavior. Console output is local process I/O and should either succeed or fail the batch. Add diversions later only if a concrete per-row formatting rejection use case appears.

### 2. Should `text` sink accept non-string field values?

Recommendation: yes, but only if they already satisfy pipeline schema and convert via ordinary `str(value)` at the final emission boundary. Do not add sink-side coercion policy beyond final formatting.

### 3. Should `sqlite` replace `database` long term?

Recommendation: no immediate replacement. Treat `sqlite` as a primitive convenience wrapper first. Revisit deduplication after implementation stabilizes.

### 4. Should `template_text` support multiple templates or header/footer wrappers?

Recommendation: no. Single-row single-template output only. Anything richer belongs in later, more specialized transforms.

## Risks and Mitigations

- **Risk:** `sqlite` duplicates too much of `database_sink.py`.
  - Mitigation: allow a small internal helper extraction if duplication exceeds trivial wrapper size.
- **Risk:** `template_text` accidentally reintroduces sink-style rendering concerns.
  - Mitigation: keep output to one new field only.
- **Risk:** `console` output formats become unstable for tests.
  - Mitigation: keep `pretty` intentionally minimal and document that `jsonl` is the machine-stable format.
- **Risk:** `null` sink returns weak artifact metadata.
  - Mitigation: define explicit artifact semantics up front and test them.

## Verification Checklist

- [ ] Each new plugin has explicit `plugin_version`
- [ ] Each new plugin is wired into `PluginConfigValidator`
- [ ] Each new plugin appears in web catalog routes/service tests
- [ ] No plugin bypasses production discovery or executor paths in tests
- [ ] Template behavior follows structural-vs-operational error split
- [ ] Sink behaviors remain audit-first and do not add redundant row logging

## Suggested Execution Strategy

- Land `null`, `console`, and `text` as one low-risk primitive sink batch if merge pressure is low.
- Land `template_text` separately because it touches DAG/schema semantics.
- Land `sqlite` last, potentially after extracting shared DB-table helpers if implementation reveals worthwhile reuse.

Plan complete and saved to `docs/plans/2026-03-30-primitive-plugin-pack.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task with code review between tasks for fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans for batch execution with checkpoints

**Which approach?**
