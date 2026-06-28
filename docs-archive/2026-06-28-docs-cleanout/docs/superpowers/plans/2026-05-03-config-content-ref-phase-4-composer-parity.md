# Phase 4 — Composer Parity + Shape 9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open the composer authorship path. Extend `_prevalidate_plugin_options` strip-before-validate to recognise the widened `blob_ref` marker on any plugin field; wire `_validate_blob_content_refs` into `validate_pipeline`; add Shape 9 (sub-pins A, B, C) to the agreement-suite registry; apply bug-verification protocol at the composer-parity-strip fix site.

**Architecture:** Two strict-additive sites in the composer / validation layer. `web/composer/tools.py:1448-1480` (the secret-ref strip block) gets a sibling block stripping widened-blob_ref markers in `inline_content` mode. `web/execution/validation.py::validate_pipeline` adds a step that calls the validate-side helper. Shape 9 lands in `tests/integration/pipeline/test_composer_runtime_agreement.py` with the three sub-pins specified in spec §9.2.

**Tech Stack:** Pydantic 2.x, FastAPI, the existing `_prevalidate_plugin_options` and `validate_pipeline` infrastructure, pytest.

---

## Pre-phase verification

- [ ] **Step 1: Confirm P3 has merged**

```bash
git log --oneline --all | grep -E "feat\(execution\): wire inline-content resolver"
```

Expected: at least one commit with that message.

- [ ] **Step 2: Confirm runtime is fail-closed**

```bash
.venv/bin/python -m pytest tests/integration/web/test_blob_inline_runtime_preflight.py -v
```

Expected: PASS. The runtime side is honouring inline_content markers; P4 opens the path that creates them via /validate.

---

## Task 1: Extend `_prevalidate_plugin_options` strip-before-validate

**Files:**
- Modify: `src/elspeth/web/composer/tools.py:1448-1480`
- Test: `tests/unit/web/composer/test_prevalidate_blob_inline.py`

- [ ] **Step 1: Re-read the existing strip block**

```bash
sed -n '1440,1485p' src/elspeth/web/composer/tools.py
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/web/composer/test_prevalidate_blob_inline.py
"""Pin: _prevalidate_plugin_options strips widened-blob_ref(inline_content) markers
before Pydantic validation, matching the secret_ref strip pattern.
"""

import pytest

from elspeth.web.composer.tools import _prevalidate_transform


VALID_HASH = "a" * 64
BLOB_ID = "5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b"


def test_prevalidate_accepts_inline_content_marker_on_required_field():
    """A field that Pydantic would reject as missing should pass when wired
    via the inline_content marker — same behaviour as the existing
    secret_ref strip.
    """
    options = {
        "system_prompt": {
            "blob_ref": BLOB_ID,
            "mode": "inline_content",
            "sha256": VALID_HASH,
        },
        "model": "gpt-4",
        "api_key": {"secret_ref": "OPENROUTER_KEY"},  # existing strip path — must still work
    }
    error = _prevalidate_transform("llm", options)
    assert error is None, f"Expected no validation error, got: {error}"


def test_prevalidate_rejects_bind_source_in_transform_options():
    """bind_source is the source-only mode; using it in a transform option
    should NOT be stripped (it's not the inline_content shape).
    """
    options = {
        "system_prompt": {
            "blob_ref": BLOB_ID,
            "mode": "bind_source",
            "path": "/data/x",
        },
    }
    # Should propagate to Pydantic which will reject the dict-shaped
    # system_prompt as a non-string value.
    error = _prevalidate_transform("llm", options)
    assert error is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_prevalidate_blob_inline.py -v`
Expected: the inline-content test FAILS (Pydantic rejects the dict-shaped value).

- [ ] **Step 4: Add the inline_content strip block**

In `src/elspeth/web/composer/tools.py`, locate the existing `secret_ref_keys` block (currently at lines 1452-1456):

```python
# EXISTING:
secret_ref_keys: set[str] = set()
for key, value in list(merged.items()):
    if isinstance(value, Mapping) and len(value) == 1 and "secret_ref" in value and isinstance(value["secret_ref"], str):
        secret_ref_keys.add(key)
        del merged[key]
```

Add a sibling block immediately after:

```python
# NEW:
from elspeth.contracts.blobs_inline import is_widened_blob_ref  # Add to imports at top of file

blob_inline_ref_keys: set[str] = set()
for key, value in list(merged.items()):
    shape = is_widened_blob_ref(value)
    if shape is not None and shape.mode == "inline_content":
        blob_inline_ref_keys.add(key)
        del merged[key]
```

Update the Pydantic-error filter (currently at line 1474):

```python
# EXISTING:
remaining = [e for e in cause.errors() if not (e["loc"] and e["loc"][0] in secret_ref_keys)]

# REPLACE WITH:
stripped_keys = secret_ref_keys | blob_inline_ref_keys
remaining = [e for e in cause.errors() if not (e["loc"] and e["loc"][0] in stripped_keys)]
```

Update the `if not secret_ref_keys` short-circuit to consider both sets:

```python
# EXISTING:
if not secret_ref_keys:
    msg = exc.cause if exc.cause is not None else str(exc)
    return f"Invalid options for {plugin_type} '{plugin_name}': {msg}"

# REPLACE WITH:
if not secret_ref_keys and not blob_inline_ref_keys:
    msg = exc.cause if exc.cause is not None else str(exc)
    return f"Invalid options for {plugin_type} '{plugin_name}': {msg}"
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_prevalidate_blob_inline.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_prevalidate_blob_inline.py
git commit -m "feat(composer): strip widened-blob_ref(inline_content) before pre-validation

Mirrors the existing secret_ref strip-before-validate pattern at
web/composer/tools.py:1448-1480 — a field carrying the inline_content
marker IS provisioned (the user wired it via wire_blob_inline_ref or
direct YAML), just deferred to runtime resolution.  Per spec §7.1 of
elspeth-fdebcaa79a.

Refs: elspeth-fdebcaa79a"
```

---

## Task 2: Wire `_validate_blob_content_refs` into `validate_pipeline`

**Files:**
- Modify: `src/elspeth/web/execution/validation.py:208`
- Test: `tests/unit/web/execution/test_validate_blob_inline.py`

- [ ] **Step 1: Re-read the existing secret-ref validation block**

```bash
sed -n '331,400p' src/elspeth/web/execution/validation.py
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/web/execution/test_validate_blob_inline.py
"""Pin: validate_pipeline returns structured violations for missing
inline-content blobs, NOT a 500."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, SourceSpec
from elspeth.web.execution.validation import validate_pipeline


def test_validate_returns_structured_violation_for_missing_blob():
    state = CompositionState(
        source=SourceSpec(plugin="csv", on_success="out", options={"path": "x.csv"}, on_validation_failure="quarantine"),
        nodes=(NodeSpec(id="classify", plugin="llm", options={
            "system_prompt": {"blob_ref": str(uuid4()), "mode": "inline_content", "sha256": "a"*64},
            "api_key": {"secret_ref": "OPENROUTER_KEY"},
        }, on_success="out"),),
        outputs=(OutputSpec(name="out", plugin="json", options={"path": "out.json"}),),
    )
    blob_service = AsyncMock()
    blob_service.get_blob.side_effect = ...  # BlobNotFoundError
    secret_service = ...  # mock with the secret available
    result = validate_pipeline(
        state, settings=..., yaml_generator=..., secret_service=secret_service, user_id="u",
        blob_service=blob_service,
    )
    assert result.is_valid is False
    assert any("missing" in err.message.lower() for err in result.errors)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/execution/test_validate_blob_inline.py -v`
Expected: FAIL.

- [ ] **Step 4: Add the validation block**

In `src/elspeth/web/execution/validation.py::validate_pipeline`, locate the secret-ref validation block (lines 331-400) and add a sibling block after it that calls the L1 helper:

```python
# After the existing secret-ref block, before YAML generation:

# Step 1c: Inline-content blob validation per spec §6.4 / ADR-021
#
# IMPORTANT — async/sync boundary:
#
#   `_call_async` lives on `ExecutionServiceImpl` (web/execution/service.py:245)
#   and is NOT available inside `validate_pipeline` (sync function in
#   validation.py:208 with no service reference).  The existing
#   `secret_service` parameter follows the sync `WebSecretResolver` Protocol
#   pattern; the validate-side blob check uses the same shape: a SYNC
#   callable injected by the caller, never an async service handle.
#
#   The runtime path keeps async fetch (parallel reads of bytes pay).  The
#   validate path is metadata-only (one get_blob per ref, sequential is
#   fine), so a sync interface is both correct AND adequate.
#
#   The sync callable is constructed by the caller — `_validate_async` in
#   service.py:555-595, which already runs `validate_pipeline` via
#   `run_in_executor`.  Inside that wrapper, the caller bridges to the
#   async blob_service via `_call_async`, then passes a closure to
#   `validate_pipeline`.

if blob_get_metadata is not None:
    import yaml as _yaml

    from elspeth.core.blobs_inline import _validate_blob_content_refs_sync

    # Convert state → YAML → dict for the discoverer.  `generate_config_dict`
    # does not exist; round-trip through YAML so the dict shape matches what
    # the runtime resolver sees.
    pipeline_yaml = yaml_generator.generate_yaml(state)
    config_dict = _yaml.safe_load(pipeline_yaml)
    if not isinstance(config_dict, dict):
        # Tier 1 — yaml_generator must produce a dict at the top level
        raise TypeError(
            f"yaml_generator.generate_yaml produced non-dict YAML "
            f"(got {type(config_dict).__name__}) — bug in the YAML generator"
        )
    blob_violations = _validate_blob_content_refs_sync(
        blob_get_metadata, config_dict,
        per_ref_byte_cap=ADR_019_PER_REF_CAP,
        aggregate_byte_cap=ADR_019_AGGREGATE_CAP,
    )
    if blob_violations:
        for violation in blob_violations:
            errors.append(ValidationError(
                component_id=_violation_component_id(violation.field_path),
                component_type=_violation_component_type(violation.field_path),
                message=f"Inline content {violation.category}: {violation.detail}",
                suggestion="Verify the blob exists, is `ready`, and the composer-pinned hash matches.",
            ))
        # Continue with downstream checks; runtime would also fail-closed,
        # so /validate must surface the violations explicitly.
```

**Update `validate_pipeline` signature** to accept the sync callable:

```python
# src/elspeth/web/execution/validation.py:208 — extend signature

def validate_pipeline(
    state: CompositionState,
    settings: ValidationSettings,
    yaml_generator: YamlGenerator,
    *,
    secret_service: WebSecretResolver | None = None,
    user_id: str | None = None,
    blob_get_metadata: Callable[[UUID], BlobRecord | None] | None = None,  # NEW
) -> ValidationResult:
    ...
```

**Update the caller** at `web/execution/service.py:_validate_async` (currently around line 555-595) to construct the sync callable from the async `BlobServiceImpl`:

```python
# web/execution/service.py — inside _validate_async, before the
# `await loop.run_in_executor(...validate_pipeline...)` call

def _blob_get_metadata_sync(blob_id: UUID) -> BlobRecord | None:
    """Sync bridge from validate_pipeline (worker thread) to async
    BlobServiceImpl.  Calls _call_async on the parent loop to issue
    the metadata fetch.  Returns None on BlobNotFoundError so the
    validate-side helper can surface it as a structured violation."""
    if self._blob_service is None:
        return None
    try:
        return self._call_async(self._blob_service.get_blob(blob_id))
    except BlobNotFoundError:
        return None

# Then pass it to validate_pipeline:
result = await loop.run_in_executor(
    self._executor,
    lambda: validate_pipeline(
        state, settings, yaml_generator,
        secret_service=self._secret_service,
        user_id=user_id,
        blob_get_metadata=_blob_get_metadata_sync,
    ),
)
```

The sync callable closes over `self._call_async` and `self._blob_service`. From inside the worker thread, calling the closure invokes `_call_async`, which uses `run_coroutine_threadsafe` against the parent loop (the loop that ran `_validate_async`) — same bridge pattern the runtime path uses.

Add helper functions to derive component_id/component_type from a canonical field_path:

```python
def _violation_component_id(field_path: str) -> str:
    """source.options.X → 'source'; node:N.options.X → 'N'; output:S.options.X → 'S'"""
    if field_path.startswith("source."):
        return "source"
    if field_path.startswith("node:"):
        return field_path.split(".", 1)[0][len("node:"):]
    if field_path.startswith("output:"):
        return field_path.split(".", 1)[0][len("output:"):]
    return field_path  # fallback


def _violation_component_type(field_path: str) -> str:
    if field_path.startswith("source."):
        return "source"
    if field_path.startswith("node:"):
        return "transform"
    if field_path.startswith("output:"):
        return "sink"
    return "unknown"
```

Pull the cap constants from a new module-level constants block:

```python
# Top of validation.py
ADR_019_PER_REF_CAP: int  # populated from the spec/ADR — one-line module constant
ADR_019_AGGREGATE_CAP: int
```

The actual numeric values land via P1's ADR. Until P1 is merged, leave them as `# noqa: pending ADR-021` and the test uses smaller caps via direct kwarg override.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/execution/test_validate_blob_inline.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/execution/validation.py tests/unit/web/execution/test_validate_blob_inline.py
git commit -m "feat(validation): wire inline-content blob validation into validate_pipeline

Per spec §6.4 of elspeth-fdebcaa79a — validate_pipeline now calls
_validate_blob_content_refs (metadata-only, never raises operational
errors) and surfaces missing/oversized/hash_mismatch/not_ready cases as
structured ValidationError rows.  Mirrors the secret-ref validation
pattern at validation.py:331-400.

Refs: elspeth-fdebcaa79a"
```

---

## Task 3: Bug-verification at the composer-parity-strip fix site

Per spec §9.3 (Sub-pin A).

- [ ] **Step 1: Manually revert the `blob_inline_ref_keys` strip block**

In `src/elspeth/web/composer/tools.py`, comment out the inline-content strip block added in Task 1.

- [ ] **Step 2: Run the test that should now fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_prevalidate_blob_inline.py::test_prevalidate_accepts_inline_content_marker_on_required_field -v
```

Expected: FAIL — Pydantic rejects the dict-shaped `system_prompt` as a non-string value, surfacing as `composer_plugin_error` 500-class through the route layer.

Capture the actual exception class.

- [ ] **Step 3: Restore + document**

Re-add the strip block. Re-run (PASS). Add the bug-verification documentation to the test's module docstring.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/web/composer/test_prevalidate_blob_inline.py
git commit -m "test(composer): document bug-verification protocol for inline-content strip

Per the agreement-suite protocol — reverting the blob_inline_ref_keys
strip block in _prevalidate_plugin_options surfaces Pydantic errors
as composer_plugin_error 500 rather than a structured
ValidationResult.  The test pins the strip as load-bearing.

Refs: elspeth-fdebcaa79a"
```

---

## Task 4: Add Shape 9 to the agreement-suite registry

**Files:**
- Modify: `tests/integration/pipeline/test_composer_runtime_agreement.py` (module docstring + new test class)

- [ ] **Step 1: Update the module docstring**

Open `tests/integration/pipeline/test_composer_runtime_agreement.py`. The closed registry of shapes lives in the module docstring (lines 11-91 today). Add a new entry after Shape 8:

```python
"""...

* Shape 9 — Phase P4 of elspeth-fdebcaa79a (widened blob_ref /
  inline_content config-content-ref capability).  Closes
  elspeth-fdebcaa79a.  Pinned by ``TestComposerRuntimeBlobInlineAgreement``
  with three sub-pins:

  * Sub-pin A — composer ``/validate`` recognises the widened
    ``blob_ref`` marker on transform/sink/gate/aggregation options
    and returns ``BlobContentResolutionError`` cases as structured
    ``ValidationResult(is_valid=False)`` (NOT a 500).  Bug verification:
    revert the ``blob_inline_ref_keys`` extension at
    ``web/composer/tools.py::_prevalidate_plugin_options``; observe
    Pydantic ``ValidationError`` propagating as composer_plugin_error.

  * Sub-pin B — runtime preflight raises
    ``BlobContentResolutionError`` / ``BlobIntegrityError`` /
    ``BlobNotFoundError`` BEFORE the first row reaches plugin
    instantiation.  Bug verification: revert the resolver wiring in
    ``web/execution/service.py::_run_pipeline`` (the
    ``_call_async(_fetch_blob_contents(...))`` call); observe the
    pipeline crash on first-row plugin call with no audit record of
    the missing/oversized/mismatched ref.

  * Sub-pin C — audit row determinism.  The ``content_hash`` recorded
    in ``blob_inline_resolutions`` for a run equals the hash
    re-derived from the blob's stored bytes via
    ``BlobServiceImpl.read_blob_content`` in the same run.  Bug
    verification: revert the audit-write site in
    ``web/execution/service.py::_run_pipeline`` (the
    ``record_blob_inline_resolutions`` call); observe that an audit
    query of ``blob_inline_resolutions`` returns no row for a
    completed run that resolved a ref.
"""
```

- [ ] **Step 2: Add `TestComposerRuntimeBlobInlineAgreement`**

Append a new test class to the same file (mirror the structure of `TestComposerRuntimeSecretRefAgreement` at line ~2153):

```python
# tests/integration/pipeline/test_composer_runtime_agreement.py — append at end

# ── Shape 9 — widened blob_ref / inline_content agreement ──────────────────

class TestComposerRuntimeBlobInlineAgreement:
    """Shape 9 — composer /validate and runtime preflight agree on
    inline-content blob ref shape errors and successes.
    """

    def test_validate_accepts_well_formed_inline_content_ref(self, ...):
        # Sub-pin A — well-formed ref → /validate returns is_valid=True
        ...

    def test_validate_returns_structured_error_for_missing_blob(self, ...):
        # Sub-pin A — missing blob → 422 ValidationResult, NOT 500
        ...

    def test_runtime_fails_closed_on_hash_mismatch(self, ...):
        # Sub-pin B — runtime BlobIntegrityError before first row
        ...

    def test_audit_row_hash_round_trips(self, ...):
        # Sub-pin C — audit content_hash equals re-derived hash
        ...
```

Implement the test bodies following the patterns in `TestComposerRuntimeSecretRefAgreement` and Shape 8 (`TestComposerRuntimeFileSinkCollisionAgreement`).

- [ ] **Step 3: Run the agreement suite**

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeBlobInlineAgreement -v
```

Expected: PASS for every sub-pin.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/pipeline/test_composer_runtime_agreement.py
git commit -m "test(agreement): add Shape 9 — widened blob_ref / inline_content

Closes elspeth-fdebcaa79a in the agreement-suite registry.  Three
sub-pins (A, B, C) per spec §9.2; bug-verification protocol applied
inline (the production reverts and observed failure modes are
documented in each sub-pin's docstring).

Refs: elspeth-fdebcaa79a"
```

---

## Task 5: Open the P4 PR

- [ ] **Step 1: Push and open**

```bash
git push -u origin <branch-name>
gh pr create --title "feat(composer): validation parity for widened blob_ref + Shape 9 agreement" --body "$(cat <<'EOF'
## Summary

- Extends \`_prevalidate_plugin_options\` strip-before-validate to recognise widened-blob_ref(inline_content) markers (spec §7.1)
- Wires \`_validate_blob_content_refs\` into \`validate_pipeline\` returning structured violations (spec §6.4)
- Adds Shape 9 to \`tests/integration/pipeline/test_composer_runtime_agreement.py\` with sub-pins A, B, C (spec §9.2)
- Bug-verification at composer-parity-strip fix site documented (spec §9.3 sub-pin A)

The composer authorship path is now open: a marker authored via direct YAML or P5's MCP tool flows through /validate (recognised, hash-checked against blob metadata) and through /runs (resolved fail-closed with audit row written before plugin instantiation).

## Test plan

- [ ] \`pytest tests/unit/web/composer/test_prevalidate_blob_inline.py tests/unit/web/execution/test_validate_blob_inline.py -v\`
- [ ] \`pytest tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeBlobInlineAgreement -v\`
- [ ] \`enforce_tier_model.py check\` exits 0
- [ ] Manual: revert the \`blob_inline_ref_keys\` strip block in \`_prevalidate_plugin_options\` and confirm \`test_prevalidate_accepts_inline_content_marker_on_required_field\` fails with the documented exception class

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Done conditions

P4 is done when:

1. `_prevalidate_plugin_options` strips widened-blob_ref(inline_content) markers before Pydantic validation.
2. `validate_pipeline` calls `_validate_blob_content_refs` and surfaces violations as structured `ValidationError` rows.
3. Shape 9 (sub-pins A, B, C) is added to the agreement-suite registry and passes.
4. Bug-verification protocol applied + documented at the composer-parity-strip fix site.
5. PR is merged.

Move to `2026-05-03-config-content-ref-phase-5-composer-tool.md` only after P4 is merged.
