# Phase 5 — Composer MCP Tool Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two MCP tools that let the chat-driven LLM author widened-blob_ref(inline_content) markers without ever seeing the underlying bytes — `list_composer_blobs` (LLM blob discovery, exact H4 visibility shape as return contract) and `wire_blob_inline_ref` (authorship). Update `set_source_from_blob` to emit explicit `mode: bind_source` so every persisted marker round-trips through the recognition function. Enforce composer-side rejection rules per spec §7.2.

**Architecture:** Two new MCP tool functions registered in the composer's tool dispatcher (`web/composer/tools.py`) with structured Pydantic-typed signatures. `list_composer_blobs` queries `BlobServiceImpl.list_blobs` and projects the result to the H4 visibility shape. `wire_blob_inline_ref` reads `BlobRecord.content_hash` and constructs the widened marker, then writes it into `composition_states.<source|nodes|outputs>.options.<field_path>`. `set_source_from_blob` emits `mode: bind_source` explicitly.

**Tech Stack:** MCP tool registration pattern, Pydantic 2.x, `CompositionState` mutation via `with_*` helpers (frozen-dataclass-style copy-on-write), pytest.

---

## Pre-phase verification

- [ ] **Step 1: Confirm P4 has merged**

```bash
git log --oneline --all | grep -E "feat\(composer\): validation parity for widened blob_ref"
```

- [ ] **Step 2: Confirm composer/runtime agreement still holds**

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeBlobInlineAgreement -v
```

Expected: PASS for every sub-pin. P5 must not regress agreement.

- [ ] **Step 3: Locate the existing MCP tool registry**

```bash
grep -n "set_source_from_blob\|wire_secret_ref\|mcp.tool\|@tool\b" src/elspeth/web/composer/tools.py | head -30
```

Capture the registration pattern for subsequent tasks.

---

## Task 1: `list_composer_blobs` MCP tool — LLM blob discovery

**Files:**
- Modify: `src/elspeth/web/composer/tools.py`
- Modify: `src/elspeth/web/composer/protocol.py` (or wherever response models live)
- Test: `tests/unit/web/composer/test_list_composer_blobs.py`

- [ ] **Step 1: Define the H4 visibility shape**

```python
# src/elspeth/web/composer/protocol.py (or appropriate models module)

@dataclass(frozen=True, slots=True)
class BlobInlineDescriptor:
    """LLM-visibility shape per spec §4 / H4 trust boundary.

    Bytes are opaque.  The LLM sees only the metadata that lets it
    reason about ref shape (which blob to use, what mime type the
    field will receive) without seeing what the content actually is.

    DELIBERATELY EXCLUDED:
    - source_description (could carry intent text)
    - resolved-content preview (would defeat the trust boundary)
    - any field carrying user-provided free text
    """

    blob_id: UUID
    mime_type: AllowedMimeType
    size_bytes: int
    content_hash: str   # SHA-256, used by wire_blob_inline_ref to pin
    filename: str       # safe — sanitized by sanitize_filename at upload
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/web/composer/test_list_composer_blobs.py
"""Pin the H4 trust boundary: list_composer_blobs returns descriptors
without source_description, preview content, or any free-text intent
field."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from elspeth.web.composer.tools import _execute_list_composer_blobs


def test_returned_descriptors_match_h4_shape():
    blob_service = AsyncMock()
    blob = ...  # construct a BlobRecord with source_description="malicious LLM-readable text"
    blob_service.list_blobs.return_value = [blob]
    result = _execute_list_composer_blobs({"session_id": str(blob.session_id)}, blob_service)
    descriptor = result.data["blobs"][0]
    assert "source_description" not in descriptor
    assert "preview" not in descriptor
    assert "content" not in descriptor
    # Required fields per H4
    assert "blob_id" in descriptor
    assert "mime_type" in descriptor
    assert "size_bytes" in descriptor
    assert "content_hash" in descriptor
    assert "filename" in descriptor


def test_only_ready_blobs_returned():
    """Pending/error blobs cannot be referenced by inline_content; the
    discovery tool filters them out so the LLM doesn't try to author a
    ref that /validate will reject."""
    blob_service = AsyncMock()
    ready_blob = ...
    pending_blob = ...
    blob_service.list_blobs.return_value = [ready_blob, pending_blob]
    result = _execute_list_composer_blobs({"session_id": str(ready_blob.session_id)}, blob_service)
    assert len(result.data["blobs"]) == 1
    assert result.data["blobs"][0]["blob_id"] == str(ready_blob.id)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_list_composer_blobs.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement the tool — match the existing sync-engine pattern**

The composer tools layer is sync and queries the SQLAlchemy `Engine` directly (see `_execute_set_source_from_blob` at `src/elspeth/web/composer/tools.py:1882-1951`, which calls a sync helper `_sync_get_blob(session_engine, blob_id, session_id)`). DO NOT call `asyncio.run()` from this layer — that's the H3 footgun the spec §6.1 warns against. The `_sync_*` helpers in tools.py are the existing pattern; mirror them.

```python
# src/elspeth/web/composer/tools.py — add a new sync helper near _sync_get_blob

def _sync_list_ready_blobs(session_engine: Engine, session_id: str) -> list[dict[str, Any]]:
    """Sync helper for list_composer_blobs — mirrors the _sync_get_blob
    pattern at line 1893.  Queries blobs_table directly instead of going
    through BlobServiceImpl.list_blobs (which is async).
    """
    from elspeth.web.sessions.models import blobs_table

    with session_engine.connect() as conn:
        rows = conn.execute(
            select(blobs_table)
            .where(blobs_table.c.session_id == session_id)
            .where(blobs_table.c.status == "ready")
            .order_by(blobs_table.c.created_at.desc())
        ).fetchall()
    return [
        {
            "id": row.id,
            "mime_type": row.mime_type,
            "size_bytes": row.size_bytes,
            "content_hash": row.content_hash,
            "filename": row.filename,
        }
        for row in rows
    ]


def _execute_list_composer_blobs(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> ToolResult:
    """List ready blobs in the session, projecting to the H4 visibility shape.

    Per spec §4 / §7.2 — the LLM sees only the descriptor fields that let it
    reason about ref shape. source_description (operator-provided free text)
    and any preview/content data are EXCLUDED to enforce the trust boundary.
    """
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    rows = _sync_list_ready_blobs(session_engine, session_id)
    descriptors = [
        {
            "blob_id": row["id"],
            "mime_type": row["mime_type"],
            "size_bytes": row["size_bytes"],
            "content_hash": row["content_hash"],
            "filename": row["filename"],
        }
        for row in rows
    ]
    # Identity-preserving: returning data does not mutate state.
    return ToolResult(success=True, data={"blobs": descriptors}, message=None, state=state)
```

**Tool registration — three sites, one CI gate to satisfy.** The codebase splits tool dispatch by surface and tracks them via a CI drift check against the composer skill doc:

1. **`_BLOB_DISCOVERY_TOOLS`** at `src/elspeth/web/composer/tools.py:3734` — the dict mapping `{tool_name: handler}` for blob-discovery surface.
2. **`get_tool_definitions`** at `src/elspeth/web/composer/tools.py:347` — returns the tool-spec list (name + JSON schema for arguments + description) emitted to the LLM. Add a new entry describing `list_composer_blobs` with required `session_id: string` argument and the H4 visibility-shape return contract.
3. **Composer skill doc** at `.claude/skills/pipeline-composer.md` (or wherever the skill lives — `find . -name 'pipeline-composer*' -path '*/skills/*'`) — kept in sync with the tool list via a CI drift gate. Add the new tool's name + one-line summary to the doc's tool reference section.

Sub-steps:

```python
# src/elspeth/web/composer/tools.py — _BLOB_DISCOVERY_TOOLS at line ~3734
_BLOB_DISCOVERY_TOOLS: dict[str, BlobToolHandler] = {
    # ... existing entries ...
    "list_composer_blobs": _execute_list_composer_blobs,  # NEW
}
```

```python
# src/elspeth/web/composer/tools.py — get_tool_definitions at line 347
# Append a new entry to the returned list:
{
    "name": "list_composer_blobs",
    "description": (
        "List ready blobs in the current session, projecting to the "
        "audited content-injection visibility shape (blob_id, mime_type, "
        "size_bytes, content_hash, filename).  Use this tool to discover "
        "blobs available for inline-content authoring via "
        "wire_blob_inline_ref.  Bytes are NOT returned."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "format": "uuid"},
        },
        "required": ["session_id"],
    },
}
```

Run the CI drift gate locally:

```bash
# The gate compares the tool list to the skill doc; locate the gate:
grep -rn "tool_definitions\|skill.*drift\|composer.*skill" scripts/cicd/ | head -5
```

Update `.claude/skills/pipeline-composer.md` to add `list_composer_blobs` (and `wire_blob_inline_ref` from Task 2) to the tool reference. Re-run the gate; it must pass.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_list_composer_blobs.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py src/elspeth/web/composer/protocol.py tests/unit/web/composer/test_list_composer_blobs.py
git commit -m "feat(composer): add list_composer_blobs MCP tool — H4 visibility shape

Per spec §4 / §7.2 of elspeth-fdebcaa79a — load-bearing site for the
H4 trust boundary.  The LLM learns blob identifiers via this tool only;
returned descriptors carry exactly the visibility shape (blob_id,
mime_type, size_bytes, content_hash, filename), excluding
source_description and any preview content.

Refs: elspeth-fdebcaa79a"
```

---

## Task 2: `wire_blob_inline_ref` MCP tool — authorship

**Files:**
- Modify: `src/elspeth/web/composer/tools.py`
- Test: `tests/unit/web/composer/test_wire_blob_inline_ref.py`

- [ ] **Step 1: Write failing test covering all rejection rules from spec §7.2**

```python
# tests/unit/web/composer/test_wire_blob_inline_ref.py
"""Pin: wire_blob_inline_ref enforces the spec §7.2 rejection rules
(no LLM-emitted bytes, no LLM-disagreeing hash, no bind_source mode,
ready-only, MIME-encoding compatibility)."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from elspeth.web.composer.tools import _execute_wire_blob_inline_ref


def test_authors_marker_with_pinned_hash_from_blob_service():
    blob_service = AsyncMock()
    blob = ...  # ready blob with content_hash="a"*64
    blob_service.get_blob.return_value = blob
    state = ...  # CompositionState with a transform named "classify"
    result = _execute_wire_blob_inline_ref(
        {
            "field_path": "node:classify.options.system_prompt",
            "blob_id": str(blob.id),
            "encoding": "utf-8",
        },
        state, blob_service,
    )
    assert result.success
    new_state = result.state
    marker = new_state.nodes[0].options["system_prompt"]
    assert marker == {
        "blob_ref": str(blob.id),
        "mode": "inline_content",
        "sha256": "a" * 64,
    }


def test_rejects_pending_blob():
    blob_service = AsyncMock()
    blob = ...  # status="pending"
    blob_service.get_blob.return_value = blob
    state = ...
    result = _execute_wire_blob_inline_ref(
        {"field_path": "node:classify.options.system_prompt", "blob_id": str(blob.id), "encoding": "utf-8"},
        state, blob_service,
    )
    assert not result.success
    assert "pending" in result.error.lower() or "not ready" in result.error.lower()


def test_rejects_llm_typed_disagreeing_hash():
    """If the LLM hand-types a sha256 that disagrees with BlobRecord.content_hash,
    reject — composer always pins from the authoritative source."""
    blob_service = AsyncMock()
    blob = ...  # content_hash="a"*64
    blob_service.get_blob.return_value = blob
    state = ...
    result = _execute_wire_blob_inline_ref(
        {
            "field_path": "node:classify.options.system_prompt",
            "blob_id": str(blob.id),
            "encoding": "utf-8",
            "sha256_override": "b" * 64,  # disagreeing hash
        },
        state, blob_service,
    )
    assert not result.success
    assert "hash" in result.error.lower()


def test_rejects_invalid_field_path():
    """Field_path must be canonical (identity-anchored)."""
    blob_service = AsyncMock()
    blob = ...
    blob_service.get_blob.return_value = blob
    state = ...
    result = _execute_wire_blob_inline_ref(
        {"field_path": "transforms[2].options.x", "blob_id": str(blob.id), "encoding": "utf-8"},
        state, blob_service,
    )
    assert not result.success
    assert "field_path" in result.error.lower()


def test_rejects_unknown_encoding():
    blob_service = AsyncMock()
    blob = ...
    blob_service.get_blob.return_value = blob
    state = ...
    result = _execute_wire_blob_inline_ref(
        {"field_path": "node:classify.options.system_prompt", "blob_id": str(blob.id), "encoding": "ascii"},
        state, blob_service,
    )
    assert not result.success
    assert "encoding" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_wire_blob_inline_ref.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the tool**

```python
# src/elspeth/web/composer/tools.py — add new tool

def _execute_wire_blob_inline_ref(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> ToolResult:
    """Author a widened-blob_ref(inline_content) marker into the composition
    state at the given canonical field_path.

    Enforces the §7.2 rejection rules:
    1. Blob must be `ready` (BlobRecord.status == "ready").
    2. Encoding must be in ALLOWED_CONTENT_ENCODINGS.
    3. Field_path must be canonical (validated by BlobInlineRef construction).
    4. Hash is auto-filled from blobs.content_hash; an LLM-emitted
       sha256_override is REJECTED if it disagrees with the authoritative
       hash (rule 2 of §7.2).

    Sync, no asyncio: matches the _execute_set_source_from_blob pattern
    at line 1882-1951 (uses _sync_get_blob, not async BlobServiceImpl).
    """
    from elspeth.contracts.blobs_inline import (
        ALLOWED_CONTENT_ENCODINGS,
        BlobInlineRef,
    )

    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    field_path = arguments["field_path"]
    try:
        blob_id = UUID(arguments["blob_id"])
    except ValueError:
        return _failure_result(state, f"blob_id {arguments['blob_id']!r} is not a valid UUID")

    encoding = arguments.get("encoding", "utf-8")
    if encoding not in ALLOWED_CONTENT_ENCODINGS:
        return _failure_result(state, f"encoding must be one of {sorted(ALLOWED_CONTENT_ENCODINGS)}, got {encoding!r}")

    # Sync DB query — mirrors _sync_get_blob at line 1893.
    blob_row = _sync_get_blob(session_engine, str(blob_id), session_id)
    if blob_row is None:
        return _failure_result(state, f"Blob {blob_id!r} not found in this session")
    if blob_row["status"] != "ready":
        return _failure_result(state, f"Cannot wire ref: blob {blob_id} status is {blob_row['status']!r}, expected 'ready'")

    # MIME-encoding compatibility (rule 5 of §7.2).
    # text/* and application/json content can be decoded as utf-8/utf-16 etc.;
    # binary mime types reject text encoding.  All ALLOWED_MIME_TYPES today
    # are text-shaped, so this is a no-op until binary types are added.
    # Document the check explicitly so future MIME additions don't bypass it.

    # Hash pinning (rule 1 of §7.2 + override-rejection rule 2).
    pinned_hash = blob_row["content_hash"]
    if pinned_hash is None:
        return _failure_result(state, f"Tier 1: ready blob {blob_id} has no content_hash; cannot pin")
    sha256_override = arguments.get("sha256_override")
    if sha256_override is not None and sha256_override != pinned_hash:
        return _failure_result(
            state,
            f"sha256 override {sha256_override[:16]}... disagrees with authoritative blob hash "
            f"{pinned_hash[:16]}... — composer always pins the authoritative value",
        )

    # Construct the marker (will validate field_path via BlobInlineRef).
    try:
        ref = BlobInlineRef(field_path=field_path, blob_id=blob_id, sha256=pinned_hash, encoding=encoding)
    except ValueError as exc:
        return _failure_result(state, f"Invalid widened-blob_ref: {exc}")

    marker: dict[str, Any] = {
        "blob_ref": str(blob_id),
        "mode": "inline_content",
        "sha256": pinned_hash,
    }
    if encoding != "utf-8":
        marker["encoding"] = encoding

    # Apply the marker to the composition state at the canonical field_path.
    new_state = _apply_marker_to_state(state, ref.field_path, marker)
    return _mutation_result(new_state, ("nodes",), data={"field_path": ref.field_path})


def _apply_marker_to_state(state: CompositionState, field_path: str, marker: dict[str, Any]) -> CompositionState:
    """Walk the canonical field_path and substitute the marker into a new
    CompositionState (frozen-dataclass copy-on-write)."""
    prefix, _, rest = field_path.partition(".options.")
    keys = rest.split(".")

    if prefix == "source":
        if state.source is None:
            raise ValueError("Cannot wire source ref: no source has been set")
        new_options = _set_nested(dict(state.source.options), keys, marker)
        return state.with_source(replace(state.source, options=new_options))

    if prefix.startswith("node:"):
        node_id = prefix[len("node:"):]
        new_nodes = []
        found = False
        for node in state.nodes:
            if node.id == node_id:
                new_options = _set_nested(dict(node.options), keys, marker)
                new_nodes.append(replace(node, options=new_options))
                found = True
            else:
                new_nodes.append(node)
        if not found:
            raise ValueError(f"Node {node_id!r} not found in composition state")
        return replace(state, nodes=tuple(new_nodes))

    if prefix.startswith("output:"):
        sink_name = prefix[len("output:"):]
        new_outputs = []
        found = False
        for output in state.outputs:
            if output.name == sink_name:
                new_options = _set_nested(dict(output.options), keys, marker)
                new_outputs.append(replace(output, options=new_options))
                found = True
            else:
                new_outputs.append(output)
        if not found:
            raise ValueError(f"Sink {sink_name!r} not found in composition state")
        return replace(state, outputs=tuple(new_outputs))

    raise ValueError(f"Unrecognised field_path prefix: {prefix!r}")


def _set_nested(container: dict[str, Any], keys: list[str], value: Any) -> dict[str, Any]:
    if len(keys) == 1:
        container[keys[0]] = value
        return container
    head, tail = keys[0], keys[1:]
    if head not in container or not isinstance(container[head], dict):
        container[head] = {}
    container[head] = _set_nested(dict(container[head]), tail, value)
    return container
```

- [ ] **Step 4: Register `wire_blob_inline_ref` at all three sites**

Same registration mechanism as `list_composer_blobs` in Task 1:

```python
# _BLOB_MUTATION_TOOLS at src/elspeth/web/composer/tools.py:3740
_BLOB_MUTATION_TOOLS: dict[str, BlobToolHandler] = {
    # ... existing entries ...
    "wire_blob_inline_ref": _execute_wire_blob_inline_ref,  # NEW
}
```

```python
# get_tool_definitions at line 347 — append:
{
    "name": "wire_blob_inline_ref",
    "description": (
        "Author a widened blob_ref(inline_content) marker into the "
        "composition state at the given canonical field_path.  The "
        "composer auto-pins the SHA-256 hash from the blob's metadata; "
        "callers MUST NOT pass content bytes or override the hash.  "
        "Use list_composer_blobs first to discover available blob_ids."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field_path": {
                "type": "string",
                "description": "Canonical identity-anchored path: source.options.<key>, node:<node_id>.options.<key>, or output:<sink_name>.options.<key>",
            },
            "blob_id": {"type": "string", "format": "uuid"},
            "encoding": {
                "type": "string",
                "enum": ["utf-8", "utf-8-sig", "utf-16", "latin-1"],
                "default": "utf-8",
            },
        },
        "required": ["field_path", "blob_id"],
    },
}
```

Update `.claude/skills/pipeline-composer.md` to document the tool. Run the skill-doc drift gate.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_wire_blob_inline_ref.py -v`
Expected: PASS for every test.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_wire_blob_inline_ref.py .claude/skills/pipeline-composer.md
git commit -m "feat(composer): add wire_blob_inline_ref MCP tool

Authors widened-blob_ref(inline_content) markers into composition
state at canonical field_paths.  Per spec §7.2 of elspeth-fdebcaa79a,
enforces:
- Blob must be ready
- Encoding in ALLOWED_CONTENT_ENCODINGS
- Field_path canonical (BlobInlineRef construction validates)
- Hash auto-filled from BlobRecord.content_hash; LLM-emitted overrides
  rejected on disagreement

Refs: elspeth-fdebcaa79a"
```

---

## Task 3: Update `set_source_from_blob` to emit explicit `mode: bind_source`

Per spec §10.1 P5 row and the closure rule "every persisted marker round-trips through the recognition function."

**Files:**
- Modify: `src/elspeth/web/composer/tools.py::_execute_set_source_from_blob` (or wherever the function lives)
- Test: `tests/unit/web/composer/test_set_source_from_blob_emits_mode.py`

- [ ] **Step 1: Locate the function**

```bash
grep -n "set_source_from_blob\|_execute_set_source_from_blob" src/elspeth/web/composer/tools.py | head -5
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/web/composer/test_set_source_from_blob_emits_mode.py
"""Pin: set_source_from_blob emits mode: bind_source explicitly so
every persisted marker round-trips through is_widened_blob_ref."""

from elspeth.contracts.blobs_inline import is_widened_blob_ref


def test_set_source_from_blob_emits_explicit_mode(blob_service, state):
    # ... compose set_source_from_blob call ...
    new_state = ...
    options = new_state.source.options
    assert "blob_ref" in options
    assert options["mode"] == "bind_source"
    # Recognition round-trip
    shape = is_widened_blob_ref({k: v for k, v in options.items() if k in {"blob_ref", "mode", "path"}})
    assert shape is not None
    assert shape.mode == "bind_source"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_set_source_from_blob_emits_mode.py -v`
Expected: FAIL — current implementation emits `{blob_ref: ID, path: ...}` without `mode`.

- [ ] **Step 4: Update the function — and the strip surfaces it depends on**

Adding `mode: bind_source` to source options is necessary but not sufficient. Two strip surfaces today filter `blob_ref` out of source options before the engine sees them — both must be widened to also strip `mode` (and any other widened-marker key) so plugin Pydantic models with `extra="forbid"` don't reject the new field.

**Sub-step 4a: Update `_execute_set_source_from_blob`** at `src/elspeth/web/composer/tools.py:1882-1951`. Locate the `merged_options` dict construction (currently around line 1931):

```python
# EXISTING (line ~1931):
merged_options = {
    **caller_options,
    **mime_extra,
    "path": blob["storage_path"],
    "blob_ref": blob["id"],
}

# REPLACE WITH:
merged_options = {
    **caller_options,
    **mime_extra,
    "path": blob["storage_path"],
    "blob_ref": blob["id"],
    "mode": "bind_source",  # NEW — explicit mode per ADR-021 closure rule
}
```

**Sub-step 4b: Widen the composer-tool strip set** at `src/elspeth/web/composer/tools.py:1483`:

```python
# EXISTING:
_WEB_ONLY_SOURCE_KEYS = frozenset({"blob_ref"})

# REPLACE WITH:
_WEB_ONLY_SOURCE_KEYS = frozenset({"blob_ref", "mode"})
```

**Sub-step 4c: Widen the YAML-generator strip set** at `src/elspeth/web/composer/yaml_generator.py:31`:

```python
# EXISTING:
_WEB_ONLY_OPTION_KEYS = frozenset({"blob_ref"})

# REPLACE WITH:
_WEB_ONLY_OPTION_KEYS = frozenset({"blob_ref", "mode"})
```

The YAML stripper at `_strip_web_metadata` (line 34-39) is shallow — it removes top-level keys from an options dict. That's correct for source.options (which carries `blob_ref` and now `mode` at the top level). For inline-content markers nested under transform/sink option fields, the L1 resolver substitutes the marker with the decoded string BEFORE settings load, so the engine never sees the marker shape — no stripping needed for nested cases.

**Sub-step 4d: Add an end-to-end strip test**

```python
# tests/unit/web/composer/test_set_source_from_blob_emits_mode.py — extend

def test_set_source_from_blob_round_trips_through_yaml_to_settings(blob_service, state, tmp_path):
    """The full path: set_source_from_blob → generate_yaml → load_settings.
    The engine's source plugin Pydantic model must NOT see `blob_ref` or
    `mode` in its options (extra='forbid' would reject them)."""
    from elspeth.core.config import load_settings_from_yaml_string
    from elspeth.web.composer.yaml_generator import generate_yaml
    new_state = ...  # call _execute_set_source_from_blob with a ready blob
    yaml_str = generate_yaml(new_state)
    # Settings load must succeed (no Pydantic ValidationError on extra keys)
    settings = load_settings_from_yaml_string(yaml_str)
    # The engine settings have no blob_ref / mode in source.options
    assert "blob_ref" not in settings.source.options
    assert "mode" not in settings.source.options
    assert settings.source.options["path"]  # path is preserved
```

Run all three tests + the existing source/blob suite:

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_set_source_from_blob_emits_mode.py tests/unit/web/composer/ tests/unit/web/blobs/ -v
```

Expected: PASS across the whole composer/blob test surface. Watch for tests that pinned the old strip set (one-key frozenset); update them to expect `frozenset({"blob_ref", "mode"})`.

- [ ] **Step 5: Run tests + run the broader composer/runtime agreement suite to confirm no regression**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_set_source_from_blob_emits_mode.py tests/unit/web/composer/ tests/integration/pipeline/test_composer_runtime_agreement.py -v
```

Expected: PASS across the whole composer test surface. Watch for tests that assumed the mode-less shape — those tests are pinning legacy behaviour and should be updated to expect the new shape (per the "no legacy code" policy).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_set_source_from_blob_emits_mode.py
git commit -m "feat(composer): emit explicit mode: bind_source in set_source_from_blob

Per spec §10.1 P5 row of elspeth-fdebcaa79a — closure rule \"every
persisted marker round-trips through is_widened_blob_ref\".  Mode-less
markers are no longer emitted by the codebase; the recognition
function rejects them as malformed at runtime resolution.

Refs: elspeth-fdebcaa79a"
```

---

## Task 4: Open the P5 PR + file the F-1 follow-up

- [ ] **Step 1: File F-1 (frontend follow-up issue)**

```bash
filigree create "Composer dashboard UI: SecretsPanel-equivalent for inline-content blob refs" \
  --type=feature \
  --priority=2 \
  --label=cluster:rc5-ux \
  --label=composer
```

Capture the new issue ID for the PR description.

- [ ] **Step 2: Push + open the PR**

```bash
git push -u origin <branch-name>
gh pr create --title "feat(composer): MCP tools list_composer_blobs + wire_blob_inline_ref" --body "$(cat <<'EOF'
## Summary

- Adds \`list_composer_blobs\` MCP tool — load-bearing site for the H4 trust boundary, returns descriptors in exactly the visibility shape (blob_id, mime_type, size_bytes, content_hash, filename)
- Adds \`wire_blob_inline_ref\` MCP tool — authorship path; auto-pins hash from \`BlobRecord.content_hash\`, rejects LLM overrides that disagree, rejects non-ready blobs, validates canonical field_path
- Updates \`set_source_from_blob\` to emit explicit \`mode: bind_source\` so every persisted marker round-trips through \`is_widened_blob_ref\`

This closes the spec's full P1–P5 work on elspeth-fdebcaa79a.  Frontend recovery surface (F-1) filed as separate follow-up issue [<NEW_ISSUE_ID>].

## Test plan

- [ ] \`pytest tests/unit/web/composer/test_list_composer_blobs.py tests/unit/web/composer/test_wire_blob_inline_ref.py tests/unit/web/composer/test_set_source_from_blob_emits_mode.py -v\`
- [ ] \`pytest tests/unit/web/composer/ tests/integration/pipeline/test_composer_runtime_agreement.py -v\` (no regression in agreement suite)
- [ ] \`enforce_tier_model.py check\` exits 0
- [ ] Manual: in a chat session, ask the LLM to wire an inline system prompt; confirm \`list_composer_blobs\` is the only path by which the LLM learns blob IDs

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Close the epic**

```bash
filigree close elspeth-fdebcaa79a --reason="P1–P5 merged.  P6 frontend recovery surface filed as <NEW_ISSUE_ID>; not blocking RC5."
```

---

## Done conditions

P5 is done when:

1. `list_composer_blobs` MCP tool exists, returns the H4 visibility shape, filters to ready blobs.
2. `wire_blob_inline_ref` MCP tool exists, enforces all spec §7.2 rejection rules, pins hash from `BlobRecord.content_hash`.
3. `set_source_from_blob` emits explicit `mode: bind_source`.
4. Composer/runtime agreement suite (Shape 9 + the rest) passes without regression.
5. F-1 (frontend follow-up issue) is filed and cited in the PR description.
6. PR is merged.
7. Epic elspeth-fdebcaa79a is closed.

The widened-`blob_ref` epic ships at this point. The frontend SecretsPanel-equivalent UI ships separately if dashboard polish is independently scheduled before RC5.
