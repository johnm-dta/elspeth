# Phase 2 — L0 Contracts + L1 Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the L0 contract module (`contracts/blobs_inline.py`) and the L1 resolver module (`core/blobs_inline.py`) with full unit-test coverage and zero behaviour change at the runtime or composer layer. P3 wires the resolver in; P2 ensures the resolver is correct in isolation first.

**Architecture:** Two new modules at the canonical layer positions. The L0 module carries closed-set Literals (`ContentEncoding`), recognition function (`is_widened_blob_ref`), value objects (`WidenedBlobRefShape`, `BlobInlineRef`, `ResolvedBlobContent`), and the batched error (`BlobContentResolutionError`). It also relocates `AllowedMimeType` from `web/blobs/protocol.py` (L3) to L0 per CLAUDE.md cross-layer resolution rule (move down before extracting). The L1 module carries the three-function resolver split (`_discover_blob_content_refs` sync, `_fetch_blob_contents` async, `_substitute_blob_content_refs` sync) plus the validate-side helper `_validate_blob_content_refs`.

**Tech Stack:** Python 3.13, `dataclasses` with `frozen=True, slots=True`, `typing.Literal` + `typing.get_args` for closed-set enforcement (matches `web/blobs/protocol.py:36-71` precedent), Hypothesis for tree-walk property tests, pytest.

---

## Pre-phase verification

- [ ] **Step 1: Confirm P1 has merged**

```bash
git log --oneline --all | grep "ADR-021"
```

Expected: at least one commit landing `docs/architecture/adr/021-config-content-ref.md`.

- [ ] **Step 2: Confirm L0/L1 module paths are free**

```bash
ls src/elspeth/contracts/blobs_inline.py 2>&1
ls src/elspeth/core/blobs_inline.py 2>&1
```

Expected: both `No such file or directory`.

- [ ] **Step 3: Confirm `enforce_tier_model.py` baseline is clean**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
echo "Exit: $?"
```

Expected: exit 0. P2 must not introduce upward imports; the layer enforcer is the gate.

---

## Task 1: Relocate `AllowedMimeType` from L3 to L0

**Files:**
- Modify: `src/elspeth/contracts/blobs.py` — create the file or extend if it exists already
- Modify: `src/elspeth/web/blobs/protocol.py:40-47` — re-export from L0 instead of defining locally

- [ ] **Step 1: Check if `contracts/blobs.py` already exists**

```bash
ls src/elspeth/contracts/blobs.py 2>&1
```

Expected: either `No such file or directory` (create) or path returned (extend in-place).

- [ ] **Step 2: Write the failing test for L0 location**

```python
# tests/unit/contracts/test_blobs.py
"""Pin AllowedMimeType's L0 location.

Per the CLAUDE.md cross-layer rule, when a value type is needed at L1 (the
core/blobs_inline.py resolver) but currently lives at L3 (web/blobs/
protocol.py), the resolution is to move it down — not to add an upward
import. This test pins the L0 location; if a future refactor moves it
back to L3, the import target here breaks.
"""

from elspeth.contracts.blobs import (
    ALLOWED_MIME_TYPES,
    AllowedMimeType,
)


def test_allowed_mime_types_at_l0() -> None:
    assert "text/csv" in ALLOWED_MIME_TYPES
    assert "application/json" in ALLOWED_MIME_TYPES
    assert "text/plain" in ALLOWED_MIME_TYPES
    assert isinstance(ALLOWED_MIME_TYPES, frozenset)


def test_allowed_mime_type_literal_get_args_consistency() -> None:
    """Anti-drift: the Literal alias and the frozenset are co-derived."""
    from typing import get_args

    assert frozenset(get_args(AllowedMimeType)) == ALLOWED_MIME_TYPES
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_blobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.contracts.blobs'`.

- [ ] **Step 4: Create `contracts/blobs.py` and move the type**

```python
# src/elspeth/contracts/blobs.py
"""Blob storage contracts shared across L1 (core) and L3 (web/cli).

Layer: L0. No upward imports.

This module hosts the MIME-type closed set used by both the web blob
service (L3) and the inline-content resolver (L1). Per CLAUDE.md
cross-layer resolution rule, when a value type is needed at L1 but
currently lives at L3, the fix is to move it down.

The original location (web/blobs/protocol.py:40-47, BlobStatus / etc.
neighbours) re-exports from this module to keep the existing import
sites at the web layer working without behavioural change.
"""

from __future__ import annotations

from typing import Literal, get_args

AllowedMimeType = Literal[
    "text/csv",
    "text/plain",
    "application/json",
    "application/x-jsonlines",
    "application/jsonl",
    "text/jsonl",
]
"""Closed set of MIME types accepted for data-oriented blob uploads.

The Literal is authoritative; ALLOWED_MIME_TYPES is derived via get_args
so adding a member is a single-site edit (matches the anti-drift pattern
used for BlobStatus, BlobRunLinkDirection, etc. in web/blobs/protocol.py).
"""

ALLOWED_MIME_TYPES: frozenset[str] = frozenset(get_args(AllowedMimeType))
```

- [ ] **Step 5: Update `web/blobs/protocol.py` to re-export**

Replace the local definition at lines 40-47 (and the matching `ALLOWED_MIME_TYPES` line at 71):

```python
# src/elspeth/web/blobs/protocol.py (replace original lines 40-47 + 71)

from elspeth.contracts.blobs import ALLOWED_MIME_TYPES, AllowedMimeType  # re-export from L0
```

Leave every other Literal in `web/blobs/protocol.py` (`BlobStatus`, `FinalizeBlobStatus`, `BlobCreator`, `BlobRunLinkDirection`) unchanged — only `AllowedMimeType` migrates because only it is needed at L1. Per the spec §4 row "L0 contract" and CLAUDE.md cross-layer resolution rule.

- [ ] **Step 6: Run all tests touching `AllowedMimeType` to confirm no behaviour change**

```bash
.venv/bin/python -m pytest tests/unit/contracts/test_blobs.py tests/unit/web/blobs/ -v
```

Expected: PASS for new test; existing web/blobs tests unchanged.

- [ ] **Step 7: Run the layer enforcer**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: exit 0. Re-exports do not violate layer constraints; the load-bearing definition has moved down.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/contracts/blobs.py src/elspeth/web/blobs/protocol.py tests/unit/contracts/test_blobs.py
git commit -m "refactor(contracts): relocate AllowedMimeType from L3 to L0

The inline-content resolver lands at L1 in P3 of elspeth-fdebcaa79a
and needs AllowedMimeType to validate MIME-encoding compatibility.
Per the CLAUDE.md cross-layer resolution rule (move code down before
extracting it), the type moves from web/blobs/protocol.py to a new
contracts/blobs.py at L0; the L3 module re-exports for backwards
compatibility within the codebase.

Refs: elspeth-fdebcaa79a"
```

---

## Task 2: L0 contract module — `contracts/blobs_inline.py`

**Files:**
- Create: `src/elspeth/contracts/blobs_inline.py`
- Test: `tests/unit/contracts/test_blobs_inline.py`

- [ ] **Step 1: Write failing tests for the value objects + recognition function**

```python
# tests/unit/contracts/test_blobs_inline.py
"""Pin the L0 contract shape for widened-blob_ref recognition + value objects."""

import pytest
from uuid import UUID, uuid4

from elspeth.contracts.blobs_inline import (
    ALLOWED_CONTENT_ENCODINGS,
    BlobContentResolutionError,
    BlobInlineRef,
    ContentEncoding,
    ResolvedBlobContent,
    WidenedBlobRefShape,
    is_widened_blob_ref,
)


VALID_HASH = "a" * 64
VALID_BLOB_ID = "5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b"


# ─── is_widened_blob_ref ──────────────────────────────────────────────

def test_is_widened_blob_ref_recognises_inline_content() -> None:
    shape = is_widened_blob_ref({
        "blob_ref": VALID_BLOB_ID,
        "mode": "inline_content",
        "sha256": VALID_HASH,
    })
    assert shape is not None
    assert shape.mode == "inline_content"
    assert shape.blob_id == UUID(VALID_BLOB_ID)
    assert shape.sha256 == VALID_HASH
    assert shape.encoding == "utf-8"  # default


def test_is_widened_blob_ref_recognises_bind_source() -> None:
    shape = is_widened_blob_ref({
        "blob_ref": VALID_BLOB_ID,
        "mode": "bind_source",
        "path": "/data/file.csv",
    })
    assert shape is not None
    assert shape.mode == "bind_source"
    assert shape.path == "/data/file.csv"


def test_is_widened_blob_ref_explicit_encoding() -> None:
    shape = is_widened_blob_ref({
        "blob_ref": VALID_BLOB_ID,
        "mode": "inline_content",
        "sha256": VALID_HASH,
        "encoding": "utf-16",
    })
    assert shape is not None
    assert shape.encoding == "utf-16"


def test_is_widened_blob_ref_rejects_mode_less() -> None:
    """Per spec §5.1, mode is required. Mode-less markers are malformed."""
    result = is_widened_blob_ref({"blob_ref": VALID_BLOB_ID})
    assert result is None or (hasattr(result, "malformed_reason") and result.malformed_reason is not None)


def test_is_widened_blob_ref_rejects_unknown_keys() -> None:
    result = is_widened_blob_ref({
        "blob_ref": VALID_BLOB_ID,
        "mode": "inline_content",
        "sha256": VALID_HASH,
        "unknown_key": "x",
    })
    assert result is None


def test_is_widened_blob_ref_rejects_inline_content_without_sha256() -> None:
    result = is_widened_blob_ref({
        "blob_ref": VALID_BLOB_ID,
        "mode": "inline_content",
    })
    assert result is None


def test_is_widened_blob_ref_rejects_bind_source_with_sha256() -> None:
    result = is_widened_blob_ref({
        "blob_ref": VALID_BLOB_ID,
        "mode": "bind_source",
        "path": "/data/x",
        "sha256": VALID_HASH,
    })
    assert result is None


def test_is_widened_blob_ref_rejects_non_marker_dicts() -> None:
    assert is_widened_blob_ref({"foo": "bar"}) is None
    assert is_widened_blob_ref({"secret_ref": "OPENROUTER_KEY"}) is None
    assert is_widened_blob_ref(None) is None
    assert is_widened_blob_ref(["x"]) is None
    assert is_widened_blob_ref("string") is None


# ─── BlobInlineRef ────────────────────────────────────────────────────

def test_blob_inline_ref_accepts_canonical_field_path() -> None:
    ref = BlobInlineRef(
        field_path="node:classify.options.system_prompt",
        blob_id=UUID(VALID_BLOB_ID),
        sha256=VALID_HASH,
        encoding="utf-8",
    )
    assert ref.field_path == "node:classify.options.system_prompt"


@pytest.mark.parametrize("path", [
    "transforms[2].options.x",  # positional (forbidden)
    "node:classify",  # missing .options.<field>
    "x.options.y",  # missing source/node:/output: prefix
    "",  # empty
])
def test_blob_inline_ref_rejects_non_canonical_path(path: str) -> None:
    with pytest.raises(ValueError, match="field_path"):
        BlobInlineRef(
            field_path=path,
            blob_id=UUID(VALID_BLOB_ID),
            sha256=VALID_HASH,
            encoding="utf-8",
        )


def test_blob_inline_ref_rejects_invalid_hash() -> None:
    with pytest.raises(ValueError, match="sha256"):
        BlobInlineRef(
            field_path="source.options.x",
            blob_id=UUID(VALID_BLOB_ID),
            sha256="not_64_hex",
            encoding="utf-8",
        )


# ─── ResolvedBlobContent ──────────────────────────────────────────────

def test_resolved_blob_content_validators() -> None:
    rec = ResolvedBlobContent(
        field_path="source.options.x",
        blob_id=UUID(VALID_BLOB_ID),
        content_hash=VALID_HASH,
        byte_length=42,
        mime_type="text/plain",
        encoding="utf-8",
    )
    assert rec.byte_length == 42
    assert rec.content_hash == VALID_HASH


def test_resolved_blob_content_rejects_negative_byte_length() -> None:
    with pytest.raises(ValueError, match="byte_length"):
        ResolvedBlobContent(
            field_path="source.options.x",
            blob_id=UUID(VALID_BLOB_ID),
            content_hash=VALID_HASH,
            byte_length=-1,
            mime_type="text/plain",
            encoding="utf-8",
        )


# ─── BlobContentResolutionError ───────────────────────────────────────

def test_blob_content_resolution_error_aggregates_all_cases() -> None:
    err = BlobContentResolutionError(
        missing=["node:c.options.system_prompt"],
        oversized=[("node:c.options.body", 100_000, 64_000)],
        undecodable=[("source.options.x", "utf-8")],
        not_ready=[("source.options.x", "pending")],
        cross_session=["output:s.options.template"],
        malformed=[("node:c.options.x", "missing mode")],
    )
    msg = str(err)
    assert "missing" in msg
    assert "oversized" in msg


# ─── ContentEncoding ──────────────────────────────────────────────────

def test_content_encoding_closed_set() -> None:
    from typing import get_args
    assert frozenset(get_args(ContentEncoding)) == ALLOWED_CONTENT_ENCODINGS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_blobs_inline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.contracts.blobs_inline'`.

- [ ] **Step 3: Implement the L0 module**

```python
# src/elspeth/contracts/blobs_inline.py
"""Widened blob_ref contracts — recognition, value objects, batched error.

Layer: L0. No upward imports.

This module hosts the L0-grade types for the widened blob_ref marker
shape per spec §5 (audited content injection, elspeth-fdebcaa79a).
The recognition function is_widened_blob_ref and the value objects
WidenedBlobRefShape / BlobInlineRef / ResolvedBlobContent are imported
by both the L1 resolver (core/blobs_inline.py) and the L3 composer
validation (web/composer/tools.py); locating them at L0 prevents the
cross-layer dependency that would otherwise force an upward import.

The companion runtime types (BlobNotFoundError, BlobIntegrityError,
etc.) remain at web/blobs/protocol.py — those are tightly bound to
the BlobService runtime contract and have no L1 callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args
from uuid import UUID

from elspeth.contracts.blobs import AllowedMimeType


# ─── Closed-set Literals ──────────────────────────────────────────────

ContentEncoding = Literal["utf-8", "utf-8-sig", "utf-16", "latin-1"]
"""Closed set of decoding modes accepted for inline_content resolution.

Strict UTF-8 is the default (spec §4 M2 reconciliation); the explicit
encoding field exists as an escape hatch for content that legitimately
isn't UTF-8 (e.g. legacy regex libraries authored in latin-1). Adding
a member requires an ADR-021 amendment per the no-new-ref-forms rule.
"""

ALLOWED_CONTENT_ENCODINGS: frozenset[str] = frozenset(get_args(ContentEncoding))

BlobRefMode = Literal["bind_source", "inline_content"]
ALLOWED_BLOB_REF_MODES: frozenset[str] = frozenset(get_args(BlobRefMode))


# ─── field_path canonical format ──────────────────────────────────────

_FIELD_PATH_PATTERN: Final = re.compile(
    r"^(?:source|node:[A-Za-z_][A-Za-z0-9_-]*|output:[A-Za-z_][A-Za-z0-9_-]*)"
    r"\.options(?:\.[A-Za-z_][A-Za-z0-9_]*)+$"
)
"""Pattern enforcing the §8.1 canonical encoding.

Allowed prefixes: `source`, `node:<id>`, `output:<name>`. After the
prefix, `.options.<key>[.<sub>...]` with identifier-shaped segments.
List indices (transforms[2].options.x) are rejected — list positions
are unstable across composer state mutations and would defeat the
audit row's cross-resume durability guarantee.
"""

_SHA256_HEX_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
"""Matches the canonical SHA-256 hex form already enforced at the blob
service layer (web/blobs/service.py:_SHA256_HEX_PATTERN). Reuses the
same shape so blob_inline_resolutions.content_hash and blobs.content_hash
hold byte-identical values."""


def _validate_field_path(owner: str, field_path: str) -> None:
    if not _FIELD_PATH_PATTERN.fullmatch(field_path):
        raise ValueError(
            f"{owner}: field_path {field_path!r} does not match canonical format "
            f"(source|node:<id>|output:<name>).options.<key>[.<sub>...] — see ADR-021 §6"
        )


def _validate_sha256(owner: str, sha256: str) -> None:
    if not _SHA256_HEX_PATTERN.fullmatch(sha256):
        raise ValueError(
            f"{owner}: sha256 must be 64-char lowercase hex (SHA-256), got {sha256!r}"
        )


# ─── Value objects ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class WidenedBlobRefShape:
    """Recognition result from is_widened_blob_ref. Carries the parsed
    mode so callers branch on mode (not on stringly-typed checks).
    """

    blob_id: UUID
    mode: BlobRefMode
    sha256: str | None = None
    path: str | None = None
    encoding: ContentEncoding = "utf-8"


@dataclass(frozen=True, slots=True)
class BlobInlineRef:
    """A single ref discovered during config tree-walk. The discoverer
    emits a list of these; the substituter consumes them.

    field_path is identity-anchored (§8.1). sha256 is the composer-pinned
    hash; the runtime substituter compares it against the bytes returned
    from BlobServiceImpl.read_blob_content.
    """

    field_path: str
    blob_id: UUID
    sha256: str
    encoding: ContentEncoding

    def __post_init__(self) -> None:
        _validate_field_path(type(self).__name__, self.field_path)
        _validate_sha256(type(self).__name__, self.sha256)
        if self.encoding not in ALLOWED_CONTENT_ENCODINGS:
            raise ValueError(
                f"{type(self).__name__}: encoding must be one of "
                f"{sorted(ALLOWED_CONTENT_ENCODINGS)}, got {self.encoding!r}"
            )


@dataclass(frozen=True, slots=True)
class ResolvedBlobContent:
    """Audit-record shape per spec §8.1 schema. One row per resolved ref."""

    field_path: str
    blob_id: UUID
    content_hash: str
    byte_length: int
    mime_type: AllowedMimeType
    encoding: ContentEncoding

    def __post_init__(self) -> None:
        _validate_field_path(type(self).__name__, self.field_path)
        _validate_sha256(type(self).__name__, self.content_hash)
        if self.byte_length < 0:
            raise ValueError(
                f"{type(self).__name__}: byte_length must be non-negative, got {self.byte_length}"
            )
        if self.encoding not in ALLOWED_CONTENT_ENCODINGS:
            raise ValueError(
                f"{type(self).__name__}: encoding must be one of "
                f"{sorted(ALLOWED_CONTENT_ENCODINGS)}, got {self.encoding!r}"
            )


# ─── Recognition function ─────────────────────────────────────────────

_ALLOWED_MARKER_KEYS: frozenset[str] = frozenset({"blob_ref", "mode", "path", "sha256", "encoding"})


def is_widened_blob_ref(value: Any) -> WidenedBlobRefShape | None:
    """Recognition predicate per spec §5.1.

    Returns a WidenedBlobRefShape if value is a widened-blob_ref marker
    that satisfies every rule in §5.1; returns None otherwise.

    Mode-less markers ({blob_ref: <UUID>} with no mode key) return None —
    they are not recognised. The caller (the discoverer) is expected to
    surface them as malformed rather than ignore them; see the
    discoverer in core/blobs_inline.py for that surfacing logic.
    """
    if not isinstance(value, dict):
        return None
    if "blob_ref" not in value:
        return None
    blob_ref_str = value["blob_ref"]
    if not isinstance(blob_ref_str, str):
        return None
    keys = set(value.keys())
    if not keys.issubset(_ALLOWED_MARKER_KEYS):
        return None
    if "mode" not in value:
        return None  # malformed — caller surfaces; recognition rejects
    mode = value["mode"]
    if mode not in ALLOWED_BLOB_REF_MODES:
        return None
    try:
        blob_id = UUID(blob_ref_str)
    except ValueError:
        return None

    if mode == "bind_source":
        if "sha256" in value or "encoding" in value:
            return None
        path = value.get("path")
        if path is not None and not isinstance(path, str):
            return None
        return WidenedBlobRefShape(blob_id=blob_id, mode="bind_source", path=path)

    # mode == "inline_content"
    if "path" in value:
        return None
    sha256 = value.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_PATTERN.fullmatch(sha256):
        return None
    encoding = value.get("encoding", "utf-8")
    if encoding not in ALLOWED_CONTENT_ENCODINGS:
        return None
    return WidenedBlobRefShape(
        blob_id=blob_id,
        mode="inline_content",
        sha256=sha256,
        encoding=encoding,
    )


# ─── Batched error ────────────────────────────────────────────────────

class BlobContentResolutionError(Exception):
    """Operationally-recoverable batched error from blob inline resolution.

    Per spec §6.5, Tier-1 anomalies (BlobIntegrityError,
    BlobContentMissingError, AuditIntegrityError) propagate immediately
    and uncaught — they are NOT included in this batched form.

    HTTP status: 422 (structured ValidationResult).
    """

    def __init__(
        self,
        *,
        missing: list[str] | None = None,
        oversized: list[tuple[str, int, int]] | None = None,
        undecodable: list[tuple[str, str]] | None = None,
        not_ready: list[tuple[str, str]] | None = None,
        cross_session: list[str] | None = None,
        malformed: list[tuple[str, str]] | None = None,
    ) -> None:
        self.missing = missing or []
        self.oversized = oversized or []
        self.undecodable = undecodable or []
        self.not_ready = not_ready or []
        self.cross_session = cross_session or []
        self.malformed = malformed or []
        parts: list[str] = []
        if self.missing:
            parts.append(f"missing: {self.missing}")
        if self.oversized:
            parts.append(f"oversized: {self.oversized}")
        if self.undecodable:
            parts.append(f"undecodable: {self.undecodable}")
        if self.not_ready:
            parts.append(f"not_ready: {self.not_ready}")
        if self.cross_session:
            parts.append(f"cross_session: {self.cross_session}")
        if self.malformed:
            parts.append(f"malformed: {self.malformed}")
        super().__init__("BlobContentResolutionError(" + "; ".join(parts) + ")")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_blobs_inline.py -v`
Expected: PASS for every test.

- [ ] **Step 5: Run the layer enforcer**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: exit 0. The new module imports only from `elspeth.contracts.blobs` (sibling L0).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/blobs_inline.py tests/unit/contracts/test_blobs_inline.py
git commit -m "feat(contracts): land widened-blob_ref L0 contracts

Adds is_widened_blob_ref recognition, WidenedBlobRefShape /
BlobInlineRef / ResolvedBlobContent value objects, ContentEncoding
closed set, and BlobContentResolutionError batched error per spec §5
and §6.5 of elspeth-fdebcaa79a.

Field_path validation enforces the §8.1 canonical (identity-anchored)
encoding via regex; positional list-index paths are rejected at
construction time.

Refs: elspeth-fdebcaa79a"
```

---

## Task 3: L1 resolver — `_discover_blob_content_refs`

**Files:**
- Create: `src/elspeth/core/blobs_inline.py`
- Test: `tests/unit/core/test_blobs_inline.py`

- [ ] **Step 1: Write failing tests for tree-walk discovery**

```python
# tests/unit/core/test_blobs_inline.py
"""Pin the three-function resolver split per spec §6.

Tree-walk parity test: the discoverer walks dict/list/Mapping the same
way core/secrets.py::_walk does (the precedent at
src/elspeth/core/secrets.py:118-159).
"""

from uuid import UUID

import pytest

from elspeth.contracts.blobs_inline import BlobInlineRef
from elspeth.core.blobs_inline import _discover_blob_content_refs


VALID_HASH = "a" * 64
BLOB1 = "5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b"
BLOB2 = "7c3a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3aaa"


def _marker(blob_id: str = BLOB1, sha: str = VALID_HASH) -> dict:
    return {"blob_ref": blob_id, "mode": "inline_content", "sha256": sha}


def test_discover_in_source_options() -> None:
    config = {"source": {"plugin": "csv", "options": {"system_prompt": _marker()}}}
    refs = _discover_blob_content_refs(config)
    assert len(refs) == 1
    assert refs[0].field_path == "source.options.system_prompt"
    assert refs[0].blob_id == UUID(BLOB1)


def test_discover_in_node_options() -> None:
    config = {
        "transforms": [
            {"name": "classify", "plugin": "llm", "options": {"system_prompt": _marker()}},
            {"name": "validate", "plugin": "rules", "options": {}},
        ]
    }
    refs = _discover_blob_content_refs(config)
    assert len(refs) == 1
    assert refs[0].field_path == "node:classify.options.system_prompt"


def test_discover_in_output_options() -> None:
    """Composer-state form uses `outputs`."""
    config = {
        "outputs": {
            "writeback": {"plugin": "json", "options": {"body_template": _marker()}},
        }
    }
    refs = _discover_blob_content_refs(config)
    assert len(refs) == 1
    assert refs[0].field_path == "output:writeback.options.body_template"


def test_discover_in_sinks_options() -> None:
    """YAML form uses `sinks` (yaml_generator.py:169 emits doc['sinks']).
    The runtime path round-trips through generated YAML, so the dict the
    discoverer sees at runtime carries `sinks`, not `outputs`.  Both
    surfaces map to the same canonical `output:<name>` field_path prefix.
    """
    config = {
        "sinks": {
            "writeback": {"plugin": "json", "options": {"body_template": _marker()}},
        }
    }
    refs = _discover_blob_content_refs(config)
    assert len(refs) == 1
    assert refs[0].field_path == "output:writeback.options.body_template"


def test_discover_nested_options() -> None:
    config = {
        "source": {
            "options": {
                "auth": {"prompt": _marker()},
            }
        }
    }
    refs = _discover_blob_content_refs(config)
    assert refs[0].field_path == "source.options.auth.prompt"


def test_discover_ignores_bind_source_mode() -> None:
    """Only inline_content mode is collected by the resolver."""
    config = {
        "source": {
            "options": {
                "blob_ref": BLOB1,
                "mode": "bind_source",
                "path": "/data/file.csv",
            }
        }
    }
    refs = _discover_blob_content_refs(config)
    assert refs == []


def test_discover_ignores_secret_refs() -> None:
    config = {"source": {"options": {"api_key": {"secret_ref": "OPENROUTER_KEY"}}}}
    refs = _discover_blob_content_refs(config)
    assert refs == []


def test_discover_multiple_refs_same_blob() -> None:
    """Same blob, multiple field_paths → multiple BlobInlineRef rows.

    Per §6.3, the link site dedupes by blob_id, but the discoverer emits
    one ref per discovered occurrence — the audit table records each
    field_path independently.
    """
    config = {
        "transforms": [
            {"name": "a", "plugin": "llm", "options": {"system_prompt": _marker(BLOB1)}},
            {"name": "b", "plugin": "llm", "options": {"system_prompt": _marker(BLOB1)}},
        ]
    }
    refs = _discover_blob_content_refs(config)
    assert len(refs) == 2
    assert {ref.field_path for ref in refs} == {
        "node:a.options.system_prompt",
        "node:b.options.system_prompt",
    }


def test_discover_surfaces_malformed_marker() -> None:
    """Per spec §5.1, mode-less markers are malformed and surface in the
    BlobContentResolutionError.malformed case.
    """
    from elspeth.contracts.blobs_inline import BlobContentResolutionError

    config = {"source": {"options": {"system_prompt": {"blob_ref": BLOB1}}}}  # no mode
    with pytest.raises(BlobContentResolutionError) as exc_info:
        _discover_blob_content_refs(config)
    assert exc_info.value.malformed
    assert exc_info.value.malformed[0][0] == "source.options.system_prompt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/test_blobs_inline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.core.blobs_inline'`.

- [ ] **Step 3: Implement the discoverer**

```python
# src/elspeth/core/blobs_inline.py (initial implementation — substitute and async fetch follow in tasks 4+5)
"""Inline-content blob resolver — three-function split per spec §6.

Layer: L1 (core). Imports L0 (contracts) only.

Three functions per spec §6.2:

- _discover_blob_content_refs: pure-sync tree walk over a YAML-shape dict.
  Recognises the widened blob_ref marker via is_widened_blob_ref.
  Collects only the inline_content mode. Surfaces malformed markers
  ({blob_ref: ID} with no mode) as BlobContentResolutionError.malformed.

- _fetch_blob_contents: async fetch via BlobServiceImpl.read_blob_content.
  Tier-1 escapes (BlobIntegrityError, BlobContentMissingError) propagate
  uncaught; operational errors collect into BlobContentResolutionError.

- _substitute_blob_content_refs: pure-sync substitution + decode + audit
  list emission. Hash mismatch → Tier-1 escape via BlobIntegrityError
  (NOT batched).

Mirrors core/secrets.py:62-159 in shape; the sync/async split is the
critical departure (H3 in elspeth-fdebcaa79a panel synthesis).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from elspeth.contracts.blobs_inline import (
    BlobContentResolutionError,
    BlobInlineRef,
    is_widened_blob_ref,
)


def _discover_blob_content_refs(config: dict[str, Any]) -> list[BlobInlineRef]:
    """Walk a config dict and return inline_content refs in canonical
    field_path form (§8.1).

    Raises BlobContentResolutionError(malformed=...) if any malformed
    marker (e.g. {blob_ref: ID} with no mode) is encountered. Other
    error categories (missing/oversized/etc.) are reserved for the
    fetch/substitute steps; the discoverer is structural-only.
    """
    refs: list[BlobInlineRef] = []
    malformed: list[tuple[str, str]] = []

    # Source
    source = config.get("source")
    if isinstance(source, Mapping):
        options = source.get("options")
        if isinstance(options, Mapping):
            _walk_options(options, "source.options", refs, malformed)

    # Transforms / nodes (mixed transform/gate/aggregation/coalesce per
    # web/composer/state.py:1348 CompositionState; see `nodes:` field at
    # state.py:1364).  The YAML-serialised form (see
    # web/composer/yaml_generator.py:155 — emits singular `coalesce`) uses
    # the keys: transforms, gates, aggregations, coalesce.  Walk all of
    # them; node identity comes from the entry's `name` field.
    for collection_key in ("transforms", "gates", "aggregations", "coalesce"):
        nodes = config.get(collection_key)
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            name = node.get("name")
            if not isinstance(name, str):
                continue
            options = node.get("options")
            if isinstance(options, Mapping):
                _walk_options(options, f"node:{name}.options", refs, malformed)

    # Outputs / sinks — TWO surfaces:
    #   - Composer-state form (CompositionState.outputs) emits the dict
    #     under the "outputs" key.
    #   - YAML form (yaml_generator.generate_yaml at line 169) emits
    #     `doc["sinks"]` — different key, same structure.
    #   The runtime path round-trips composer state → generate_yaml →
    #   yaml.safe_load → resolved_dict, so the dict reaching the
    #   discoverer at runtime carries `sinks`, not `outputs`.  The
    #   validate path may use either depending on whether the caller
    #   round-trips through YAML first.  Walk BOTH keys; field_path
    #   uses the canonical `output:<sink_name>` prefix in both cases
    #   (the audit row's identity is the sink name, not the dict key).
    for sinks_key in ("outputs", "sinks"):
        sinks = config.get(sinks_key)
        if isinstance(sinks, Mapping):
            for sink_name, sink in sinks.items():
                if not isinstance(sink, Mapping) or not isinstance(sink_name, str):
                    continue
                options = sink.get("options")
                if isinstance(options, Mapping):
                    _walk_options(options, f"output:{sink_name}.options", refs, malformed)

    if malformed:
        raise BlobContentResolutionError(malformed=malformed)
    return refs


def _walk_options(
    obj: Any,
    field_path: str,
    refs: list[BlobInlineRef],
    malformed: list[tuple[str, str]],
) -> None:
    """Recursive walk over an options subtree.

    Mirrors core/secrets.py::_walk shape (sibling pattern). Recognises
    the widened blob_ref marker; only collects inline_content mode;
    surfaces malformed markers (has blob_ref key, fails recognition).
    """
    if isinstance(obj, Mapping):
        # Detect malformed marker before recursing.  A mapping with a
        # blob_ref key that fails recognition is malformed; it must
        # surface, not be silently walked into.
        if "blob_ref" in obj:
            shape = is_widened_blob_ref(obj)
            if shape is None:
                malformed.append((field_path, _malformed_reason(obj)))
                return
            if shape.mode == "inline_content":
                refs.append(BlobInlineRef(
                    field_path=field_path,
                    blob_id=shape.blob_id,
                    sha256=shape.sha256,  # type: ignore[arg-type]  # inline_content guarantees sha256 is set
                    encoding=shape.encoding,
                ))
            # bind_source: skip (collected by source-binding path, not here)
            return
        for key, value in obj.items():
            if not isinstance(key, str):
                continue
            _walk_options(value, f"{field_path}.{key}", refs, malformed)
    elif isinstance(obj, list):
        # Lists inside options are valid (e.g. allowlists), but we don't
        # emit positional field_paths for items inside them; if a marker
        # appears inside a list, surface as malformed (the canonical form
        # has no positional encoding).
        for item in obj:
            if isinstance(item, Mapping) and "blob_ref" in item:
                malformed.append((
                    field_path,
                    "marker inside list — positional field_paths are forbidden",
                ))
                return


def _malformed_reason(obj: Mapping[str, Any]) -> str:
    """Best-effort diagnostic reason for a marker that failed recognition."""
    if "mode" not in obj:
        return "missing mode key (required per ADR-021 §6)"
    mode = obj.get("mode")
    if mode not in ("bind_source", "inline_content"):
        return f"unknown mode {mode!r}"
    if mode == "inline_content" and "sha256" not in obj:
        return "inline_content requires sha256 hash"
    if mode == "bind_source" and ("sha256" in obj or "encoding" in obj):
        return "bind_source must not carry sha256/encoding"
    blob_ref = obj.get("blob_ref")
    if not isinstance(blob_ref, str):
        return "blob_ref must be a UUID string"
    try:
        UUID(blob_ref)
    except ValueError:
        return f"blob_ref {blob_ref!r} is not a valid UUID"
    return "marker shape rejected by is_widened_blob_ref"
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/core/test_blobs_inline.py -v`
Expected: PASS for every test.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/blobs_inline.py tests/unit/core/test_blobs_inline.py
git commit -m "feat(core): land L1 inline-content discoverer

Implements _discover_blob_content_refs per spec §6.2 — pure-sync tree
walk emitting BlobInlineRef objects with canonical field_paths.
Handles source / node / output surfaces. Surfaces malformed markers
({blob_ref: ID} with no mode key) via BlobContentResolutionError.

Mirrors core/secrets.py:_walk shape; the sync/async split (fetch and
substitute steps) follows in subsequent tasks.

Refs: elspeth-fdebcaa79a"
```

---

## Task 4: L1 resolver — `_fetch_blob_contents`

**Files:**
- Modify: `src/elspeth/core/blobs_inline.py` — extend with the async fetch helper
- Modify: `tests/unit/core/test_blobs_inline.py` — add fetch coverage

- [ ] **Step 1: Write failing tests for the async fetch**

```python
# tests/unit/core/test_blobs_inline.py — add the following
import asyncio
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from elspeth.contracts.blobs_inline import BlobInlineRef
from elspeth.core.blobs_inline import _fetch_blob_contents


def _ref(blob_id: str, field_path: str = "source.options.x") -> BlobInlineRef:
    return BlobInlineRef(field_path=field_path, blob_id=UUID(blob_id), sha256="a" * 64, encoding="utf-8")


@pytest.mark.asyncio
async def test_fetch_blob_contents_returns_dict() -> None:
    blob_service = AsyncMock()
    blob_service.read_blob_content.return_value = b"content"
    refs = [_ref(BLOB1)]
    fetched = await _fetch_blob_contents(blob_service, refs)
    assert fetched[refs[0]] == b"content"


@pytest.mark.asyncio
async def test_fetch_blob_contents_dedupes_by_blob_id() -> None:
    """Two refs to the same blob must result in one read_blob_content call."""
    blob_service = AsyncMock()
    blob_service.read_blob_content.return_value = b"content"
    refs = [_ref(BLOB1, "source.options.a"), _ref(BLOB1, "source.options.b")]
    fetched = await _fetch_blob_contents(blob_service, refs)
    blob_service.read_blob_content.assert_called_once_with(UUID(BLOB1))
    assert fetched[refs[0]] == b"content"
    assert fetched[refs[1]] == b"content"


@pytest.mark.asyncio
async def test_fetch_blob_contents_propagates_integrity_error() -> None:
    """BlobIntegrityError is a Tier-1 escape — never batched."""
    from elspeth.web.blobs.protocol import BlobIntegrityError

    blob_service = AsyncMock()
    blob_service.read_blob_content.side_effect = BlobIntegrityError(BLOB1, expected="x"*64, actual="y"*64)
    refs = [_ref(BLOB1)]
    with pytest.raises(BlobIntegrityError):
        await _fetch_blob_contents(blob_service, refs)


@pytest.mark.asyncio
async def test_fetch_blob_contents_collects_not_found() -> None:
    from elspeth.contracts.blobs_inline import BlobContentResolutionError
    from elspeth.web.blobs.protocol import BlobNotFoundError

    blob_service = AsyncMock()
    blob_service.read_blob_content.side_effect = BlobNotFoundError(BLOB1)
    refs = [_ref(BLOB1)]
    with pytest.raises(BlobContentResolutionError) as exc_info:
        await _fetch_blob_contents(blob_service, refs)
    assert exc_info.value.missing == ["source.options.x"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/test_blobs_inline.py::test_fetch_blob_contents_returns_dict -v`
Expected: FAIL with `ImportError: cannot import name '_fetch_blob_contents'`.

- [ ] **Step 3: Add the async fetch helper to `core/blobs_inline.py`**

```python
# Append to src/elspeth/core/blobs_inline.py

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.web.blobs.protocol import BlobServiceProtocol


async def _fetch_blob_contents(
    blob_service: "BlobServiceProtocol",
    refs: list[BlobInlineRef],
) -> dict[BlobInlineRef, bytes]:
    """Async fetch per spec §6.2.

    Issues read_blob_content in parallel via asyncio.gather. Dedupes by
    blob_id (the blob service guarantees identical bytes for identical
    IDs, so a single fetch covers all refs to that blob).

    Tier-1 escapes (BlobIntegrityError, BlobContentMissingError)
    propagate uncaught — they MUST NOT be silently aggregated. Operational
    errors (BlobNotFoundError, BlobStateError) collect into
    BlobContentResolutionError and raise once at end.
    """
    from elspeth.web.blobs.protocol import (
        BlobIntegrityError,
        BlobContentMissingError,
        BlobNotFoundError,
        BlobStateError,
    )

    unique_blob_ids = list({ref.blob_id for ref in refs})
    coros = [blob_service.read_blob_content(blob_id) for blob_id in unique_blob_ids]
    # return_exceptions=True so we collect Tier-2 errors per-blob; Tier-1
    # escapes still propagate because we re-raise them below before the
    # batched-error path.
    results = await asyncio.gather(*coros, return_exceptions=True)

    bytes_by_blob_id: dict[UUID, bytes] = {}
    missing: list[str] = []
    not_ready: list[tuple[str, str]] = []
    refs_for_blob: dict[UUID, list[BlobInlineRef]] = {}
    for ref in refs:
        refs_for_blob.setdefault(ref.blob_id, []).append(ref)

    for blob_id, result in zip(unique_blob_ids, results):
        if isinstance(result, BlobIntegrityError) or isinstance(result, BlobContentMissingError):
            # Tier-1 escape: re-raise immediately, do not batch.
            raise result
        if isinstance(result, BlobNotFoundError):
            for ref in refs_for_blob[blob_id]:
                missing.append(ref.field_path)
            continue
        if isinstance(result, BlobStateError):
            # Status != "ready" — operational, collect into not_ready.
            for ref in refs_for_blob[blob_id]:
                not_ready.append((ref.field_path, "not_ready"))
            continue
        if isinstance(result, BaseException):
            # Anything else propagates — programmer bug, not Tier-2 operational.
            raise result
        bytes_by_blob_id[blob_id] = result

    if missing or not_ready:
        raise BlobContentResolutionError(missing=missing, not_ready=not_ready)

    return {ref: bytes_by_blob_id[ref.blob_id] for ref in refs}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/core/test_blobs_inline.py -v`
Expected: PASS for every test.

- [ ] **Step 5: Relocate blob exceptions + protocol to L0 (UNCONDITIONAL — not a follow-up)**

The L1 resolver imports `BlobIntegrityError`, `BlobContentMissingError`, `BlobNotFoundError`, `BlobStateError` from `web/blobs/protocol.py` (L3) for the Tier-1 escape paths. That is an L1→L3 violation. The CLAUDE.md cross-layer rule is: move down before extracting. Apply the same fix Task 1 used for `AllowedMimeType`.

Move the four exception classes — and `BlobServiceProtocol` itself — into `contracts/blobs.py` (L0). Re-export from `web/blobs/protocol.py` so existing L3 import sites continue to resolve.

Sub-steps:

```bash
# Edit contracts/blobs.py: add the exception family + Protocol from
# web/blobs/protocol.py (preserving frozen-attr machinery).
# Edit web/blobs/protocol.py: replace the local definitions with
#     from elspeth.contracts.blobs import (
#         BlobNotFoundError, BlobActiveRunError, BlobQuotaExceededError,
#         BlobStateError, BlobIntegrityError, BlobContentMissingError,
#         BlobError, BlobServiceProtocol,
#     )
# Re-run the existing tests touching these symbols:
.venv/bin/python -m pytest tests/unit/web/blobs/ tests/unit/contracts/ -v
```

Then remove the `TYPE_CHECKING` shim in `core/blobs_inline.py` (the `BlobServiceProtocol` is now an L0 type) and the per-function `from elspeth.web.blobs.protocol import ...` lines (replace with `from elspeth.contracts.blobs import ...`).

- [ ] **Step 6: Run the layer enforcer**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: exit 0. After the relocation, the L1 resolver imports only from L0; no allowlist entry needed.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/core/blobs_inline.py tests/unit/core/test_blobs_inline.py
git commit -m "feat(core): land async fetch helper for inline-content resolver

Implements _fetch_blob_contents per spec §6.2 — async parallel fetch
with dedupe by blob_id. Tier-1 escapes (BlobIntegrityError,
BlobContentMissingError) propagate uncaught; operational errors
(BlobNotFoundError → missing, BlobStateError → not_ready) collect
into BlobContentResolutionError.

Refs: elspeth-fdebcaa79a"
```

---

## Task 5: L1 resolver — `_substitute_blob_content_refs`

**Files:**
- Modify: `src/elspeth/core/blobs_inline.py` — add the substituter
- Modify: `tests/unit/core/test_blobs_inline.py` — add substitute coverage

- [ ] **Step 1: Write failing tests for substitute + decode + hash verification**

```python
# tests/unit/core/test_blobs_inline.py — add the following

import hashlib
from copy import deepcopy

from elspeth.contracts.blobs_inline import BlobInlineRef, ResolvedBlobContent
from elspeth.core.blobs_inline import _substitute_blob_content_refs


def _ref_with_hash(content: bytes, field_path: str = "source.options.system_prompt", blob_id: str = BLOB1) -> BlobInlineRef:
    return BlobInlineRef(
        field_path=field_path,
        blob_id=UUID(blob_id),
        sha256=hashlib.sha256(content).hexdigest(),
        encoding="utf-8",
    )


def test_substitute_replaces_marker_with_decoded_string() -> None:
    content = b"You are a helpful assistant."
    ref = _ref_with_hash(content)
    config = {"source": {"options": {"system_prompt": {"blob_ref": BLOB1, "mode": "inline_content", "sha256": ref.sha256}}}}
    substituted, audit = _substitute_blob_content_refs(
        deepcopy(config), {ref: content}, refs=[ref], blob_metadata={ref.blob_id: ("text/plain", len(content))},
    )
    assert substituted["source"]["options"]["system_prompt"] == "You are a helpful assistant."
    assert len(audit) == 1
    assert isinstance(audit[0], ResolvedBlobContent)
    assert audit[0].byte_length == len(content)


def test_substitute_raises_on_hash_mismatch() -> None:
    """Hash mismatch is a Tier-1 escape — never batched."""
    from elspeth.web.blobs.protocol import BlobIntegrityError

    content = b"You are a helpful assistant."
    ref = BlobInlineRef(
        field_path="source.options.system_prompt",
        blob_id=UUID(BLOB1),
        sha256="z" * 64,  # wrong hash
        encoding="utf-8",
    )
    config = {"source": {"options": {"system_prompt": {"blob_ref": BLOB1, "mode": "inline_content", "sha256": ref.sha256}}}}
    with pytest.raises(BlobIntegrityError):
        _substitute_blob_content_refs(
            config, {ref: content}, refs=[ref], blob_metadata={ref.blob_id: ("text/plain", len(content))},
        )


def test_substitute_collects_decode_failures() -> None:
    from elspeth.contracts.blobs_inline import BlobContentResolutionError

    content = b"\xff\xfe\xfd"  # invalid UTF-8
    ref = BlobInlineRef(
        field_path="source.options.x",
        blob_id=UUID(BLOB1),
        sha256=hashlib.sha256(content).hexdigest(),
        encoding="utf-8",
    )
    config = {"source": {"options": {"x": {"blob_ref": BLOB1, "mode": "inline_content", "sha256": ref.sha256}}}}
    with pytest.raises(BlobContentResolutionError) as exc_info:
        _substitute_blob_content_refs(
            config, {ref: content}, refs=[ref], blob_metadata={ref.blob_id: ("text/plain", len(content))},
        )
    assert exc_info.value.undecodable == [("source.options.x", "utf-8")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/test_blobs_inline.py::test_substitute_replaces_marker_with_decoded_string -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add the substituter**

```python
# Append to src/elspeth/core/blobs_inline.py

import hmac
from collections.abc import MutableMapping

from elspeth.contracts.blobs_inline import ResolvedBlobContent


def _substitute_blob_content_refs(
    config: dict[str, Any],
    fetched: dict[BlobInlineRef, bytes],
    *,
    refs: list[BlobInlineRef],
    blob_metadata: dict[UUID, tuple[str, int]],
) -> tuple[dict[str, Any], list[ResolvedBlobContent]]:
    """Substitute markers with decoded bytes per spec §6.2.

    Hash verification: each fetched ref's bytes are re-hashed and
    compared (via hmac.compare_digest) against the marker's pinned
    sha256. Mismatch → BlobIntegrityError (Tier-1 escape, NOT batched).

    Decode failures collect into BlobContentResolutionError.undecodable.

    Per §8.1 audit-row schema, blob_metadata carries (mime_type, byte_length)
    per blob_id so the ResolvedBlobContent can be constructed without a
    second blob-service round-trip. The caller is responsible for
    populating it (typically from BlobServiceImpl.get_blob; runtime path
    has already paid the cost during lifecycle pinning).
    """
    from elspeth.web.blobs.protocol import BlobIntegrityError

    audit: list[ResolvedBlobContent] = []
    undecodable: list[tuple[str, str]] = []

    for ref in refs:
        content = fetched[ref]
        actual_hash = hashlib.sha256(content).hexdigest()
        if not hmac.compare_digest(actual_hash, ref.sha256):
            raise BlobIntegrityError(str(ref.blob_id), expected=ref.sha256, actual=actual_hash)

        try:
            decoded = content.decode(ref.encoding)
        except UnicodeDecodeError:
            undecodable.append((ref.field_path, ref.encoding))
            continue

        # Substitute at the canonical field_path.
        _substitute_at_path(config, ref.field_path, decoded)

        mime_type, byte_length = blob_metadata[ref.blob_id]
        audit.append(ResolvedBlobContent(
            field_path=ref.field_path,
            blob_id=ref.blob_id,
            content_hash=actual_hash,
            byte_length=byte_length,
            mime_type=mime_type,  # type: ignore[arg-type]  # blob service guarantees AllowedMimeType
            encoding=ref.encoding,
        ))

    if undecodable:
        raise BlobContentResolutionError(undecodable=undecodable)

    return config, audit


def _substitute_at_path(config: MutableMapping[str, Any], field_path: str, value: str) -> None:
    """Walk to the marker location identified by canonical field_path
    and replace it with the decoded string.

    Field_path syntax (§8.1):
      source.options.<key>[.<sub>...]
      node:<id>.options.<key>[.<sub>...]
      output:<name>.options.<key>[.<sub>...]
    """
    prefix, _, rest = field_path.partition(".options.")
    keys = rest.split(".")

    if prefix == "source":
        container = config["source"]["options"]
    elif prefix.startswith("node:"):
        node_id = prefix[len("node:"):]
        container = _find_node_options(config, node_id)
    elif prefix.startswith("output:"):
        sink_name = prefix[len("output:"):]
        # Both `outputs` (composer state) and `sinks` (YAML form) map to
        # the same canonical `output:<name>` prefix — substitute into
        # whichever key is present.
        for sinks_key in ("outputs", "sinks"):
            sinks = config.get(sinks_key)
            if isinstance(sinks, dict) and sink_name in sinks:
                container = sinks[sink_name]["options"]
                break
        else:
            raise KeyError(
                f"Sink {sink_name!r} not found under either 'outputs' or 'sinks' key"
            )
    else:
        raise ValueError(f"Unrecognised field_path prefix: {prefix!r}")

    for key in keys[:-1]:
        container = container[key]
    container[keys[-1]] = value


def _find_node_options(config: MutableMapping[str, Any], node_id: str) -> MutableMapping[str, Any]:
    for collection_key in ("transforms", "gates", "aggregations", "coalesce"):
        nodes = config.get(collection_key)
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if isinstance(node, dict) and node.get("name") == node_id:
                return node["options"]
    raise KeyError(f"Node {node_id!r} not found in any of transforms/gates/aggregations/coalesce")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/core/test_blobs_inline.py -v`
Expected: PASS for every test.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/blobs_inline.py tests/unit/core/test_blobs_inline.py
git commit -m "feat(core): land substituter + hash verification for inline-content resolver

Implements _substitute_blob_content_refs per spec §6.2 — sync
substitution that:
- Re-hashes fetched bytes; mismatch → BlobIntegrityError (Tier-1
  escape, NOT batched)
- Decodes per the ref's encoding; failures collect into
  BlobContentResolutionError.undecodable
- Walks the canonical field_path (source/node:/output:) and writes
  the decoded string in place
- Emits ResolvedBlobContent audit records keyed by canonical
  field_path

Closes the L1 resolver three-function split.

Refs: elspeth-fdebcaa79a"
```

---

## Task 6: Validate-side helper — `_validate_blob_content_refs`

**Files:**
- Modify: `src/elspeth/core/blobs_inline.py` — add the validate-path helper
- Modify: `tests/unit/core/test_blobs_inline.py` — coverage

- [ ] **Step 1: Write failing test**

```python
# tests/unit/core/test_blobs_inline.py — add

from unittest.mock import AsyncMock

from elspeth.core.blobs_inline import _validate_blob_content_refs


@pytest.mark.asyncio
async def test_validate_blob_content_refs_returns_violations_does_not_raise() -> None:
    """Per spec §6.4 — validate path returns structured violations,
    never raises operational errors. Tier-1 anomalies still propagate.
    """
    blob_service = AsyncMock()
    blob_service.get_blob.side_effect = BlobNotFoundError(BLOB1)
    config = {"source": {"options": {"x": {"blob_ref": BLOB1, "mode": "inline_content", "sha256": "a"*64}}}}
    violations = await _validate_blob_content_refs(blob_service, config, user_id="u")
    # Returns structured violations, doesn't raise
    assert any("missing" in v.category for v in violations)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/test_blobs_inline.py::test_validate_blob_content_refs_returns_violations_does_not_raise -v`
Expected: FAIL.

- [ ] **Step 3: Add the validate helper to `core/blobs_inline.py`**

```python
# Append to src/elspeth/core/blobs_inline.py

@dataclass(frozen=True, slots=True)
class BlobInlineValidationViolation:
    """Structured validation violation (returned, never raised) per §6.4."""
    category: str  # one of: missing, oversized, not_ready, hash_mismatch, malformed, mime_encoding_incompatible
    field_path: str
    detail: str


async def _validate_blob_content_refs(
    blob_service: "BlobServiceProtocol",
    config: dict[str, Any],
    *,
    user_id: str,
    per_ref_byte_cap: int | None = None,  # if None, uses ADR-021 cap
    aggregate_byte_cap: int | None = None,  # if None, uses ADR-021 cap
) -> list[BlobInlineValidationViolation]:
    """Validate-path helper per spec §6.4.

    Returns a list of violations; NEVER raises on operational errors
    (the caller assembles them into ValidationResult). Tier-1 anomalies
    still propagate — those are runtime contract violations the
    composer cannot recover from.
    """
    from elspeth.web.blobs.protocol import (
        BlobNotFoundError,
        BlobStateError,
    )
    try:
        refs = _discover_blob_content_refs(config)
    except BlobContentResolutionError as exc:
        # Malformed markers surface as violations, not raised errors.
        return [
            BlobInlineValidationViolation(category="malformed", field_path=fp, detail=reason)
            for fp, reason in exc.malformed
        ]

    violations: list[BlobInlineValidationViolation] = []
    aggregate_bytes = 0
    for ref in refs:
        try:
            record = await blob_service.get_blob(ref.blob_id)
        except BlobNotFoundError:
            violations.append(BlobInlineValidationViolation(
                category="missing", field_path=ref.field_path, detail=f"blob {ref.blob_id} not found",
            ))
            continue
        except BlobStateError as exc:
            violations.append(BlobInlineValidationViolation(
                category="not_ready", field_path=ref.field_path, detail=str(exc),
            ))
            continue
        if record.content_hash != ref.sha256:
            violations.append(BlobInlineValidationViolation(
                category="hash_mismatch",
                field_path=ref.field_path,
                detail=f"composer-pinned hash {ref.sha256[:16]}... != blob content_hash {record.content_hash[:16]}...",
            ))
            continue
        if per_ref_byte_cap is not None and record.size_bytes > per_ref_byte_cap:
            violations.append(BlobInlineValidationViolation(
                category="oversized", field_path=ref.field_path,
                detail=f"{record.size_bytes} bytes exceeds per-ref cap {per_ref_byte_cap}",
            ))
        aggregate_bytes += record.size_bytes

    if aggregate_byte_cap is not None and aggregate_bytes > aggregate_byte_cap:
        violations.append(BlobInlineValidationViolation(
            category="oversized",
            field_path="(aggregate)",
            detail=f"total resolved bytes {aggregate_bytes} exceeds aggregate cap {aggregate_byte_cap}",
        ))

    return violations
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/core/test_blobs_inline.py -v`
Expected: PASS.

- [ ] **Step 5: Add the sync sibling `_validate_blob_content_refs_sync`**

`validate_pipeline` (`web/execution/validation.py:208`) is sync and has no `_call_async` access — it lives at L2/L3 and runs in a worker thread spawned by `_validate_async`.  The sync helper takes a sync callable `(blob_id) -> BlobRecord | None` and the caller at `_validate_async` constructs that closure (it can reach `_call_async`).

```python
# Append to src/elspeth/core/blobs_inline.py

from collections.abc import Callable

def _validate_blob_content_refs_sync(
    blob_get_metadata: Callable[[UUID], "BlobRecord | None"],
    config: dict[str, Any],
    *,
    per_ref_byte_cap: int | None = None,
    aggregate_byte_cap: int | None = None,
) -> list[BlobInlineValidationViolation]:
    """Sync sibling of `_validate_blob_content_refs` per spec §6.4.

    Takes a sync callable for blob metadata lookup so the validate path
    (worker thread, no `_call_async` available) can use the same
    discovery + violation-aggregation logic without touching asyncio.

    Returns a list of violations; NEVER raises on operational errors
    (caller assembles them into ValidationResult). Tier-1 anomalies
    still propagate.
    """
    try:
        refs = _discover_blob_content_refs(config)
    except BlobContentResolutionError as exc:
        return [
            BlobInlineValidationViolation(category="malformed", field_path=fp, detail=reason)
            for fp, reason in exc.malformed
        ]

    violations: list[BlobInlineValidationViolation] = []
    aggregate_bytes = 0
    for ref in refs:
        record = blob_get_metadata(ref.blob_id)
        if record is None:
            violations.append(BlobInlineValidationViolation(
                category="missing", field_path=ref.field_path,
                detail=f"blob {ref.blob_id} not found",
            ))
            continue
        if record.status != "ready":
            violations.append(BlobInlineValidationViolation(
                category="not_ready", field_path=ref.field_path,
                detail=f"blob {ref.blob_id} status is {record.status!r}",
            ))
            continue
        if record.content_hash != ref.sha256:
            violations.append(BlobInlineValidationViolation(
                category="hash_mismatch", field_path=ref.field_path,
                detail=f"composer-pinned hash {ref.sha256[:16]}... != blob {record.content_hash[:16]}...",
            ))
            continue
        if per_ref_byte_cap is not None and record.size_bytes > per_ref_byte_cap:
            violations.append(BlobInlineValidationViolation(
                category="oversized", field_path=ref.field_path,
                detail=f"{record.size_bytes} bytes > per-ref cap {per_ref_byte_cap}",
            ))
        aggregate_bytes += record.size_bytes

    if aggregate_byte_cap is not None and aggregate_bytes > aggregate_byte_cap:
        violations.append(BlobInlineValidationViolation(
            category="oversized", field_path="(aggregate)",
            detail=f"total resolved bytes {aggregate_bytes} > aggregate cap {aggregate_byte_cap}",
        ))

    return violations
```

Add a unit test:

```python
# tests/unit/core/test_blobs_inline.py — add

def test_validate_sync_returns_violations_does_not_raise():
    """Sync sibling for the validate path — same shape as async, no asyncio."""
    from elspeth.core.blobs_inline import _validate_blob_content_refs_sync
    config = {"source": {"plugin": "csv", "options": {
        "x": {"blob_ref": BLOB1, "mode": "inline_content", "sha256": "a"*64},
    }}}
    violations = _validate_blob_content_refs_sync(
        blob_get_metadata=lambda _bid: None,  # missing
        config=config,
    )
    assert any(v.category == "missing" for v in violations)
```

- [ ] **Step 6: Commit + open PR**

```bash
git add src/elspeth/core/blobs_inline.py tests/unit/core/test_blobs_inline.py
git commit -m "feat(core): land validate-side helpers for inline-content resolver

Implements _validate_blob_content_refs (async, runtime-path-compatible)
and _validate_blob_content_refs_sync (sync, validate-path; takes a
sync get_metadata callable bridged by the caller).  Both return
BlobInlineValidationViolation rows without raising on operational
errors per spec §6.4.

Closes P2 of elspeth-fdebcaa79a (L0 contracts + L1 resolver).
Next phase: P3 wires the resolver into _run_pipeline.

Refs: elspeth-fdebcaa79a"

git push -u origin <branch-name>
gh pr create --title "feat(blobs): widened blob_ref L0/L1 resolver foundation" --body "$(cat <<'EOF'
## Summary

- Adds L0 contracts module (\`contracts/blobs_inline.py\`): \`is_widened_blob_ref\` recognition, \`WidenedBlobRefShape\` / \`BlobInlineRef\` / \`ResolvedBlobContent\` value objects, \`ContentEncoding\` closed set, \`BlobContentResolutionError\` batched error
- Adds L1 resolver module (\`core/blobs_inline.py\`): three-function split (\`_discover_blob_content_refs\` sync, \`_fetch_blob_contents\` async, \`_substitute_blob_content_refs\` sync) plus validate-path helper
- Relocates \`AllowedMimeType\` from L3 to L0 per CLAUDE.md cross-layer resolution rule
- No behaviour change at runtime or composer layer — P3 wires the resolver in

## Test plan

- [ ] All new unit tests pass: \`pytest tests/unit/contracts/test_blobs_inline.py tests/unit/core/test_blobs_inline.py -v\`
- [ ] \`enforce_tier_model.py check\` exits 0
- [ ] Existing \`tests/unit/web/blobs/\` tests unchanged (no regression from \`AllowedMimeType\` relocation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Done conditions

P2 is done when:

1. `src/elspeth/contracts/blobs_inline.py` exists with full coverage.
2. `src/elspeth/core/blobs_inline.py` exists with all four functions: `_discover_blob_content_refs`, `_fetch_blob_contents`, `_substitute_blob_content_refs`, `_validate_blob_content_refs`.
3. `AllowedMimeType` is relocated to `contracts/blobs.py` and re-exported from `web/blobs/protocol.py`.
4. `pytest tests/unit/contracts/test_blobs_inline.py tests/unit/core/test_blobs_inline.py -v` passes.
5. `pytest tests/unit/web/blobs/ -v` passes (no regression from the relocation).
6. `enforce_tier_model.py check` exits 0.
7. PR is merged.

Move to `2026-05-03-config-content-ref-phase-3-runtime-preflight.md` only after P2 is merged.
