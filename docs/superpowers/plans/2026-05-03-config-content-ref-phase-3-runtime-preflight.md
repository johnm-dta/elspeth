# Phase 3 — Runtime Preflight + Audit + Lifecycle Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the L1 resolver into `_run_pipeline` fail-closed; add the `blob_inline_resolutions` audit table; extend lifecycle pinning to non-source blob refs; apply bug-verification protocol at the runtime-resolver and audit-write fix sites. Composer authoring path remains closed (P4) — P3 ships a runtime that *would* honour widened-blob_ref markers if any existed in `composition_states`, but no marker emission path is open yet.

**Architecture:** Two new sites in `src/elspeth/web/execution/service.py::_run_pipeline` immediately after the existing `resolve_secret_refs` block (line 727 today). One new SQLAlchemy table + service method (`SessionsService.record_blob_inline_resolutions`). Lifecycle pinning extends from "source blob references" to "every blob reference discovered in the resolved config." Audit-primacy invariant: the audit row exists in DB before the bytes flow into plugin instantiation.

**Tech Stack:** SQLAlchemy 2.x (sync `Engine`), Pydantic 2.x, `BlobServiceImpl` async-over-sync pattern (matches `web/blobs/service.py`), OpenTelemetry counters, pytest.

---

## Pre-phase verification gate (mandatory — fails P3 if any check fails)

Per spec §7.3, P3 cannot begin until two structural checks pass.

- [ ] **Step 1: Direction-segregation check**

```bash
grep -nR --include='*.py' "blob_run_links\b" src/elspeth/ | grep -E '\.direction|direction\b' | tee /tmp/p3-direction-check.txt
```

Expected findings (per spec §7.3): only `_assert_blob_run_same_session`, `link_blob_to_run` (write-side), `finalize_run_output_blobs` (filters `direction == 'output'` — doesn't conflict with `direction == 'input'` reuse), `_row_to_link_record` (read-side guard). No new query that filters by direction in a way that would make config-content reads invisible to existing source-data tooling.

If a new direction-segregating query is found:
- The S5 reuse claim collapses.
- P3 introduces a new column (`link_kind` ENUM(`source_data`, `inline_content`)) on `blob_run_links` instead of reusing direction.
- Update spec §4 (decision row "Lifecycle pinning") with the new column rationale.
- Update this plan with the column-add steps.
- File a follow-up issue documenting the schema diversion.

- [ ] **Step 2: Unique-constraint check**

```bash
grep -n "uq_blob_run_link\|UniqueConstraint.*blob_run_links" src/elspeth/web/sessions/models.py
```

Expected output (verified at spec authoring time): `UniqueConstraint("blob_id", "run_id", "direction", name="uq_blob_run_link")` at `src/elspeth/web/sessions/models.py:240`.

If the constraint shape diverges from `(blob_id, run_id, direction)`:
- The dedupe-by-blob-id strategy in spec §6.3 is invalid for this codebase.
- Either tighten the constraint OR adopt SAVEPOINT-and-IGNORE-on-conflict at the link site.
- Update this plan with the chosen fallback.

- [ ] **Step 3: Mark the gate complete**

```bash
git add /tmp/p3-direction-check.txt
echo "P3 verification gate cleared at $(date -u +%FT%TZ)" > docs/superpowers/plans/2026-05-03-config-content-ref-phase-3-runtime-preflight.gate.md
git add docs/superpowers/plans/2026-05-03-config-content-ref-phase-3-runtime-preflight.gate.md
git commit -m "chore(plan): clear P3 verification gate for direction reuse + unique-constraint shape

Refs: elspeth-fdebcaa79a"
```

If either check failed, do NOT commit the gate file — instead, file the diversion follow-up and return to the gate after the upstream fix.

---

## Task 1: Add `blob_inline_resolutions` table

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`
- Test: `tests/unit/web/sessions/test_blob_inline_resolutions_schema.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/web/sessions/test_blob_inline_resolutions_schema.py
"""Pin the blob_inline_resolutions table shape per spec §8.1."""

from datetime import datetime, UTC
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions.models import (
    blob_inline_resolutions_table,
    metadata,
)


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    return eng


def test_blob_inline_resolutions_round_trip(engine) -> None:
    run_id = str(uuid4())
    blob_id = str(uuid4())
    with engine.begin() as conn:
        conn.execute(blob_inline_resolutions_table.insert().values(
            run_id=run_id,
            attempt=1,
            field_path="source.options.system_prompt",
            blob_id=blob_id,
            content_hash="a" * 64,
            byte_length=42,
            mime_type="text/plain",
            encoding="utf-8",
            resolved_at=datetime.now(UTC),
        ))
    with engine.connect() as conn:
        rows = conn.execute(select(blob_inline_resolutions_table)).fetchall()
        assert len(rows) == 1
        assert rows[0].field_path == "source.options.system_prompt"


def test_blob_inline_resolutions_field_path_check_rejects_positional(engine) -> None:
    run_id = str(uuid4())
    blob_id = str(uuid4())
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(blob_inline_resolutions_table.insert().values(
                run_id=run_id,
                attempt=1,
                field_path="transforms[2].options.x",  # positional — forbidden
                blob_id=blob_id,
                content_hash="a" * 64,
                byte_length=10,
                mime_type="text/plain",
                encoding="utf-8",
                resolved_at=datetime.now(UTC),
            ))


def test_blob_inline_resolutions_encoding_check_rejects_unknown(engine) -> None:
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(blob_inline_resolutions_table.insert().values(
                run_id=str(uuid4()),
                attempt=1,
                field_path="source.options.x",
                blob_id=str(uuid4()),
                content_hash="a" * 64,
                byte_length=10,
                mime_type="text/plain",
                encoding="ascii",  # not in closed set
                resolved_at=datetime.now(UTC),
            ))


def test_blob_inline_resolutions_attempt_supports_resume(engine) -> None:
    """Same (run_id, field_path, blob_id) with different attempt = OK."""
    run_id = str(uuid4())
    blob_id = str(uuid4())
    with engine.begin() as conn:
        for attempt in (1, 2):
            conn.execute(blob_inline_resolutions_table.insert().values(
                run_id=run_id, attempt=attempt,
                field_path="source.options.x",
                blob_id=blob_id,
                content_hash="a" * 64,
                byte_length=10,
                mime_type="text/plain",
                encoding="utf-8",
                resolved_at=datetime.now(UTC),
            ))
    with engine.connect() as conn:
        rows = conn.execute(select(blob_inline_resolutions_table)).fetchall()
        assert len(rows) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_blob_inline_resolutions_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'blob_inline_resolutions_table'`.

- [ ] **Step 3: Add the table to `web/sessions/models.py`**

Locate the existing `blob_run_links_table = Table(...)` definition (currently `web/sessions/models.py:224-247`) and append the new table after it:

```python
# src/elspeth/web/sessions/models.py — append after blob_run_links_table

blob_inline_resolutions_table = Table(
    "blob_inline_resolutions",
    metadata,
    Column("run_id", String, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
    Column("attempt", Integer, nullable=False, server_default="1"),
    Column("field_path", String, nullable=False),
    Column("blob_id", String, ForeignKey("blobs.id"), nullable=False),
    Column("content_hash", String, nullable=False),
    Column("byte_length", Integer, nullable=False),
    Column("mime_type", String, nullable=False),
    Column("encoding", String, nullable=False),
    Column("resolved_at", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint("run_id", "field_path", "blob_id", "attempt", name="pk_blob_inline_resolutions"),
    CheckConstraint(
        "length(content_hash) = 64",
        name="ck_blob_inline_resolutions_hash_format",
    ),
    CheckConstraint(
        "encoding IN ('utf-8', 'utf-8-sig', 'utf-16', 'latin-1')",
        name="ck_blob_inline_resolutions_encoding",
    ),
    CheckConstraint(
        "field_path LIKE 'source.options.%' "
        "OR field_path LIKE 'node:%.options.%' "
        "OR field_path LIKE 'output:%.options.%'",
        name="ck_blob_inline_resolutions_field_path",
    ),
    CheckConstraint("byte_length >= 0", name="ck_blob_inline_resolutions_byte_length"),
)

Index("ix_blob_inline_resolutions_blob_id", blob_inline_resolutions_table.c.blob_id)
Index("ix_blob_inline_resolutions_run_id", blob_inline_resolutions_table.c.run_id)
```

Add the imports at the top of the file if missing: `CheckConstraint`, `PrimaryKeyConstraint`, `DateTime`, `Integer`, `String`, `ForeignKey`, `Table`, `Column`, `Index` should already exist; `from sqlalchemy import` covers them.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_blob_inline_resolutions_schema.py -v`
Expected: PASS for every test.

- [ ] **Step 5: Update the dev DB note**

Per the project's `db_migration_policy` (operator deletes the dev/staging DB), no Alembic migration is needed. Add a one-line operator note to the deploy runbook:

```bash
# Edit the staging deploy runbook to note the schema change
grep -l "DELETE FROM" docs/runbooks/*.md
```

If no runbook covers this case, add a section to the most relevant existing runbook:

```markdown
## P3 of elspeth-fdebcaa79a (widened blob_ref) — staging deploy step

Before deploying P3:

```sql
DROP TABLE IF EXISTS blob_inline_resolutions;
-- The new schema is created on next service start by metadata.create_all().
```

Production deploy: pre-RC5 there are no production deployments to migrate.
```

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/models.py tests/unit/web/sessions/test_blob_inline_resolutions_schema.py docs/runbooks/
git commit -m "feat(sessions): add blob_inline_resolutions audit table

Per spec §8.1 of elspeth-fdebcaa79a — new Tier-1 audit table
recording (run_id, attempt, field_path, blob_id, content_hash,
byte_length, mime_type, encoding, resolved_at) for every inline-content
ref resolved at run time. Composite PK supports elspeth resume
semantics via the attempt column. CHECK constraints pin the canonical
field_path format and the closed-set encoding values.

Refs: elspeth-fdebcaa79a"
```

---

## Task 2: `SessionsService.record_blob_inline_resolutions` service method

**Files:**
- Modify: `src/elspeth/web/sessions/service.py` (or wherever `SessionsService` lives — verify path)
- Modify: `src/elspeth/web/sessions/protocol.py` (add to Protocol if applicable)
- Test: `tests/unit/web/sessions/test_record_blob_inline_resolutions.py`

- [ ] **Step 1: Locate SessionsService**

```bash
grep -rn "class SessionsServiceImpl\|class SessionsService" src/elspeth/web/sessions/ | head -5
```

Use the located paths in subsequent steps.

- [ ] **Step 2: Write failing test**

```python
# tests/unit/web/sessions/test_record_blob_inline_resolutions.py
"""Pin record_blob_inline_resolutions: audit-write primacy + Tier-1 escape."""

import asyncio
from datetime import datetime, UTC
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from elspeth.contracts.blobs_inline import ResolvedBlobContent
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.sessions.models import blob_inline_resolutions_table, metadata
from elspeth.web.sessions.service import SessionsServiceImpl


@pytest.mark.asyncio
async def test_record_writes_one_row_per_resolution() -> None:
    eng = create_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    service = SessionsServiceImpl(engine=eng)
    run_id = uuid4()
    blob_id = uuid4()
    resolutions = [ResolvedBlobContent(
        field_path="source.options.system_prompt",
        blob_id=blob_id,
        content_hash="a" * 64,
        byte_length=42,
        mime_type="text/plain",
        encoding="utf-8",
    )]
    # Pre-create a runs row so the FK doesn't fire (skip if test infra
    # already provides this fixture).
    # … fixture detail …
    await service.record_blob_inline_resolutions(run_id=run_id, resolutions=resolutions, attempt=1)
    with eng.connect() as conn:
        rows = conn.execute(select(blob_inline_resolutions_table)).fetchall()
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_record_raises_audit_integrity_error_on_db_failure() -> None:
    """A DB-level failure to write the audit row is a Tier-1 anomaly —
    the run cannot proceed, bytes must not flow into plugin instantiation.
    """
    eng = create_engine("sqlite:///:memory:")
    # Don't run metadata.create_all → table doesn't exist → INSERT fails
    service = SessionsServiceImpl(engine=eng)
    with pytest.raises(AuditIntegrityError):
        await service.record_blob_inline_resolutions(
            run_id=uuid4(),
            resolutions=[ResolvedBlobContent(
                field_path="source.options.x", blob_id=uuid4(),
                content_hash="a"*64, byte_length=1, mime_type="text/plain", encoding="utf-8",
            )],
            attempt=1,
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_record_blob_inline_resolutions.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement the method on `SessionsServiceImpl`**

```python
# src/elspeth/web/sessions/service.py — add method to SessionsServiceImpl

async def record_blob_inline_resolutions(
    self,
    *,
    run_id: UUID,
    resolutions: list[ResolvedBlobContent],
    attempt: int = 1,
) -> None:
    """Write Tier-1 audit rows for resolved inline-content refs.

    Per spec §8.2: this MUST land in DB before the resolved bytes flow
    into plugin instantiation. The caller in _run_pipeline invokes this
    via _call_async immediately after _substitute_blob_content_refs;
    failure here propagates to the run-failed branch and the bytes
    never reach plugin construction.

    Raises AuditIntegrityError on any DB-level failure (Tier-1 escape).
    """
    if not resolutions:
        return

    run_id_str = str(run_id)
    now = datetime.now(UTC)

    def _sync() -> None:
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    blob_inline_resolutions_table.insert(),
                    [
                        {
                            "run_id": run_id_str,
                            "attempt": attempt,
                            "field_path": r.field_path,
                            "blob_id": str(r.blob_id),
                            "content_hash": r.content_hash,
                            "byte_length": r.byte_length,
                            "mime_type": r.mime_type,
                            "encoding": r.encoding,
                            "resolved_at": now,
                        }
                        for r in resolutions
                    ],
                )
        except SQLAlchemyError as exc:
            raise AuditIntegrityError(
                f"Tier 1: failed to record blob_inline_resolutions for run {run_id_str}: {exc}"
            ) from exc

    await self._run_sync(_sync)
```

Add the matching async signature to the `SessionsServiceProtocol` Protocol if one exists.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_record_blob_inline_resolutions.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py tests/unit/web/sessions/test_record_blob_inline_resolutions.py
git commit -m "feat(sessions): add record_blob_inline_resolutions service method

Async-over-sync writer for the new blob_inline_resolutions audit table.
DB failure raises AuditIntegrityError (Tier-1 escape), propagating to
_run_pipeline's run-failed branch — bytes never reach plugin
instantiation if the audit row cannot be written.

Refs: elspeth-fdebcaa79a"
```

---

## Task 3: Wire the resolver into `_run_pipeline`

**Files:**
- Modify: `src/elspeth/web/execution/service.py:_run_pipeline` (resolver wiring after `resolve_secret_refs` block at line ~733)
- Test: `tests/integration/web/test_blob_inline_runtime_preflight.py`

- [ ] **Step 1: Re-read the existing `_run_pipeline` block**

```bash
sed -n '710,770p' src/elspeth/web/execution/service.py
```

Confirm the call site shape: `resolved_dict, resolutions = resolve_secret_refs(...)` → `resolved_yaml = _yaml.dump(resolved_dict, ...)` → `settings = load_settings_from_yaml_string(resolved_yaml)`. The new block lands between the secret resolve and the YAML dump.

- [ ] **Step 2: Write failing integration test**

```python
# tests/integration/web/test_blob_inline_runtime_preflight.py
"""End-to-end: composer-pinned blob_inline_ref resolves at runtime.

This is the integration-level pin for spec §6.3 (composition site)
and §8.2 (audit-row primacy).
"""

import hashlib
from uuid import UUID

import pytest

from elspeth.web.sessions.models import blob_inline_resolutions_table


@pytest.mark.integration
def test_runtime_resolves_inline_content_ref(client, blob_service, session_service):
    # Arrange: create a session, upload a blob with known content
    content = b"You are a helpful assistant."
    sha256 = hashlib.sha256(content).hexdigest()
    session = session_service.create_session_sync(user_id="u")
    blob = blob_service.create_blob_sync(
        session_id=session.id, filename="prompt.txt", content=content,
        mime_type="text/plain", created_by="user",
    )
    # Compose pipeline with the widened marker
    state = {
        "source": {"plugin": "csv", "options": {"path": "test.csv"}, "on_success": "out"},
        "transforms": [{
            "name": "classify", "plugin": "llm",
            "options": {
                "system_prompt": {
                    "blob_ref": str(blob.id),
                    "mode": "inline_content",
                    "sha256": sha256,
                },
                "api_key": {"secret_ref": "OPENROUTER_KEY"},
            },
            "on_success": "out",
        }],
        "outputs": {"out": {"plugin": "json", "options": {"path": "out.json"}}},
    }
    # Act: persist state, kick off run
    response = client.post(f"/api/sessions/{session.id}/runs", json={"state": state})
    run_id = response.json()["run_id"]

    # Assert: the run completed and the audit row exists
    audit_rows = session_service.query_audit(blob_inline_resolutions_table, run_id=run_id)
    assert len(audit_rows) == 1
    assert audit_rows[0].field_path == "node:classify.options.system_prompt"
    assert audit_rows[0].content_hash == sha256
    assert audit_rows[0].byte_length == len(content)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/web/test_blob_inline_runtime_preflight.py -v`
Expected: FAIL — the resolver isn't wired in yet.

- [ ] **Step 4: Add the wiring in `_run_pipeline`**

Locate the existing block in `src/elspeth/web/execution/service.py` (currently around line 727-733):

```python
# EXISTING:
resolved_dict, resolutions = resolve_secret_refs(
    config_dict, self._secret_service, user_id, env_ref_names=env_ref_names,
)
resolved_yaml = _yaml.dump(resolved_dict, default_flow_style=False)
```

Insert the inline-content resolver between the secret-resolve and the YAML dump:

```python
# NEW BLOCK — insert before resolved_yaml = _yaml.dump(...)

import asyncio  # Add to imports at top of file if not already imported

from elspeth.core.blobs_inline import (
    _discover_blob_content_refs,
    _fetch_blob_contents,
    _substitute_blob_content_refs,
)

# Discover inline-content refs (sync) — see spec §6.3 for the composition
# site rationale.
inline_refs = _discover_blob_content_refs(resolved_dict)

# Lifecycle pinning — link every UNIQUE ref'd blob to this run BEFORE
# fetching bytes.  Dedupe by blob_id: blob_run_links has
# UniqueConstraint(blob_id, run_id, direction) at uq_blob_run_link;
# multiple field_path occurrences of the same blob produce ONE row.
# Per-field_path lifecycle is captured in blob_inline_resolutions
# (Task 1 schema).
unique_blob_ids = sorted({ref.blob_id for ref in inline_refs})  # sort for deterministic ordering
if unique_blob_ids and self._blob_service is not None:
    # `_call_async(coro, ...)` calls `asyncio.run_coroutine_threadsafe` with
    # its first argument.  `asyncio.gather(...)` returns a Future, NOT a
    # coroutine — wrap in a thin `async def` so the argument is a real
    # coroutine.  Same wrap applies to the metadata-fetch site below.
    async def _link_all_blobs() -> None:
        await asyncio.gather(*[
            self._blob_service.link_blob_to_run(
                blob_id=blob_id, run_id=run_uuid, direction="input",
            ) for blob_id in unique_blob_ids
        ])
    self._call_async(_link_all_blobs())

# Fetch bytes (async) — _fetch_blob_contents uses asyncio.gather
# internally so this is a single worker→loop→worker round-trip.
if inline_refs and self._blob_service is not None:
    fetched = self._call_async(_fetch_blob_contents(self._blob_service, inline_refs))

    # Build (mime_type, byte_length) per blob for the audit-row substituter.
    # Same async-def wrap pattern as the link site: gather() returns a
    # Future, _call_async needs a coroutine.
    async def _gather_blob_metadata() -> list[Any]:
        return await asyncio.gather(*[
            self._blob_service.get_blob(blob_id) for blob_id in unique_blob_ids
        ])
    metadata_records = self._call_async(_gather_blob_metadata())
    blob_metadata: dict[UUID, tuple[str, int]] = {
        blob_id: (record.mime_type, record.size_bytes)
        for blob_id, record in zip(unique_blob_ids, metadata_records)
    }

    resolved_dict, blob_resolutions = _substitute_blob_content_refs(
        resolved_dict, fetched, refs=inline_refs, blob_metadata=blob_metadata,
    )

    # Audit primacy: write Tier-1 rows BEFORE bytes flow into plugin
    # instantiation.  AuditIntegrityError propagates uncaught — the run
    # fails, plugins are never constructed against the resolved bytes.
    self._call_async(self._session_service.record_blob_inline_resolutions(
        run_id=run_uuid, resolutions=blob_resolutions, attempt=1,
    ))

resolved_yaml = _yaml.dump(resolved_dict, default_flow_style=False)
```

- [ ] **Step 5: Run integration test**

Run: `.venv/bin/python -m pytest tests/integration/web/test_blob_inline_runtime_preflight.py -v`
Expected: PASS.

- [ ] **Step 6: Add OpenTelemetry counters per spec §1.4**

Locate the existing counter declarations in `web/execution/service.py` (or wherever `composer.requests.*` counters live):

```python
# Add near other counter declarations (search for `Counter(` in service.py)
_BLOB_INLINE_HASH_MISMATCH_TOTAL = meter.create_counter(
    name="composer.blob_inline.hash_mismatch_total",
    description="composer-pinned hash != runtime-fetched hash; SLO threshold = 0",
)
_BLOB_INLINE_AUDIT_ROW_TIER1_VIOLATION_TOTAL = meter.create_counter(
    name="composer.blob_inline.audit_row_tier1_violation_total",
    description="resolved ref produced no audit row; SLO threshold = 0",
)
```

Increment from the appropriate sites: `_BLOB_INLINE_HASH_MISMATCH_TOTAL` in the `BlobIntegrityError` exception path; `_BLOB_INLINE_AUDIT_ROW_TIER1_VIOLATION_TOTAL` only ever fires from a Hypothesis-property-test post-condition assertion (in production it MUST stay zero).

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/execution/service.py tests/integration/web/test_blob_inline_runtime_preflight.py
git commit -m "feat(execution): wire inline-content resolver into _run_pipeline

Per spec §6.3 — discoverer → lifecycle pinning (dedupe by blob_id) →
async fetch → substituter → audit row → plugin instantiation. Audit-
write primacy: bytes never reach plugin construction if the audit row
cannot be written.

OTel counters added: composer.blob_inline.hash_mismatch_total,
composer.blob_inline.audit_row_tier1_violation_total — both with
SLO threshold = 0.

Refs: elspeth-fdebcaa79a"
```

---

## Task 4: Bug-verification at the runtime-resolver fix site

Per spec §9.3 (Sub-pin B) and the agreement-suite docstring's protocol (lines 78-90 of `tests/integration/pipeline/test_composer_runtime_agreement.py`).

- [ ] **Step 1: Manual revert of the resolver wiring**

In `src/elspeth/web/execution/service.py`, comment out the `inline_refs = _discover_blob_content_refs(resolved_dict)` line and the entire block that depends on it.

- [ ] **Step 2: Run the integration test that should now fail**

```bash
.venv/bin/python -m pytest tests/integration/web/test_blob_inline_runtime_preflight.py -v
```

Expected: FAIL — but observe carefully WHAT failure surfaces. The expected mode is "first-row plugin call crashes when the LLM plugin tries to use a dict-shaped `system_prompt` value, with no audit row of why."

Capture the actual exception class and message in scratch notes.

- [ ] **Step 3: Restore the wiring**

Re-add the resolver block. Re-run the test:

```bash
.venv/bin/python -m pytest tests/integration/web/test_blob_inline_runtime_preflight.py -v
```

Expected: PASS.

- [ ] **Step 4: Document the bug-verification in the test's docstring**

```python
# Add to tests/integration/web/test_blob_inline_runtime_preflight.py module docstring

"""...

Bug verification (per agreement-suite protocol): manually revert the
resolver block in src/elspeth/web/execution/service.py::_run_pipeline
(remove the `inline_refs = _discover_blob_content_refs(resolved_dict)`
line and dependent block).  Running test_runtime_resolves_inline_content_ref
without the resolver block produces <ACTUAL EXCEPTION CLASS observed
in Step 2>.  The test pins the resolver wiring as load-bearing.
"""
```

- [ ] **Step 5: Commit the bug-verification documentation**

```bash
git add tests/integration/web/test_blob_inline_runtime_preflight.py
git commit -m "test(execution): document bug-verification protocol for resolver wiring

Per the agreement-suite protocol — the resolver-wiring block in
_run_pipeline is load-bearing.  Reverting it produces a documented
failure mode the integration test pins.

Refs: elspeth-fdebcaa79a"
```

---

## Task 5: Bug-verification at the audit-write fix site

Per spec §9.3 (Sub-pin C).

- [ ] **Step 1: Add a dedicated test pinning audit-row primacy**

```python
# tests/integration/web/test_blob_inline_audit_primacy.py
"""Pin: every resolved inline-content ref produces an audit row.

Per spec §8.2 / §9.3 sub-pin C — audit-write primacy. If this test
fails, the audit-write site has been removed or weakened.
"""

import pytest


@pytest.mark.integration
def test_resolved_ref_always_produces_audit_row(client, blob_service, session_service):
    # ... compose + run as in test_blob_inline_runtime_preflight.py ...
    # Then assert exactly one audit row per resolved ref.
    # ... (full body following the runtime-preflight test structure)
```

- [ ] **Step 2: Manually revert the `record_blob_inline_resolutions` call**

In `_run_pipeline`, comment out the `self._call_async(self._session_service.record_blob_inline_resolutions(...))` line.

- [ ] **Step 3: Run the test, observe failure**

```bash
.venv/bin/python -m pytest tests/integration/web/test_blob_inline_audit_primacy.py -v
```

Expected: FAIL — the audit table is empty for the completed run, which is exactly the audit-fraud pattern this site exists to prevent.

- [ ] **Step 4: Restore + document**

Re-add the audit-write call. Re-run the test (PASS). Add the bug-verification documentation to the test's module docstring naming the production line reverted.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/web/test_blob_inline_audit_primacy.py
git commit -m "test(execution): pin audit-write primacy for inline-content refs

Per spec §9.3 sub-pin C — every resolved inline-content ref MUST
produce a blob_inline_resolutions row.  Bug-verification protocol
documented in module docstring; reverting the audit-write call in
_run_pipeline produces a completed run with no audit row, which is
the exact audit-fraud pattern the audit-primacy invariant exists
to prevent.

Refs: elspeth-fdebcaa79a"
```

---

## Task 5b: Close the pre-link gap in `_source_references_blob`

The existing `delete_blob` guard at `web/blobs/service.py:85-116` walks `composition_states.source.options` for `blob_ref`/`path`/`file` matches to detect blobs referenced by an active run between `create_run` and the source-data `link_blob_to_run` call. That walker is hardcoded to source.options. Without extending it, a blob referenced ONLY from a transform/sink option (no source binding) can be deleted in the window between `create_run` and the new resolver's `link_blob_to_run` calls in `_run_pipeline` — the exact silent-data-loss pattern the existing source-only walker was added to close (per the `_source_references_blob` docstring rationale).

**Files:**
- Modify: `src/elspeth/web/blobs/service.py:85-116` — extend `_source_references_blob` to walk the full config tree
- Test: `tests/unit/web/blobs/test_source_references_blob_inline.py`

> **Phase dependency:** the walker introduced below operates on YAML-form input. P2b (`2026-05-03-config-content-ref-phase-2b-state-adapter.md`) ships the canonical `generate_pipeline_dict(state)` adapter; the production call site in this task uses `generate_pipeline_dict(state_from_record(record))` to bridge from DB-form to YAML-form. The DB-vs-YAML shape gap that previously required a per-implementer caveat is closed by P2b.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/web/blobs/test_source_references_blob_inline.py
"""Pin: a blob referenced ONLY from a transform option (no source
binding) is detected by _source_references_blob — closes the pre-link
gap for inline-content refs.
"""

from elspeth.web.blobs.service import _source_references_blob


def test_walker_detects_inline_content_ref_in_transforms():
    """A composition_state with no source.blob_ref but a transform
    option carrying a widened-blob_ref(inline_content) marker must
    register as referencing the blob."""
    composition_state = {
        "source": {"plugin": "csv", "options": {"path": "x.csv"}},  # no blob_ref
        "transforms": [{
            "name": "classify",
            "plugin": "llm",
            "options": {
                "system_prompt": {
                    "blob_ref": "5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b",
                    "mode": "inline_content",
                    "sha256": "a" * 64,
                },
            },
        }],
    }
    # Note: _source_references_blob today takes (source, blob_id, storage_path);
    # the new signature accepts the full composition_state so it can walk
    # transforms/outputs.  Adapter task may rename the function.
    assert _composition_references_blob(
        composition_state,
        blob_id="5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b",
        storage_path="/never-used-for-inline-content",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/blobs/test_source_references_blob_inline.py -v`
Expected: FAIL — function doesn't exist yet (or the existing function returns False because it only walks source).

- [ ] **Step 3: Extend the walker**

Replace `_source_references_blob` (currently at `web/blobs/service.py:85-116`) with a wider walker `_composition_references_blob` that walks source.options, transforms[].options, gates[].options, aggregations[].options, coalesce[].options, and outputs.<sink>.options. Keep the same Tier-1 read guards (`AuditIntegrityError` if `composition_states.source` is wrong type, etc.).

```python
# src/elspeth/web/blobs/service.py — replace _source_references_blob

from elspeth.contracts.blobs_inline import is_widened_blob_ref


def _composition_references_blob(
    composition_state: Any,
    blob_id: str,
    storage_path: str,
) -> bool:
    """Check whether ANY part of a composition state references a specific blob.

    Walks source.options, transforms/gates/aggregations/coalesce[].options,
    and outputs.<sink>.options for either:
    - a widened blob_ref marker (any mode) carrying the matching blob_id, OR
    - a `path` or `file` option matching the blob's storage_path (covers
      the legacy source-only set_source path that doesn't go through
      set_source_from_blob)

    Tier 1 guards: malformed shapes raise AuditIntegrityError, mirroring
    the original _source_references_blob behaviour.
    """
    if composition_state is None:
        return False
    if not isinstance(composition_state, dict):
        raise AuditIntegrityError(
            f"Tier 1: composition_states is {type(composition_state).__name__}, expected dict"
        )

    # Walk source
    source = composition_state.get("source")
    if isinstance(source, dict):
        if _options_reference_blob(source.get("options"), blob_id, storage_path):
            return True

    # Walk all node collections
    for collection_key in ("transforms", "gates", "aggregations", "coalesce"):
        nodes = composition_state.get(collection_key)
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if isinstance(node, dict) and _options_reference_blob(node.get("options"), blob_id, storage_path):
                return True

    # Walk outputs
    outputs = composition_state.get("outputs")
    if isinstance(outputs, dict):
        for sink in outputs.values():
            if isinstance(sink, dict) and _options_reference_blob(sink.get("options"), blob_id, storage_path):
                return True

    return False


def _options_reference_blob(options: Any, blob_id: str, storage_path: str) -> bool:
    """Recursively walk an options subtree for blob references."""
    if options is None:
        return False
    if isinstance(options, dict):
        # Widened blob_ref marker (any mode)?
        if "blob_ref" in options and options.get("blob_ref") == blob_id:
            return True
        # Source-style path/file legacy match
        if any(options.get(key) == storage_path for key in ("path", "file")):
            return True
        # Recurse
        for v in options.values():
            if _options_reference_blob(v, blob_id, storage_path):
                return True
    elif isinstance(options, list):
        for item in options:
            if _options_reference_blob(item, blob_id, storage_path):
                return True
    return False
```

Update the call sites — `delete_blob` at `web/blobs/service.py:501` calls `_source_references_blob(active_run.source, blob_id_str, row.storage_path)` today. Change the call to `_composition_references_blob(active_run, blob_id_str, row.storage_path)` and update the SELECT to fetch the full composition_state row (currently fetches `composition_states_table.c.source` — needs to fetch the whole row OR a wider projection).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/blobs/test_source_references_blob_inline.py tests/unit/web/blobs/ -v`
Expected: PASS for the new test; no regression in existing `delete_blob` tests.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/blobs/service.py tests/unit/web/blobs/test_source_references_blob_inline.py
git commit -m "fix(blobs): extend pre-link reference walker to inline-content refs

Closes the pre-link gap for transform/sink option blob refs: a blob
referenced ONLY via inline_content from a transform option could be
deleted in the window between create_run and the resolver's
link_blob_to_run call.  The walker now covers source.options,
transforms/gates/aggregations/coalesce[].options, and
outputs.<sink>.options for any blob_ref marker.

Refs: elspeth-fdebcaa79a"
```

---

## Task 5c: Lifecycle-pinning integration test

Per spec done condition #4 ("a referenced blob cannot be GC'd while the run is pending/running"). Pins both the active-run guard (Task 5b) and the `link_blob_to_run` retention path (Task 3).

**Files:**
- Test: `tests/integration/web/test_blob_inline_lifecycle_pinning.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/web/test_blob_inline_lifecycle_pinning.py
"""Lifecycle pinning: an inline-content blob ref keeps the blob alive
during a pending/running run.  Pins spec done-condition #4."""

import hashlib
import time
from threading import Event

import pytest

from elspeth.web.blobs.protocol import BlobActiveRunError


@pytest.mark.integration
def test_inline_content_ref_blocks_blob_deletion(client, blob_service, session_service):
    # Arrange: ready blob in a session, used as inline_content in a transform option
    content = b"You are a helpful assistant."
    sha256 = hashlib.sha256(content).hexdigest()
    session = session_service.create_session_sync(user_id="u")
    blob = blob_service.create_blob_sync(
        session_id=session.id, filename="prompt.txt", content=content,
        mime_type="text/plain", created_by="user",
    )

    # Compose a pipeline that references the blob ONLY from a transform option
    # (no source binding).  Run it with a long-running source so the run
    # stays pending.
    state = {
        "source": {"plugin": "csv", "options": {"path": "long_running.csv"}, "on_success": "out"},
        "transforms": [{
            "name": "classify", "plugin": "llm",
            "options": {
                "system_prompt": {
                    "blob_ref": str(blob.id), "mode": "inline_content", "sha256": sha256,
                },
                "api_key": {"secret_ref": "OPENROUTER_KEY"},
            },
            "on_success": "out",
        }],
        "outputs": {"out": {"plugin": "json", "options": {"path": "out.json"}}},
    }
    run_response = client.post(f"/api/sessions/{session.id}/runs", json={"state": state})
    run_id = run_response.json()["run_id"]

    # Wait until the run is at least registered as pending in the DB
    _wait_until_run_status_in(session_service, run_id, ("pending", "running"), timeout_s=5)

    # Act: attempt to delete the blob while the run is pending/running
    delete_response = client.delete(f"/api/blobs/{blob.id}")

    # Assert: 409 Conflict (BlobActiveRunError), blob row still present
    assert delete_response.status_code == 409
    assert blob_service.get_blob_sync(blob.id) is not None  # row still exists


def _wait_until_run_status_in(session_service, run_id, statuses, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        record = session_service.get_run_sync(run_id)
        if record.status in statuses:
            return
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} never reached {statuses} within {timeout_s}s")
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest tests/integration/web/test_blob_inline_lifecycle_pinning.py -v
```

Expected: PASS. Both Task 5b's walker extension and Task 3's `link_blob_to_run` calls combine to keep the blob retained.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/web/test_blob_inline_lifecycle_pinning.py
git commit -m "test(blobs): pin inline-content lifecycle retention

Pins spec done-condition #4 of elspeth-fdebcaa79a — a referenced blob
cannot be GC'd while the run is pending/running.  Exercises both the
extended composition-walker (Task 5b) and the link_blob_to_run pin
(Task 3) at integration level.

Refs: elspeth-fdebcaa79a"
```

---

## Task 6: Hash-determinism property test

Per spec §1.4 NFR "Hash-pin determinism (round-trip)" — the audit-row `content_hash` re-derived from the blob's stored bytes equals the value composer pinned at submit time.

- [ ] **Step 1: Write the property test**

```python
# tests/integration/web/test_blob_inline_hash_determinism.py
"""Hash-determinism: audit-recorded content_hash equals re-derived hash."""

import hashlib

import pytest
from hypothesis import given, settings, strategies as st

from elspeth.web.blobs.service import content_hash as compute_content_hash


@settings(max_examples=50, deadline=10_000)
@given(content=st.binary(min_size=1, max_size=8192))
@pytest.mark.integration
def test_audit_hash_equals_rederived_hash(content, blob_service, session_service, client):
    # ... composer-authored ref → run → query audit row → re-read blob
    #     bytes via BlobServiceImpl.read_blob_content → re-derive hash →
    #     assert byte-identical match
    pass  # Full body following test_blob_inline_runtime_preflight.py
```

- [ ] **Step 2: Implement the property test body, run, commit**

```bash
.venv/bin/python -m pytest tests/integration/web/test_blob_inline_hash_determinism.py -v
```

Expected: PASS across all 50 generated cases.

```bash
git add tests/integration/web/test_blob_inline_hash_determinism.py
git commit -m "test(execution): pin inline-content hash-determinism property

Refs: elspeth-fdebcaa79a"
```

---

## Task 7: Open the P3 PR

- [ ] **Step 1: Push and open**

```bash
git push -u origin <branch-name>
gh pr create --title "feat(execution): runtime preflight + audit + lifecycle pinning for inline-content refs" --body "$(cat <<'EOF'
## Summary

- Adds \`blob_inline_resolutions\` audit table per spec §8.1
- Adds \`SessionsService.record_blob_inline_resolutions\` async writer (Tier-1; AuditIntegrityError on DB failure)
- Wires the L1 resolver into \`_run_pipeline\` after \`resolve_secret_refs\` per spec §6.3 — discoverer → lifecycle pinning (dedupe by blob_id) → async fetch → substituter → audit row → plugin instantiation
- Adds OTel counters \`composer.blob_inline.hash_mismatch_total\` / \`audit_row_tier1_violation_total\` (both SLO threshold = 0)
- Bug-verification protocol applied at runtime-resolver and audit-write fix sites (spec §9.3 sub-pins B + C)

P4 opens the composer authorship path (validation parity + Shape 9). P3 ships fail-closed: nothing emits inline-content refs yet.

## Pre-phase verification gate

- [x] Direction-segregation check (no new query filters direction in a way that segregates source-data from config-content reads)
- [x] Unique-constraint check confirmed: \`UniqueConstraint(blob_id, run_id, direction)\` covers the dedupe strategy

## Test plan

- [ ] \`pytest tests/unit/web/sessions/test_blob_inline_resolutions_schema.py tests/unit/web/sessions/test_record_blob_inline_resolutions.py -v\`
- [ ] \`pytest tests/integration/web/test_blob_inline_runtime_preflight.py tests/integration/web/test_blob_inline_audit_primacy.py tests/integration/web/test_blob_inline_hash_determinism.py -v\`
- [ ] \`enforce_tier_model.py check\` exits 0
- [ ] Manual: revert the resolver block in \`_run_pipeline\` and confirm \`test_blob_inline_runtime_preflight\` fails with the documented exception class

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Done conditions

P3 is done when:

1. The verification gate file is committed (or a documented schema diversion is filed and applied).
2. `blob_inline_resolutions` table + service method exist with full coverage.
3. The L1 resolver is wired into `_run_pipeline` per spec §6.3.
4. Bug-verification protocol applied + documented at the runtime-resolver and audit-write fix sites.
5. Hash-determinism property test passes.
6. OTel counters are in place.
7. PR is merged.

Move to `2026-05-03-config-content-ref-phase-4-composer-parity.md` only after P3 is merged.
