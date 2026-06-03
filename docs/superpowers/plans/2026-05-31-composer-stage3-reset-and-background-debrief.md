# Stage 3 — Reset + Background Debrief Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the composer a user-initiated "hard reset / start over" that opens a fresh session with an empty graph while preserving the prior session as an immutable audit record, and seed a compact background debrief (lessons learned, correct topology skeleton, blocked-pending-operator items) into the new attempt's LLM context — never a user-facing card.

> **SCOPE NOTE (2026-05-31):** "Emit the debrief on *any* terminal stop" (formerly Task 6) is **split out to a separate follow-up plan** — it is the riskiest piece (touches the hot compose-loop finalize path and the tutorial-invisibility guarantee, needs a real test and a verified hook point). This plan lands the reset feature itself: reset → fresh session → debrief seeded into the *new* session's context, plus the audit record of the reset. Task 6 below is retained only as a DO-NOT-EXECUTE pointer to the follow-up.

## Operator gates & pre-execution corrections (2026-05-31 review)

Read this before executing Task 1. Two items are operator-gated (the agent must STOP and obtain sign-off, per the project's governance and Git-safety doctrine), and three are factual corrections to the citations below that the executing agent must apply.

- **GATE-1 — closed-enum extension: OPERATOR-APPROVED 2026-05-31.** Adding `session_reset` to `ck_composer_completion_events_type` (and to the Python mirror — see CORR-1) is a governance-gated closed-enum extension. The operator **explicitly approved this specific extension on 2026-05-31** during plan review. The executing agent may add `session_reset` without re-stopping for this enum. Scope of the approval is narrow: it authorizes **only** the `session_reset` value carrying NULL `payload_digest`/`expires_at` (like `export_yaml`) in `ck_composer_completion_events_type` and its Python mirror. Any *other* closed-enum change discovered during execution still requires a fresh stop.
- **GATE-2 — the staging DB recreate is a destructive OPERATOR ACTION.** Task 1 is a schema change. Per project policy there is no migration: the operator deletes & recreates `data/sessions.db`. This is destructive shared-state on staging and must be surfaced and gated **before** the destructive step (see `feedback_operator_gate_destructive_actions`). The agent may land the code; it must **not** delete/recreate any staging DB without explicit operator go-ahead, and must not leave staging in a broken state if that go-ahead is withheld.
- **CORR-1 — the `event_type` Python mirror is NOT in `contracts/`.** It is `_CompletionVerb = Literal["mark_ready_for_review", "export_yaml"]` plus `_KNOWN_COMPLETION_VERBS: frozenset[str]` at `src/elspeth/web/composer/telemetry_phase8.py:145-146`. Add `session_reset` to **both** the `Literal` and the frozenset there, in the same commit as the CHECK-constraint change in `models.py`. (There is no `StrEnum` in `contracts/`; the prior pointer was wrong.)
- **CORR-2 — there is NO `service.record_completion_event(...)` method.** Completion events are written **inline at the route layer** via a direct insert into `composer_completion_events_table` — see the live `export_yaml` write at `src/elspeth/web/sessions/routes/composer.py:1218` (and the `completion_verb="export_yaml"` telemetry call at `:1235`). Task 1's test and Task 3 must NOT call `service.record_completion_event`. Choose one and apply it consistently across Tasks 1/3: **(a)** add a real `record_completion_event` service method (mirroring the route's insert) and reuse it, OR **(b)** write the `session_reset` row inline in `reset_session` exactly as `routes/composer.py:1218` writes `export_yaml`. Option (a) is preferred (keeps the writer in the service layer where `reset_session` lives); confirm against `routes/composer.py:1202-1235` before writing.
- **CORR-3 — bump `SESSION_SCHEMA_EPOCH`.** Adding the `reset_from_session_id` column is a schema bump. Increment `SESSION_SCHEMA_EPOCH` (`src/elspeth/web/sessions/schema.py`) in the same commit so the startup sentinel guard (`_assert_schema_sentinels`, `schema.py:126-168`) rejects a stale staging DB with the actionable "Delete the session DB file and restart" message. Without this bump, a stale DB fails later in obscure SQLAlchemy errors — defeating the very gate GATE-2 relies on.

**Architecture:** A new `reset_session` service method writes a single `session_reset` event to `composer_completion_events` for the prior session (forcing `archive_session` onto its soft-archive branch so the prior session is never hard-deleted, and serving as the durable audit record), creates a new session with an empty seed state whose `composer_meta` carries the structured debrief, and carries forward only user-uploaded blobs + accepted interpretation decisions + the original intent. The debrief reaches the next LLM attempt through the existing compose-time context channel: a new `build_context_string` kwarg sourced from the seed state's `composer_meta` (the exact pattern `schemas_loaded` uses) — no new chat role, no transcript row, no frontend change.

**Tech Stack:** Python 3.13, pytest, SQLAlchemy Core (`web/sessions/`), FastAPI routes, LiteLLM chokepoint (`_litellm_acompletion`).

**Spec:** `docs/superpowers/specs/2026-05-31-composer-reset-debrief-and-rootcause-fixes-design.md` (Stage 3 + Post-recon C1/C2 resolutions: no new role; one `session_reset` event does double duty; reuse `session_seed` provenance; user-uploaded = `created_by=='user'`; accepted = `choice IN ('accepted_as_drafted','amended')`).

---

## Verified facts (citations)

- `create_session` (`web/sessions/service.py:2052-2091`) inserts a row; takes `forked_from_session_id` / `forked_from_message_id` (no `reset_from`, no `trust_mode`).
- `archive_session` (`:2206-2232`) is **dual-mode**: soft-archives (sets `archived_at`) when a `runs` OR `composer_completion_events` row exists for the session; **hard-deletes** otherwise. Stages blob dir to `.archive_quarantine/<sid>` first.
- `fork_session` (`:4618-4912`) — single transaction; copies pre-send state, copies messages `[:fork_idx]`, appends a synthetic `role="system"` notice (`:4739-4757`, `writer_principal="session_fork"`), sets `forked_from_*`, inserts state at `version=1` provenance `session_fork`, reserves sequence range. Blob copy + source-ref rewrite happen in the **route** (`routes/sessions.py:311-347` via `blob_service.copy_blobs_for_fork`), not the service.
- `add_message` (`:3597-3609`) — `writer_principal` required keyword-only; allocates sequence under the write lock. Live `system` example: `routes/composer.py:1111-1116` (`writer_principal="route_system_message"`).
- Models (`web/sessions/models.py`): `sessions` (`:121-208`) — no `reset_from` column. `chat_messages.role` CHECK `('user','assistant','system','tool','audit')` (`:259-262`); `writer_principal` CHECK `('compose_loop','route_user_message','route_system_message','admin_tool','session_fork')` (`:271-274`). `composition_states.provenance` closed enum (`:413`) includes `session_seed`, `session_fork`. `composer_completion_events.event_type` CHECK `('mark_ready_for_review','export_yaml')`; biconditional CHECKs require `payload_digest`/`expires_at` **iff** `mark_ready_for_review` (so a new type with NULLs is legal, like `export_yaml`).
- Blobs: `created_by` CHECK `('user','assistant','pipeline')` (`:1208`); `copy_blobs_for_fork` (`web/blobs/service.py:947-1007`) copies **all** ready blobs preserving `created_by`. user-uploaded = `created_by == 'user'`.
- Interpretation events: `choice` CHECK `('pending','accepted_as_drafted','amended','opted_out','abandoned')` (`:680`); `list_interpretation_events` (`service.py:3313`) status filter is `Literal["pending","all"]` only — accepted selection is a Python-side filter. Composite same-session FK `(composition_state_id, session_id)` (`:666`) — carry-forward must repoint or NULL `composition_state_id`.
- LLM synth: do NOT reuse `_call_advisor_with_audit` (`web/composer/service.py:3104`, hardcoded stuck-prompt). Use the chokepoint `_litellm_acompletion` with `self._settings.composer_advisor_model` (`:3141`), `composer_advisor_timeout_seconds`, `composer_advisor_max_completion_tokens`.
- Context channel: `build_context_string(state, catalog, *, schemas_loaded=...)` (`web/composer/prompts.py:207`), called at `prompts.py:439`. `CompositionState.to_dict()` does NOT carry `composer_meta` → the debrief must be threaded as a new kwarg.
- Test style: `tests/unit/web/sessions/test_fork.py:31-73` (engine/service fixtures, fork assertions); `tests/unit/web/sessions/test_service.py` `TestSessionCRUD` (`:62`), `test_archive_session_hides_session_with_durable_completion_history` (`:137`).
- DB policy: no migrations — operator deletes & recreates the session DB (memory `project_db_migration_policy`, `project_phase9_sqlite_only`).

---

## Task 1: Schema — `reset_from_session_id` column + `session_reset` event type

**Files:**
- Modify: `src/elspeth/web/sessions/models.py` (sessions table; `ck_composer_completion_events_type`)
- Modify: `src/elspeth/web/composer/telemetry_phase8.py:145-146` — the `event_type` Python mirror: `_CompletionVerb` `Literal` + `_KNOWN_COMPLETION_VERBS` frozenset (see CORR-1; **NOT** `contracts/`)
- Modify: `src/elspeth/web/sessions/schema.py` — bump `SESSION_SCHEMA_EPOCH` (see CORR-3)
- Modify: `src/elspeth/web/sessions/protocol.py` (`SessionRecord` dataclass — add `reset_from_session_id`)
- Test: `tests/unit/web/sessions/test_schema.py` (or the nearest schema test module)

- [ ] **Step 1: Write the failing test**

```python
def test_sessions_table_has_reset_from_column(engine):
    from sqlalchemy import inspect
    cols = {c["name"] for c in inspect(engine).get_columns("sessions")}
    assert "reset_from_session_id" in cols


def test_completion_event_type_allows_session_reset(engine):
    # Inserting a session_reset completion event must satisfy the CHECK.
    # NOTE (CORR-2): there is no service.record_completion_event method. Drive
    # the CHECK directly against the table — mirror the inline insert at
    # routes/composer.py:1218 (event_type="export_yaml"), substituting
    # "session_reset" with NULL payload_digest/expires_at.
    from sqlalchemy import insert
    from elspeth.web.sessions.models import composer_completion_events_table
    with engine.begin() as conn:
        conn.execute(
            insert(composer_completion_events_table).values(
                id="evt-reset-1", session_id="s1", event_type="session_reset",
                payload_digest=None, expires_at=None,
            )
        )  # must NOT raise the ck_composer_completion_events_type CHECK
```

- [ ] **Step 2: Run it (fails)**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_schema.py -k "reset_from or session_reset" -v`
Expected: FAIL (`no such column`, CHECK violation, or missing writer).

- [ ] **Step 3: Add the column and enum value**

In `sessions_table` (`models.py:121-208`), add alongside `forked_from_session_id`:

```python
    Column("reset_from_session_id", String, ForeignKey("sessions.id"), nullable=True),
```

In `ck_composer_completion_events_type` change the allowed set to include `'session_reset'`:

```python
        "event_type IN ('mark_ready_for_review', 'export_yaml', 'session_reset')",
```

The two biconditional CHECKs (`payload_digest`/`expires_at` iff `mark_ready_for_review`) need no change — `session_reset` carries NULLs like `export_yaml`. **Mirror `session_reset` into the Python type (CORR-1):** add it to both `_CompletionVerb = Literal[...]` and `_KNOWN_COMPLETION_VERBS` at `telemetry_phase8.py:145-146` — there is no `StrEnum` in `contracts/`. Add `reset_from_session_id` to the `SessionRecord` dataclass (`protocol.py`) defaulting to `None`, and set it in `create_session`'s returned record.

- [ ] **Step 4: Bump `SESSION_SCHEMA_EPOCH` + DB recreate (no migration — project policy)**

Bump `SESSION_SCHEMA_EPOCH` (`src/elspeth/web/sessions/schema.py`) in this same commit (CORR-3) so the startup sentinel guard `_assert_schema_sentinels` (`schema.py:126-168`) rejects a stale staging DB with the actionable "Delete the session DB file and restart" message. Without the bump, a stale DB fails later in obscure SQLAlchemy errors instead of cleanly.

This is a schema change; per project policy there is no migration. The operator deletes & recreates the session DB (`data/sessions.db`) on deploy. **This is GATE-2 — a destructive OPERATOR ACTION:** surface it and obtain explicit go-ahead before any staging DB is deleted; do not leave staging in a broken state if go-ahead is withheld. Add the recreate note to the PR description.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_schema.py -k "reset_from or session_reset" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/models.py src/elspeth/web/composer/telemetry_phase8.py src/elspeth/web/sessions/schema.py src/elspeth/web/sessions/protocol.py tests/unit/web/sessions/test_schema.py
git commit -m "feat(sessions): add reset_from_session_id column + session_reset event type"
```

---

## Task 2: Deterministic debrief builder (+ thin optional LLM narrative)

**Files:**
- Create: `src/elspeth/web/composer/debrief.py`
- Test: `tests/unit/web/composer/test_debrief.py`

The builder reads a session's audit data and produces the structured debrief dict (the LLM seed shape from the spec). Deterministic facts only here; the one-line narrative is a separate optional async call (Step 3) so the builder stays pure and testable.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/web/composer/test_debrief.py
from elspeth.web.composer.debrief import build_debrief


def test_build_debrief_separates_rejected_from_blocked():
    # Inputs are plain data structures (validation-error history, interp events,
    # cost) so the builder is pure. See build_debrief signature in Step 3.
    debrief = build_debrief(
        prior_session_id="2e6d5e3e",
        cost={"turns": 13, "tool_calls": 14, "slowest_call_s": 118},
        validation_history=[
            {"version": 1, "errors": ["Invalid options for source 'csv': schema: Observed schemas cannot have explicit field definitions."]},
            {"version": 7, "errors": ["web_scrape.http.abuse_contact has domain 'example.com' — RFC 2606/6761 reserved ..."]},
        ],
        topology="csv -> web_scrape -> llm -> json",
        resolved_reviews=["llm_prompt_template:color_analysis"],
    )
    # The abuse_contact item is a correct refusal -> blocked_pending_human, NOT rejected (D3).
    assert any("abuse_contact" in b["field"] for b in debrief["blocked_pending_human"])
    assert all("abuse_contact" not in c for c in debrief["constraints_learned"] if "rejected" in c)
    # The observed+fields contradiction is a model mistake -> a learned constraint.
    assert any("observed" in c.lower() for c in debrief["constraints_learned"])
    # Never the 5KB schema (D1).
    assert "json_schema" not in str(debrief)
    assert debrief["synthesized"] is True
```

- [ ] **Step 2: Run it (fails)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_debrief.py::test_build_debrief_separates_rejected_from_blocked -v`
Expected: FAIL — `ModuleNotFoundError: ... debrief`.

- [ ] **Step 3: Implement the pure builder**

```python
# src/elspeth/web/composer/debrief.py
"""Background composer debrief — LLM context seed + audit artifact.

NOT user-facing (operator directive 2026-05-31). Separates `rejected`
(model mistakes, learn from) from `blocked_pending_human` (correct refusals
like abuse_contact — D3: must never be filed as errors or the next attempt
learns to fabricate). Never carries the full plugin schema (D1).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_ABUSE_PATTERNS = ("abuse_contact", "scraping_reason")


def build_debrief(
    *,
    prior_session_id: str,
    cost: Mapping[str, int],
    validation_history: Sequence[Mapping[str, Any]],
    topology: str,
    resolved_reviews: Sequence[str],
) -> dict[str, Any]:
    constraints: list[str] = []
    blocked: list[dict[str, str]] = []
    for entry in validation_history:
        for msg in entry["errors"]:
            low = msg.lower()
            if any(p in low for p in _ABUSE_PATTERNS) and ("reserved" in low or "must come from" in low or "example.com" in low):
                blocked.append({"field": "web_scrape.http.abuse_contact", "reason": msg})
            elif "observed schemas cannot have explicit field" in low:
                constraints.append("csv source: mode:observed forbids explicit fields[]; use guaranteed_fields/required_fields")
            elif "references unknown sink" in low:
                constraints.append("every transform on_error/on_success must reference a declared sink")
    # Dedupe, stable order.
    constraints = sorted(set(constraints))
    return {
        "prior_session": prior_session_id,
        "cost": dict(cost),
        "correct_topology_skeleton": topology,
        "constraints_learned": constraints,
        "blocked_pending_human": blocked,
        "resolved_reviews": list(resolved_reviews),
        "synthesized": True,
    }
```

- [ ] **Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_debrief.py::test_build_debrief_separates_rejected_from_blocked -v`
Expected: PASS

- [ ] **Step 5: Add the optional one-line narrative synth (thin litellm helper)**

```python
async def synthesize_debrief_narrative(debrief: Mapping[str, Any], *, litellm_acompletion, model: str, timeout: float, max_tokens: int) -> str:
    """One-line 'why' narrative. Marked synthesized; never asserted as audit fact.

    Uses the LiteLLM chokepoint + composer_advisor_model — NOT
    _call_advisor_with_audit (which hardcodes the stuck-prompt shape).
    """
    import asyncio
    messages = [
        {"role": "system", "content": "Summarise, in ONE sentence, why the prior composition attempt did not complete. Be concrete; do not invent."},
        {"role": "user", "content": str({k: debrief[k] for k in ("constraints_learned", "blocked_pending_human")})},
    ]
    resp = await asyncio.wait_for(
        litellm_acompletion(model=model, messages=messages, temperature=0, max_tokens=max_tokens),
        timeout=timeout,
    )
    return resp.choices[0].message.content.strip()
```

Test with a stub `litellm_acompletion` returning a canned response (mirror how existing composer tests stub LiteLLM). Assert the narrative is a string and the call uses `model`.

**Tier-3 best-effort contract (concern #2, 2026-05-31 review).** The narrative is Tier-3 LLM output and is NON-LOAD-BEARING — the reset and the deterministic debrief must succeed without it. The *caller* (Task 3 `reset_session` / Task 4 route) MUST wrap `synthesize_debrief_narrative` so that timeout, exception, or empty/blank content resolves to **absence (`None`)**, never fabrication and never a failed reset:

```python
narrative: str | None
try:
    text = await synthesize_debrief_narrative(debrief, litellm_acompletion=_litellm_acompletion,
                                              model=self._settings.composer_advisor_model,
                                              timeout=self._settings.composer_advisor_timeout_seconds,
                                              max_tokens=self._settings.composer_advisor_max_completion_tokens)
    narrative = text or None          # empty/blank -> None (absence, not "")
except (TimeoutError, Exception):     # best-effort: synth failure must not fail the reset
    narrative = None                  # Tier-3 absence recorded honestly; NOT fabricated
debrief["narrative"] = narrative      # may be None; consumers must handle absence
```

Doctrine: per the three-tier model, a Tier-3 source's silence is recorded as `None`, not invented; `synthesized: true` already flags the LLM-authored portion so an auditor never mistakes it for a recorded fact. Add a test driving a raising/timing-out stub and asserting (a) `reset_session` still returns a valid new session + seed state and (b) `debrief["narrative"] is None`. (If you prefer, the bare `except Exception` may be narrowed once the real `_litellm_acompletion` failure modes are confirmed — but the reset MUST NOT propagate a synth failure.)

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/debrief.py tests/unit/web/composer/test_debrief.py
git commit -m "feat(composer): pure debrief builder + thin narrative synth (D1/D3)"
```

---

## Task 3: `reset_session` service method

**Files:**
- Modify: `src/elspeth/web/sessions/service.py` (new `reset_session`)
- Modify: `src/elspeth/web/sessions/protocol.py` (add to `SessionServiceProtocol`)
- Test: `tests/unit/web/sessions/test_reset.py`

`reset_session` (single transaction where possible, mirroring `fork_session`'s discipline):
1. Write a `session_reset` `composer_completion_events` row for the prior session (durable marker → soft-archive; + audit record).
2. `create_session(... )` for the new session; set `reset_from_session_id = prior`.
3. Insert an **empty** composition state at `version=1`, `provenance="session_seed"`, with `composer_meta = {"reset_debrief": <debrief dict>}`.
4. Copy the original first user message (the first `role=="user"` row of the prior session) into the new session (`writer_principal="session_fork"`, `composition_state_id=None`).
5. Carry forward accepted interpretation decisions: select prior `interpretation_events` with `choice IN ('accepted_as_drafted','amended')` (Python filter on `status="all"`), re-attach to the new session with `composition_state_id` NULLed or repointed to the seed state (respect the composite FK `:666`).
6. `archive_session(prior)` (now takes the soft-archive branch because of step 1).

(Blob carry-forward is route-layer — Task 4.)

- [ ] **Step 1: Write the failing test (mirror test_fork.py)**

```python
# tests/unit/web/sessions/test_reset.py
import pytest


@pytest.mark.asyncio
async def test_reset_creates_empty_session_referencing_prior(service):
    prior = await service.create_session("alice", "CSV colour analysis", "local")
    await service.add_message(prior.id, "user", "rate gov pages", writer_principal="route_user_message")
    debrief = {"prior_session": str(prior.id), "constraints_learned": ["x"], "blocked_pending_human": [], "synthesized": True}

    new_session, seed_state = await service.reset_session(prior.id, debrief=debrief, user_id="alice", auth_provider_type="local")

    assert new_session.reset_from_session_id == prior.id
    assert seed_state.version == 1
    # Empty graph.
    assert seed_state.nodes in ({}, [], None) or len(seed_state.nodes) == 0
    # Debrief carried in composer_meta.
    assert seed_state.composer_meta["reset_debrief"]["constraints_learned"] == ["x"]
    # Prior session preserved (soft-archived, not hard-deleted).
    prior_after = await service.get_session(prior.id)
    assert prior_after is not None and prior_after.archived_at is not None


@pytest.mark.asyncio
async def test_reset_carries_forward_first_user_message(service):
    prior = await service.create_session("alice", "t", "local")
    await service.add_message(prior.id, "user", "ORIGINAL INTENT", writer_principal="route_user_message")
    await service.add_message(prior.id, "assistant", "...", writer_principal="compose_loop")
    new_session, _ = await service.reset_session(prior.id, debrief={"synthesized": True}, user_id="alice", auth_provider_type="local")
    msgs = await service.get_messages(new_session.id)
    assert [m.content for m in msgs if m.role == "user"] == ["ORIGINAL INTENT"]
```

- [ ] **Step 2: Run it (fails — no reset_session)**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_reset.py -v`
Expected: FAIL — `AttributeError: ... reset_session`.

- [ ] **Step 3: Implement `reset_session`**

Model it on `fork_session` (`service.py:4618`) for transaction discipline and `_insert_composition_state` / `_reserve_sequence_range` usage, but seed an **empty** `CompositionStateData` and set `composer_meta`. Write the `session_reset` completion event before `archive_session`. Add the protocol method to `SessionServiceProtocol` (`protocol.py`, near `fork_session` `:1095`):

```python
    async def reset_session(
        self,
        prior_session_id: UUID,
        *,
        debrief: Mapping[str, Any],
        user_id: str,
        auth_provider_type: AuthProviderType,
    ) -> tuple[SessionRecord, CompositionStateRecord]:
        """Open a fresh session (empty graph) referencing an archived prior
        session; seed the debrief into the new state's composer_meta.
        """
        ...
```

Write the `session_reset` row per CORR-2 (there is **no** `record_completion_event` method): either add one to the service mirroring the inline insert at `routes/composer.py:1218`, or insert into `composer_completion_events_table` directly inside `reset_session` (preferred — keeps it in the same transaction as the rest of the reset). For accepted-interpretation carry-forward, select via `list_interpretation_events(prior, status="all")` then Python-filter `choice IN ('accepted_as_drafted','amended')`; insert into the new session with `composition_state_id=None` (avoids the composite FK violation) — or repoint to the seed state id.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_reset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py tests/unit/web/sessions/test_reset.py
git commit -m "feat(sessions): reset_session — empty new session + session_reset event + carry-forward"
```

---

## Task 4: Reset route + user-uploaded blob carry-forward

**Files:**
- Modify: `src/elspeth/web/sessions/routes/sessions.py` (new `POST /{session_id}/reset`)
- Modify: `src/elspeth/web/blobs/service.py` (a `created_by`-filtered copy, or filter route-side)
- Test: `tests/integration/web/test_reset_route.py`

Mirror the `fork_from_message` route (`sessions.py:273-347`): ownership-verify, active-run guard, call `service.reset_session(...)`, then the post-commit blob phase — but copy **only** `created_by == 'user'` blobs.

- [ ] **Step 1: Write the failing test**

```python
def test_reset_route_returns_new_session_and_copies_user_blobs(client, seeded_session_with_user_blob):
    sid = seeded_session_with_user_blob
    resp = client.post(f"/api/sessions/{sid}/reset", json={})
    assert resp.status_code == 201
    body = resp.json()
    assert body["reset_from_session_id"] == sid
    # The user-uploaded CSV is carried into the new session; assistant/pipeline blobs are not.
    new_blobs = client.get(f"/api/sessions/{body['id']}/blobs").json()
    assert all(b["created_by"] == "user" for b in new_blobs)
```

- [ ] **Step 2: Run it (fails — no route)**

Run: `.venv/bin/python -m pytest tests/integration/web/test_reset_route.py -v`
Expected: FAIL (404 / route missing).

- [ ] **Step 3: Implement the route + filtered blob copy**

Add (mirroring `fork_from_message` exactly, `:273-347`):

```python
    @router.post("/{session_id}/reset", status_code=201, response_model=SessionResponse)
    async def reset_session(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> SessionResponse:
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        active_run = await service.get_active_run(session.id)
        if active_run is not None:
            raise HTTPException(status_code=409, detail="Cannot reset while a pipeline run is active.")
        debrief = await _build_session_debrief(session.id, request)  # Task 2 builder over audit data
        new_session, _seed = await service.reset_session(
            session.id, debrief=debrief, user_id=user.user_id,
            auth_provider_type=request.app.state.settings.auth_provider,
        )
        blob_service = request.app.state.blob_service
        await blob_service.copy_blobs_for_fork(session.id, new_session.id, only_created_by="user")
        return _session_response(new_session)
```

Add an `only_created_by: str | None = None` filter param to `copy_blobs_for_fork` (`blobs/service.py:947`), defaulting to today's behaviour (copy all) when `None`. `_build_session_debrief` reads the prior session's composition-state validation history + interp events + cost and calls `build_debrief` (Task 2).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/integration/web/test_reset_route.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/routes/sessions.py src/elspeth/web/blobs/service.py tests/integration/web/test_reset_route.py
git commit -m "feat(web): POST /sessions/{id}/reset + user-uploaded blob carry-forward"
```

---

## Task 5: Thread the debrief into the LLM context (no new role)

**Files:**
- Modify: `src/elspeth/web/composer/prompts.py` (`build_context_string` + the `:439` call site)
- Test: `tests/unit/web/composer/test_prompts_debrief.py`

The debrief lives in the seed state's `composer_meta["reset_debrief"]`. `CompositionState.to_dict()` does not surface `composer_meta`, so add a `reset_debrief: Mapping | None = None` kwarg to `build_context_string` (mirroring `schemas_loaded`) and render it into the context block. The compose loop passes `state.composer_meta.get("reset_debrief")` when assembling the context.

- [ ] **Step 1: Write the failing test**

```python
def test_context_string_includes_reset_debrief(catalog):
    from elspeth.web.composer.state import CompositionState
    from elspeth.web.composer.prompts import build_context_string
    debrief = {"constraints_learned": ["csv source: mode:observed forbids explicit fields[]"], "blocked_pending_human": []}
    ctx = build_context_string(CompositionState.empty(), catalog, reset_debrief=debrief)
    assert "mode:observed forbids explicit fields" in ctx
    # Absent by default.
    assert "reset_debrief" not in build_context_string(CompositionState.empty(), catalog)
```

- [ ] **Step 2: Run it (fails — no kwarg)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_prompts_debrief.py -v`
Expected: FAIL — unexpected kwarg / assertion.

- [ ] **Step 3: Add the kwarg + render it; thread at the call site**

In `build_context_string` add `reset_debrief: Mapping[str, Any] | None = None`; when present, append a `prior_attempt_lessons` block (constraints + blocked-pending-operator) to the returned context string. At `prompts.py:439`, pass `reset_debrief=...` sourced from the composing state's `composer_meta` (thread it from the compose loop the same way `schemas_loaded` is sourced from the service).

- [ ] **Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_prompts_debrief.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/prompts.py tests/unit/web/composer/test_prompts_debrief.py
git commit -m "feat(composer): inject reset debrief into compose-time context (no new role)"
```

---

## Task 6: Emit the debrief on any terminal stop — ⛔ DO NOT EXECUTE (split to follow-up)

> **STATUS (2026-05-31 review): SPLIT OUT — DO NOT EXECUTE IN THIS PLAN.** This task is the riskiest in Stage 3: it touches the hot compose-loop finalize path *and* the tutorial-invisibility guarantee, its test is a stub, and its finalize hook point is unverified against code. It is **out of scope for this plan**. Stage 3 lands the reset feature (Tasks 1–5, 7) without it. The notes below are retained only as the seed for a separate, properly-spec'd follow-up plan that must add: (a) a real (non-stub) test driving the production compose-loop entry; (b) a hook point verified against `_persist_turn_audit` / `persist_compose_turn_async`; (c) an explicit tutorial-path-silence assertion. **Do not check these boxes.**

**Files (follow-up only):**
- Modify: the compose-loop finalize/persist path (`web/composer/no_tool_finalize.py` and/or `turn_audit.py` — locate via `persist_compose_turn_async` / `_persist_turn_audit`, P4 in `_compose_loop_carriers.py`)
- Test: `tests/unit/web/composer/test_debrief_on_terminal.py`

When a turn ends in a terminal surface (completed / draft+approvals / info-gap), build the debrief from the current state and record it to audit (and make it available to the same session's subsequent context). Background only — no human-facing surface; tutorial path stays silent.

- [ ] **Step 1: Write the failing test**

```python
def test_terminal_blocked_stop_records_debrief(...):
    # Drive a compose turn that ends blocked_pending_human; assert a debrief
    # artifact is recorded (audit) and NO user-facing message is emitted.
    ...
```

(Construct using the production compose-loop entry the existing turn-audit tests use; assert on the recorded audit artifact and the absence of a new visible chat row.)

- [ ] **Step 2: Run it (fails)** — Expected FAIL.

- [ ] **Step 3: Hook `build_debrief` into the finalize path**

At the terminal-surface finalize (where `_persist_turn_audit` runs), build the debrief from the current state's validation history and record it via the audit writer. Do NOT add a chat message. Reuse `_build_session_debrief`/`build_debrief` from Tasks 2/4. Gate on terminal-surface only (not every turn).

- [ ] **Step 4: Run test** — Expected PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(composer): record background debrief on terminal stops"
```

---

## Task 7: Frontend reset button (minimal) + verification gate

**Files:**
- Modify: the composer toolbar/header component (`web/frontend/src/...`); add a "Start over" button calling `POST /api/sessions/{id}/reset` then navigating to the new session id.
- Test: a vitest unit + the existing route integration test (Task 4) cover the contract.

- [ ] **Step 1:** Add the button in the composer header next to existing session controls; on click, POST reset, then route to `#/<new_session_id>`. Confirm with the user where the control should sit before styling (per UX-redesign conventions).

- [ ] **Step 2: Full verification gate**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/sessions/ tests/unit/web/composer/test_debrief.py tests/unit/web/composer/test_prompts_debrief.py -q
.venv/bin/python -m pytest tests/integration/web/test_reset_route.py -q
.venv/bin/python -m mypy src/elspeth/web/sessions/ src/elspeth/web/composer/debrief.py src/elspeth/web/composer/prompts.py
.venv/bin/python -m ruff check src/elspeth/web/
(cd src/elspeth/web/frontend && npm run typecheck && npx vitest run)
```
Expected: clean / PASS

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(composer): reset/start-over button + stage-3 verification"
```

---

## Self-review checklist (completed by plan author)

- **Spec coverage:** schema for reset_from + session_reset event (Task 1); debrief builder splitting rejected/blocked, no 5KB (Task 2); reset_session = empty new session + double-duty event + carry-forward (Task 3); route + user-uploaded blob filter (Task 4); context injection no-new-role (Task 5); button + gate (Task 7). **Task 6 (terminal-stop emission) is SPLIT OUT to a follow-up — not in scope.** C1+C2 resolutions honoured. ✓
- **2026-05-31 review corrections applied:** GATE-1 (enum operator-APPROVED, narrow scope), GATE-2 (DB recreate = gated destructive op), CORR-1 (Python mirror is `telemetry_phase8.py:145-146`, not `contracts/`), CORR-2 (no `record_completion_event` — inline insert per `routes/composer.py:1218`), CORR-3 (`SESSION_SCHEMA_EPOCH` bump), Tier-3 best-effort narrative (synth failure → `None`, never fails the reset). ✓
- **Type consistency:** `reset_session`, `build_debrief`, `_build_session_debrief`, `reset_from_session_id`, `composer_meta["reset_debrief"]`, `reset_debrief` kwarg, `only_created_by` used identically across tasks. ✓

## Risks

- **C1 hard-delete:** mitigated by writing the `session_reset` event *before* `archive_session` (Task 3) + a test asserting the prior session is soft-archived (`archived_at` set), mirroring `test_archive_session_hides_session_with_durable_completion_history`.
- **Composite interpretation FK (`:666`):** carry-forward NULLs/repoints `composition_state_id` (Task 3 Step 3).
- **Schema change:** no migration — operator deletes/recreates `data/sessions.db` (Task 1 Step 4), gated as a destructive OPERATOR ACTION (GATE-2); `SESSION_SCHEMA_EPOCH` bumped so a stale DB fails cleanly; note in PR.
- **Tutorial invisibility (C2 intent):** debrief is context + audit only, never a chat row. (The terminal-stop path that most stresses this — former Task 6 — is split to a follow-up that owns the tutorial-silence assertion.)
- **Tier-3 narrative is best-effort:** synth timeout/exception/empty → `None`, never fabricated, never fails the reset (Task 2; per the three-tier trust model).
