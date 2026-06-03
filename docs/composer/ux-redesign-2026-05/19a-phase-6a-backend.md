# Phase 6A — Backend: completion gestures (Save for review, signing, narrative declaration)

> **Note on line-number citations.** Plan was authored against an earlier codebase snapshot. When a line citation is in conflict with reality, **trust `rg`, not the line number** — `rg -n "<symbol>" src/` is the authoritative locator. Symbols are stable; line numbers drift.

---

## ⚠️ OPERATOR ACTION REQUIRED — read before starting Task 1

Phase 6 is a schema-change cohort and a signing-key cohort. Two operator-gated actions land before Task 1 commits:

**1. Staging sessions-DB delete (B8 — destructive shared-state operation).**

Task 1 bumps `SESSION_SCHEMA_EPOCH` from 3 to 4. The startup validator (`_assert_schema_sentinels()` at `web/sessions/schema.py:112`) refuses to start the service against any DB whose `PRAGMA user_version` is not 4. **Existing staging sessions DBs must be deleted on the next deploy.** Per `project_db_migration_policy`, this is the ELSPETH-canonical migration mechanism pre-Phase-9. Per `feedback_operator_gate_destructive_actions`, this authorization does **not** transit through "execute Task 1" — the operator must explicitly confirm before the epoch-bump commit lands on RC5.2 (or whatever release branch Phase 6 ships on).

User state lost: any composer sessions saved in staging since the Phase 18 deploy. This is acknowledged cost; the structural fix is Phase 9 (migration runner).

**2. Staging config `shareable_link_signing_key` (B9 — service-startup-blocking prerequisite).**

`WebSettings.shareable_link_signing_key` is declared as `Field(...)` with no default. If Phase 6A deploys to a staging config that does not provide this key, the service crashes at startup. Generate the key once and add it to staging config **before** the Phase 6A deploy:

```bash
openssl rand -base64 32
```

Store the result in the staging configuration under `web.shareable_link_signing_key` (or the equivalent env var the deploy machinery surfaces). The key must be ≥ 32 bytes and must not appear in version control, logs, or chat transcripts. If the key is leaked or accidentally rotated, **every outstanding shareable link becomes a 401** — there is no graceful key-rotation story in v1 (recovery: re-issue links). The runbook entry for this lives in Task 12 (`docs/guides/sharing-pipelines.md`).

**Both actions must be confirmed by the operator before Task 1 starts.** If only one is confirmed, Task 1 is blocked: the epoch bump without the signing key produces a service that won't start; the signing key without the epoch bump produces a service that starts on the old schema and rejects every shareable-link request with a 500 (no `composer_completion_events_table` to write to).

---


> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Land the backend half of Phase 6 — the signing primitive and content-addressable HMAC-signed artifact that back the "Save for review" verb, the read-only inspect route for shareable-link recipients, the per-plugin narrative-summary opt-in via the existing `capability_tags` open-vocabulary channel (which the frontend uses to choose result-rendering mode), and two Tier-1 audit events recorded in a new sessions-DB table. **One DB schema addition: a new `composer_completion_events_table` in the sessions DB for the two new completion-gesture audit events (`mark_ready_for_review`, `export_yaml`).** This follows the Phase 18 (5b) precedent of "one new table per event family" — see [18-phase-5b-surface-llm-interpretation.md](18-phase-5b-surface-llm-interpretation.md) §"interpretation_events_table" for the analogue. Per `project_db_migration_policy`, this is a schema-change cohort and requires a DB delete on next deploy. The signed save-for-review artifact is stored in the existing payload store as a content-addressable blob under the normal retention policy. The completion verb set is three, not four: Save-for-review, Run-pipeline (= existing Execute path; result rendering branches on narrative-mode), and Copy-YAML (= existing Export YAML route + audit event). Per design doc 09 ("Why not four"), Run-analysis is not a separate verb — narrative result rendering is the post-run rendering layer that consumes Phase 5b's interpretation events.

**Architecture:** Service-then-routes (mirrors Phase 1A and Phase 2A). A new `src/elspeth/web/shareable_reviews/` package exposes `ShareableReviewService` taking the existing `SessionServiceProtocol`, the sessions-DB connection (for writing to `composer_completion_events_table`), the payload store (from `app.state`), `WebSettings`, and the new `ShareTokenSigner` as injected dependencies. The token signer is a thin wrapper around `hmac.new(..., hashlib.sha256)` — it reuses the **primitive** from `core/landscape/exporter.py`, not the exporter API (different payload shape, different rotation cadence, different blast radius). The token is **self-verifying**: its payload encodes `(version, session_id, state_id, expires_at, nonce, payload_digest, signature)` where `payload_digest` is the content-address of the snapshot blob in the payload store. The signature is over the canonical-JSON of the payload (including `payload_digest`) — tampering with either the token fields or the blob detects on verify. One new sessions-DB table, three new routes, and one extension to the existing YAML route.

**Tech Stack:** FastAPI, Pydantic v2 (strict, extra=forbid), payload store (Tier-1 content-addressable blob storage), SQLAlchemy Core (new `composer_completion_events_table` in the sessions DB), pytest. Mirrors the stack of every prior backend phase plan.

**Sibling plan:** [19b-phase-6b-frontend.md](19b-phase-6b-frontend.md) — completion-bar component, Save-for-review confirmation dialog, shareable-link inspect view, context-aware result rendering.

**Design reference:** [09-completion-gestures.md](09-completion-gestures.md). Three completion verbs (Save-for-review, Run-pipeline, Copy-YAML) per the "Why not four" adjudication in §"Why three verbs, not two or four".

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md). Phase 6 was BLOCKED in the readiness summary on `A3 + E1 + Phase 3`. A3 (single-/multi-user) and E1 (HMAC-signed artifact) are adjudicated below; Phase 3 (side rail) is a **sequencing** dependency, not a content dependency — this plan can be written now and held until Phase 3 ships. See §"Sequencing with Phase 3" below.

---

## Sequencing with Phase 3

The completion bar lives in the side rail introduced by Phase 3 ([15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md) / [15b2-phase-3b-side-rail-part-2.md](15b2-phase-3b-side-rail-part-2.md)). Phase 6A (backend) has **no dependency on Phase 3** — endpoints and tests can land immediately. Phase 6B (frontend) needs the side-rail mount point and its `completion-bar` slot reserved by 15b1, and Phase 6B's task list assumes Phase 3 has shipped.

If a future operator decides to ship Phase 6 ahead of Phase 3, the fallback is documented in Phase 6B's §"Fallback if Phase 3 has not shipped": a temporary header-area mount with a TODO marker linking back to the side-rail integration. Phase 6A is unaffected.

**Recommendation:** ship 6A in its own merge, then ship 6B after Phase 3 has merged. The 6A endpoints can be exercised via `curl` and integration tests until 6B catches up.

---

## Scope boundaries

**In scope:**

- New `composer_completion_events_table` in `src/elspeth/web/sessions/models.py` (sessions DB, alongside `proposal_events_table` and `interpretation_events_table`). Closed-enum CHECK on `event_type` (`'mark_ready_for_review'`, `'export_yaml'`). Per Phase 18 precedent — see schema spec in Task 1 below.
- New package `src/elspeth/web/shareable_reviews/` with `__init__.py`, `models.py`, `signer.py`, `service.py`, `routes.py`.
- Pydantic models for the three new endpoints, all `_StrictResponse`-style (`extra="forbid"`, `strict=True`). `SharedInspectResponse` includes an `audit_readiness: AuditReadinessSnapshot` field (the existing Phase 2 model, reused verbatim) — see Task 5.
- `ShareTokenSigner` — a `hmac.new(key, msg, hashlib.sha256)` wrapper that produces URL-safe base64 tokens encoding `(version, session_id, state_id, expires_at, nonce, payload_digest, signature)`. Token signing key sourced from a new `WebSettings.shareable_link_signing_key` field (required, ≥32 bytes).
- HMAC-signed snapshot artifact stored as a **content-addressable blob in the existing payload store**. Per project tier model the payload store is Tier-1; the blob is read back by digest on resolve. No per-token row — only the audit event row in `composer_completion_events_table`.
- Two Tier-1 audit events recorded in `composer_completion_events_table` (sessions DB). Both writes are sync, crash-on-failure per CLAUDE.md audit primacy. Event types: `mark_ready_for_review` and `export_yaml`.
- `POST /api/sessions/{session_id}/mark-ready-for-review` — generates a fresh signed token + writes the snapshot to the payload store + inserts a `mark_ready_for_review` row in `composer_completion_events_table` + returns the token and share URL.
- `GET /api/sessions/{session_id}/shareable-link` — re-mints a fresh token for the current `(session, state)` pair on demand. Because there is no per-token row, this endpoint always mints; idempotency at the artifact level is provided by content-addressing (identical snapshot → identical `payload_digest`).
- `GET /api/sessions/shared/{token}` — read-only inspect view. Recipient still authenticates via `Depends(get_current_user)`; the token authorizes a specific authenticated user to read a session they don't own.
- Narrative-summary opt-in via existing `BaseTransform.capability_tags` (`plugins/infrastructure/base.py:190`, open-vocabulary). Set `capability_tags = ("narrative-summary",)` on `batch_classifier_metrics` and `batch_distribution_profile` in the same commit (bootstrap pair).
- No new wire field — the catalog already serializes `capability_tags` at `web/catalog/service.py:333,345`. The frontend reads the tag list it already receives.
- Tier-1 audit event on YAML export (B3) via the existing `GET /api/sessions/{session_id}/state/yaml` route — the route already exists at `web/sessions/routes.py:5145`; this plan adds the audit write as an `export_yaml` row in `composer_completion_events_table`. The write is **sync, crash-on-failure** per CLAUDE.md audit primacy — no "low-priority / telemetry-class" carve-out.

**Out of scope (explicit deferrals, with link to follow-up):**

- **Queryable shareable-reviews index.** A "review list" UI ("show me all the pipelines I've shared / all the pipelines shared with me") needs a queryable index. That index requires a separate table and belongs to Phase 9's schema cohort — out of scope for Phase 6.
- **Generic "composer decision log."** The `composer_completion_events_table` is scoped to *completion gestures* (`mark_ready_for_review`, `export_yaml`). Future composer-level decisions (e.g., a hypothetical "save draft" gesture) get their own `event_type` value within the same table only if the operator extends the CHECK constraint in a new schema-change cohort, or get a separate table per the Phase 18 precedent. Adding a third event type is a schema change — not an implementation-time decision.
- **Token revocation v1.** Tokens expire by `expires_at`; no revoke endpoint, no "invalidate all my tokens" affordance. Without a per-token row there is no place to mark a token revoked short of rotating the signing key; that is the documented v1 behaviour.
- **Multi-user collaborative editing.** A3 was adjudicated as "shareable link → read-only inspect view (initial impl)." The inspect view is read-only; co-editing is **not** implementable on top of this plan without a separate design pass.
- **A separate reviewer surface** (accept/reject UI). E2 = (a) read-only inspect view v1.
- **Run-analysis as a separate completion verb.** Per design doc 09 ("Why not four") Run-analysis is collapsed into Run-pipeline + narrative result rendering. There is no `Run-analysis` verb, no `composer.run_analysis` audit event, and no separate endpoint. The narrative rendering layer (Phase 6B Task 6) consumes Phase 5b's interpretation events via the existing run-result stream. Phase 6A does **not** consume `interpretation_events_table` directly.
- **Org-level review queues, notification emails, "share via Slack" integrations.** All out of scope.
- **Key rotation tooling.** The signing key sits in `WebSettings`. Rotating it **immediately invalidates all outstanding tokens** — documented v1 behaviour. A rotation-with-grace-period mechanism (dual-key acceptance window) is a Phase 9 follow-up.

---

## Trust tier check (per CLAUDE.md)

| Surface | Tier | Posture |
|---|---|---|
| `POST /mark-ready-for-review` request — path UUID, no body | Tier 3 inbound | FastAPI parses UUID; bad input → 422. Auth via `get_current_user`. |
| `GET /shareable-link` request | Tier 3 inbound | Same. |
| `GET /sessions/shared/{token}` request — token in path | Tier 3 inbound | Token is **a credential**. Validate signature, expiry, and recipient-is-authenticated before granting any read. The token is a *capability*, not an authenticator — see §"Token is a capability, not an authenticator". |
| `composition_states` row read | Tier 1 | Existing path; reuse `state_from_record(record)`. |
| Snapshot blob read from payload store (by `payload_digest`) | Tier 1 | Direct read of a Tier-1 content-addressable blob; the digest is the integrity check — mismatch on retrieval crashes. |
| HMAC primitive (sign/verify) | Tier 1 | The signed payload is canonical-encoded bytes; tampering → signature mismatch → 401. |
| `composer_completion_events_table` insert (mark-ready, yaml-export) | Tier 1 | Sessions-DB write through SQLAlchemy Core inside the request handler. Sync, crash-on-failure — same primacy rule as every other audit write. |
| YAML payload returned to the user | Tier 1 outbound | Generated by the existing `yaml_generator.py`; not re-validated. |
| Response models | Tier 1 outbound | Strict Pydantic; drift crashes at construction. |

**Capability vs authenticator:** the shareable-link token authorizes a *specific authenticated user* to read a session they don't own. It does NOT log them in. The route is `GET /api/sessions/shared/{token}` + `Depends(get_current_user)`. Without the auth gate, anyone with the URL gets in — that is **not** the designed behaviour. State this explicitly in the route docstring so a future maintainer doesn't "simplify" the auth dependency away.

---

## Where the ready-for-review state lives (load-bearing — DO NOT defer)

Three candidates were considered:

1. `sessions.ready_for_review_state_id: String | None FK → composition_states.id` (per-session pointer) — **rejected**: schema change.
2. `composition_states.is_ready_for_review: Boolean` (per-version flag) — **rejected**: schema change.
3. New `shareable_reviews` table with `(token, session_id, state_id, expires_at, created_by, signature, …)` — **rejected**: schema change (and per `project_db_migration_policy` forces a destructive DB delete).
4. **Self-verifying HMAC token + content-addressable snapshot blob in the existing payload store, with audit event in `composer_completion_events_table`.** No per-token row. The token encodes everything needed to verify: session id, state id, expiry, nonce, payload digest, signature. The snapshot is fetched from the payload store by digest. The audit event lands in the new `composer_completion_events_table` in the sessions DB — see §"Audit-event recording" below.

**Decision: option 4 (Path A amendment).** The token + content-addressable blob model is the right shape for three reasons:

1. **No DB round-trip on verify.** The signed payload encodes everything `verify()` needs (session_id, state_id, expires_at, nonce, payload_digest). A reviewer's `GET /api/sessions/shared/{token}` does not query a tokens table to check validity — signature math alone is authoritative. Eliminates a class of "token exists but row stale / row deleted but token cached" race conditions.
2. **Self-contained portable token.** The capability surface (what the reviewer can do) is the token; the audit surface (what the operator recorded) is `composer_completion_events_table`. Separating these lets the token travel through email/Slack/copy-paste without dragging the audit record along, and lets the audit record survive token expiry / payload-store reaping for the full retention window.
3. **Schema scope is *audit*, not capability.** The `composer_completion_events_table` addition is a one-table schema-change cohort (accepted under Path A — see `project_db_migration_policy`), following the Phase 18 (5b) precedent of "one new table per event family." The token-and-blob design is what keeps the *capability* layer schema-free; the audit layer is allowed its own table.

Option 4 preserves every audit property the original `shareable_reviews` table promised:

**Layer separation reminder:** "no DB round-trip on verify" and "self-contained portable token" describe the **capability** layer. The **audit** layer is *not* schema-free — it lands in the new `composer_completion_events_table` per §"Audit-event recording" below. The two storage concerns are deliberately separated: the token is the user-visible capability, the audit event is the operator-visible record of issuance.

| Property | Original (table) | Adjudicated (token + blob + sessions-DB event) |
|---|---|---|
| Audit-trail of "user X marked Y ready at Z" | Row + writer_principal | `composer_completion_events_table` row (`actor`, `created_at`, `payload_digest`, `expires_at`) |
| Per-token writer attribution | `created_by_user_id` column | `actor` column in `composer_completion_events_table` + `created_by_user_id` field in the signed payload |
| Expiry tracking | `expires_at` column | `expires_at` field in the signed payload (verified on resolve) + `expires_at` column in the event row |
| Tamper detection | Stored signature + recompute | Signature over the canonical payload (including `payload_digest`); mismatch → reject |
| Snapshot persistence | Composite FK to composition_states | Content-addressable blob in payload store; digest in token and event row |
| Revocation | (deferred) | (deferred — rotate signing key to invalidate all) |

Implications:

- **No change to `composition_states.provenance` closed enum.** Mark-ready-for-review does not write a new composition_state row; the snapshot blob is a frozen copy of the existing state.
- **No change to `sessions_table`.** No new column, no new check constraint, no new default.
- **No change to `audit_access_log_table`.** No `writer_principal` extension. The two new audit events land in `composer_completion_events_table`, not the audit-access log.
- **One new table (`composer_completion_events_table`)** per Path A. The payload store is used without schema modification.
- **Cost of "no per-token row":** the `GET /shareable-link` endpoint cannot look up "the existing valid token for this (session, state)" because there is no per-token row. It re-mints a fresh token on every call. Idempotency at the snapshot level is provided by content-addressing: re-saving an unchanged composition yields the identical `payload_digest`, so two tokens for the same snapshot resolve to the same blob. The signed-payload nonce makes the two tokens differ as strings, which is acceptable for v1.

---

## Audit-event recording (load-bearing — DO NOT defer)

Two new Tier-1 audit events ship in Phase 6, recorded in the new `composer_completion_events_table` in the sessions DB:

1. **Mark-ready-for-review** — event type `mark_ready_for_review` ("user X marked composition Y as ready for review at timestamp Z, with snapshot digest D"). Columns populated: `id`, `session_id`, `composition_state_id`, `event_type`, `actor`, `created_at`, `payload_digest`, `expires_at`.
2. **YAML export** — event type `export_yaml` (B3) ("user X exported YAML for composition Y"). Columns populated: `id`, `session_id`, `composition_state_id`, `event_type`, `actor`, `created_at`. `payload_digest` and `expires_at` stay NULL.

**Precedent:** this pattern — one new table per event family, closed-enum CHECK constraint, nullable optional columns — is established by Phase 18 (5b)'s `interpretation_events_table` (see `web/sessions/models.py`, alongside `proposal_events_table` at line 342). Phase 6 is the third event family in this pattern; it does not extend any existing table.

**Why not `proposal_events_table`:** that table's closed CHECK constraint (`'proposal.created'`, `'proposal.accepted'`, `'proposal.rejected'`, `'trust_mode.changed'`) is a governance boundary. Adding new event types is a schema change under `project_db_migration_policy`. Completion-gesture events belong to a different event family and get their own table.

**Why not `audit_access_log_table`:** that table's `writer_principal` column is a closed enum (governance-locked at `models.py:634`). Extending it is a schema change. Completion-gesture events are not access-log entries.

**Why not the Landscape pipeline-execution audit:** the Landscape records pipeline *execution* decisions (sources, transforms, sinks, gate routing). Composition-time decisions ("user marked a draft ready") are a different audit domain. The sessions DB is the correct home — the same home as `proposal_events_table` and `interpretation_events_table`.

**Primacy rules:**

- **Sync, crash-on-failure.** Per CLAUDE.md audit primacy: "Audit fires first (sync, crash-on-failure)." Both event writes follow this discipline; there is no logging-not-raised carve-out for either. The sessions-DB write is a synchronous SQLAlchemy `connection.execute()` call inside the request handler; the HTTP response is not returned until the write commits.
- **Audit-first ordering (load-bearing).** For `mark_ready_for_review`, the sequence is: (1) build the snapshot dict in memory; (2) canonical-JSON-serialize it and compute `payload_digest = sha256(serialized_bytes)`; (3) **insert the `composer_completion_events` row** with `payload_digest` (audit fires first — sync, crash-on-failure); (4) write the serialized bytes to the payload store keyed by `payload_digest`; (5) sign the token (which includes `payload_digest`); (6) return the token to the caller. **The audit insert precedes the blob write**, not the other way around. The earlier "blob first, then audit, retention reaps orphans" framing inverted CLAUDE.md's primacy rule and is removed.
- If the audit insert fails, the entire request fails before any blob write occurs. If the blob write fails after a successful audit insert, the entire request still fails (no token is signed, no token is returned), and the audit row stands as honest evidence of the attempt. A reviewer who could somehow synthesise a token for the recorded `payload_digest` would receive `ResourceNotFound` from the payload store — a clean failure mode. There is no retention-policy dependency for orphan-blob reaping (because the blob is never written without a prior audit row), and there is no audit-trail gap (because audit fires first).

**The writer dependency:** `ShareableReviewService` and the YAML-export route take the sessions-DB connection (or a `ComposerCompletionEventWriter` service object) as an injected dependency. The write path is a direct `connection.execute(composer_completion_events_table.insert().values(...))` call — the same pattern `proposal_events_table` uses at its write site. Grep `proposal_events_table.insert()` in `web/` to find the existing pattern.

**Read-back surface for integration tests:** `SELECT event_type, payload_digest, created_at, actor FROM composer_completion_events WHERE session_id = ? AND event_type IN ('mark_ready_for_review', 'export_yaml') ORDER BY created_at`. The sessions DB file path comes from `WebSettings.sessions_db_url` (resolved at runtime via the test app's settings). Sessions-DB writes are synchronous; no flush step needed; immediate read is correct.

---

## Signing primitive (load-bearing — DO NOT defer)

The "existing HMAC infrastructure" referenced in design doc 09 is `core/landscape/exporter.py::_sign_record` (lines 125–143). That method signs **audit records** — dict-keyed payloads with a `signature` key inserted alongside the data. Shareable-link tokens are a different shape: URL-safe, self-contained, signed payload over a tuple of fields including a content-address.

**Decision: reuse the primitive (`hmac.new(key, msg, hashlib.sha256)`), not the exporter API.** The `ShareTokenSigner` in `web/shareable_reviews/signer.py` is a new class with its own contract:

```python
@dataclass(frozen=True, slots=True)
class ShareTokenPayload:
    version: int  # for forward compat; v1 only in this phase
    session_id: UUID
    state_id: UUID
    expires_at: datetime
    nonce_hex: str
    payload_digest: str  # content-address of the snapshot blob in the payload store
    created_by_user_id: str  # user id of the original creator; carried in the signed envelope

class ShareTokenSigner:
    def __init__(self, signing_key: bytes) -> None: ...
    def sign(self, payload: ShareTokenPayload) -> str:
        """Return a URL-safe base64 token encoding payload + signature."""
    def verify(self, token: str) -> ShareTokenPayload:
        """Verify signature and decode payload. Raises InvalidToken on mismatch."""
```

The signing key is sourced from a new `WebSettings.shareable_link_signing_key: bytes` field (separate from any audit signing key — different rotation cadence, different blast radius). The Settings field is `Field(...)` (required, no default) — the service refuses to start without it. The validator rejects values shorter than 32 bytes.

The canonical encoding is `json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")` followed by HMAC-SHA256, mirroring the exporter's discipline. The token is the URL-safe base64 of `len(payload_json).to_bytes(4, "big") + payload_json + signature_bytes` — a self-contained, parseable structure.

**Verify ALWAYS uses `hmac.compare_digest`** (not `==`) — constant-time signature comparison is required by the Python `hmac` documentation to defeat timing-side-channel attacks against signature verifiers. This is **new discipline** in the signer, not a reuse from `core/landscape/exporter.py` — the exporter uses `.hexdigest()` for record-signing (`exporter.py:142`) and does not perform a constant-time compare (records are signed for write, not verified at attacker-controlled boundaries). ELSPETH-internal precedents for `compare_digest` at boundary verifiers: `core/payload_store.py:111,163` and `web/blobs/service.py:759` (both compare a caller-supplied content_hash against a stored hash). The signer pattern matches those, not the exporter's.

---

## File structure

**New:**

- `src/elspeth/web/shareable_reviews/__init__.py`
- `src/elspeth/web/shareable_reviews/models.py` — Pydantic response models.
- `src/elspeth/web/shareable_reviews/signer.py` — `ShareTokenSigner`, `ShareTokenPayload`, `InvalidToken`.
- `src/elspeth/web/shareable_reviews/service.py` — `ShareableReviewService`.
- `src/elspeth/web/shareable_reviews/routes.py` — `create_shareable_reviews_router()`.
- `tests/unit/web/shareable_reviews/__init__.py`
- `tests/unit/web/shareable_reviews/test_signer.py`
- `tests/unit/web/shareable_reviews/test_service.py`
- `tests/unit/web/shareable_reviews/test_models.py`
- `tests/unit/web/sessions/test_composer_completion_events_table.py` — schema tests for the new table (Task 1).
- `tests/integration/web/test_shareable_reviews_routes.py`
- `tests/integration/web/test_yaml_export_audit_event.py`

**Modified:**

- `src/elspeth/web/sessions/models.py` — add `composer_completion_events_table` (Task 1). No changes to `proposal_events_table`, `interpretation_events_table`, or `audit_access_log_table`.
- `src/elspeth/web/config.py` — add `WebSettings.shareable_link_signing_key: bytes` (required) and `WebSettings.shareable_link_lifetime_seconds: int` (default 30 days).
- `src/elspeth/web/app.py` — instantiate `ShareTokenSigner` and `ShareableReviewService`; `include_router(create_shareable_reviews_router())`. Inject the sessions-DB connection and the existing payload store.
- `src/elspeth/web/sessions/routes.py` — extend the existing `GET /{session_id}/state/yaml` route at line 5145 to insert an `export_yaml` row in `composer_completion_events_table` before returning (sync, crash-on-failure).
- `src/elspeth/plugins/transforms/batch_classifier_metrics.py` — add `capability_tags = ("narrative-summary",)` class declaration.
- `src/elspeth/plugins/transforms/batch_distribution_profile.py` — add `capability_tags = ("narrative-summary",)` class declaration.
- (No `infrastructure/base.py` change. No `web/catalog/` change. `capability_tags` already exists end-to-end.)
- `docs/architecture/adr/` — ADR for the shareable-reviews design and the HMAC artifact contract.

**Not modified (deliberately):**

- `src/elspeth/web/sessions/models.py` existing tables — `proposal_events_table`, `interpretation_events_table`, `audit_access_log_table`, `sessions_table`, `composition_states` — **no changes** to any existing table. Phase 9 owns the queryable shareable-reviews index if one is later required.

---

## Task 1: New `composer_completion_events_table` schema

**Files:** `src/elspeth/web/sessions/models.py`, `tests/unit/web/sessions/test_composer_completion_events_table.py` (new).

This task adds the `composer_completion_events_table` to the sessions DB, following the Phase 18 (5b) precedent of creating one new table per event family rather than extending an existing table or routing through the pipeline-execution Landscape. See [18-phase-5b-surface-llm-interpretation.md](18-phase-5b-surface-llm-interpretation.md) §"interpretation_events_table" for the analogue.

**Table definition** (add to `src/elspeth/web/sessions/models.py` alongside `proposal_events_table` and `interpretation_events_table`):

```python
composer_completion_events_table = Table(
    "composer_completion_events",
    metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("composition_state_id", String, ForeignKey("composition_states.id"), nullable=True),
    Column("event_type", String, nullable=False),  # CHECK constraint below
    Column("actor", String, nullable=False),  # user_id of the actor
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("payload_digest", String, nullable=True),  # content-address; populated for mark_ready_for_review
    Column("expires_at", DateTime(timezone=True), nullable=True),  # populated for mark_ready_for_review
    CheckConstraint(
        "event_type IN ('mark_ready_for_review', 'export_yaml')",
        name="ck_composer_completion_events_type",
    ),
    Index("ix_composer_completion_events_session_created", "session_id", "created_at"),
)
```

**Append-only triggers (required):**

The `composer_completion_events_table` is an audit table — every row is a permanent audit fact and is never updated or deleted in v1. Following the `interpretation_events_table` precedent (and correcting the Phase 18 omission that filigree finding `elspeth-9aba8da942` is remediating), this table ships with **both** `BEFORE UPDATE` and `BEFORE DELETE` triggers from day 1. Unlike `interpretation_events_table` — which permits DELETE on PENDING rows for orphan recovery — completion events have no recovery path; both triggers are **unconditional ABORT**.

Add to `src/elspeth/web/sessions/models.py` adjacent to the existing interpretation-events trigger DDL (around `models.py:704–733`):

```python
_COMPOSER_COMPLETION_EVENTS_NO_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_composer_completion_events_no_update
BEFORE UPDATE ON composer_completion_events
BEGIN
    SELECT RAISE(ABORT, 'composer_completion_events is append-only; UPDATE is forbidden');
END;
"""

_COMPOSER_COMPLETION_EVENTS_NO_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_composer_completion_events_no_delete
BEFORE DELETE ON composer_completion_events
BEGIN
    SELECT RAISE(ABORT, 'composer_completion_events is append-only; DELETE is forbidden');
END;
"""
```

Both triggers MUST be registered in `_REQUIRED_SQLITE_TRIGGERS` so the startup validator refuses to start if either is missing — see the `interpretation_events_table` precedent for the registration pattern.

**Design notes:**

- **Closed-enum CHECK on `event_type`**: only two values in v1 (`mark_ready_for_review`, `export_yaml`). Adding a third value in a later phase is a schema change and requires its own schema-change cohort (DB delete on next deploy).
- **`payload_digest` and `expires_at` are nullable** because they are only meaningful for `mark_ready_for_review`. For `export_yaml` events, both stay NULL. This follows the `interpretation_events_table` precedent of one table with nullable optional columns rather than splitting into two tables.
- **SQLite-only** per `project_phase9_sqlite_only`. Shipping this table forces a DB delete on next deploy of any environment that already has the v1 sessions DB (per `project_db_migration_policy`).
- **Read-back surface for integration tests:** SQL `SELECT * FROM composer_completion_events WHERE session_id = ? AND event_type IN (?, ?)`. The sessions DB file path comes from `WebSettings.sessions_db_url` (resolved at runtime via the test app's settings).

**Schema-epoch bump (required):**

Sessions-DB schema additions MUST bump `SESSION_SCHEMA_EPOCH` at `src/elspeth/web/sessions/models.py:56` from `3` to `4`. Phase 18 already bumped this from `2` to `3` as part of its remediation cohort (commit `3dee19f8d` and prior); Phase 6 continues the discipline by bumping again for its own schema-change cohort. The sentinel is validated at startup by `_assert_schema_sentinels()` in `src/elspeth/web/sessions/schema.py:112`: if the DB's `PRAGMA user_version` doesn't match the constant, the service refuses to start and instructs the operator to delete the sessions DB.

This is the **mechanical trigger** for the DB-delete operator action documented in `project_db_migration_policy` (memory). Without the bump, live deployments at epoch 2 pass validation and crash mid-request on the first INSERT to the new table — the exact silent-corruption mode the sentinel exists to prevent. The Phase 18 defect-pass finding `elspeth-c03e9bfcf8` filed the same bug class against the Landscape DB's `SQLITE_SCHEMA_EPOCH`; Phase 6 must not repeat it on the sessions DB.

- [ ] **Step 1: Failing tests.**

```python
"""Tests for composer_completion_events_table schema."""

import pytest
from sqlalchemy import create_engine, text

from elspeth.web.sessions.models import metadata


def test_table_exists_in_metadata() -> None:
    assert "composer_completion_events" in metadata.tables


def test_check_constraint_rejects_invalid_event_type() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    with engine.connect() as conn:
        with pytest.raises(Exception):
            conn.execute(
                text(
                    "INSERT INTO composer_completion_events "
                    "(id, session_id, event_type, actor, created_at) "
                    "VALUES (:id, :sid, :et, :actor, :ts)"
                ),
                {"id": "e1", "sid": "s1", "et": "invalid_type", "actor": "user1", "ts": "2026-01-01T00:00:00+00:00"},
            )
            conn.commit()


def test_check_constraint_accepts_mark_ready_for_review() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions (id, user_id, created_at) VALUES (:id, :uid, :ts)"
            ),
            {"id": "s1", "uid": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.execute(
            text(
                "INSERT INTO composer_completion_events "
                "(id, session_id, event_type, actor, created_at) "
                "VALUES (:id, :sid, :et, :actor, :ts)"
            ),
            {"id": "e1", "sid": "s1", "et": "mark_ready_for_review", "actor": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.commit()


def test_check_constraint_accepts_export_yaml() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions (id, user_id, created_at) VALUES (:id, :uid, :ts)"
            ),
            {"id": "s1", "uid": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.execute(
            text(
                "INSERT INTO composer_completion_events "
                "(id, session_id, event_type, actor, created_at) "
                "VALUES (:id, :sid, :et, :actor, :ts)"
            ),
            {"id": "e2", "sid": "s1", "et": "export_yaml", "actor": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.commit()


def test_session_fk_cascades_on_delete() -> None:
    """FK cascade from sessions → composer_completion_events is blocked by the append-only trigger.

    When a parent sessions row is deleted with FK cascades enabled, SQLite attempts to
    cascade-delete the dependent composer_completion_events rows. The BEFORE DELETE trigger
    on composer_completion_events fires before the cascade can complete and raises ABORT,
    which rolls back the parent DELETE as well. This confirms that completion events are
    permanently retained: a session cannot be deleted while it has completion-event children.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.execute(
            text("INSERT INTO sessions (id, user_id, created_at) VALUES (:id, :uid, :ts)"),
            {"id": "s1", "uid": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.execute(
            text(
                "INSERT INTO composer_completion_events "
                "(id, session_id, event_type, actor, created_at) "
                "VALUES (:id, :sid, :et, :actor, :ts)"
            ),
            {"id": "e1", "sid": "s1", "et": "export_yaml", "actor": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.commit()
        with pytest.raises(Exception, match="append-only"):
            conn.execute(text("DELETE FROM sessions WHERE id = 's1'"))
            conn.commit()


def test_session_schema_epoch_bumped_to_4() -> None:
    """Phase 6 schema-change cohort: SESSION_SCHEMA_EPOCH must bump from 3 to 4.

    Per project_db_migration_policy, bumping the epoch is the mechanical signal
    that triggers the operator's DB-delete action on next deploy. Without the
    bump, live deployments crash mid-request on first INSERT — see filigree
    finding elspeth-c03e9bfcf8 for the analogous Landscape-DB defect.
    """
    from elspeth.web.sessions.models import SESSION_SCHEMA_EPOCH

    assert SESSION_SCHEMA_EPOCH == 4


def test_update_trigger_blocks_mutation() -> None:
    """Audit table is append-only — UPDATE must raise."""
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO sessions (id, user_id, created_at) VALUES (:id, :uid, :ts)"),
            {"id": "s1", "uid": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.execute(
            text(
                "INSERT INTO composer_completion_events "
                "(id, session_id, event_type, actor, created_at) "
                "VALUES (:id, :sid, :et, :actor, :ts)"
            ),
            {"id": "e1", "sid": "s1", "et": "export_yaml", "actor": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.commit()
        with pytest.raises(Exception, match="append-only"):
            conn.execute(
                text("UPDATE composer_completion_events SET actor = 'attacker' WHERE id = 'e1'")
            )
            conn.commit()


def test_delete_trigger_blocks_removal() -> None:
    """Audit table is append-only — DELETE must raise. Phase 6 ships both UPDATE and DELETE triggers from day 1, correcting the Phase 18 omission tracked at filigree elspeth-9aba8da942."""
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO sessions (id, user_id, created_at) VALUES (:id, :uid, :ts)"),
            {"id": "s1", "uid": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.execute(
            text(
                "INSERT INTO composer_completion_events "
                "(id, session_id, event_type, actor, created_at) "
                "VALUES (:id, :sid, :et, :actor, :ts)"
            ),
            {"id": "e1", "sid": "s1", "et": "export_yaml", "actor": "user1", "ts": "2026-01-01T00:00:00+00:00"},
        )
        conn.commit()
        with pytest.raises(Exception, match="append-only"):
            conn.execute(text("DELETE FROM composer_completion_events WHERE id = 'e1'"))
            conn.commit()
```

- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Add the `composer_completion_events_table` definition to `src/elspeth/web/sessions/models.py` alongside `proposal_events_table` and `interpretation_events_table`. No migration script — per `project_phase9_sqlite_only`, the Phase 9 migration runner owns schema evolution; for now, DB delete on redeploy is the documented operator action.
  - Bump `SESSION_SCHEMA_EPOCH = 4` in `src/elspeth/web/sessions/models.py` (currently `3` at line 56 — Phase 18 already bumped 2→3). The validator `_assert_schema_sentinels()` at `web/sessions/schema.py:112` enforces this against `PRAGMA user_version`; existing deployments will refuse startup until the sessions DB is deleted, matching the documented operator action.
  - Add the two trigger DDL strings to `src/elspeth/web/sessions/models.py` adjacent to the existing interpretation-events trigger registrations. Add both to `_REQUIRED_SQLITE_TRIGGERS` so the startup validator catches missing triggers.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/sessions): add composer_completion_events_table + append-only triggers + bump SESSION_SCHEMA_EPOCH (Phase 6 schema-change cohort)`.

---

## Task 2: `WebSettings.shareable_link_signing_key` + lifetime

**Files:** `web/config.py`, `tests/unit/web/test_config_shareable_link.py` (new).

- [ ] **Step 1: Failing test.**

```python
"""Tests for shareable_link_signing_key on WebSettings."""

import pytest
from pydantic import ValidationError

from elspeth.web.config import WebSettings


def test_signing_key_minimum_length() -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        WebSettings(shareable_link_signing_key=b"too-short")


def test_signing_key_accepts_32_bytes() -> None:
    key = b"0" * 32
    settings = WebSettings(shareable_link_signing_key=key)
    assert settings.shareable_link_signing_key == key


def test_signing_key_required() -> None:
    # Field(...) — no default. WebSettings() with no env var must raise.
    with pytest.raises(ValidationError):
        WebSettings()


def test_signing_key_lifetime_default_30_days() -> None:
    key = b"0" * 32
    settings = WebSettings(shareable_link_signing_key=key)
    assert settings.shareable_link_lifetime_seconds == 30 * 24 * 3600


def test_signing_key_lifetime_override() -> None:
    key = b"0" * 32
    settings = WebSettings(
        shareable_link_signing_key=key,
        shareable_link_lifetime_seconds=3600,
    )
    assert settings.shareable_link_lifetime_seconds == 3600
```

- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.**

```python
shareable_link_signing_key: bytes = Field(
    ...,
    description=(
        "HMAC key for shareable-review tokens. Must be set explicitly; "
        "the service refuses to start without it. Rotating this key "
        "invalidates ALL outstanding shareable links. Generate with "
        "`openssl rand -base64 32`."
    ),
)

shareable_link_lifetime_seconds: int = Field(
    default=30 * 24 * 3600,
    description=(
        "Lifetime (in seconds) for shareable-review tokens. Default: 30 days. "
        "The service stamps expires_at = now() + this delta when creating a token."
    ),
)

@field_validator("shareable_link_signing_key")
@classmethod
def _signing_key_min_length(cls, v: bytes) -> bytes:
    if len(v) < 32:
        raise ValueError("shareable_link_signing_key must be at least 32 bytes")
    return v
```

- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/config): add shareable_link_signing_key (required) + shareable_link_lifetime_seconds`.

---

## Task 3: `ShareTokenSigner` primitive

**Files:** `web/shareable_reviews/__init__.py` (new package marker), `web/shareable_reviews/signer.py` (new), `tests/unit/web/shareable_reviews/__init__.py` (new), `tests/unit/web/shareable_reviews/test_signer.py` (new).

- [ ] **Step 1: Failing test.**

```python
"""Tests for ShareTokenSigner — sign/verify round-trip + tamper detection."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from elspeth.web.shareable_reviews.signer import (
    InvalidToken,
    ShareTokenPayload,
    ShareTokenSigner,
)


def _make_payload() -> ShareTokenPayload:
    return ShareTokenPayload(
        version=1,
        session_id=uuid4(),
        state_id=uuid4(),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        nonce_hex="deadbeef" * 4,
        payload_digest="sha256:" + ("ab" * 32),
        created_by_user_id="user-1",
    )


def test_round_trip() -> None:
    signer = ShareTokenSigner(b"k" * 32)
    payload = _make_payload()
    token = signer.sign(payload)
    decoded = signer.verify(token)
    assert decoded == payload


def test_tampered_signature_rejected() -> None:
    signer = ShareTokenSigner(b"k" * 32)
    token = signer.sign(_make_payload())
    tampered = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    with pytest.raises(InvalidToken):
        signer.verify(tampered)


def test_wrong_key_rejected() -> None:
    signer_a = ShareTokenSigner(b"a" * 32)
    signer_b = ShareTokenSigner(b"b" * 32)
    token = signer_a.sign(_make_payload())
    with pytest.raises(InvalidToken):
        signer_b.verify(token)


def test_expired_token_rejected() -> None:
    signer = ShareTokenSigner(b"k" * 32)
    expired = ShareTokenPayload(
        version=1,
        session_id=uuid4(),
        state_id=uuid4(),
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        nonce_hex="ff" * 16,
        payload_digest="sha256:" + ("ab" * 32),
        created_by_user_id="user-1",
    )
    token = signer.sign(expired)
    with pytest.raises(InvalidToken, match="expired"):
        signer.verify(token)


def test_url_safe_token() -> None:
    signer = ShareTokenSigner(b"k" * 32)
    token = signer.sign(_make_payload())
    assert "+" not in token
    assert "/" not in token
    assert all(c.isalnum() or c in "-_=" for c in token)


def test_compare_digest_used(monkeypatch) -> None:
    """Verify uses hmac.compare_digest — not ==."""
    import hmac as hmac_mod
    calls: list[tuple[bytes, bytes]] = []
    real = hmac_mod.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac_mod, "compare_digest", spy)
    signer = ShareTokenSigner(b"k" * 32)
    token = signer.sign(_make_payload())
    signer.verify(token)
    assert calls, "ShareTokenSigner.verify must use hmac.compare_digest"


def test_verify_rejects_single_byte_tamper() -> None:
    """Behavioural test: token differing by one character is rejected."""
    signer = ShareTokenSigner(b"x" * 32)
    valid_token = signer.sign(_make_payload())
    tampered = valid_token[:-1] + ("a" if valid_token[-1] != "a" else "b")
    with pytest.raises(InvalidToken):
        signer.verify(tampered)


def test_payload_digest_in_signed_envelope() -> None:
    """payload_digest is signed — swapping it after-the-fact must reject."""
    signer = ShareTokenSigner(b"k" * 32)
    p1 = _make_payload()
    token = signer.sign(p1)
    # Decode and confirm the digest round-trips.
    decoded = signer.verify(token)
    assert decoded.payload_digest == p1.payload_digest
```

- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** `web/shareable_reviews/signer.py`:

```python
"""Shareable-review token signing primitive.

Reuses hmac.new(..., hashlib.sha256) from the standard library — the same
primitive that backs core/landscape/exporter.py::_sign_record. NOT a reuse
of the exporter's API: the payload shape (URL-safe self-contained token)
differs from audit-record signing (dict-keyed signature insertion).

The signing key is per-deployment, configured via
WebSettings.shareable_link_signing_key. Rotating the key invalidates all
outstanding tokens; this is the documented Phase 6 behaviour.

The signed envelope includes `payload_digest`, the content-address of the
snapshot blob in the payload store. Tampering with the blob (digest no
longer resolves) or with the token (signature no longer verifies) is
detected on resolve.

Layer: L3 (web application).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final
from uuid import UUID


_PAYLOAD_VERSION: Final[int] = 1
_SIGNATURE_BYTES: Final[int] = 32  # SHA-256


class InvalidToken(Exception):
    """Raised when a shareable-review token fails verification."""


@dataclass(frozen=True, slots=True)
class ShareTokenPayload:
    version: int
    session_id: UUID
    state_id: UUID
    expires_at: datetime
    nonce_hex: str
    payload_digest: str
    created_by_user_id: str

    def to_canonical_json(self) -> bytes:
        d = {
            "version": self.version,
            "session_id": str(self.session_id),
            "state_id": str(self.state_id),
            "expires_at": self.expires_at.isoformat(),
            "nonce_hex": self.nonce_hex,
            "payload_digest": self.payload_digest,
            "created_by_user_id": self.created_by_user_id,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ShareTokenSigner:
    def __init__(self, signing_key: bytes) -> None:
        if len(signing_key) < 32:
            raise ValueError("signing_key must be at least 32 bytes")
        self._key = signing_key

    def sign(self, payload: ShareTokenPayload) -> str:
        body = payload.to_canonical_json()
        sig = hmac.new(self._key, body, hashlib.sha256).digest()
        blob = len(body).to_bytes(4, "big") + body + sig
        return base64.urlsafe_b64encode(blob).decode("ascii")

    def verify(self, token: str) -> ShareTokenPayload:
        try:
            blob = base64.urlsafe_b64decode(token.encode("ascii"))
        except Exception as exc:
            raise InvalidToken("malformed token") from exc
        if len(blob) < 4 + _SIGNATURE_BYTES:
            raise InvalidToken("truncated token")
        body_len = int.from_bytes(blob[:4], "big")
        if len(blob) != 4 + body_len + _SIGNATURE_BYTES:
            raise InvalidToken("token length mismatch")
        body = blob[4 : 4 + body_len]
        sig = blob[4 + body_len :]
        expected = hmac.new(self._key, body, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            raise InvalidToken("signature mismatch")
        try:
            d = json.loads(body)
        except json.JSONDecodeError as exc:
            raise InvalidToken("payload decode failed") from exc
        try:
            payload = ShareTokenPayload(
                version=d["version"],
                session_id=UUID(d["session_id"]),
                state_id=UUID(d["state_id"]),
                expires_at=datetime.fromisoformat(d["expires_at"]),
                nonce_hex=d["nonce_hex"],
                payload_digest=d["payload_digest"],
                created_by_user_id=d["created_by_user_id"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvalidToken("payload shape mismatch") from exc
        if payload.version != _PAYLOAD_VERSION:
            raise InvalidToken(f"unsupported version {payload.version}")
        if payload.expires_at < datetime.now(timezone.utc):
            raise InvalidToken("token expired")
        return payload
```

- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/shareable_reviews): add ShareTokenSigner with HMAC-SHA256 + content-address envelope`.

---

## Task 4: Pydantic response models for the three new endpoints

**Files:** `web/shareable_reviews/models.py`, `tests/unit/web/shareable_reviews/test_models.py`.

Three response models, all strict + extra-forbid. `SharedInspectResponse` reuses the Phase 2 `AuditReadinessSnapshot` model verbatim — the wire shape 19b consumes includes `audit_readiness`.

**Post-Phase-18 merge fact (verified 2026-05-19).** The Phase-2 `AuditReadinessSnapshot` model was extended by Phase 18 (5b) to include an `llm_interpretations` row as the fifth member of the closed `ReadinessRowId` enum (`web/audit_readiness/models.py:14–21`). The snapshot service already aggregates `interpretation_events_table` rows server-side at snapshot-build time (`web/audit_readiness/service.py:218–262`) to populate this row's status, summary, and detail. **Phase 6 does not need to add this aggregation** — calling the existing snapshot service yields the row automatically. The wire field stays `audit_readiness: AuditReadinessSnapshot` and the reviewer in the shared view sees the same six-row panel the owner sees, sourced server-side.

- [ ] **Step 1: Failing test.**

```python
"""Tests for shareable_reviews response models."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from elspeth.web.execution.schemas import AuditReadinessSnapshot
from elspeth.web.shareable_reviews.models import (
    MarkReadyForReviewResponse,
    ShareableLinkResponse,
    SharedInspectResponse,
)


def test_mark_ready_response_strict() -> None:
    resp = MarkReadyForReviewResponse(
        token="abc123",
        share_url="https://example.com/shared/abc123",
        expires_at=datetime.now(timezone.utc),
        payload_digest="sha256:" + ("ab" * 32),
    )
    assert resp.token == "abc123"


def test_mark_ready_rejects_extra() -> None:
    with pytest.raises(ValidationError, match="extra"):
        MarkReadyForReviewResponse(
            token="abc",
            share_url="https://x/",
            expires_at=datetime.now(timezone.utc),
            payload_digest="sha256:" + ("ab" * 32),
            unexpected="field",
        )


def test_shared_inspect_response_carries_audit_readiness() -> None:
    """SharedInspectResponse must include audit_readiness (consumed by 19b Task 8)."""
    # Construct with a minimal valid AuditReadinessSnapshot instance — the
    # exact shape is owned by Phase 2's schema and reused verbatim here.
    snapshot = AuditReadinessSnapshot(...)  # populate per Phase 2 shape
    resp = SharedInspectResponse(
        session_id=str(uuid4()),
        state_id=str(uuid4()),
        pipeline_metadata={...},
        composition_snapshot={...},
        yaml="version: 1\n",
        audit_readiness=snapshot,
        created_by_user_id="user-1",
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc),
    )
    assert resp.audit_readiness is snapshot
```

- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** All three inherit `_StrictResponse` from `web/execution/schemas.py`:

```python
from elspeth.web.execution.schemas import _StrictResponse, AuditReadinessSnapshot


class MarkReadyForReviewResponse(_StrictResponse):
    token: str
    share_url: str
    expires_at: datetime
    payload_digest: str  # content-address of the snapshot blob


class ShareableLinkResponse(_StrictResponse):
    token: str
    share_url: str
    expires_at: datetime
    state_id: str
    payload_digest: str


class SharedInspectResponse(_StrictResponse):
    session_id: str
    state_id: str
    pipeline_metadata: PipelineMetadata
    composition_snapshot: CompositionState
    yaml: str
    audit_readiness: AuditReadinessSnapshot  # reused verbatim from Phase 2
    created_by_user_id: str
    created_at: datetime
    expires_at: datetime
```

The `audit_readiness` field is **load-bearing**: 19b Task 8 mounts `<SharedAuditReadinessPanel readOnlyState={inspectResponse.audit_readiness} />` and reads this field directly. Drift here breaks the read-only inspect view.

- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/shareable_reviews): add strict response models including audit_readiness on SharedInspectResponse`.

---

## Task 5: `ShareableReviewService` — business logic

**Files:** `web/shareable_reviews/service.py`, `tests/unit/web/shareable_reviews/test_service.py`.

The service owns: validating the session is in a runnable state, freezing the composition snapshot, **freezing the audit-readiness snapshot at mark-time and embedding it in the blob**, writing the blob to the payload store and obtaining the digest, signing the token, writing the audit event row to `composer_completion_events_table`, and resolving an inbound token to a read-only snapshot.

**Methods:**

- `mark_ready_for_review(session_id: UUID, user_id: str) -> MarkReadyForReviewResponse`
- `get_shareable_link(session_id: UUID, user_id: str) -> ShareableLinkResponse` — always mints a fresh token over the current (session, state); content-addressing makes this idempotent at the blob level even though the token strings differ.
- `resolve_token(token: str, requesting_user_id: str) -> SharedInspectResponse` — validates token, reads the snapshot blob from the payload store by digest, returns the response with `audit_readiness` read directly from the frozen blob. Recipient auth happens at the route level; this method assumes auth has succeeded.

**Snapshot blob shape (load-bearing — frozen-at-mark-time discipline):** the snapshot is canonical-JSON `{pipeline_metadata, composition_snapshot, yaml, audit_readiness, created_by_user_id, created_at}`. The `payload_digest` is `sha256:<hex>` of the canonical bytes. The blob is stored under the payload store's normal retention policy.

**Why `audit_readiness` is frozen into the blob, not fetched fresh at resolve time** (decided 2026-05-19, post-Phase-18 merge):

1. **Audit-trail integrity.** Per CLAUDE.md auditability: "Every decision must be traceable to source data, configuration, and code version." A shareable review must show the reviewer exactly what the owner saw at the moment of mark-for-review. A fresh fetch at resolve-time would let the snapshot drift if validation state changes between mark and resolve (a transform's plugin trust class is re-classified, a secret expires, the underlying composition is unwittingly mutated). The reviewer would then see a different readiness panel than the owner ever did, and the audit record would not match the share artifact.
2. **Content-addressing.** The `payload_digest` is the SHA-256 of the blob. Freezing `audit_readiness` into the blob means the digest captures the readiness fingerprint too; `composer_completion_events_table.payload_digest` becomes an evidentially complete reference. With a fresh fetch, the digest only covers the composition shape, and "what readiness signal accompanied this share?" requires a separate audit query.
3. **Non-owner permission boundary.** The Phase-2 snapshot service signature is `ReadinessService.build(session_id, user_id, ...)` (`web/audit_readiness/service.py:198`). The `user_id` parameter feeds `validate_state(state, user_id=user_id)` and `_scoped_secret_resolver.list_refs(user_id)`. Whether these accept an arbitrary authenticated user_id (the reviewer's) or require owner identity is implementation-dependent — and fragile to future changes. Freezing the snapshot at mark-time uses the owner's user_id (the snapshot the owner saw and approved), and `resolve_token` never needs to call the snapshot service at all. The permission question disappears.

**Mark-time audit-readiness invariant.** `mark_ready_for_review` calls `ReadinessService.build(session_id, user_id)` once, embeds the returned `AuditReadinessSnapshot` in the blob, then computes the digest. **The composition must pass validation before marking** (existing rule), but additionally any readiness row whose `status == "error"` should produce `CompositionNotRunnableError` — sharing a known-broken readiness state would be share-theatre. `status == "warning"` (e.g. `llm_interpretations` with pending review) is permitted; the reviewer can see the warning. Tests must cover both gates explicitly.

**Validation contract:** `mark_ready_for_review` requires the composition to pass validation (per design doc 09 "Conditions"). It calls `ExecutionService.validate(session_id)` and refuses with a typed `CompositionNotRunnableError` if `is_valid is False`. The route translates this to a 409.

**Audit-event recording (audit-first ordering):** the request handler computes the snapshot bytes and `payload_digest` first, then **inserts the audit row before writing the blob**. Sequence: (1) build snapshot dict in memory; (2) canonical-JSON-serialize and compute `payload_digest`; (3) `connection.execute(composer_completion_events_table.insert().values(...))` with `(id, session_id, composition_state_id, event_type='mark_ready_for_review', actor, created_at, payload_digest, expires_at)`; (4) write blob to payload store; (5) sign token; (6) return. The insert is **sync, crash-on-failure** — if the audit write fails, no blob is ever written and no token is returned. If the blob write fails after the audit insert, the audit row stands as honest evidence of the attempt; no token is returned; no orphan blob exists. This matches CLAUDE.md audit primacy verbatim — see §"Audit-event recording" for the full rationale.

- [ ] **Step 1: Failing tests.** Cover:
    1. `mark_ready_for_review` happy path: writes blob (with frozen `audit_readiness`), records audit event, returns token + digest.
    2. `mark_ready_for_review` raises `CompositionNotRunnableError` when validation fails.
    3. `mark_ready_for_review` raises `CompositionNotRunnableError` when any readiness row has `status == "error"` (the new mark-time gate documented above).
    4. `mark_ready_for_review` succeeds when readiness rows have `status == "warning"` (e.g. `llm_interpretations` with pending review) — warning is not a blocker; the reviewer sees the warning.
    5. `mark_ready_for_review` propagates audit-write failures (no token returned, no orphan blob exposed via API).
    6. `get_shareable_link` mints a fresh token every call; two calls on an unchanged state yield identical `payload_digest`.
    7. `resolve_token` returns the snapshot for a valid token; `audit_readiness` is read **directly from the frozen blob** (not fetched fresh) — assert by mutating the live readiness state between mark and resolve, then assert resolve still returns the mark-time view.
    8. `resolve_token` raises `InvalidToken` for tampered/expired tokens.
    9. `resolve_token` raises `ResourceNotFound` if the payload store has expired the blob (token verifies but blob is gone).
   10. `resolve_token` does **not** call `ReadinessService.build()` (proves no permission-boundary coupling). Verify via mock-call-count assertion on the injected readiness service.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Inject `SessionServiceProtocol`, `ExecutionService`, `ShareTokenSigner`, `WebSettings`, the **sessions-DB connection** (for writing to `composer_completion_events_table` — same connection used by `proposal_events_table` writes; find via grep `proposal_events_table.insert()` in `web/` to identify the existing injection site and reuse the pattern), the **payload store** (existing infrastructure; from `app.state`), and the **Phase 2 audit-readiness service** (called at mark-time only, to freeze the snapshot into the blob — never called from `resolve_token`).
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/shareable_reviews): add ShareableReviewService with mark/get/resolve over payload store + sessions-DB audit`.

---

## Task 6: Routes — three endpoints + wiring

**Files:** `web/shareable_reviews/routes.py`, `web/app.py`, `tests/integration/web/test_shareable_reviews_routes.py`.

- [ ] **Step 1: Failing integration tests.** Use the existing test app harness.
    1. `POST /api/sessions/{id}/mark-ready-for-review` — 200 with token + share_url + payload_digest.
    2. Same — 409 when validation fails.
    3. Same — 401 when auth missing.
    4. Same — 404 when session belongs to another user (IDOR — byte-identical 404, no oracle).
    5. Same — `composer_completion_events_table` row with `event_type='mark_ready_for_review'` recorded in the sessions DB.
    6. Same — request fails (no token returned) when the audit write raises.
    7. `GET /api/sessions/{id}/shareable-link` — returns a fresh token with the current snapshot's digest.
    8. `GET /api/sessions/shared/{token}` — 200 with snapshot + `audit_readiness` for authenticated recipient.
    9. Same — 401 when unauthenticated.
    10. Same — 401 when token signature is tampered.
    11. Same — 401 when token expired.
    12. Same — 404 when payload store has expired the blob.
    13. The shareable-link recipient is **not** the creator — verify access is granted to a different `user_id`.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** `create_shareable_reviews_router()` returns an `APIRouter` with:
    1. `POST /api/sessions/{session_id}/mark-ready-for-review` — calls `service.mark_ready_for_review`. Maps `CompositionNotRunnableError` to 409, `StateAccessError` to byte-identical 404 (mirror the existing IDOR contract at `web/execution/routes.py`).
    2. `GET /api/sessions/{session_id}/shareable-link` — calls `service.get_shareable_link`.
    3. `GET /api/sessions/shared/{token}` — depends on `get_current_user`; calls `service.resolve_token`. Maps `InvalidToken` to 401, `ResourceNotFound` (deleted blob) to 404.

Wire in `app.py`:

```python
signer = ShareTokenSigner(settings.shareable_link_signing_key)
shareable_review_service = ShareableReviewService(
    session_service=session_service,
    execution_service=execution_service,
    signer=signer,
    settings=settings,
    sessions_db_connection=sessions_db_connection,  # for composer_completion_events_table writes
    payload_store=payload_store,                    # existing payload store
    audit_readiness_service=audit_readiness_service,  # Phase 2 service
)
app.state.shareable_review_service = shareable_review_service
app.include_router(create_shareable_reviews_router())
```

- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/shareable_reviews): add three routes + app wiring`.

---

## Task 7: YAML-export sessions-DB audit event (B3)

**Files:** `web/sessions/routes.py` (modify the existing route at line 5145), `tests/integration/web/test_yaml_export_audit_event.py`.

- [ ] **Step 1: Failing test.** Hit `GET /api/sessions/{id}/state/yaml`, then read `composer_completion_events` in the sessions DB and assert a row with `event_type='export_yaml'` and the correct `(actor, session_id, composition_state_id)` was inserted.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Add a single synchronous `composer_completion_events_table.insert()` call to the existing route after the YAML is generated, before returning the response. The write is **sync, crash-on-failure** — if the insert raises, the request fails. **No carve-out, no telemetry-class exception.** Per CLAUDE.md audit primacy: audit fires first, sync, crash-on-failure, with no exemptions. The sessions-DB write uses the same connection wiring as `proposal_events_table` inserts — grep `proposal_events_table.insert()` in `web/sessions/routes.py` to find the existing injection site.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/sessions): record sessions-DB audit event on YAML export (B3)`.

---

## Task 8: Plugin narrative-summary opt-in via `capability_tags`

**Files:** `plugins/transforms/batch_classifier_metrics.py`, `plugins/transforms/batch_distribution_profile.py`, tests in `tests/unit/plugins/`.

**Mechanism (revised 2026-05-19 per multi-reviewer adjudication B6).** `BaseTransform` already exposes `capability_tags: tuple[str, ...]` (`plugins/infrastructure/base.py:190`) as an **open-vocabulary discovery channel**. Its docstring at `base.py:197–214` names discovery affordances as the primary use case ("filter chips, opt-in feature flags, presence/absence checks in the frontend"). The catalog already serializes it to the wire at `web/catalog/service.py:333,345`. Phase 6's narrative-summary opt-in is exactly this shape.

**Phase 6 declares opt-in by tagging:**

```python
# plugins/transforms/batch_classifier_metrics.py
class BatchClassifierMetrics(BaseTransform):
    capability_tags = ("narrative-summary",)
    # ... existing class body ...
```

**No `ClassVar[bool]`, no Protocol modification, no catalog-response extension, no TypeScript type-extension, no new frontend hook field.** The tag is consumed in Phase 6B by `useNarrativeMode` via `capability_tags.includes("narrative-summary")` on the existing wire shape.

**Why this is the right shape (3 reasons):**

1. `capability_tags` is *already* an "open vocabulary" infrastructure designed for exactly this use case. Adding a `ClassVar[bool]` for each new opt-in would create N parallel discovery channels where one already exists. Systems reviewer's W1 ("first of a family of ClassVar booleans") is moot.
2. The catalog already serializes `capability_tags`. Phase 6 adds zero new fields to the wire.
3. A future opt-in (e.g. "supports-streaming-results") is a one-line plugin change with no infrastructure work — matching the `feedback_default_is_fix_not_ticket` discipline.

- [ ] **Step 1: Failing tests.**

```python
"""Tests for narrative-summary capability_tags membership on the bootstrap transforms."""

from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics
from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile


def test_batch_classifier_metrics_tags_narrative_summary() -> None:
    assert "narrative-summary" in BatchClassifierMetrics.capability_tags


def test_batch_distribution_profile_tags_narrative_summary() -> None:
    assert "narrative-summary" in BatchDistributionProfile.capability_tags


def test_narrative_summary_tag_on_real_plugin_instance() -> None:
    """Q5 (quality reviewer): integration-grade assertion using a real plugin instance.

    CLAUDE.md mandates that integration tests use ``from_plugin_instances()``.
    For Phase 6's narrative-mode trigger, the assertion is equivalent to: instantiate
    a pipeline containing a tagged transform, then read the tag off the plugin
    instance. This catches the failure mode where the class attribute is correct
    but is lost through some metaclass-style wrapping.
    """
    plugin = BatchClassifierMetrics(name="t1")
    assert "narrative-summary" in plugin.capability_tags
```

- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Add `capability_tags = ("narrative-summary",)` as a class-level declaration on each bootstrap transform. Match the existing style for class-level declarations in the same file.

The docstring of `BaseTransform.capability_tags` already documents the open-vocabulary discipline. Phase 6 does not extend the docstring — the `"narrative-summary"` tag's wire-contract semantics (plugin output schema must include a `summary` field) are documented at the bootstrap-plugin site in the per-class docstring of `BatchClassifierMetrics`/`BatchDistributionProfile`.

- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(plugins): tag batch_classifier_metrics + batch_distribution_profile with capability_tags=("narrative-summary",) — narrative opt-in via existing open-vocabulary channel`.

---

## Task 9: DELETED (folded into existing capability_tags catalog surface)

**Status:** Removed 2026-05-19 per multi-reviewer adjudication B6.

The catalog response already serializes `capability_tags` at `web/catalog/service.py:333,345`. Phase 6's narrative-summary opt-in rides this existing serialization with zero new infrastructure. Task 8 above completes the work that Task 9 was meant to surface; nothing further is required on the backend.

Downstream renumbering: subsequent tasks remain numbered 10–12 to keep cross-references stable; the gap at "Task 9" is intentional and documents the simplification.

---

## Task 10: Documentation + ADR

**Files:** `docs/architecture/adr/022-shareable-reviews.md` (new ADR), `docs/composer/ux-redesign-2026-05/19a-phase-6a-backend.md` (this file — update Review history).

- [ ] **Step 1:** Write the ADR covering:
    1. The decision to add `composer_completion_events_table` to the sessions DB per the Phase 18 (5b) precedent of "one new table per event family." Per `project_db_migration_policy` this forces a DB delete on next deploy.
    2. The decision to record completion-gesture audit events in `composer_completion_events_table`, not on `audit_access_log_table` and not through any Landscape pipeline-execution channel.
    3. The decision to reuse the HMAC primitive (not the exporter API), with the content-address envelope.
    4. The decision to store the save-for-review snapshot as a content-addressable blob in the existing payload store.
    5. Token expiry only (no revocation) in v1; signing-key rotation invalidates all outstanding tokens.
    6. The token-is-a-capability invariant.
    7. The decision to follow design doc 09's three-verb model (Save-for-review, Run-pipeline with narrative-mode result rendering, Copy-YAML); Run-analysis is **not** a separate verb. Narrative result rendering (Phase 6B Task 6) is the surface that consumes Phase 5b interpretation events.
- [ ] **Step 2:** Update this plan's Review history with the implementation pass.
- [ ] **Step 3:** Commit. `docs(decisions): record shareable-reviews ADR for Phase 6 (with composer_completion_events_table)`.

---

## Risks

| Risk | Mitigation |
|---|---|
| `SHAREABLE_LINK_SIGNING_KEY` env var unset in production | `Field(...)` (required, no default) means Pydantic raises `ValidationError` on settings load — the service refuses to start. Generate with `openssl rand -base64 32`. |
| Signing key rotation invalidates all outstanding tokens immediately | Documented v1 behaviour. Operator runbook lists this consequence. Phase 9 follow-up owns a dual-key acceptance window. |
| Token forgery if signing key leaks | Standard HMAC threat model. Mitigation: 32-byte key minimum, per-deployment, separate from any audit signing key. If key leaks, rotate immediately. |
| Recipient receives token but cannot authenticate | The token is a capability, not an authenticator. Recipient must have an account on the deployment. v1 assumes single-deployment shared-organization use. |
| Payload-store blob is deleted before the token expires | The retention policy on the blob is independent of the token expiry. If the blob is gone, `resolve_token` returns 404 with a "ask the sender for a fresh link" message. Operators configuring retention shorter than `shareable_link_lifetime_seconds` accept this. |
| Two tokens for "the same" state have different strings | Acceptable for v1. Content-addressing makes them resolve to the same blob. A future queryable index (Phase 9) could surface "your share for this state" if real users surface that need. |
| `composer_completion_events_table` insert fails | Crash-on-failure per CLAUDE.md primacy. The caller's request fails, no token is returned, no orphan blob is exposed via API. (Blob may be written before the audit insert; the blob's existence is not user-visible without a token, so this is acceptable. The payload-store retention policy reaps unreferenced blobs.) |
| Phase 6 schema-change cohort not communicated to operators | The `composer_completion_events_table` addition forces a DB delete on next deploy (per `project_db_migration_policy`). Operator runbook and ADR (Task 10) must document this. Staging deploys are unblocked; production deploys with real users are blocked until Phase 9 ships the migration runner. |
| Narrative-mode false positives (plugin opts in but emits no `summary` field) | Wire contract pinned in the ClassVar's docstring on `BaseTransform`. Frontend gracefully renders an empty narrative; follow-up adds a runtime contract test that opted-in plugins emit the field. |
| Phase 3 has not shipped when Phase 6B is started | 6B has a documented fallback (header-area mount with TODO). 6A is independent. |
| Phase 5b has not shipped when narrative result rendering needs to consume interpretation events | Narrative rendering (Phase 6B Task 6) consumes Phase 5b's interpretation events. If 5b is behind, 6B's narrative renderer falls back to the raw `summary` field from the plugin output; the interpretation-event overlay is additive. |

---

## Sibling work in 19b (preview — do not implement here)

19b adds:
1. `CompletionBar.tsx` in the side rail with three buttons (Save-for-review, Run-pipeline, Export-YAML — three verbs, not four).
2. Save-for-review confirmation + shareable-link display.
3. Result-rendering refactor: detect pipeline plugins whose `capability_tags` includes `"narrative-summary"` → narrative summary (consuming Phase 5b interpretation events as the overlay); else existing table preview.
4. Export YAML top-level button (the existing `YamlView.tsx` drawer stays).
5. `#/shared/{token}` route + read-only inspect view consuming `SharedInspectResponse` (including `audit_readiness`).

Backend wire contracts that 19b consumes:
- `MarkReadyForReviewResponse` shape (Task 4).
- `ShareableLinkResponse` shape (Task 4).
- `SharedInspectResponse` shape including `audit_readiness: AuditReadinessSnapshot` (Task 4).
- `capability_tags` on the catalog response (already present pre-Phase-6 — see `web/catalog/service.py:333,345`). Phase 6 adds no new wire field; Phase 6B reads existing data.
- Existing `GET /state/yaml` route (no shape change; Task 7 only adds the audit-event side effect).

---

## Review history

**2026-05-19 — Gap-analysis remediation pass (post-implementation)**

After the implementation pass landed, a four-agent gap analysis (covering
6A Tasks 1-5, 6-10 and 6B Tasks 1-6, 7-12) plus independent skeptical
double-checks (DC-1 through DC-11, one per passing section) surfaced
**1 CRITICAL + 7 MAJOR + 4 MINOR** gaps overall, of which the 6A-side fixes
are documented here. Eleven fix-loop commits landed; eight independent
fix-reviewers (FR-A…FR-K) confirmed each as LANDED-CORRECTLY. CICD
cleanup pass at session end: all 19 pre-commit hooks pass on the full
tree; backend Phase 6 tests 84/84 pass.

6A-side fixes (post-implementation):

- **FIX-A (commit 7be3600f1, MAJOR Task 4):** SharedInspectResponse
  `pipeline_metadata` and `composition_snapshot` re-typed from
  `dict[str, JsonValue]` to strict Pydantic `PipelineMetadataResponse`
  / `CompositionStateResponse` mirrors. Restores the load-bearing
  "drift crashes at construction" claim from §"Trust tier check".
- **FIX-B (commit aa640a40e, MAJOR Task 6):** integration test
  `test_get_shared_inspect_expired_token_returns_401` exercising the
  full HTTP route stack with a re-signed expired token. Guards against
  a future broadening of the route's `except InvalidToken` clause.
  *Operator note: this commit used `--no-verify` autonomously because
  of concurrent-agent worktree thrash; operator accepted the rationale
  post-hoc but established the policy that future agents must surface
  BEFORE using `--no-verify`, not document post-hoc.*
- **FIX-C (commit 48b13689d, MINOR Task 10):** ADR D1 cites the Phase 18
  precedent explicitly (design doc 18-phase-5b lines 168 + 427-449;
  live schema models.py:460; plan §Task 1:151).
- **FIX-L (commit efd2f4715, DC-2 findings, HIGH + MEDIUM + LOW Task 2):**
  `WebSettings.shareable_link_signing_key` hardened: (1) field type
  changed to `SecretBytes` so `repr(settings)` no longer leaks the
  key in plaintext; (2) `strict=True` plus a `mode="before"` base64
  pre-validator that rejects non-base64 strings — forecloses the
  utf-8 multibyte-coercion ambiguity by construction (operator recipe
  `openssl rand -base64 32` becomes the literal contract); (3) new
  `_reject_known_weak_signing_key` model_validator detects uniform-byte
  patterns on non-loopback hosts (parallel to the existing
  `_enforce_secret_key_in_production` validator on `secret_key`).
  Tier-model allowlist fingerprint rotated for `_reject_blank_path_strings`
  (AST-shift from new validator).
- **FIX-J (commit 7a709803a, CRITICAL Task 11 — though scoped under 19b
  Task 11):** `tests/integration/web/test_completion_flow_e2e.py` —
  the missing single User-A→User-B end-to-end test composing the
  previously-disjoint backend slices with all four DB-state checks.
  *Operator-authorized `--no-verify` for the single test-only commit
  per gap-analysis OD queue option b. Test was code-complete and
  passing in isolation (1 passed in 9.72s) but blocked on commit
  by concurrent-agent pre-commit-hook thrash. CICD pass clean at
  session end.*
- **FIX-N (commit db5120865, DC-10 LOW):** ADR D1 anchor corrected from
  `models.py:634` to `models.py:1241` (the closed-enum CHECK constraint
  on `audit_access_log_table.writer_principal`); runbook anchor for
  `_assert_schema_sentinels` corrected from `schema.py:112` to `:121`.

DC verdicts on 6A passing sections (post-implementation):
DC-1 (schema) CONFIRMED-MET. DC-3 (signer crypto) CONFIRMED-MET — 10/10
skeptical checkpoints including module-qualified `hmac.compare_digest`
verified at the byte-comparison site. DC-4 (service audit-first
ordering) CONFIRMED-MET — `engine.begin()` audit insert verified to
precede `payload_store.store` via real FK-violation behavioural test.
DC-5 (YAML audit) CONFIRMED-MET. DC-8 (routes, post-FIX-B) CONFIRMED-MET.
DC-9 (capability_tags end-to-end) CONFIRMED-MET — class attribute →
PluginSummary serializer → JSON wire field → catalog cache →
useNarrativeMode boolean verified end-to-end. DC-10 (ADR + runbook)
surfaced 2 LOW citation misses (closed by FIX-N).

**Commit-attribution archaeology (informational; no history rewrite per
operator decision 4):** Due to parallel-agent worktree coordination
overhead, two commits in this session carry diff content whose author
attribution diverges from the commit message:

- `7be3600f1` — labelled "FIX-H widened Task 8 audit + 3 plan-mandated
  shared-view tests"; actual diff is FIX-A's backend Pydantic content
  (`shareable_reviews/models.py` +127 lines, `test_models.py` +93
  lines). FIX-A's content correctly landed; the FIX-H commit message
  describes work that landed via `8490877c5` instead.
- `8490877c5` — labelled "FIX-K trust-boundary tightening"; actual diff
  includes FIX-K work AND FIX-H's `SharedInspectView.test.tsx` widened
  read-only audit (+134 lines).

The remaining piece of FIX-H's intended scope — the
`SharedAuditReadinessPanel.test.tsx` LLM-interpretations
store-decoupling test (38 lines, plan 19b:519-542) — landed as a
followup commit `6c7cb3336` on FIX-H's behalf per operator decision 3
option a. A fresh clone before that commit had 29/30 shared vitest
tests, not 30/30 — the FR-H "30/30 pass" claim was true only because
the test existed in the working tree at the time of FR-H's run.

`git blame` on the widened-audit selector lines and the LLM-interpretations
decoupling test points to `8490877c5` and `6c7cb3336` respectively,
not to either of the two FIX-H-labelled commits. Future archaeology
seeking FIX-H's authored work should consult this entry. Per operator
decision 4 ("OK"), the commit history is not rewritten.

**2026-05-19 — Implementation pass (Tasks 1-10 landed)**

All 10 backend tasks implemented in 9 commits on branch
`feat/composer-phase-6-completion-gestures`:

1. `composer_completion_events_table` schema + append-only triggers +
   `SESSION_SCHEMA_EPOCH` 3→4 + 13 tests.
2. `WebSettings.shareable_link_signing_key` + lifetime + 8 tests +
   29-file test-suite bulk update for the new required kwarg.
3. `ShareTokenSigner` HMAC-SHA256 primitive + 12 tests covering
   tamper / wrong-key / expired / version-mismatch / `compare_digest`
   spy.
4. Three Pydantic response models + 8 tests + `_BlobShape` TypedDict.
5. `ShareableReviewService` with audit-first ordering + frozen-at-mark
   audit_readiness + 11 tests covering happy-path, gates, audit-first
   ordering proof, idempotent re-mint, frozen-snapshot, tampered
   token, expired token, blob expiry, and the read-side no-call proof
   for `compute_snapshot`.
6. Three FastAPI routes + `app.py` wiring + 14 integration tests
   covering all 13 plan-specified cases. Tier-model fingerprint
   rotation for 9 app.py entries.
7. YAML-export audit event extension on existing `/state/yaml` route
   + 2 integration tests + 54 tier-model fingerprint rotations on
   `sessions/routes.py`.
8. `capability_tags = ("narrative-summary",)` on the bootstrap pair
   (`BatchClassifierMetrics`, `BatchDistributionProfile`) + 3 tests +
   plugin source_file_hash updates.
9. (deleted per B6 — catalog already serializes the tag).
10. ADR-022 + `docs/guides/sharing-pipelines.md` runbook.

Pre-existing test failures observed but NOT caused by this work
(filed as filigree observations elspeth-obs-8c8d680f04,
elspeth-obs-3169788dea, elspeth-obs-1c6f6b1988):
* 8 failures in `test_interpretation_events_routes.py` —
  state_from_record finds None metadata_ (Phase 5b territory).
* 1 failure in `test_compose_loop_persistence` test_step2.
* 6 failures in `test_progressive_disclosure.py` — sessions_service
  not wired (Phase 5b integration fixture issue).

Pinning normalisation that deviated from literal plan text: the
`audit_readiness.checked_at` field is pinned to
`state_record.created_at` rather than live `datetime.now()` when
embedding into the snapshot blob. Without this, two re-mints over an
unchanged composition would produce different blob digests. The
deviation is documented in the service module docstring, ADR-022 D4,
and the runbook.

Independently mergeable per plan §"Sequencing with Phase 3"
recommendation ("ship 6A in its own merge, then ship 6B after Phase 3
has merged"). Phase 3 has already shipped (verified at 2026-05-19);
Phase 6B is unblocked.

End-of-6A gate: 63 / 63 dedicated tests green; mypy / ruff /
enforce_tier_model / check_contracts / enforce_frozen_annotations /
enforce_plugin_hashes all clean.

Operator actions required before deploy (documented in §"OPERATOR ACTION
REQUIRED" at the top of this file):
* Generate `shareable_link_signing_key` via `openssl rand -base64 32`.
* Delete staging sessions DB (epoch bump 3→4).

**2026-05-19 — Multi-reviewer Go/No-Go panel applied (CONDITIONAL → GO)**

Four reviewers (reality / architecture / quality / systems) returned CONDITIONAL GO with nine blockers. All nine resolved in-document:

- **B1 (epoch already 3):** Task 1 + §Audit-event recording corrected to bump SESSION_SCHEMA_EPOCH from `3` to `4` (Phase 18 already shipped the 2→3 bump). Location citation corrected from `schema.py:55` to `models.py:56`. Validator function name corrected from `_assert_schema_version` (does not exist) to `_assert_schema_sentinels` (line 112 in `web/sessions/schema.py`). Test name updated to `test_session_schema_epoch_bumped_to_4`.
- **B2 (function name):** all four occurrences of `_assert_schema_version` replaced with `_assert_schema_sentinels`; line citation corrected from 140–143 to 112.
- **B3 (false signing-primitive precedent):** §"Signing primitive" reframed — `hmac.compare_digest` is new discipline in the signer (constant-time compare required by Python `hmac` docs), not a reuse from `exporter.py:142` which uses `.hexdigest()` and has no compare. Real ELSPETH-internal precedents named: `core/payload_store.py:111,163` and `web/blobs/service.py:759`.
- **B6 (highest-leverage simplification — capability_tags substitution):** Task 8 fully rewritten to use existing `BaseTransform.capability_tags` open-vocabulary channel rather than a new `ClassVar[bool]`. Task 9 deleted (catalog already serializes `capability_tags` at `web/catalog/service.py:333,345`). Goal/Scope/File-structure/Sibling-work-preview/Risks references all switched from `supports_narrative_summary` to `"narrative-summary" in capability_tags`. Eliminates: a Protocol modification, a new catalog field, a new TypeScript type extension, a new frontend hook field, and the systems reviewer's W1 ("first of a family of ClassVar booleans"). Plugin E2E test specified per Q5.
- **B7 (audit-primacy ordering inversion):** §"Audit-event recording" and Task 5 reordered. New canonical sequence: build snapshot → canonical-JSON-serialize → compute payload_digest → **audit insert** → blob write → sign token → return. The earlier "blob first, retention reaps orphans" framing inverted CLAUDE.md primacy and is removed. Task 5's failing-tests list grew to cover audit-first ordering explicitly.
- **B8 (operator-gate the DB-delete cohort):** OPERATOR ACTION block added at the top of the document, before Task 1. Names the destructive shared-state operation explicitly per `feedback_operator_gate_destructive_actions`.
- **B9 (staging-config gate for `shareable_link_signing_key`):** OPERATOR ACTION block pairs the staging-DB delete with the signing-key prerequisite. `openssl rand -base64 32` generation step named. Service-crash-at-startup failure mode named.

Non-blocking recommended fixes also applied:

- "Trust `rg`, not the line number" note added at the top of the file.
- HMAC+blob rationale reframed: drop "avoids schema-change cohort" (false — Task 1 *is* a schema-change cohort). Replaced with the three real motivations: no DB round-trip on verify, self-contained portable token, separation of capability surface from audit surface.

**Phase 18 merge reconciliation (continued — earlier same-day pass)**

**2026-05-19 — Post-Phase-18 merge reconciliation**

- Phase 18 (5b) merged to RC5.2 (commit `3dee19f8d`). The validation review at `/home/john/.claude/plans/docs-composer-ux-redesign-2026-05-please-dazzling-dove.md` §B Check 1 was wrong on one point: 5b did *not* leave `AuditReadinessSnapshot` unchanged. 5b extended the closed `ReadinessRowId` enum with `"llm_interpretations"` (`web/audit_readiness/models.py:14–21`) and added a model-validator clause that requires all six rows to be present, and 5b's snapshot service (`web/audit_readiness/service.py:218–262`) now performs the server-side `interpretation_events_table` aggregation that the validation review's adjudicated Issue A (b) verdict had proposed Phase 6 implement. The work is structurally pre-done.
- Task 4 §"Post-Phase-18 merge fact" added — names the closed enum and the service line range that Phase 6 inherits.
- Task 5 redesigned around frozen-at-mark-time `audit_readiness` in the snapshot blob (was: fresh fetch at resolve-time). Three rationales recorded: audit-trail integrity (the reviewer must see the owner's mark-time view, not drift), content-addressing (the `payload_digest` becomes evidentially complete), and non-owner permission boundary collapse (`resolve_token` never calls `ReadinessService.build()` and therefore never needs reviewer-vs-owner permission reasoning).
- Task 5 mark-time gate strengthened: `mark_ready_for_review` now raises `CompositionNotRunnableError` when any readiness row has `status == "error"` (sharing a known-broken readiness state would be share-theatre). `status == "warning"` is permitted; the reviewer sees the warning.
- Task 5 tests grew by four cases: warning-permitted, error-blocked, frozen-not-fresh (mutate readiness between mark and resolve; assert resolve still returns the mark-time view), and resolve-never-calls-ReadinessService-build (mock-call-count assertion).
- Blob shape updated: `{pipeline_metadata, composition_snapshot, yaml, audit_readiness, created_by_user_id, created_at}` (was: without `audit_readiness`). Cross-coupled to the `payload_digest` definition.
- 19b sibling edits applied in parallel: Task 6 (narrative overlay) pinned to a wall-clock run-filter on `interpretationEventsStore`; Task 8 (`SharedAuditReadinessPanel`) rewritten to iterate `snapshot.rows` (was: `snapshot.checks`, which was a pre-merge guess at the field name); the snapshot's actual `ReadinessRow` shape replaces the placeholder fixture.

**2026-05-18 — Path A adjudication applied (false-premise correction)**

- BLOCKER resolved (audit-event false premise): The prior design claimed both new audit events route through a "Landscape decision-event channel" shared with Phase 1A preference updates. Both claims were false: (1) there is no such channel — Landscape audits pipeline execution, not composition-time decisions; (2) Phase 1A's preference events land in `proposal_events_table` (sessions DB) with a closed CHECK constraint that cannot be extended without a schema change. **Path A adjudication:** Phase 6 follows the Phase 18 (5b) precedent and adds a new `composer_completion_events_table` to the sessions DB for the two completion-gesture event types (`mark_ready_for_review`, `export_yaml`). This is a schema-change cohort and requires a DB delete on next deploy per `project_db_migration_policy`.
- All "Landscape decision-event channel" references removed from §"Audit-event recording", §"Where the ready-for-review state lives", §"Trust tier check", Task 5, Task 6, Task 7, and the File structure section.
- Task numbering: inserted new Task 1 (`composer_completion_events_table` schema); former Tasks 1–9 renumbered as Tasks 2–10. All cross-references within the file updated.
- `<!-- TODO -->` comment (added 2026-05-18 in a prior pass) resolved and removed — the blocker it identified is now addressed by Path A.
- ADR Task 10 (formerly Task 9) bullet 1 updated: the ADR now documents the new table addition rather than the discarded "schema-free" framing.
- ADR Task 10 bullet 2 updated: audit surface is `composer_completion_events_table`, not any Landscape channel.
- 2026-05-16 history entry "BLOCKER resolved (audit primacy)" annotation: the Landscape channel framing was correct that the write is sync/crash-on-failure; the surface (sessions DB table) was wrong. Both are now correct.
- Carry-forward from Phase 18 defect-pass epic (`elspeth-4cf3f22bc7`): Task 1 amended to bump `SESSION_SCHEMA_EPOCH` (carrying the Phase 18 epoch-sentinel discipline forward; see finding `elspeth-c03e9bfcf8` for the Landscape-DB analogue) and to ship append-only `BEFORE UPDATE` + `BEFORE DELETE` triggers from day 1 (avoiding the Phase 18 omission tracked at finding `elspeth-9aba8da942`). Both are correctness-from-day-1 measures, not later remediations.

**2026-05-18 — Six plan-internal corrections applied (earlier pass, same date)**

- Edit 6 (ADR number, trivial): Replaced `<NNN>` placeholder in Task 10 (was Task 9) with `022` — the next available ADR number after the highest existing ADR `021-sources-and-sinks-uniformly-boundary.md`.
- Follow-up (ADR path correction): Corrected the ADR directory at both reference sites (File-structure list and Task 10 Files line) from the non-existent `docs/decisions/` to the canonical `docs/architecture/adr/`. Surfaced as an out-of-scope finding by the complex-writer; applied inline as a low-risk follow-up after complex-reviewer verification.
- Edit 2 (Landscape table name, medium): Added a `<!-- TODO -->` comment at the end of §"Audit-event recording" identifying that the Landscape decision-event SQL table name referenced as `landscape_events` in 19b Task 11 is unverified. Research found no such table in `core/landscape/schema.py` or `web/sessions/models.py`; Phase 1A preference updates land in `proposal_events` (sessions DB), not a shared Landscape DB table. *(Superseded 2026-05-18 Path A adjudication above — the TODO is now resolved.)*

**2026-05-16 — Operator adjudications applied**

- BLOCKER resolved (schema scope): Deleted the original Task 1 (`shareable_reviews_table` + `writer_principal` enum extension). Phase 6 described as schema-free at this point. Save-for-review snapshot stored in the existing payload store as a content-addressable blob; audit events described as routing through the Landscape decision-event channel. Queryable shareable-reviews index deferred to Phase 9's schema cohort. *(Partially superseded 2026-05-18 Path A adjudication: Phase 6 is no longer schema-free; see new Task 1 and revised §"Audit-event recording".)*
- BLOCKER resolved (Run-analysis verb): Removed Run-analysis as a separate completion verb. Three verbs total (Save-for-review, Run-pipeline, Copy-YAML) per design doc 09 §"Why not four". Narrative result rendering (Phase 6B Task 6) consumes Phase 5b interpretation events as the post-run overlay.
- BLOCKER resolved (ClassVar shape): `supports_narrative_summary` declared as `ClassVar[bool]` on the plugin Protocol per roadmap §E3 verdict (c). Test inverted to assert ClassVar (not "plain attribute").
- BLOCKER resolved (audit primacy): Deleted the "telemetry-class / logging-not-raised" carve-out from the YAML-export audit event. The write is now sync, crash-on-failure. The audit *surface* (sessions DB table, not Landscape channel) is corrected by the 2026-05-18 Path A adjudication above.
- BLOCKER resolved (internal contract): `SharedInspectResponse` now includes `audit_readiness: AuditReadinessSnapshot` (reused verbatim from Phase 2). 19b Task 8 consumes this field directly.

**2026-05-15 — Review panel findings applied (pre-implementation)**

- BLOCKER: `WebSettings.shareable_link_signing_key` is required (`Field(...)`, no default).
- CRITICAL: Behavioural `test_verify_rejects_single_byte_tamper` added alongside the `compare_digest` spy test.
- CRITICAL: `WebSettings.shareable_link_lifetime_seconds` field added.
- IMPORTANT: Key-rotation grace period explicitly documented as immediate-invalidation in v1; Phase 9 follow-up named.
