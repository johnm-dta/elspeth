# ADR-022: Shareable Reviews — Completion Gestures, Signed Tokens, and the Composer Completion Events Table

**Date:** 2026-05-19
**Status:** Accepted
**Deciders:** ELSPETH maintainers
**Tags:** ux-redesign-2026-05, phase-6, composer, audit, hmac, capability-tokens

## Context

Phase 6 of the UX redesign 2026-05 (see `docs/composer/ux-redesign-2026-05/09-completion-gestures.md`) introduces three completion gestures the composer surface offers a user once a pipeline is ready:

1. **Save for review** — freeze a snapshot of the composition and mint a signed link the user can share with a reviewer.
2. **Run pipeline** — the existing Execute path; the result-rendering layer branches on a per-transform capability tag to render a narrative summary when available.
3. **Copy YAML** — the existing YAML-export route, extended to record a Tier-1 audit event on every export.

Run-analysis is **not** a fourth verb. Per design doc 09 ("Why three verbs, not two or four"), narrative result rendering is a post-run rendering layer that consumes Phase 5b interpretation events; it is not a separate completion gesture.

This ADR records the architectural decisions made during Phase 6A (backend implementation) — plan at `docs/composer/ux-redesign-2026-05/19a-phase-6a-backend.md`. The frontend (Phase 6B) consumes the wire shapes defined here.

## Decisions

### D1. Add `composer_completion_events_table` to the sessions DB

**Decision:** Phase 6 ships a new SQLite table `composer_completion_events` in the sessions database. Two event types in v1: `mark_ready_for_review` and `export_yaml`. Closed-enum CHECK constraint on `event_type`; nullable `payload_digest` and `expires_at` columns populated only for `mark_ready_for_review`.

**Why a new table, not an extension:**

* `proposal_events_table.event_type` is a governance-locked closed enum (`'proposal.created'`, `'proposal.accepted'`, `'proposal.rejected'`, `'trust_mode.changed'`). Extending it would conflate composer-time proposal decisions with completion-gesture decisions; each event family has its own writer path and its own audit invariants.
* `audit_access_log_table.writer_principal` is also a closed enum (`models.py:1241`). Completion-gesture events are not access-log entries.
* The Landscape audit DB records pipeline *execution* decisions (sources, transforms, sinks, gate routing). Composition-time decisions ("user marked a draft ready") belong to the composer audit domain, not the Landscape.

The pattern is **established** by Phase 18 (5b)'s `interpretation_events_table` — one new table per event family, closed-enum CHECK, nullable optional columns. Phase 6 is the third event family in this pattern.

**Precedent citation.** The "one new table per event family, closed-enum CHECK on `event_type`, append-only via `BEFORE UPDATE` / `BEFORE DELETE` triggers" pattern is established by Phase 18 (5b). The verifiable artifacts are:

* **Design doc:** `docs/composer/ux-redesign-2026-05/18-phase-5b-surface-llm-interpretation.md` line 168 ("A **new `interpretation_events_table`** in `web/sessions/models.py`…") and lines 427–449 (event-type vocabulary, audit-table semantics).
* **Live schema:** `src/elspeth/web/sessions/models.py:460` (the `interpretation_events_table` definition itself, alongside `proposal_events_table` at line 423).
* **Plan reference:** Phase 6A plan `docs/composer/ux-redesign-2026-05/19a-phase-6a-backend.md` §Task 1 (line 151) names this same precedent and the same `models.py` anchor when specifying the `composer_completion_events_table` schema.

Phase 6 follows the precedent with one deliberate sharpening: where `interpretation_events_table` permits DELETE on PENDING rows for orphan recovery, `composer_completion_events_table` is fully append-only — both `BEFORE UPDATE` and `BEFORE DELETE` triggers are unconditional ABORT from day 1 (plan 19a:268, correcting the Phase 18 omission tracked under filigree `elspeth-9aba8da942`).

**Consequences:**

* This is a schema-change cohort. Per `project_db_migration_policy`, sessions DBs at any earlier `SESSION_SCHEMA_EPOCH` must be deleted on next deploy. The validator (`web/sessions/schema.py:_assert_schema_sentinels`) enforces this — the service refuses to start against a stale DB.
* `SESSION_SCHEMA_EPOCH` bumps from `3` to `4`. Operator runbook (Task 12 / `docs/guides/sharing-pipelines.md`) documents the DB-delete requirement.

### D2. Audit-first ordering, no telemetry-class carve-out

**Decision:** Both new audit events are recorded **synchronously, crash-on-failure** before the request returns. For `mark_ready_for_review` specifically, the sequence is:

1. Build the snapshot dict in memory.
2. Canonical-JSON-serialize and compute `payload_digest = sha256(bytes)`.
3. **INSERT the `composer_completion_events` row** (sync, crash-on-failure).
4. Write the blob to the payload store.
5. Sign the token.
6. Return the response.

**Why audit precedes blob:**

* If the audit insert fails, no blob is ever written. The user sees an error; no token is returned.
* If the blob write fails after the audit insert, the audit row stands as honest evidence of the attempt; no token is returned. A future reviewer who could somehow synthesize a token for the recorded `payload_digest` would receive `ResourceNotFound` from the payload store — a clean failure mode.
* The alternative ("blob first, audit second, retention reaps orphans") inverts CLAUDE.md's audit primacy and is explicitly forbidden. The blob's existence is not user-visible without a token, so a never-audited blob is operationally invisible — but a blob without an audit row is exactly the "evidence-without-record" anti-pattern the primacy rule exists to prevent.

For `export_yaml`, there is no blob; the audit row is the entire side effect. The discipline is the same: sync write before response.

### D3. Reuse the HMAC primitive, not the exporter API

**Decision:** The signing primitive is `hmac.new(key, msg, hashlib.sha256)` — the same primitive that backs `core/landscape/exporter.py::_sign_record`. The signing **API** is a new class (`ShareTokenSigner` in `web/shareable_reviews/signer.py`) — the exporter signs dict-keyed audit records by inserting a `signature` key alongside the data, while the shareable-link token is a URL-safe self-contained envelope with a length-prefixed wire format.

**Why a new API, not exporter reuse:**

* Payload shapes differ. The exporter signs records that get inserted into the L1 audit DB; the shareable-link token is a capability that travels in a URL.
* Threat models differ. The exporter signs records for **write integrity** (the record never reaches an attacker-controlled boundary before verification). The shareable-link token is **verified at an attacker-controlled boundary** — every token an arbitrary user submits to `GET /api/sessions/shared/{token}` goes through `verify`. This requires constant-time signature comparison via `hmac.compare_digest`, which the exporter does not use (it calls `.hexdigest()` and never verifies).
* Rotation cadences differ. The exporter signing key rotates with the audit DB; the shareable-link key is per-deployment and rotating it invalidates **all** outstanding shareable links (documented v1 behaviour).

The precedent for `hmac.compare_digest` at boundary verifiers is `core/payload_store.py:111,163` and `web/blobs/service.py:759`, both of which compare a caller-supplied content_hash against a stored hash. `ShareTokenSigner.verify` follows that pattern.

### D4. Store the save-for-review snapshot in the existing payload store

**Decision:** The frozen snapshot is a content-addressable blob in the existing `FilesystemPayloadStore`. The `payload_digest` (sha256 hex of the canonical-JSON bytes) is the blob's address; it also rides in the signed token envelope so tamper detection covers both the token AND the blob.

**Why payload store, not a new table:**

* The payload store already provides content-addressable, integrity-verified blob storage with a retention policy. Building a parallel storage mechanism would duplicate infrastructure.
* The blob's lifetime is bounded by the retention policy, not by the token's expiry. If retention is shorter than `shareable_link_lifetime_seconds`, the token may verify but the blob may be gone — the route returns 404 with a "ask the sender for a fresh link" message. Operators choosing retention < lifetime accept this.
* Content-addressing dedupes identical snapshots automatically. Two re-mints over an unchanged composition produce the same `payload_digest` and reuse the same blob.

**Frozen-at-mark-time discipline.** The snapshot blob carries the entire `AuditReadinessSnapshot` from the moment of marking. `resolve_token` reads `audit_readiness` directly from the blob; it never re-calls `ReadinessService.compute_snapshot`. Three reasons:

* **Audit-trail integrity.** A reviewer must see exactly what the owner saw at mark-time. A fresh readiness fetch could drift if validation state changes between mark and resolve.
* **Content-addressing completeness.** Freezing readiness into the blob means the `payload_digest` fingerprint covers the readiness panel too; `composer_completion_events_table.payload_digest` becomes an evidentially complete reference.
* **Permission-boundary collapse.** `resolve_token` never calls the readiness service, so the reviewer-vs-owner permission question never arises at resolve time.

### D4a. Pin `audit_readiness.checked_at` to `state_record.created_at`

**Decision:** Inside `_build_snapshot` (`web/shareable_reviews/service.py:215`), the `audit_readiness.checked_at` field is **normalised** before the blob is canonical-JSON-serialised: it is overwritten with `state_record.created_at.isoformat()` rather than carrying the live `datetime.now()` that `ReadinessService.compute_snapshot` stamps on each call.

**Why this pin is load-bearing for the content-addressing claim in D4:**

* `ReadinessService.compute_snapshot` stamps a fresh `datetime.now()` UTC timestamp on every call — by design, because the readiness panel in normal UI flow describes "as of right now."
* Without normalisation, the canonical-JSON bytes that feed `sha256(...)` would include a wall-clock timestamp that changes per call. Two `get_shareable_link` requests over an unchanged composition would produce **different** `payload_digest` values, breaking the content-addressing dedupe claim in D4 and undermining the evidential completeness claim in the Frozen-at-mark-time discipline note.
* `state_record.created_at` is the semantically defensible pin. The readiness panel inside the blob describes "what readiness signal accompanied the state when it was committed," not "when was this readiness query run." That framing matches what a reviewer needs to see: the readiness signal at the moment the composition was marked, frozen alongside the composition itself.

**Why this lives in the service, not in `ReadinessService`:** Bare `compute_snapshot` callers (the live UI panel, audit-readiness CLI) want the wall-clock timestamp. Only the share-mint path needs the pin. Pushing the normalisation up into `ReadinessService` would conflate "what time is it now" with "what time was the state committed," which is exactly the question this pin answers.

**Consequence — what an auditor sees:** `composer_completion_events_table.payload_digest` is a stable fingerprint of the (composition + readiness) pair. Re-minting the same composition twice yields the same digest, the same payload-store blob (content-addressed dedupe), and the same audit row payload_digest. A divergence between two digests over what should be the same composition is evidence of a real change.

### D5. Token expiry only, no revocation in v1

**Decision:** Tokens carry an `expires_at` field that the signer verifies. There is no per-token row in any database, no revoke endpoint, and no "invalidate all my tokens" affordance. Rotating the signing key invalidates **every** outstanding token immediately.

**Why no revocation in v1:**

* Per-token revocation would require a queryable index of outstanding tokens — that index is a schema change deferred to Phase 9 with the rest of the queryable shareable-reviews surface.
* Signing-key rotation is the v1 emergency mechanism if a key is leaked. The operator runbook (`docs/guides/sharing-pipelines.md`) documents this.
* A future dual-key acceptance window (graceful key rotation) is a Phase 9 follow-up.

### D6. Token is a capability, NOT an authenticator

**Decision:** `GET /api/sessions/shared/{token}` requires authentication (`Depends(get_current_user)`). The token authorizes a *specific authenticated user* to read a session they don't own — it does not log them in.

**Why explicit authentication:**

* A capability without authentication is a public URL. Anyone who learns the URL gets the data.
* v1 assumes a single-deployment shared-organization use case: the recipient must have an account on the deployment.
* The route's docstring explicitly forbids "simplifying" the auth dependency away. A future maintainer who removes the dependency converts every shareable link into a public URL.

### D7. Three-verb completion model

**Decision:** Save-for-review, Run-pipeline (existing Execute path), and Copy-YAML are the three completion verbs. Run-analysis is **not** a fourth verb; narrative result rendering (Phase 6B Task 6) is the post-run rendering layer that consumes Phase 5b's `interpretation_events_table`.

**Why three, not four:**

* Per design doc 09 §"Why three verbs, not two or four": a fourth verb fragments the user's "I'm done composing" decision into two: "I want to run this" vs. "I want to analyse this." In practice the user wants both, and the rendering layer is the natural place to surface the narrative.
* Narrative rendering depends on per-plugin opt-in via `capability_tags`. Phase 6A tags `BatchClassifierMetrics` and `BatchDistributionProfile` as the bootstrap pair; future transforms add the tag to opt in.

### D8. Open-vocabulary `capability_tags`, not a closed ClassVar boolean

**Decision:** The narrative-summary opt-in rides the existing open-vocabulary `BaseTransform.capability_tags` channel (`plugins/infrastructure/base.py:190`). The catalog already serializes the tag list to the wire (`web/catalog/service.py:333,345`). The frontend reads `capability_tags.includes("narrative-summary")` on the existing catalog response shape.

**Why this is the right shape:**

* `capability_tags` is *already* an open-vocabulary infrastructure designed for exactly this use case. Adding a `ClassVar[bool]` for each new opt-in would create N parallel discovery channels where one already exists.
* The catalog already serializes `capability_tags`. Phase 6 adds zero new fields to the wire — no Protocol modification, no TypeScript type extension, no new frontend hook field.
* A future opt-in (e.g. "supports-streaming-results") is a one-line plugin change with no infrastructure work.

## Consequences

### What this commits us to

* The two-event-type `composer_completion_events_table` is now the canonical home for composer completion-gesture audit events. Adding a third event type (e.g. a hypothetical "save draft" gesture) requires extending the closed CHECK constraint in a schema-change cohort, OR adding a separate table per the Phase 18 precedent.
* The HMAC-SHA256 + content-address envelope is the v1 signed-token contract. Bumping `ShareTokenPayload.version` is the forward-compat hook; a future v2 envelope would coexist via dual-version verify.
* The payload store is now a Tier-1 dependency of the shareable-reviews path. Operators must provision sufficient retention for the configured `shareable_link_lifetime_seconds`.

### What this defers

* **Queryable shareable-reviews index** — "show me all the pipelines I've shared." Requires a separate table; Phase 9.
* **Per-token revocation** — rotate the signing key to revoke all; Phase 9 for graceful rotation.
* **Multi-user collaborative editing** — out of scope; the inspect view is read-only.
* **Org-level review queues / notification emails / Slack integrations** — out of scope.

### Operator actions on first deploy

Two deploy-time gates documented in `docs/composer/ux-redesign-2026-05/19a-phase-6a-backend.md` §"OPERATOR ACTION REQUIRED":

1. **Delete the staging sessions DB** before deploying — `SESSION_SCHEMA_EPOCH` bumps from 3 to 4 and the validator refuses to start against the old DB. Any composer sessions saved since the Phase 18 deploy are lost; this is acknowledged cost.
2. **Add `web.shareable_link_signing_key`** to staging config (generate with `openssl rand -base64 32`). The field is `Field(...)` (required, no default) — the service refuses to start without it.

The signing key is per-deployment, MUST be ≥32 bytes (HMAC-SHA256's digest size; the natural entropy floor for a tag produced by this hash — *not* the 64-byte block size), and MUST NOT appear in version control. Rotating it invalidates every outstanding shareable link.

## References

* Plan: [docs/composer/ux-redesign-2026-05/19a-phase-6a-backend.md](../../../docs-archive/2026-06-28-docs-cleanout/docs/composer/ux-redesign-2026-05/19a-phase-6a-backend.md)
* Sibling plan: [docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md](../../../docs-archive/2026-06-28-docs-cleanout/docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md)
* Design: [docs/composer/ux-redesign-2026-05/09-completion-gestures.md](../../../docs-archive/2026-06-28-docs-cleanout/docs/composer/ux-redesign-2026-05/09-completion-gestures.md)
* Precedent: [docs/composer/ux-redesign-2026-05/18-phase-5b-surface-llm-interpretation.md](../../../docs-archive/2026-06-28-docs-cleanout/docs/composer/ux-redesign-2026-05/18-phase-5b-surface-llm-interpretation.md) (interpretation_events_table)
* Runbook: [docs/guides/sharing-pipelines.md](../../guides/sharing-pipelines.md)
