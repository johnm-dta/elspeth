# Sharing Pipelines for Review

Operator runbook for the Phase 6 shareable-reviews feature. Covers the
deploy-time configuration, the lifecycle of a shareable link, and the
recovery procedure when a signing key is compromised or rotated.

## When to use this feature

A composer user with a valid composition state can mint a **shareable
link** that grants a different authenticated user read-only access to a
frozen snapshot of the pipeline. The reviewer sees:

* The composition graph (sources, transforms, sinks).
* The rendered YAML.
* The full six-row Audit Readiness panel as it stood at mark-time.
* Who shared the pipeline and when, plus the link's expiry.

The reviewer cannot edit, execute, or fork the shared session. The
inspect view is purely read-only.

## Deployment requirements

### Required: `web.shareable_link_signing_key`

The web service refuses to start without this key configured. Generate
it once per deployment:

```bash
openssl rand -base64 32
```

The output is a 44-character base64 string (32 bytes of entropy). Store
the result in your staging / production configuration under
`web.shareable_link_signing_key` or set the environment variable
`ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY` to the same value.

The key must:

* Be at least 32 bytes (utf-8 encoded). Shorter keys are rejected at
  service startup.
* Not appear in version control, logs, chat transcripts, or screenshots.
* Be different from any audit-signing key the deployment uses — the
  shareable-link key has a different rotation cadence and a different
  blast radius.

**Rotation invalidates every outstanding shareable link.** There is no
dual-key acceptance window in v1; once you rotate, every link a user
previously copied returns 401. Document the planned rotation cadence
with your operations team, and prefer NOT to rotate unless the key has
been leaked.

### Optional: `web.shareable_link_lifetime_seconds`

Default: **30 days** (`2592000`).

This is the absolute expiry stamped onto every newly minted token. The
signer's `verify()` rejects tokens past this point regardless of their
HMAC signature. Operators may lower or raise this to match their review
cadence; values must be `> 0`.

The lifetime is independent of the payload store retention policy. If
your retention is shorter than the lifetime, the token may verify but
the underlying blob may be gone — the route returns 404 with a "ask the
sender for a fresh link" message. Operators choosing
`payload_store_retention_days * 86400 < shareable_link_lifetime_seconds`
accept this trade-off.

## First-deploy operator action

For 0.7.1, shareable-review state is part of the broader web session database
contract. The release expects `SESSION_SCHEMA_EPOCH=28` and
`SQLITE_SCHEMA_EPOCH=27`. When upgrading from an older pre-1.0 build, stop and
uninstall the web service, archive/export evidence when required, recreate the
configured session and Landscape databases, then reinstall and initialize this
ELSPETH version. No SQLite or PostgreSQL predecessor schema is transformed in
place; PostgreSQL recreation remains a schema-owner operation. Deployments
crossing from an older release must account for the historical 0.7.0 boundary
as well.

Use [the staging session DB recreation runbook](../runbooks/staging-session-db-recreation.md)
as the operational source of truth. It covers the matched SQLite sidecars,
Landscape reset, `data/auth.db` preservation, archive handling, and
`settings.secret_key` rotation decision.

The shareable-link signing key is still required. Generate and configure it
before returning the service to users.

## Lifecycle of a shareable link

A shareable link goes through three distinct lifecycle phases.

### Mark-time

The owner clicks **Save for review** in the composer UI. The backend:

1. Runs validation against the current composition state. Validation
   failures return HTTP 409 — the user fixes the errors and tries again.
2. Computes the Audit Readiness snapshot. If any row's `status` is
   `error`, returns 409 — sharing a known-broken state is share-theatre.
3. Freezes the entire snapshot (composition + YAML + readiness panel)
   into a canonical-JSON blob and computes its sha256 content-address.
4. **Inserts an audit row** into `composer_completion_events_table` with
   `event_type='mark_ready_for_review'` (sync, crash-on-failure).
5. **Writes the blob** to the payload store, addressed by the digest.
6. **Signs a token** encoding `(session, state, expiry, digest, owner)`
   with HMAC-SHA256 over the canonical envelope.
7. Returns `{token, share_url, expires_at, payload_digest}` to the UI.

The audit row carries the digest and the expiry. The token carries the
same fields plus a random nonce that distinguishes two tokens minted at
the same instant.

### Re-mint-time (no new audit row)

If the owner clicks **Save for review** again on the same composition
state — for example, because they lost the URL — the backend reads the
current state, rebuilds the snapshot, and mints a fresh token. Because
the content is identical, the `payload_digest` is identical too;
content-addressing in the payload store dedupes the blob automatically.
The token string differs (different nonce + different mint-time), but it
resolves to the same blob.

No new audit row is written on re-mint — only the original mark-time
write is the auditable decision. Re-mints are UI affordances.

The `GET /api/sessions/{session_id}/shareable-link` route is the
re-mint entry point. It re-runs the readiness snapshot, but normalises
the snapshot's `checked_at` to the composition state's `created_at` so
two re-mints on an unchanged composition produce the same digest.

### Resolve-time

The reviewer follows the share URL. The frontend opens the SPA at the
hash route `#/shared/{token}` and makes a backend call to
`GET /api/sessions/shared/{token}`. The backend:

1. **Authenticates the reviewer.** The token is a CAPABILITY, not an
   authenticator — the reviewer must be logged in. Anyone with the URL
   but no account on the deployment gets 401.
2. **Verifies the token signature** via constant-time HMAC comparison.
   Any tampering, signature mismatch, malformed envelope, or expired
   `expires_at` returns 401 with a generic "invalid or expired" message
   (the failure mode is deliberately not disclosed).
3. **Reads the blob** from the payload store by `payload_digest`. If
   the blob has been reaped by the retention policy, returns 404 with a
   "ask the sender for a fresh link" message.
4. **Returns the frozen snapshot** including the mark-time readiness
   panel. The reviewer sees exactly what the owner saw at mark-time.

The resolve path does NOT call `ReadinessService.compute_snapshot` —
the audit_readiness field is read directly from the frozen blob. This
guarantees the reviewer sees the owner's mark-time view, not a drifted
fresh-fetch view.

## YAML export audit

Every call to `GET /api/sessions/{session_id}/state/yaml` records an
`event_type='export_yaml'` row in `composer_completion_events_table`.
The row carries the actor (user_id), the composition_state_id, and the
timestamp. `payload_digest` and `expires_at` stay NULL — YAML export
does not mint a token or write a blob; the audit row is the entire side
effect.

Two YAML exports of the same state produce two distinct audit rows
(append-only audit table, no deduplication).

## Recovery procedures

### A signing key has been leaked

**Symptoms:** unauthorized share URLs appearing; an attacker has
unauthenticated access (note: they still need an account on your
deployment to call the inspect route — the token grants *which session*
they can read, not *whether they can read at all*).

**Procedure:**

1. **Generate a new signing key** with `openssl rand -base64 32`.
2. **Update the deployment configuration** with the new value (env var
   or config file).
3. **Restart the web service.** The service will load the new key and
   reject every outstanding token immediately.
4. **Notify users** that previously shared links have been invalidated;
   they must re-mint by clicking **Save for review** again.

There is no per-token revocation in v1 — rotation invalidates all.

### The payload store has reaped a blob a user needs

**Symptoms:** the reviewer's GET returns 404 with "ask the sender for
a fresh link."

**Procedure:**

1. The owner of the original session opens the composer at that
   session.
2. The owner clicks **Save for review** again. This re-mints a fresh
   token (the payload store stores the blob again under the same
   digest).
3. The owner sends the new URL to the reviewer.

If the underlying composition has changed since the original mark, the
new digest will differ from the original. The audit trail records both
events distinctly.

### The sessions DB has been deleted (data loss event)

Outstanding shareable links remain valid as long as:

* The signing key has not rotated.
* The payload store blob still exists.

However, the `composer_completion_events_table` rows are lost. An
operator cannot enumerate "who shared what" after a DB delete. If
auditable provenance matters, **back up the sessions DB before any
operator action that deletes it.**

## Troubleshooting

### Service refuses to start with "shareable_link_signing_key Field required"

The `web.shareable_link_signing_key` field is not set. Generate one
with `openssl rand -base64 32` and add it to the configuration.

### Service refuses to start with "shareable_link_signing_key must be at least 32 bytes"

The configured key is shorter than 32 bytes (utf-8 encoded). Regenerate
with `openssl rand -base64 32` and replace.

### Service refuses to start with a `SESSION_SCHEMA_EPOCH` mismatch

The sessions DB predates the running code. Archive/export evidence when
required, stop and uninstall the deployment, recreate both stale databases,
then reinstall. Writable, read-only, and inspection opens do not migrate any
predecessor Landscape epoch. PostgreSQL recreation requires the schema-owner
path. Do not roll older code over a database initialized by newer code; restore
the matched archive with the old code instead.

### `POST /mark-ready-for-review` returns 409 with "composition validation failed"

The composition has validation errors. Fix them in the composer and try
again.

### `POST /mark-ready-for-review` returns 409 with "readiness panel reports an error"

The audit-readiness panel has at least one row with `status == "error"`
— typically an unresolved validation issue, a missing secret reference,
or a plugin-trust gap. Resolve the error and try again. (Warnings, e.g.
pending LLM interpretations, do NOT block sharing.)

### `GET /sessions/shared/{token}` returns 401 unexpectedly

* The token has expired (`shareable_link_lifetime_seconds` past).
* The signing key has rotated since the token was minted.
* The token was truncated or tampered in transit (URL-decoding issues
  in email clients are a common cause).

Ask the sender to re-mint and resend.

### `GET /sessions/shared/{token}` returns 404

The payload store has reaped the blob. Ask the sender to re-mint.

## Frontend surfaces (post-Phase-6B)

The user-facing entry points landed by Phase 6B:

* **Completion bar** — `components/composer/CompletionBar.tsx`. Mounted
  in the side rail's `completionBarSlot`. Three co-equal verbs: Save
  for review, Run pipeline, Export YAML. The Save-for-review button is
  client-side-disabled when the composition's validation is invalid or
  has not run; the backend would also 409 on submission, but the
  client preview is friendlier.
* **Save-for-review dialog** —
  `components/composer/SaveForReviewDialog.tsx`. Mounted at app-root so
  the verb can open it from any focused view. Three observable states:
  spinner (in-flight), error banner (with Try-again button), or
  success panel (share URL + copy-to-clipboard + Open-in-new-tab link
  + expiry timestamp). The share URL is the absolute form
  `${location.origin}/#/shared/{token}` — the backend returns the
  path-only suffix and the frontend prepends `location.origin`.
* **Shared inspect view** — `components/shared/SharedInspectView.tsx`.
  Mounted at app-root by a top-level branch in `App.tsx` when the URL
  hash matches `#/shared/{token}`. Renders read-only: pipeline
  metadata, the six-row audit-readiness panel served verbatim from the
  frozen blob, and the rendered YAML. The composer chat panel, run
  controls, and edit affordances are deliberately absent.
* **Narrative results panel** —
  `components/composer/NarrativeResults.tsx`, conditionally rendered
  by `InlineRunResults` when `useNarrativeMode().narrativeMode` is
  true. Surfaces the plugin-emitted `summary` field; the tabular
  `RunOutputsPanel` remains visible alongside.

The hash-router extension (`useSharedToken` + `useHashRouter` guards)
ensures the shared-route URL is preserved across the entire shared-view
lifecycle. The session router will not mutate the hash while
`#/shared/*` is live; navigating away (via the "Return to my
workspace" link) returns control to the regular session router.

## References

* ADR: [docs/architecture/adr/022-shareable-reviews.md](../architecture/adr/022-shareable-reviews.md)
* Historical Phase 6A/6B implementation plans are preserved in git history or
  maintainer-local archives.
