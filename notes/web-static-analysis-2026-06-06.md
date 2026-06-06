# `src/elspeth/web` — Static Desktop Security & Correctness Analysis

**Date:** 2026-06-06
**Branch:** release/0.5.3
**Scope:** `src/elspeth/web` (read-only desktop analysis — no code or config changed)
**Method:** Multi-agent fan-out (ultracode) — 24 lens-matched per-file review bundles
covering all 140 substantive source files + 6 cross-cutting taint/authz traces →
per-finding adversarial verification → single-rubric synthesis. Headline findings then
**checked against the live staging deployment and the filigree tracker** (see
"Live-deployment verification" and corrected labels below).
**Run:** `wf_9529cdde-7a2` — 60 agents, 4.58M agent tokens, 1113 tool calls, ~24 min.
**Result:** 29 raw findings → **20 confirmed P0–P2** after verification (0 hard dedups).

> **Two corrections were applied after the agent run, against real data the agents
> could not see:** (1) the rank-1 "P0 live auth bypass" is **NOT live** on the actual
> staging box — `SECRET_KEY` is a real 64-char key and `HOST=0.0.0.0` (so the weak-key
> gate doesn't even bypass); it is a *latent* code defect in a *deliberate, already-
> dispositioned* design area. (2) The synthesis agent's "LIKELY-TRACKED" labels were
> title-matches, not tracker lookups; the corrected NEW-vs-tracked status is in the
> tables. Severities and the "top risk" are revised accordingly.

---

## Bottom line (revised after verification)

**There is no confirmed live P0 on the actual deployment.** The single finding that
looked like a career-ender — the default-JWT-key host-gate — does not fire on staging
(`HOST=0.0.0.0`, non-default key), the loopback exemption is a deliberate design choice
with two **closed** prior bugs in the same code, and exploiting it would require a
future operator to set `HOST=127.0.0.1` behind a TLS proxy *and* leave the key at
default — a triple-misconfiguration, not the current state.

The real residue is a cluster of **NEW, not-yet-filed** Tier-3 egress and audit-provenance
defects (ranks 2, 5, 6, 7) — mostly the user's *own* data echoed back to that user, P1
and below — that should be filed and fixed but are not "outsider impersonates anyone."
The honest answer to *"is there a showstopper that ends my career in six months?"* is:
**not in `web/` as deployed today** — but file the four NEW items below before they rot,
and close the latent host-gate foot-gun so a future deploy can't step on it.

---

## Findings by severity (with corrected tracked-status)

Legend — **Status**: `LIVE` = exploitable on the deployment as configured; `LATENT` =
real code defect, not reachable in the current deployment/config; `CONTINGENT` = needs an
external precondition (IdP/library/proxy misconfig). **Track**: `NEW (not filed)` /
`TRACKED <id>` (verified) / `CLOSED-AREA <id>` (prior bug closed here). `Sev↕` shows
synthesis-vs-verifier where they split.

### P0 candidates — both downgraded after verification (2)

| # | Title | Location | Verified status |
|---|-------|----------|-----------------|
| 1 | Default JWT signing key accepted for loopback-bound hosts → would-be auth bypass | `config.py:496-538` (+`app.py:769`) | **LATENT, not live.** Staging `HOST=0.0.0.0` (gate not bypassed) + non-default 64-char key. Loopback exemption is deliberate; CLOSED-AREA `1d0be35f23`, `e816796ddf`. **Revised → P2 (latent foot-gun / defense-in-depth).** |
| 2 | Raw Tier-3 per-row failure messages leak into sessions-DB `runs.error`, HTTP response, SSE stream | `execution/service.py:1263-1283,1366,765` | **Real, NEW (not filed).** Self-exposure (own run data). Partly sanctioned by `COMPOSER_EXPOSE_PROVIDER_ERRORS=true` on staging. **Sev↕ P0(synth)/P1(verifier) → settle P1.** |

### P1 — serious correctness / integrity (verified)

| # | Title | Location | Sev↕ | Track |
|---|-------|----------|------|-------|
| 3 | LLM fanout/spend guard **fails open** for unestimable sources | `execution/fanout_guard.py:165-174` | P1 | **NEW** (parent feature `40fd60fb3a` DONE; this hole is new) |
| 4 | T3 (sample_rows + chat content) exposed in HTTP 400 body via `ToolResult` repr | `sessions/routes/composer.py:2723-2727` | P1 | TRACKED epic `e1ab67e55a` (specific bug not confirmed filed) |
| 5 | Declarative response redactor is shallow: nested node-options bypass redaction into `chat_messages.tool_calls` | `composer/redaction.py:3072-3087` | P1(synth)/**P2(verifier)** | **NEW** (no epic for redaction internals) |
| 6 | `_count_calls_for_run` counts HTTP+LLM+SQL but writes total as `llm_call_count` | `composer/tutorial_service.py:361-372` | P1 | **NEW (not filed)** |
| 7 | `output_file_hash` populated from input-side `source_data_hash` — wrong audit provenance | `sessions/audit_story_service.py:64-68` | P1 | **NEW (not filed)** |
| 8 | Post-compose state saves use `provenance='session_seed'` — mis-attribution | `sessions/routes/messages.py:656-702` (+`sessions.py:503`) | P1 | **TRACKED `elspeth-24a7fb8e54`** (confirmed bug, *not* the expiring obs) |

### P2 — contained correctness / contingent hardening (10)

| # | Title | Location | Track |
|---|-------|----------|-------|
| 9 | B-4D-3 bonus LLM call drops terminal tool-call requests from the audit trail | `composer/service.py:2075-2083` | epic `bf85fc8349` (not confirmed filed) |
| 10 | Recipe fast-path silently omits audit records when arg canonicalization fails | `composer/service.py:1735-1764` | epic `e1ab67e55a` |
| 11 | `prune_state_versions` omits 3 referencing tables → raw `IntegrityError` | `sessions/service.py:4551-4637` | epic `ef52049338` (no prod caller yet) |
| 12 | `accept_composition_proposal` holds no `compose_lock` → lost-update race | `sessions/routes/composer.py:347-525` | epic `e1ab67e55a` |
| 13 | Provider HTTP error bodies reach diagnostics-LLM prompt unfiltered (stored injection) | `execution/routes.py:697` | epic `528bde62bb`; softened by `EXPOSE_PROVIDER_ERRORS=true` |
| 14 | Missing `blob_id` type guard → `TypeError` mis-disposed as T1 500 | `composer/tools/sources.py:635-660` | epic `e1ab67e55a` |
| 15 | `_state_matches_cached_topology` raises on gate/coalesce instead of returning `False` | `composer/tutorial_service.py:128` | epic `e1ab67e55a` |
| 16 | `discarded_row_count` not stored in `TutorialCacheEntry` → replays under-report discards | `preferences/tutorial_cache.py:79-96` | epic `e1ab67e55a` |
| 17 | OIDC/Entra decode does not `require` `exp` → tokens without `exp` never expire | `auth/oidc.py:406-412` | CONTINGENT (IdP); epic `250f698aaf` |
| 18 | JWT `algorithms` taken from attacker-controlled header, not pinned to the JWK | `auth/oidc.py:392-412` | CONTINGENT (PyJWT regression); epic `250f698aaf` |
| 19 | Pre-auth login rate limiter keys on `client.host` (XFF) → brute-force evadable | `middleware/rate_limit.py:148-160` | epic `250f698aaf` |
| 20 | JWT auth token transmitted as URL query param on the WebSocket endpoint | `execution/routes.py:797-801` | epic `250f698aaf` |

> P2 tracked labels for ranks 9–20 are epic-level (the epic exists and plausibly covers
> the area); child-issue existence was **not** individually verified (advisor scoped
> verification to the P0s and P1s).

---

## Live-deployment verification (the decisive correction)

Read from the real box (`/etc/systemd/system/elspeth-web.service`,
`deploy/elspeth-web.env`) — diagnostic read, nothing changed:

- **Bind:** unix domain socket (`--uds /run/elspeth/uvicorn.sock --proxy-headers
  --forwarded-allow-ips=/run/elspeth/uvicorn.sock`) behind Caddy — *not* a loopback TCP
  port.
- **`ELSPETH_WEB__SECRET_KEY`** — set, 64 chars, **not** the `change-me-in-production`
  default. → rank 1's core premise is false on staging.
- **`ELSPETH_WEB__HOST = 0.0.0.0`** — **not** in `_LOCAL_HOSTS` (`{127.0.0.1, localhost,
  ::1}`), so the weak-key gate at `config.py:498/528` does **not** short-circuit; the
  production enforcement path runs and the strong key passes it.
- **`ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY`** — set, 44 chars, not default.
- **`ELSPETH_WEB__AUTH_PROVIDER = local`**, `REGISTRATION_MODE = closed`.
- **`ELSPETH_WEB__COMPOSER_EXPOSE_PROVIDER_ERRORS = true`** — provider-error exposure to
  the authenticated user is *sanctioned config*, which softens ranks 2 and 13 (they are
  partly an opted-in behaviour, not purely an accidental leak — though rank 2 also carries
  raw *row* data, which is broader than provider errors and still warrants redaction).

**Net effect on rank 1:** the only way the host-gate bites is a *future* deploy that sets
`HOST` to a loopback value behind a TLS-terminating proxy **and** leaves `secret_key`
default. That is a real latent foot-gun worth removing (the gate shouldn't equate
"loopback bind" with "unreachable"), but it is **not a live P0**, and the area is already
dispositioned (two closed bugs). Severity revised P0 → **P2 (latent)**.

---

## Severity disagreements worth an operator eyeball

- **Rank 2 — P0(synth) vs P1(verifier) → P1.** Verifier held P1 (owner's own run data).
  Synthesis promoted to P0 on multi-sink egress (sessions DB + SSE + HTTP, outside the
  audit boundary). With `EXPOSE_PROVIDER_ERRORS=true` sanctioning part of this, **P1 is
  the honest settle** — but the raw *row* payload fragments are broader than provider
  errors and should still be redacted on egress. It is **NEW / not filed.**
- **Rank 5 — P1(reviewer/synth) vs P2(verifier).** The **only finding with no epic at
  all** (composer redaction internals) — most likely to fall through tracking cracks,
  **file it explicitly.** The structural asymmetry is confirmed (declarative response path
  less-protected than type-driven path for the identical leaf); the verifier's downgrade
  rests on a reachability proof (`blob_ref`/`bind_source` is source-only, repair
  suggestions act on consumer nodes, so no actually-Sensitive value is reachable *today*).
  Close the asymmetry; dispute is P1-vs-P2 on current reachability.
- **Rank 13 — held P2, with a tripwire.** Provider error text reaches the diagnostics LLM
  unscrubbed, but `_call_text_llm` passes **no tools** (`composer/service.py:2712-2739`),
  so injection only corrupts an advisory explanation. **If diagnostics ever gains tool
  access, this jumps to P0** — add a comment at that call site.

---

## Coverage & caveats (honest provenance)

- **Coverage:** all **140 substantive `.py` files** reviewed at least once (24 lens-matched
  bundles); the remaining 18 files are package `__init__.py` re-exports with no logic. **6
  cross-cutting traces** (redaction-leak, IDOR/tenancy, SSRF, secret-exfil, audit-integrity,
  auth-token) ran in parallel.
- **IDOR/tenancy and SSRF traces produced no new confirmed P0–P2.** The IDOR primitive
  (`sessions/ownership.py`: 404-for-everything, checks `user_id` *and* `auth_provider_type`)
  is correctly applied at the handlers checked; the `${VAR}`-exfil class (previously fixed)
  was not found re-openable. *Absence of a trace finding is not proof of absence of the
  class.*
- **Spot-verified against source:** the two P0 candidates, the fail-open guard, the
  redaction asymmetry, and the failure-samples path all matched the candidate evidence.
  My own independent reads of `auth/oidc.py` and `sessions/ownership.py` converged with the
  agents (I predicted the rank-18 alg-confusion gap before the run finished).
- **Live-config verified** (this revision): staging env + systemd unit read directly;
  rank-1 premise refuted; `EXPOSE_PROVIDER_ERRORS` flag found.
- **Tracker status verified** for the P0s and P1s via `filigree search`; corrected in the
  tables above. Ranks 9–20 carry epic-level labels only.
- **Verifier drops audited (done):** the findings the verifier refuted were checked from
  the per-agent transcripts. **All 18 refutations are `confirmed_severity: P3`, `confidence:
  high`, `needs_human_look: false` — zero refuted at P0/P1, none flagged high-impact-
  uncertain.** The needs-human-look bucket functioned (it preserved finding 6's flag); no
  important finding was silently wrong-killed. The drops were sub-threshold polish /
  doctrine false-positives, correctly excluded.
- **No code was executed** (read-only); the "PyJWT `prepare_key` guard currently blocks
  rank 18" claim is the verifier's, not reproduced here.

---

## Appendix A — round-1 sub-threshold findings (the 9 the verifier dropped)

You asked for *all* faults documented, not just the 20 above the P2 floor. Round 1 emitted
29 raw findings; these 9 were dropped from the main report because the adversarial verifier
refuted them or scored them P3. **All 9 drops are sound** (I re-read every verdict; none was
a wrong-kill — see "Verifier drops audited" above). Several are nonetheless legitimate
**defense-in-depth / consistency** items worth a hardening ticket. Disposition is mine, on
the verifier's reasoning.

| Title | Location | Verdict | Worth fixing? |
|-------|----------|---------|---------------|
| Gate-condition injection via unvalidated `field` slot (`row['{field}'] >= {threshold}`) | `composer/recipes.py:344` | **Refuted → P3.** The composer LLM can already author any gate condition directly via `upsert_node`/`set_pipeline` through the *identical* validation path — no privilege/tenant boundary is crossed — and `ExpressionParser` is a hardened AST whitelist (no calls/attr/dunder, fail-closed). Audit is faithful. | Optional hardening: coerce `field` to a bare identifier / quote-escape it, so the recipe scaffold can't *look* like an injection in the audit record. |
| `update_run_status` terminal-state **TOCTOU** (stale read + unconditional UPDATE, DEFERRED txn) | `sessions/service.py:4178-4228` | **Refuted → P3.** A DB-layer last-write-wins window is constructable, but every concurrent-writer path is closed by app design: `cancel_orphaned_runs` (the cited vector) is test-only; production uses single-writer-per-run + the sweep excludes live runs for their whole lifecycle. | **Yes, cheap hardening.** Add `WHERE status == <expected>` to the UPDATE (or `BEGIN IMMEDIATE`) so the invariant is enforced at the DB layer, not just by call-site discipline that a future caller could break. |
| `InterpretationEventResponse` (+4 wrappers) omit `strict=True`, contradicting the module's audit-integrity docstring | `sessions/schemas.py:470-688` | **Refuted → P3.** The only constructor (`_interpretation_event_record_from_row`) already validates every drift-prone field with crashing UUID/enum/datetime constructors *before* the wire model is built, so lax mode has nothing to coerce. | Optional: add `strict=True` to honour the docstring and harden against a *future* construction path that skips the record layer. |
| T3 `tool_name` interpolated unescaped into the audit-persisted system-hint message | `composer/anti_anchor.py:73` | **Refuted → P3.** `role`/`content` are separate DB columns (Core param-bound); an injected newline/`role:` token stays inside the `content` value and cannot escape to the `role` column. The `[ELSPETH-SYSTEM-HINT]` marker precedes the interpolation. | No — refuted on structure. Note only if a future reader re-parses `content` into roles. |
| Blob verbatim-provenance classifier uses substring match (`content in user_message`) not equality | `composer/tools/blobs.py:570-595` | **Refuted → P3.** Substring is a *deliberate, tested* design (LLM strips an instruction prefix, passes the literal content); `created_from_message_id` is required and **fail-closed** (raises `AuditIntegrityError` without a message anchor). The LLM cannot fabricate novel content under VERBATIM, only relabel bytes already in the user message. | No — working as designed. The enum docstring's "byte-identical" phrasing is stale; tighten the *doc*, not the code. |
| `POST /api/tutorial/run` missing the composer rate limiter | `composer/tutorial_run_routes.py:17-28` | **Refuted → P3.** Premise was wrong (`/execute` has no rate limiter either; `ComposerRateLimiter` is chat-turn-scoped). Concurrency is governed by per-session `asyncio.Lock` + `RunAlreadyActiveError` + `max_workers=1`. | Minor: the route *blocks* up to 120s holding a connection slot — a per-IP cap on this blocking endpoint would bound slot exhaustion. Low priority. |
| Internal filesystem paths echoed in HTTP 4xx error bodies (artifact/preview endpoints) | `execution/routes.py:1042-1072` | **Refuted → P3.** Ownership-gated; the *same* `path_or_uri` is a documented happy-path contract field on the `/outputs` manifest (the "audit-evidence retrieval surface"). The read side re-resolves + allowlist-checks and never re-accepts the echoed string. | Optional consistency: one sibling handler redacts a (different) path; align the style. Not a leak. |
| `blob_inline_resolutions.content_hash` CHECK enforces length-64 but omits the lowercase-hex char-class (sibling `blobs_table` has both) | `sessions/models.py:1359-1362` | **Refuted → P3.** Unreachable: the only writer produces `sha256().hexdigest()` by construction, and `ResolvedBlobContent.__post_init__` enforces `^[0-9a-f]{64}$` and raises before insert. App validation is strictly stronger than the DB CHECK. | **Yes, trivial.** Tighten the CHECK to match the sibling — pure defense-in-depth, one line, removes the asymmetry. |
| `ComposerServiceError` `str(exc)` passed to the 502 body without the `_litellm_error_detail` redaction wrapper | `sessions/routes/messages.py:530-533` | **Refuted → P3.** No `raise ComposerServiceError` site embeds `str()` of an underlying exception — all use `type(exc).__name__`, static literals, or `_availability.reason` (model/provider/missing-env-key *names*, never secret values or T3). LiteLLM raw exceptions are caught by handlers *above* this one. | Minor consistency: those config-state strings aren't gated behind `composer_expose_provider_errors` the way the LiteLLM handlers are; gate them for uniformity. |

**Net:** the four worth a (small) ticket are the terminal-state TOCTOU DB-precondition
(`service.py:4178`), the `content_hash` CHECK tightening (`models.py:1359`), the `strict=True`
docstring-honouring (`schemas.py:470`), and the `recipes.py:344` identifier-escaping. None
changes the security posture; all are latent-foot-gun / consistency hardening.

> **Round 2 in progress** (run `wf_591e1ac0-56f`): a deliberately lower-floored (P0–P3)
> sweep with rotated lenses + easy-to-miss fault-class traces (error-swallowing, missing-
> await, resource leaks, unbounded results, datetime/numeric, cache-staleness, FSM
> completeness, lifespan/DI ordering), deduped against all 29 round-1 findings. Appendix B
> will capture whatever it surfaces, plus the synthesizer's dry-signal on whether a round 3
> is warranted.

---

## Appendix B — round 2 (lower-floored, rotated-lens sweep)

**⚠ Coverage warning — round 2 attempt 1 was infrastructure-truncated.** Of the 31 finder
units, **29 failed with server-side rate-limiting** ("Server is temporarily limiting
requests — not your usage limit"), including *every* monster re-read (`service.py`,
`redaction.py`, `_helpers.py`, `composer.py`, `tool_batch.py`, `models.py`, `blobs.py`,
`generation.py`, `_common.py`, `sessions.py`, `validation.py`), *all 10* fault-class traces,
and *all 7* correctness bundles. Only the `state.py` lens completed. **The synthesizer's
"seam is dry" signal is an artifact of that truncation, not a coverage result** — it judged
dryness from the 2 findings that survived without knowing 29 finders never ran. (Cause:
two large back-to-back workflows, ~6M tokens, tripped a transient global API throttle.)
A re-run of the failed coverage (round 2b) was launched.

### New finding from round 2 (the one unit that completed)

| Title | Location | Sev | Class |
|-------|----------|-----|-------|
| B1 | `state.py` docstrings overclaim deep-freeze for **tuple-typed** container fields (`NodeSpec.fork_to`, `CompositionState.nodes/edges/outputs`) that the file's deliberate rule leaves unfrozen | `composer/state.py:3-4, 1765, 147-156, 1787-1789` | **P3** | documented-invariant-not-enforced |

**B1 detail (confirmed, latent):** the module docstring lists `fork_to` among "deep-frozen"
container fields and `CompositionState`'s docstring asserts "all container fields are
deep-frozen," but `__post_init__` freezes only the **Mapping** fields (`options/routes/
branches/trigger`) — tuple fields are left to their type annotation. This is **not** a
behavioural inconsistency: the inline comment at `state.py:148-149` ("scalar, enum, and
tuple fields are already immutable and need no guard") is the file's deliberate, uniform
rule, and the sibling specs (`SourceSpec`, `OutputSpec`) follow it. The defect is in the
**docstring claims**, not the runtime — every current construction site passes real tuples
(`NodeSpec.from_dict:180`, `tools/transforms.py:464`, `tools/sessions.py:514`), so there is
no live mutable-container escape. The foot-gun is a future caller who trusts the docstring
and passes a `list` (which would already violate the `tuple[...]` annotation). *Fix:* amend
the two docstrings to scope "deep-frozen" to the Mapping fields, **or** add tuple-coercion in
`__post_init__` so code matches the documented guarantee. Pure hardening; for a compliance
tool the "immutable versioned snapshot" claim is provenance-relevant, which is why it clears
the P3 floor rather than being pure style.

### Dry-signal — NOT ESTABLISHED (round 2 abandoned on repeated rate-limiting)

**Final status: the lower-floored (P3) sweep was not completed.** Attempt 1 lost 29/31
finders to a server-side throttle; the re-run (round 2b) hit the same global rate-limit and
was **stopped by the operator**. So the question *"is there material below P2 that round 1's
floor suppressed?"* is **open, not answered.** Do **not** read either truncated round-2 run
as evidence the codebase is clean below P2 — that conclusion was never earned.

**What round 2 did yield before truncation:** exactly one new item, **B1** (`state.py`
docstring overclaims deep-freeze, P3, latent). Across ~33 finder-units that *did* run between
the two attempts (the `state.py` lens both times, plus whatever completed in 2b before the
stop), **no new P0–P2 surfaced** — a weak positive signal, but far from the clean exhaustive
pass that was intended.

**If you want to close the lower-floor seam later:** re-run `web-audit-round2.js` when the
API is not under load, ideally with the concurrency cap lowered (e.g. run the 14 monster
re-reads, 10 traces, and 7 bundles as smaller batches across separate invocations rather than
31 units at once) to stay under the throttle. The round-1 P0–P2 result and Appendix A are
unaffected and stand on their own.

---

## Recommended remediation order (revised)

1. **File the four NEW, not-yet-tracked items now** — ranks 2 (failure-samples Tier-3
   egress), 5 (redaction asymmetry, *no epic exists*), 6 (`llm_call_count` inflation), 7
   (`output_file_hash` wrong provenance). These are the things genuinely *not known* and
   most likely to rot.
2. **Rank 3 (P1)** — make the fanout guard fail *closed* on unknown cardinality (matches
   its own docstring); parent feature is DONE so this is a regression-class hole.
3. **Rank 1 (now P2, latent)** — remove the loopback-equals-unreachable exemption (or gate
   on an explicit deployment-mode flag) so no future deploy can step on it. Not urgent —
   staging is safe — but cheap and it kills a foot-gun for good.
4. **Ranks 4, 8 (P1)** — the repr leak (model on the adjacent PR#37 fixes) and the
   `session_seed` mis-attribution (already tracked `24a7fb8e54`).
5. **Ranks 9–20 (P2)** — schedule against existing web-hardening epics; add the rank-13
   diagnostics-LLM tripwire comment.
6. **Cheap insurance:** skim the 9 verifier-dropped findings from the run transcripts to
   confirm no weak wrong-kill.
