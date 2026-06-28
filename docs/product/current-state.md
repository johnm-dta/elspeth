# Current State — ELSPETH        Checkpoint: 2026-06-28 · branch release/0.7.0

> ⚠️ **This workspace is ~2 weeks stale.** It was last written at the 2026-06-14
> bootstrap; only the tier-model-signing delta below was checkpointed since.
> Reality has moved to the `release/0.7.0` line (composer UX phases 1–8, passive
> tutorial e2e, website, design system — see project memory), none of which is
> reflected here. **The next session must run `/own-product` for a full RESUME +
> re-orient and re-confirm the Now bet with the owner — do not treat the bet below
> as current.**

## The bet right now (stated, NOT re-confirmed this session)

**Web hardening to GA** — the 2026-06-14 owner-confirmed primary (close the five
Web-surface assurance clusters). It was **not advanced this session** and predates
the 0.7.0 line; treat it as unverified until the next RESUME re-confirms or replaces
it. Metric: north-star (run assurance completeness) + Web-GA input.

## In flight

- **Turnkey tier-model signing (this session) — BUILT, not pushed, e2e owed.** The
  stage→sign-bundle seam: agent stages key-free via the `elspeth-judge` MCP server;
  operator fires `sign-bundle` / `rekey` with the HMAC key. Committed to
  `release/0.7.0`: feature `6e0d66f9`, spec `b87829bc3`, plan `d59fbccdd`, refinements
  `af30dfc6e` / `786a609ab`. Green (70 new tests; 5 security invariants pinned).
  tracker: **elspeth-281582acc9** (still `building` — built, not closed pending e2e).
  See `decisions/0002` (proposed) + `docs/judge-signature-handoff.md`.
- **Web hardening clusters** — still Now per roadmap, still un-dispatched. tracker:
  elspeth-250f698aaf, elspeth-ef52049338, elspeth-0fd9dfcb7e, elspeth-16ddaa7d02,
  elspeth-248536c9e6. (Reconcile against 0.7.0 reality at next RESUME.)

## Open questions / blocked-on-owner

- **[owner sign-off — PDR-0002] Tier-model signing is an escalate-before-acting
  domain.** Three calls gate to you: (a) adopt the stage→sign-bundle seam as the
  *sanctioned* signing workflow; (b) confirm the runbook removal stands (the
  `notes/060-*` runbooks are gone from disk — gitignored scratch; `sign_accept_backlog.py`
  is git-recoverable); (c) authorize the **push** of `release/0.7.0`.
- **[owner] Live-judge e2e not yet run** — the new path is unit-tested with a
  monkeypatched judge only; it has not been exercised `stage_scan → sign-bundle`
  against a real `ELSPETH_JUDGE_METADATA_HMAC_KEY` + live LLM. Run before relying on it.
- **[next RESUME] Workspace staleness** — roadmap still says "Ship the 0.6.0 line" as
  Now; we're on 0.7.0. The whole workspace needs reconciliation against the actual
  shipped line and a re-confirmed Now bet.
- **[carried from bootstrap, unresolved]** metrics uninstrumented (every north-star/
  input is a placeholder); tracker drift (C1/C2/C3 fixed-in-branch vs P0-READY); 5
  stale claims; Landscape MCP epoch 11 vs DB epoch 21.

## Last checkpoint did

- Recorded the tier-model signing seam as **PDR-0002 (proposed)** — built + shipped
  to `release/0.7.0` this session under direct owner direction; flagged the three
  signing-domain items for owner sign-off (adopt / runbook-removal / push).
- Refreshed `metrics.md`: red-gate guardrail **strengthened** (structural operator-only
  custody + CI-never-signs); no reversal trigger crossed; full battery not re-run.
- Did **not** touch `roadmap.md` (no roadmap bet changed horizon — the signing seam was
  an owner-directed side task, not a tracked bet).

## Next session, start here

Run **`/own-product`** for a full RESUME — the workspace is stale and the Now bet must
be re-confirmed against the 0.7.0 reality. Then get owner sign-off on **PDR-0002**
(signing workflow + push) and run the live-judge e2e before closing elspeth-281582acc9.
