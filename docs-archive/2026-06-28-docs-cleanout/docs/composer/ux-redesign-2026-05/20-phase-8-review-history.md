# Phase 8 — Review History

This file accumulates the pass-by-pass review findings for the Phase 8
plan ([`20-phase-8-polish-and-telemetry.md`](20-phase-8-polish-and-telemetry.md)).
It is **not** part of the implementer's load-bearing instructions —
those live in the plan file itself. This file exists so future review
cycles can see the trajectory of prior reviews without reading meta-
commentary inline with the spec.

**How to read this file:** entries are chronological, newest at the
bottom. Each entry names the date, the review pass, and the findings
applied (or rationale for declining to apply).

**How the plan refers to this file:** the plan's body contains
`Bn`/`Wn`/`Sn` markers and `(B1-r3 — load-bearing)` style references.
This file is the trail of where those markers came from.

---

## 2026-05-15 — review-panel findings applied (5 items)

- **F1 (Critical/Systems):** Partial-telemetry signals can mislead. Added Task 0
  (Probe-mechanism tests) before Task 1 to verify each upstream phase probe returns
  the expected signal before conditional branches run. Added a fail-safe policy to
  §Verification approach: conditional tasks whose probe returns "phase not shipped"
  must emit `phase_8_probe_failed{phase, probe}` via telemetry rather than silently
  emitting degraded metrics. Added `phase_8_probe_failed` to the OTel counter list.
- **F2 (Important/Architecture+Systems):** "WE HAVE NO USERS YET" tension. Added
  §Caretaker logic policy explaining why caretaker logic is currently nullified by
  the delete-the-DB policy and that Phase 8 wires it correctly so Phase 9's
  migration-runner gives it permanent meaning.
- **F3 (Important/Architecture):** Probe mechanism not tested. Resolved by F1 above.
- **F4 (Important/Quality):** Some telemetry belongs inline with earlier phases.
  Original resolution added §"Minimum instrumentation bar" listing six metrics
  asserted to be emitted by Phases 2, 5b, 6 — that framing was incorrect
  (see B3 below); the section has been reshaped.
- **F5 (Suggestion/Architecture):** Dead-code sweep at end of Phase 8 is archaeology.
  Added a Risks row recommending a CI lint rule on exported-but-unused constants
  post-Phase-8 and noting that future phases should clean up after themselves.

## 2026-05-19 — B3 reshape (cross-phase telemetry ownership)

- **B3 (Blocking):** Phase 8 §"Minimum instrumentation bar" listed six metrics
  expected to have been emitted by Phases 2C, 5b, and 6, but direct grep against
  all three sibling plans confirmed **none** of the six was actually named.
  Phase 8 could not "verify their presence" — it was creating a requirement
  against shipped or mid-delivery siblings without telling them. Resolution:
  reshape the section into §"Cross-phase telemetry — cohort split (B3 reshape)"
  with three cohorts:
  - **Cohort (a)** (`composer.share.token_verify_failure_total`,
    `composer.share.link_expiry_hit_total`) → Phase 8 owns the emits at
    Phase 6's verify-failure and expiry-hit code sites; conditional on
    Phase 6's token-verify path having shipped (NEW Task 1 Sub-task 7d;
    probe in Task 0; new Risks row for Phase 6/8 timing collision).
  - **Cohort (b1)** (`composer.interpretation.opt_out_total`) → Phase 8 owns
    the emit at Phase 5b's opt-out route (audit-derivable from
    `interpretation_source='auto_interpreted_opt_out'`; superset rule
    satisfied) (NEW Task 1 Sub-task 7e).
  - **Cohort (b2)** (`composer.audit.fetch_failure_total`) → Phase 8 owns
    the emit at Phase 2C's audit-readiness fetch site (telemetry-only signal
    on a non-decision read; superset exception applies) (NEW Task 1 Sub-task
    7f).
  - **Cohort (c)** (`composer.audit.render_duration`,
    `composer.interpretation.resolve_duration`) → pure perf signals; reopening
    shipped phases for perf instrumentation is out-of-scope. Filed as Phase 9
    follow-ups via new Task 0 Step 4 that seeds `21-phase-9-followups.md`.

  Updated: counter container (Task 1 Step 5) extended with four new counters;
  Task 0 probe-mechanism table extended with three new probe rows; §"Files
  this plan modifies" extended with three new emit-site bullets; §"Out of
  scope" extended with the cohort (c) deferral; Risks table extended with
  the Phase 6 timing-collision row.

## 2026-05-19 — pass-2 panel findings applied

Pass-2 found 3 blockers + 4 warnings across the four-reviewer panel
(reality, architecture, quality, systems). The Systems reviewer caught
the most novel defect: a B1-r2 vocabulary collision that would have
500'd the per-session preferences PATCH after the audit row committed.
Findings landed inline as `(B1-r2 — load-bearing)`,
`(B2.b — load-bearing)`, etc. — see the plan body for the load-bearing
sections each finding spawned.

Notable: pass-2 introduced the operator-gate sections (Decision 1
phase split, Decision 2 tutorial-replay boundary) that pass-3 then
resolved (see below).

## 2026-05-19 — pass-3 panel findings applied + Decisions resolved

Pass-3 found 4 blockers and 3 path-only warnings. The four blockers
were all the same shape — **the plan made commitments whose
preconditions weren't in place**:

- **B1-r3:** No `MeterProvider` is wired in `src/`; the A9 tracer-
  bullet cannot validate against a NoOp meter. **Resolution:** Wire
  a MeterProvider + Prometheus reader into the FastAPI app factory
  as Phase 8a's first task. See §"MeterProvider precondition
  (B1-r3 — load-bearing)" in the plan body.
- **B2-r3:** Decision 2 Option A (telemetry-only tutorial-replay
  counter) violates CLAUDE.md audit primacy — the superset
  exception is for read-path operational health, not for
  deliberate user-write-intent. **Resolution:** Adopt Decision 2
  Option C — defer the counter and the boundary question to
  Phase 9. See the Decision 2 resolution block in the plan body.
- **B3-r3:** W4 denominator prose described a transition-rate but
  the `mode_changed` attribute at `preferences/service.py:296`
  is a set-rate (fires whenever the PATCH body includes
  `default_mode`). A `freeform → freeform` re-PATCH inflated both
  numerator and denominator without flipping anything.
  **Resolution:** Rewrite W4 prose to honestly describe a
  set-rate. The `(B3-r3 — load-bearing)` paragraph in §"How to
  read these counters" makes the semantic explicit.
- **B4-r3:** A4 commit-trailer MUST was unenforced; per the
  `cicd-allowlist-audit` memory unenforced conventions decay
  (51% growth in 32d on the L3 allowlist). **Resolution:** Phase
  8a installs a commit-msg hook + CI backstop that mechanically
  rejects cohort-touching commits without the right trailer. See
  the `(B4-r3 enforcement — load-bearing)` block under
  §"Cohort attribution via commit trailers (A4 — load-bearing)".

Also resolved at this pass: **Decision 1** (phase split into 8a /
8b / 8c) — operator chose Option A. The B1-r3 MeterProvider work and
the B4-r3 commit-msg hook both land in 8a, which formalises the
split with concrete content rather than convention.

The synthesizer's pass-3 recommendation was explicit: do not run a
fourth fix cycle. The pattern from pass-1 → pass-2 → pass-3 (7 → 3
→ 4 blockers, each cycle's fixes seeding the next cycle's findings)
showed diminishing returns at exactly this point. The four pass-3
blockers were operator-decision shaped, not mechanical-fix shaped;
pass-4 would have produced a different four blockers from the same
convergent dynamic. Phase 8 ships on the pass-3 resolutions.

## 2026-05-19 — cross-plan contract amendment (Phase 4 ↔ Phase 8 P1)

Phase 4 pass-1 review found a Systems S1 contract rupture between
Phase 4 and Phase 8 Task 6 over the `tutorial_completed_at` field
shape and PATCH semantics. Phase 8 Task 6 originally tested
`PATCH {"tutorial_completed": false}` against a boolean
`tutorial_completed` field that Phase 4 was not in fact going to
ship; Phase 4 ships `tutorial_completed_at: datetime | None` with
the original draft disallowing nullification via PATCH — the very
operation Phase 8 needed.

Synthesizer-adopted resolution: Option (a) — single column, single
contract; Phase 4 permits explicit `null` in the PATCH body to
clear the column; Phase 8 retake PATCHes
`{"tutorial_completed_at": null}`. The Pydantic three-state
discrimination (absent / datetime / null) is implemented via
`model_fields_set`. The boolean frontend accessor
(`tutorialCompleted = (tutorialCompletedAt !== null)`) is unchanged.

Edits applied in the Phase 8 plan:

- §Pre-flight check probe table: probe target updated from
  `'tutorial_completed'` to `'tutorial_completed_at'`.
- §Sibling plans: Phase 4 line updated to link the shipped plan
  and the cross-plan contract section.
- §Scope boundaries: Task 6 bullet rewritten to name the new PATCH
  body and cite the cross-plan contract.
- §Decision 2 prose: PATCH body string updated.
- §Trust tier check: Tier-1 entry renamed from
  `composer.tutorial_completed` (bool) to
  `user_preferences_table.tutorial_completed_at` (datetime | None)
  with the Phase 4 Tier-1 read-side guard cited. Tier-3 entry
  reshaped from "Pydantic `Literal[False]` / `bool`" to "Pydantic
  `datetime | None` accepts null, rejects other types with 422".
- §B2 service-signature reshape: transition predicate updated from
  `prior.tutorial_completed and not current.tutorial_completed` to
  `prior.tutorial_completed_at is not None and current.tutorial_completed_at is None`.
- §Cohort (c) Phase 9 follow-up text: `True → False` transition
  rewritten as `not-None → None` (retake); the Q6 transition-edge
  test names updated from `true_to_false`/`false_to_true` to
  `set_to_null`/`null_to_set`.
- Task 6 Step 1 (probe): grep target updated; case-A/B language
  shifted from "flag" to "column".
- Task 6 Step 2 (wire shape): rewritten to cite Phase 4's
  cross-plan contract directly; PATCH body updated.
- Task 6 Step 3 (Q5 Pydantic boundary tests): the three original
  bool-rejection tests (`string_false`, `integer_zero`, `null`)
  removed (those rejections now live in Phase 4); replaced with
  two retake-specific tests (`with_explicit_null_clears`,
  `distinguishes_absent_from_null`).
- Task 6 Step 5 (store action): API call corrected to
  `api.updateComposerPreferences({ tutorial_completed_at: null })`;
  TypeScript request type contract pinned to allow `null`; a
  prohibition added on inventing a separate
  `clearTutorialCompleted` store action (single shared API
  surface).
- Task 6 deferred-Step-7 block: surrounding prose updated; new
  paragraph added naming the audit-emit boundary question for
  Phase 9 (where the audit event lives, prior-timestamp capture,
  Landscape-row-versus-counter choice).

No change to the Decision 2 Option C outcome itself
(`composer.tutorial.replayed_total` remains deferred to Phase 9);
only the field-name, type, and PATCH-body shape were corrected.
