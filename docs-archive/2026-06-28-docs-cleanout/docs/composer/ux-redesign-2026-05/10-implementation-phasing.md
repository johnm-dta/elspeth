# 10 — Implementation Phasing

## Principle

This redesign is not a single big-bang implementation. The recommendations
group into phases with internal coherence — each phase produces a coherent
intermediate state of the composer. Shipping phase boundaries (rather than
individual features) avoids the "half-redesigned UI" trap where some
surfaces have been updated and others haven't.

The phases below are ordered by dependency, value-per-effort, and risk.
Each is described with: what it includes, what it doesn't, what it
requires from the backend, and what state the composer is in afterward.

## Phase 0 — Adjudication

**Not implementation work.** A working session with the operator to
confirm the recommendations in this redesign and resolve the open
questions in [11-open-questions.md](11-open-questions.md). No phase below
can start without this.

**Output:** A signed-off recommendation list. Decisions on each open
question (or explicit "punt to phase X" call).

**Estimated time:** A few hours, distributed across the team.

## Phase 1 — Default-guided + opt-out

**Smallest cohesive change. Lowest risk. High value for the target audience.**

**Includes:**
- Add `composer.default_mode` user preference (default value: `guided`).
- New session creation honors the preference.
- User settings menu with composer-preferences pane.
- Inline opt-out affordance on guided mode UI.
- Migration: existing users default to `freeform` to avoid surprise.

**Doesn't include:**
- Hello-world tutorial (Phase 4).
- Inline opt-out tooltip nuances.
- Telemetry for opt-out rates (Phase 8 polish).

**Backend dependencies:**
- User preferences storage (likely exists; if not, this phase adds it).

**Composer state after Phase 1:**
- New sessions for new users start in guided mode. Existing users
  unchanged. Marcus and Dev can opt out from a visible affordance.

**Reference:** [05-modes-and-opt-out.md](05-modes-and-opt-out.md)

## Phase 2 — Audit-readiness panel

**The single highest-impact change. Makes ELSPETH's defining feature
visible during composition.**

**Includes:**
- New backend endpoint aggregating: validation result + plugin trust
  tiers + provenance check + retention config + secrets status + LLM
  interpretations status (conditional rows on the latter two).
- New frontend persistent panel in side rail.
- Click-to-explain detail view.
- Vocabulary mapping per
  [07-audit-readiness-panel.md](07-audit-readiness-panel.md).
- Remove the standalone Validate button (subsumed into the panel's
  Validation row).

**Doesn't include:**
- The full Explain narrative view (a richer version comes in Phase 7).

**Backend dependencies:**
- The aggregation endpoint composes existing checks; no new audit-trail
  work required.

**Composer state after Phase 2:**
- Audit readiness is visible at all times during composition. Linda's
  primary trust mechanism is in place. The standalone Validate button is
  gone.

**Reference:** [07-audit-readiness-panel.md](07-audit-readiness-panel.md)

## Phase 3 — IA cleanup (kill Spec / Runs tabs, side-rail reorganisation)

**Removes obsolete chrome; reorganises the inspector into the side-rail
shape.**

**Includes:**
- Remove the Spec tab and its component.
- Remove the Runs tab as a tab; move run-results to inline-after-Execute
  rendering.
- Move Graph view to a persistent mini-view in the side rail (clickable
  to expand).
- Move YAML view behind an "Export YAML" button in the side rail.
- Move Catalog button from inspector header to side rail (chrome change
  only — Phase 5 reshapes its content).
- Replace inspector header with the simpler shape described in
  [03-target-information-architecture.md](03-target-information-architecture.md).

**Doesn't include:**
- Reshape of Catalog drawer content (Phase 5).
- Completion bar with new verbs (Phase 6).

**Backend dependencies:**
- None. This is frontend layout work.

**Composer state after Phase 3:**
- The IA shape matches [03-target-information-architecture.md](03-target-information-architecture.md).
  Catalog still shows the old toolkit-style drawer (its reshape is Phase 5);
  Execute is still a separate button in the side rail (the completion bar
  with three verbs is Phase 6).

**Reference:** [03-target-information-architecture.md](03-target-information-architecture.md)

## Phase 4 — Hello-world tutorial

**Onboards every new user with the canonical pipeline + the audit story.**

**Includes:**
- Detect "first session" state per user.
- Sequential tutorial turns per
  [04-first-run-tutorial.md](04-first-run-tutorial.md).
- Tutorial uses dynamic-source-from-chat (Phase 5b dependency).
- Tutorial uses the surface-the-LLM's-interpretation affordance
  (Phase 5b dependency).
- Tutorial preserves the produced pipeline as a real session.
- Final tutorial step sets the `composer.default_mode` preference
  (Phase 1 dependency).

**Doesn't include:**
- A "redo the tutorial" affordance in settings (optional polish in Phase 8).
- Tutorial localisation.

**Backend dependencies:**
- Phase 5b features must be in place.
- A `composer.tutorial_completed` user-level flag.

**Composer state after Phase 4:**
- Every new user sees the tutorial on first login. The mode-default
  preference is set after informed exposure. The composer's defining
  feature (audit) is introduced explicitly.

**Reference:** [04-first-run-tutorial.md](04-first-run-tutorial.md)

## Phase 5 — Chat-as-data-entry features

**Two sub-phases. 5a is the smaller; 5b is needed by Phase 4.**

### Phase 5a — Dynamic-source-from-chat

**Includes:**
- Composer chat input accepts data directly (URL, sentence, short list).
- Composer creates a dynamic source from the chat content.
- Empty-state chat placeholder primes the user that this is possible.
- Verify audit recorder treats dynamic-source-from-chat content
  identically to file-source content for provenance.

**Backend dependencies:**
- Composer tool surface needs (or already has) a tool that creates an
  inline-content source.

### Phase 5b — Surface-the-LLM's-interpretation

**Includes:**
- New turn-widget pattern / chat affordance for presenting LLM
  interpretations.
- "Use mine / Change it" UI; inline editor pre-filled with LLM draft.
- Audit-recorder records the user's accepted/amended interpretation as
  a discrete event.

**Backend dependencies:**
- Audit-recorder change: new event type for interpretation acceptance.

**Composer state after Phase 5:**
- Both features available everywhere — guided, freeform, tutorial. The
  composer feels meaningfully more conversational. The audit story
  gains an explicit human-in-the-loop record.

**Reference:** [06-chat-as-data-entry.md](06-chat-as-data-entry.md)

## Phase 6 — Completion gestures

**Replaces the single Execute button with the three-verb completion bar.**

**Includes:**
- Side-rail completion bar component.
- "Save for review" backend endpoint and persistence.
- "Save for review" shareable-link generation.
- "Run pipeline" with context-aware result rendering (narrative if
  `batch_*` analytics present; otherwise table preview).
- "Export YAML" as a co-equal completion verb (already exists as a
  drawer, just gets a top-level button).

**Backend dependencies:**
- New endpoints for save-for-review state and shareable links.
- Optional: signed snapshot for save-for-review handoffs.

**Composer state after Phase 6:**
- Linda's flow has an honest verb. Sarah's run produces narrative
  results when appropriate. Marcus's flow is unchanged structurally.
  Dev's flow has a clear primary verb.

**Reference:** [09-completion-gestures.md](09-completion-gestures.md)

## Phase 7 — Catalog reshape

**Lowest urgency, but valuable. Reshapes the catalog from toolkit to
reference.**

**Includes:**
- Plugin metadata extension: "when you'd use this" / "when you wouldn't"
  / "example use" / audit-characteristic flags / trust tier (exposed).
- Frontend redesign of catalog cards.
- Filter chips (capability, trust tier, audit characteristics).
- Search extension (across prose + tags).
- "Inline data from chat" entry as the first source option.

**Doesn't include:**
- Recipes / examples tab (optional addition; defer).
- Catalog-from-keyboard-shortcut reorganisation (optional; defer).

**Backend dependencies:**
- Plugin metadata documentation work — per-plugin, can be done
  incrementally even before the frontend ships.

**Composer state after Phase 7:**
- Catalog is honest reference. Personas with orientation needs (Linda,
  Sarah, Marcus, Dev) all find what they need. No more "what is this
  button even for?" energy.

**Reference:** [08-catalog-reshape.md](08-catalog-reshape.md)

## Phase 8 — Polish and telemetry

**Cleanup, instrumentation, edge cases. Not required for the redesign
to be functional; required for it to be durable.**

**Includes:**
- Templates → README's Example Use Cases mapping.
- Session sidebar → header switcher (delayed; deferring to here because
  the IA change is independent and shouldn't block other phases).
- Mode-related telemetry (opt-out rate, completion rate, etc.).
- Keyboard-shortcut audit and reorganisation.
- Tutorial-replay affordance in settings.
- Accessibility audit (focus management, keyboard navigation,
  screen-reader labelling) on all new components.

**Backend dependencies:**
- Telemetry endpoints (likely exist; this hooks new metrics in).

**Composer state after Phase 8:**
- Full target state per [03-target-information-architecture.md](03-target-information-architecture.md).
  Telemetry tells the team whether the call (default-guided, opt-out, etc.)
  was right.

## Critical-path summary

The phases have these hard dependencies:

```text
   Phase 0 — Adjudication
        │
        ▼
   Phase 1 — Mode default + opt-out  ────┐
        │                                 │
        ▼                                 │
   Phase 2 — Audit-readiness panel   ────┤
        │                                 │
        ▼                                 │
   Phase 3 — IA cleanup              ────┤   independent;
        │                                 │   could run in
        ▼                                 │   parallel
   Phase 5a — Dynamic-source-from-chat   │
        │                                 │
        ▼                                 │
   Phase 5b — Interpretation surface ────┘
        │
        ▼
   Phase 4 — Hello-world tutorial   (needs 1, 5a, 5b)
        │
        ▼
   Phase 6 — Completion gestures
        │
        ▼
   Phase 7 — Catalog reshape
        │
        ▼
   Phase 8 — Polish + telemetry
```

The earliest demo-shippable state is **end of Phase 5b** (or end of
Phase 6 if Linda's completion verb is demo-load-bearing). The audit-readiness
panel + the dynamic-source-from-chat + the interpretation surfacing
together produce a coherent ELSPETH narrative even without the tutorial
or catalog reshape.

The latest demo-required state is **end of Phase 4** if the demo audience
includes first-time users.

## Risk by phase

| Phase | Risk | Severity |
|---|---|---|
| 1 | Migration: existing users default to freeform vs guided. Some confusion. | Low — affordance is visible. |
| 2 | Panel disagrees with the actual run. Audit-readiness lies. | Medium — same underlying logic must be used. |
| 3 | "I can't find the Spec tab" — power users miss it. | Low — there's no actual loss; engine debugging uses CLI. |
| 4 | Tutorial fails for an edge-case prompt | Medium — fallback to canonical seed text. |
| 5a | Audit recorder doesn't treat inline content as a real source | High — needs verification before shipping. |
| 5b | Audit recorder doesn't record interpretation acceptance | High — the audit story collapses without this. |
| 6 | Save-for-review semantics are unclear without a reviewer surface on the other end | Medium — initial implementation is single-user friendly. |
| 7 | Plugin docs are incomplete | Low — graceful fallback to generic descriptions. |
| 8 | Telemetry reveals the default was wrong | Acceptable — that's what telemetry is for. |

## What can ship independently

- **Phase 1 alone** is shippable as a small product change. No other
  phase depends on its UI surface except the tutorial's final turn.
- **Phase 2 alone** is shippable as a feature flag. The audit-readiness
  panel can appear next to the current inspector tabs before the IA
  cleanup happens.
- **Phase 5a alone** is shippable; users can type data into chat for
  any source.
- **Phase 7 alone** is shippable; the catalog reshape doesn't require
  any other phase.

**Phases that must ship together** (or via feature flag tied to all):
- Phase 3 (IA cleanup) and Phase 6 (completion gestures). The IA cleanup
  removes the Execute button from the inspector header, but the
  completion bar isn't where Execute lives until Phase 6. Ship them
  together or feature-flag the layout.

## What this redesign does NOT phase

- **Backend audit-recorder changes** for Phase 5b (interpretation events).
  These need to land before Phase 5b's frontend can be useful. Treat as
  a sub-task of Phase 5b.
- **Plugin metadata documentation work** for Phase 7. Per-plugin
  documentation is a continuous task; the catalog reshape can ship with
  partial docs and improve incrementally.

## Memory references

- All seven of the composer-design memory entries (see README index).
