# 09 — Completion Gestures

## The problem

The current composer has a single completion gesture: **Execute**. It runs
the pipeline immediately. This cleanly serves one persona (Marcus, the
action-oriented marketing-ops user) and serves the rest poorly:

- **Linda** doesn't run her own pipelines — she composes to hand off to a
  colleague. Today she has no honest completion verb; "Execute" feels
  wrong (and may be wrong for her workflow), so she falls back to "copy
  the YAML and email it." That's not a designed flow.
- **Sarah** runs her pipelines to see results, but "Execute" doesn't
  convey "and I want narrative findings" — she gets a JSONL she has to
  interpret.
- **Dev** mostly bypasses the composer to the CLI. The Execute button is
  ignored.

The persona analysis in
[02-personas-and-audiences.md](02-personas-and-audiences.md) shows each
persona has a different completion verb. The current single-Execute
treatment is incomplete.

## The recommendation

A **completion bar** in the side rail (visible whenever the pipeline is
composable, i.e., past the first composition state) offering three
persona-aware verbs side-by-side:

```text
  ┌─ COMPLETION ─────────────────────────────┐
  │                                          │
  │  [💾 Save for review]                     │
  │  [▶  Run pipeline ]                       │
  │  [⬇  Export YAML  ]                       │
  │                                          │
  └──────────────────────────────────────────┘
```

The user picks the verb that matches their intent. The buttons are
co-equal — none is "primary" — because the right verb depends on the
user, not on the tool.

## Verb-by-verb specifications

### Save for review

**For Linda's flow.** Captures: "this pipeline is ready for a colleague
to review and run. It is not running yet."

**Behaviour:**

- Persists the current composition state with a "ready for review" flag.
- Generates the YAML and stores it alongside the session.
- Produces a shareable session link the user can send to a colleague.
- Optionally: produces a *signed* artifact (using the existing
  HMAC-signing infrastructure for audit exports) that the colleague can
  verify came from this user without tampering.
- Surfaces a confirmation: "Saved. Your colleague can review at
  [link]. They'll see the audit-readiness panel and the YAML you wrote."

**State produced:** A persisted, reviewable session. Not yet executed.

**Audit recording:** "User X marked composition Y as ready for review at
timestamp Z."

**Conditions:** Available when the composition is in a runnable state
(passes validation). Not available for compositions with ✗ status — the
user fixes those first.

### Run pipeline

**For Sarah and Marcus's flows.** Captures: "execute this pipeline now,
show me the results."

**Behaviour:**

- The existing Execute behaviour. Starts a background run, streams
  progress over WebSocket, surfaces results inline.
- **Result framing depends on the pipeline shape:**
  - If outputs are numerical / structured (Marcus): show a table preview
    with download link.
  - If outputs are categorisations / text (Sarah): show a narrative
    summary if a `batch_*` analytic transform is in the pipeline;
    otherwise the table preview.

**State produced:** A run with outputs, audit records, and a results view.

**Audit recording:** The existing run-accounting record. (No new audit
work; this is the current Execute path.)

**Conditions:** Available when the composition is runnable.

### Export YAML

**For Dev's flow.** Captures: "give me the YAML, I'll take it from here."

**Behaviour:**

- Generates the YAML (existing functionality).
- Offers: copy to clipboard, download as file, view in modal with syntax
  highlighting.
- Optionally: surfaces a `elspeth run --settings <path>` command snippet
  ready to paste into a terminal.

**State produced:** YAML in the user's hand. Nothing else changes.

**Audit recording:** Optional — "User X exported YAML for composition Y."
This is debatable; the YAML itself is not the audit-bearing artifact
(the run is). See [11-open-questions.md](11-open-questions.md).

**Conditions:** Available always (even with ⚠ status — power users may
want to export a draft).

## Why three verbs, not two or four

**Why not two:** Linda's flow is too different from Sarah/Marcus's to
collapse into one Run. Save-for-review is structurally distinct — it
produces a different artifact and means a different thing.

**Why not four:** I considered splitting Run into "Run analysis" (Sarah)
vs "Execute" (Marcus). Both produce the same backend action; the
difference is purely cosmetic (result rendering). A single Run verb with
context-aware result rendering captures this without forcing the user to
pick between two synonyms.

Dev's "Copy YAML" is the same action as Sarah/Linda's "Export YAML
to share with colleague" — single verb, different downstream use.

## What happens after each gesture

| Verb | Immediate result | Next action available |
|---|---|---|
| Save for review | Confirmation toast + shareable link | "Open in a new session" / "Share via email" / "Run anyway" |
| Run pipeline | Progress indicator → results view | "View audit trail" / "Re-run with changes" / "Download output" |
| Export YAML | YAML in hand | "Run via CLI" snippet / "Save as a new file" |

## Layout and visual hierarchy

The completion bar lives in the side rail, **below** the audit-readiness
panel and the graph mini-view, but **above** the Catalog and Export YAML
buttons (Export YAML is *also* a completion gesture; it appears in both
places, since some users will find it in the side rail menu and others
will find it in the completion bar).

```text
  ┌─ side rail ───────────────────────────┐
  │  AUDIT READINESS                       │
  │  GRAPH (mini)                          │
  │  ─────────────────                     │
  │  [💾 Save for review]                   │  ← completion bar
  │  [▶  Run pipeline   ]                   │
  │  [⬇  Export YAML    ]                   │
  │  ─────────────────                     │
  │  [📋 Catalog       ]                    │  ← reference / discovery
  └────────────────────────────────────────┘
```

No single button is "primary" via color/size emphasis. The audit-readiness
panel above it carries the priority signal — a green panel means all
three are safe to use; a yellow/red panel means the user should resolve
the issue before any completion gesture.

## Mode independence

The completion bar is **the same in guided and freeform modes**. Both
modes produce the same underlying composition state, so both modes can
offer the same completion gestures.

The completion bar appears once the composition has enough structure to
be runnable. In guided mode, this is typically after the last wizard step
(or once the user reaches the "Ready" state). In freeform mode, this is
once a source-transform-sink chain has been composed.

## Persona × completion-gesture summary

This restates the matrix from
[02-personas-and-audiences.md](02-personas-and-audiences.md) for
convenience:

| Persona | Primary | Secondary |
|---|---|---|
| Linda | Save for review | Export YAML (to attach to email) |
| Sarah | Run pipeline (with narrative results) | Save for review (if she wants colleague to verify before run) |
| Marcus | Run pipeline | None |
| Dev | Export YAML | None |

## Implementation notes

| Component | Touch-point |
|---|---|
| Backend | New endpoint: `POST /api/sessions/{id}/mark-ready-for-review` — flips a flag on the session and generates a signed snapshot of the composition state |
| Backend | New endpoint: `GET /api/sessions/{id}/shareable-link` — returns a URL that another authenticated user can open to inspect a saved-for-review session |
| Backend | Optional: extend audit recorder to log "marked for review" events |
| Backend | Run pipeline endpoint: no changes (existing Execute path) |
| Backend | Export YAML: no changes (existing path) |
| Frontend | New completion bar component in side rail |
| Frontend | "Run pipeline" result rendering: detect presence of `batch_*` analytic transforms; show narrative summary if present, table preview otherwise |
| Frontend | "Save for review" confirmation + shareable link dialog |

## Risks

| Risk | Mitigation |
|---|---|
| Three buttons looks cluttered | Group visually as a single "Done with this?" cluster; lighter button styling than full primary buttons |
| Users don't know which to pick | Tooltip explanations; the audit-readiness panel above sets context ("All checks pass — safe to run") |
| Save-for-review semantics are unclear without a multi-user culture | The first version targets single-user-with-colleague flows; org-level review queues are out of scope |
| Run pipeline result rendering complexity | Start with the simplest split: presence of `batch_distribution_profile` / `batch_classifier_metrics` etc. triggers narrative mode. Tune over time. |
| "Save for review" without a real review surface on the other end | Initial implementation: the link opens the session in read-only inspect mode. A full reviewer surface (with accept/reject) can come later. |

## Memory references

- `project_composer_personas` (the four completion verbs)
- `project_composer_two_audiences` (Linda's compose-then-hand-off journey)
