# Guided decision: read-only summary + lead-with-rationale

**Date:** 2026-06-29
**Status:** Design (awaiting approval → implementation)
**Scope:** Frontend presentation/interaction. No sub-web-layer change.

## Problem

The guided "Current decision" surface (`SchemaFormTurn`, rendered inside
`ChatPanel`'s guided branch) presents a plugin's config as a **full-width
editable form** — every knob a live `<input>`/`<select>`/`<textarea>`, each with
a help line — even in the passive tutorial where the learner edits nothing. From
a live source-stage screenshot:

1. **Poor space use.** Short values (`discard`, `utf-8`, `0`, `,`) each occupy a
   full-width input + description line; the box is very tall and very wide.
2. **Editability illusion.** The fields render as enabled inputs, but the passive
   learner cannot/should not edit them — the UI lies about what's interactive.
3. **Buried rationale.** The dynamic per-build explanation (the LLM's "Source
   created as a 3-row CSV…", a `ChatTurn`) sits as a small muted transcript line
   *above* the giant form; the prominent thing should be *why this was built*,
   not an edit grid. (The bottom "worked example / discard" line is **static**
   teaching copy — `TUTORIAL_VALIDATION_FAILURE_CAVEAT` — not dynamic.)

## Goals (confirmed with operator)

- **Read-only summary by default for ALL guided decisions** (not tutorial-only):
  the decision renders as a compact, read-only summary; the interactive composer
  exposes an **Edit** affordance to switch to the existing editable form.
- **Lead with the dynamic build rationale**, prominent and larger.
- **Demote the static caveat** to a small contextual note beside its field.

## Non-goals (sub-web layer unchanged)

- No change to the guided `/respond` contract, the `GuidedRespondRequest` submit
  shape, `canSubmit`/validation semantics, or any backend. The editable form is
  **preserved verbatim** behind the Edit toggle; "Continue" submits the same
  values it does today (prefilled when unedited, edited values when edited).
- No change to the run/resolve/audit paths.

## Design

### `SchemaFormTurn` — summary-first with an edit toggle

A new **default summary view** for both modes (`plugin_options`,
`recipe_decision`):

- **Compact read-only summary**: a definition list of the visible knobs —
  `field.label: value`. Short scalars inline; JSON/object values through the
  shared **`CodeBlock`** (`prettyJson`, the component built for the acknowledge
  stack); `string-list` as comma/line-joined text; checkbox as Yes/No; enum as
  the chosen option. The `path` knob reuses `friendlyBlobRef` so the absolute
  blob storage_path never leaks (the existing tutorial mask, now applied in the
  summary for every mode).
- **Edit toggle (non-tutorial only):** an `Edit` button switches the summary to
  the existing editable `KnobFieldRenderer` form (unchanged). A `Done editing`
  control returns to the summary; submitting also exits edit. **Tutorial mode
  renders no Edit button** — passive learners get summary + Continue only.
- **Primary unchanged:** `Continue` / `Apply recipe` submit exactly as today
  (`handleContinue` reads `values`, which equal `prefilled` until edited). The
  required/validity gate (`canSubmit`) is unchanged; in summary mode the values
  are the prefilled ones, which the backend already produced as valid.
- **Static caveat demoted:** `TUTORIAL_VALIDATION_FAILURE_CAVEAT` renders as a
  small contextual note attached to the `on_validation_failure` summary row, not
  a full-width paragraph above Continue.

### Lead with the dynamic rationale (`ChatPanel` guided branch)

Surface the current step's **latest assistant `ChatTurn`** (from
`guidedSession.chat_history`, filtered to `role === "assistant"` and the current
`step`) as the **decision headline** — prominent and larger — at the top of the
`guided-current-decision` block, above the summary. This replaces the small
muted transcript echo as the thing the eye lands on. When no assistant rationale
exists yet for the step (e.g. a server-emitted turn), fall back to the existing
`GUIDED_STEP_PURPOSES[step]` copy (no empty headline).

### Styling / a11y

- Summary is a real `<dl>` (or labelled rows); the Edit/Done toggles are real
  `<button>`s; Continue stays reachable and keyboard-operable.
- Design-system tokens; compact spacing; `colorContrast` stays green.
- The headline is a heading-level element so AT users land on the rationale.

## Testing (frontend only)

- `SchemaFormTurn`: summary view renders each visible knob read-only with its
  value; JSON via `CodeBlock`; path masked; static caveat beside
  `on_validation_failure`.
- Edit toggle: non-tutorial reveals the editable form and back; **tutorial shows
  no Edit button**.
- Submit parity: `Continue` from the summary (unedited) submits the prefilled
  values; editing then continuing submits the edited values — same
  `GuidedRespondRequest` shape as today (assert the onSubmit payload unchanged).
- `ChatPanel`: the decision headline renders the latest assistant rationale for
  the step; falls back to the step purpose when absent.
- a11y (jest-axe) + `colorContrast` stay green.

## Risks

- **Interactive editing is now one click behind a summary.** Confirmed in-scope
  by the operator ("all guided decisions"). The Edit affordance must be obvious
  so power users aren't surprised the form isn't immediately editable.
- **Rationale sourcing**: relies on `chat_history` carrying an assistant turn for
  the step; the `GUIDED_STEP_PURPOSES` fallback covers server-emitted/empty cases
  so the headline is never blank.
