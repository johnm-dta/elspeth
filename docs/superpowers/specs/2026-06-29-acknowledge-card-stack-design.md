# Acknowledge card stack — guided interpretation reviews redesign

**Date:** 2026-06-29
**Status:** Design (awaiting approval → implementation plan)
**Scope:** Frontend presentation only. No sub-web-layer change.

## Problem

The guided composer surfaces LLM-authored decisions ("interpretation reviews")
for the operator. Today they render inline in the chat turn as verbose,
whitespace-heavy panels: an `<h3>` heading + a separate sentence + a large
`16rem` bordered box even for one-line values, with primary buttons labelled
"Use prompt template / Use this model / Use pipeline decision".

Three problems, from a live review of the transforms-stage screenshot:

1. **Wrong frame.** "Use X" reads as an *approval gate*, but there is no "No" —
   the only path is acknowledge-then-change-via-conversation. The data model
   already encodes exactly this (`accepted_as_drafted` vs `amended`); the copy
   misrepresents it.
2. **Wasted space.** Heading + body + oversized boxes make four short decisions
   fill the viewport. Short values (a model id, a one-line decision) get a
   scrollable code box sized for paragraphs.
3. **Opaque + plain.** Internal node ids (`guided_xform_1`) leak into
   user-facing copy; structured values (invented source data) render as flat,
   un-highlighted text.

## Goals

- Reframe as **acknowledgement**, not approval: one `Acknowledge` button per
  decision; acknowledging clears the card so the operator can proceed.
- **Compact, card-based, top-of-view** placement: pending decisions appear as a
  stack of cards pinned at the top of the chat column; the stack disappears when
  empty.
- **Punchy, LLM-attributed copy** with **humanised step names**.
- **Pretty, colour-coded values**: JSON pretty-printed and syntax-highlighted;
  long prompt templates in clean structured monospace behind a `View` expander.
- Fix the AA-contrast risk in the muted secondary text / placeholder as part of
  the restyle.

## Non-goals (the "sub-web layer" stays untouched)

- No change to the resolve endpoint (`POST …/interpretations/{id}/resolve`),
  `InterpretationResolveRequest`, the `InterpretationChoice` values, or the
  audit-primary `interpretation_events` write. **Acknowledge == today's accept
  (`accepted_as_drafted`).**
- No change to `useInterpretationResolver`, the `interpretationEventsStore`
  actions (`resolveEvent` / `optOut` / refresh), the 8 KB amend byte cap, or the
  error-mapping. Behaviour is reused verbatim; only rendering changes.
- No backend, contract, migration, or `audit-readiness` change.

## Design

### Placement & structure

A new container renders the pending reviews as a **pinned stack at the top of
the chat column**, driven by the existing `pendingBySession[sessionId]`
projection. One card per pending event. When the projection empties, the
container renders nothing and the conversation is unobstructed ("clear it to
proceed"). This **replaces** the inline mounts in both modes:

- Guided: today via `GuidedTurn` / `GuidedInterpretationReviews`.
- Freeform: today via `ChatPanel` inline `InterpretationReviewInlineMessage`.

Both modes unify on the single top stack so the surfaces cannot drift.

New / changed components:

- `AcknowledgementStack` (new) — subscribes to the pending projection, renders
  the header ("N decisions the LLM made — acknowledge each"), the ordered cards,
  and ONE foot-of-stack opt-out link. Ordering: by pipeline step then
  `created_at` (stable).
- `AcknowledgementCard` (new, extracted from `InterpretationReviewTurn`'s
  presentation) — one decision. Uses `useInterpretationResolver` unchanged.
- `InterpretationReviewTurn` / `InterpretationReviewInlineMessage` are retired
  as mount points (logic already lives in the hook); their behaviour-bearing
  pieces (scroll-gate, focus rules, ARIA) move into the card / stack.

### The card

Layout, compact:

```
┌───────────────────────────────────────────────┐
│ Summarise step · model                          │
│ The LLM picked openai/gpt-4o-mini  [ Acknowledge ]│
└───────────────────────────────────────────────┘
```

- **Title row:** humanised step label · kind (e.g. "Summarise step · prompt").
- **Body:** one punchy LLM-attributed line.
- **Value:** inline for short values; behind a `View` expander for long ones.
- **Action:** right-aligned primary `Acknowledge` (existing accept handler),
  with the existing in-flight spinner ("Saving…") and inline error banner.
- **Amend:** kinds that support it today (`vague_term` / null) keep the inline
  "Change…" edit; other kinds change via the conversation (no inline edit) —
  unchanged from today.

### Copy (per-kind rewrite of `getReviewPresentation`)

| kind | title | line |
|---|---|---|
| `llm_prompt_template` | `<step> · prompt` | "The LLM wrote the instruction for this step." (value behind `View`) |
| `pipeline_decision` | `<step> · decision` | the decision text (`llm_draft`) inline |
| `llm_model_choice` | `<step> · model` | "The LLM picked `<model>`." |
| `invented_source` | "Source data" | "The LLM invented this source data — review before fetching." (value pretty-printed) |
| `vague_term` / null | "Interpretation" | "You said '<term>'; the LLM read it as '<draft>'." (+ inline Change) |

**Humanised step name:** resolve `affected_node_id` → label via the node's
plugin in the current composition state (`llm`→"Summarise", `web_scrape`→"Fetch",
`field_mapper`→"Output", …), falling back to the raw id when the node is absent.
Presentational only (reads existing store state).

### Value rendering ("prettify text fields")

A shared `CodeBlock` component built on the **already-present
`prism-react-renderer`** (the same highlighter `MarkdownRenderer` uses):

- **JSON-shaped values** (e.g. invented source data): `JSON.parse` → 2-space
  `JSON.stringify` → Prism `json` grammar (colour-coded). On parse failure, fall
  back to plain monospace (never throw; never fabricate structure).
- **Prompt template:** clean structured monospace `<pre>`; the existing
  *scroll-to-end-before-acknowledge* review gate moves **inside** the `View`
  expander.
- **Model id / decision sentence:** plain inline text (no code box).

### Accessibility

- `role="status"` live-region announces the stack on appearance and on
  count change ("N decisions to acknowledge").
- Real `<button>`s with `aria-label`s naming the decision (kept from today).
- **Do not auto-steal focus.** Unlike today's guided card (focus-on-mount), a
  persistent top stack must not yank focus from someone typing; announce only.
- Scroll-gate's `aria-describedby` hint preserved within the expander.
- Stack and cards fully keyboard reachable; `View` is a real toggle button.

### Styling

- Design-system tokens; compact padding; remove the `16rem` box for short
  values. Fix the muted secondary-text / placeholder contrast to pass
  `colorContrast.test.ts` (AA). Mirror any token change into the website palette
  per the shared-palette rule if a token value changes (likely not needed —
  prefer existing AA-safe tokens).

## Testing

Frontend only (no backend test changes — mechanics unchanged):

- Port/extend `GuidedInterpretationReviews.test.tsx` and the
  `InterpretationReviewTurn` behaviour tests onto `AcknowledgementStack` /
  `AcknowledgementCard`: acknowledge → resolve called with
  `accepted_as_drafted`; vague-term amend → `amended`; opt-out flow; error
  mapping (409/422/other); prompt scroll-gate disables Acknowledge until scrolled.
- New: stack renders one card per pending event, ordered; empties to nothing;
  announces count via `role="status"`; does NOT move focus on mount.
- New: `CodeBlock` pretty-prints + highlights valid JSON; falls back to plain
  monospace on invalid JSON.
- a11y assertions (jest-axe) on the stack; `colorContrast.test.ts` stays green.

## Risks / open items

- **Focus model** in a persistent stack — announce-only is the chosen default;
  revisit if keyboard users report they can't find the stack (a "review N
  decisions" skip-link is a fallback).
- **Freeform relocation** — moving freeform reviews from inline to the top stack
  changes their in-conversation context; acceptable given the unify decision,
  but called out for review.
- **Card ordering** — by pipeline step then created_at; confirm step order is
  derivable from the composition state.
