# 03 — Target Information Architecture

This document specifies the target state of the composer's surfaces. It is
the central design artifact of the redesign — every later document
elaborates a surface or feature defined here.

## Layout sketch

The target layout keeps the high-level shape of the current UI (header,
main work area, side rail) but rebalances what each region carries.

```text
┌──────────────────────────────────────────────────────────────────────┐
│  ELSPETH  ▾ Session: cool-government-pages-1            👤 Settings ☾│  ← thin header:
│                                                                       │     session switcher,
├──────────────────────────────────────────────────────────────────────┤     user menu, theme
│                                                                       │
│  GUIDED MODE STEPPER (when in guided mode):                           │     ← visible only in
│  Source ▸ Sink ▸ Recipe ▸ Transforms ▸ Ready                          │       guided mode
│                                                                       │
├───────────────────────────────────────────┬──────────────────────────┤
│                                            │                          │
│   PRIMARY WORK AREA                        │  ┌─ AUDIT READINESS ──┐ │
│   (current decision turn, or freeform      │  │  Validation:   ✓   │ │
│    chat stream)                            │  │  Trust tiers:  ✓   │ │
│                                            │  │  Provenance:   ⚠   │ │
│   Includes:                                │  │  Retention:    —   │ │
│   • Per-step chat scrollback               │  │                    │ │
│   • Turn widget (guided) or messages       │  │  [Explain →]       │ │
│     (freeform)                             │  └────────────────────┘ │
│   • Inline proposal accept/reject          │                          │
│                                            │  ┌─ GRAPH (mini) ─────┐ │
│                                            │  │  src → tx → sink   │ │
│                                            │  └────────────────────┘ │
│                                            │                          │
│                                            │  [📋 Catalog (reference)│
│                                            │  [⬇  Export YAML]      │
│   ┌──────────────────────────────────┐    │                          │
│   │  Chat input — placeholder primes  │    │  COMPLETION BAR:        │
│   │  dynamic-source-from-chat...      │    │  [Save for review]      │
│   └──────────────────────────────────┘    │  [Run pipeline →]       │
└───────────────────────────────────────────┴──────────────────────────┘
```

Notes:
- No always-on session sidebar. The session switcher is a header dropdown.
- No Spec / Runs tabs separate from the inspector. The audit-readiness panel
  and graph mini-view live persistently in the side rail; the run results
  appear inline in the primary work area after Execute fires.
- No separate Validate button. The audit-readiness panel's validation row
  carries that signal.
- Completion bar offers persona-appropriate verbs side-by-side; the user
  picks the one that matches their intent.

## Surface inventory

The table below catalogs every meaningful surface and gives the verdict.
Items marked **NEW** do not exist in the current UI; items marked
**REMOVED** are in the current UI but go away.

| Surface | Current state | Target state | Why |
|---|---|---|---|
| **Session switcher** | Always-on 200px sidebar | Header dropdown | Persona need is "pick a session occasionally," not "browse my sessions all the time." See [02-personas-and-audiences.md](02-personas-and-audiences.md) §matrix. |
| **Theme toggle** | Top-left, in collapsed-sidebar mode | Top-right, in user menu | Cheap; centralizes account-related controls. |
| **User menu (new)** | Implicit via login | Top-right dropdown — settings, account, sign out, **default-mode preference** | The opt-out for default-guided needs a permanent home. See [05-modes-and-opt-out.md](05-modes-and-opt-out.md). |
| **Guided stepper** | Top of chat panel when in guided mode | Same place, unchanged | Works. |
| **Chat stream** | Center column | Center column | Primary work area unchanged in shape. |
| **Chat input** | Bottom of center column | Same place, with **primed placeholder** for dynamic-source-from-chat | See [06-chat-as-data-entry.md](06-chat-as-data-entry.md). |
| **Per-step chat** (guided mode) | Sidecar below main widget | Same | Works. |
| **Turn widgets** (SingleSelect, etc.) | In chat stream | Same | Works. |
| **Pending proposal banner** | Above chat input | Same | Works. |
| **Audit-readiness panel** | Does not exist | **NEW** — persistent in side rail | Linda's load-bearing surface. See [07-audit-readiness-panel.md](07-audit-readiness-panel.md). |
| **Graph view (mini)** | Inspector tab | Persistent in side rail, small | Verification surface; valuable continuously, not on-demand. |
| **Graph view (full)** | Inspector tab | Modal / expanded view on click of mini | Edge cases (large pipelines) deserve the full view; common case is the mini. |
| **YAML view** | Inspector tab | "Export YAML" button in side rail; opens a modal | Reframes from "code preview" to "export artifact." |
| **Spec tab** | Inspector tab | **REMOVED** | Leaky abstraction; no documented persona needs it. |
| **Runs tab** | Inspector tab | **REMOVED as tab**; run results appear inline after Execute | Compose vs run separation respected; results land where the user just clicked Run. |
| **Validate button** | Inspector header | **REMOVED** | Theater — the indicator already does the work. |
| **Execute button** | Inspector header | Moved to **completion bar** in side rail | Persona-aware completion. |
| **Catalog button** | Inspector header | Side rail, below export YAML | Kept; reshaped as reference. See [08-catalog-reshape.md](08-catalog-reshape.md). |
| **Catalog drawer** | Right-edge slide-over | Same, with **reference framing** (search-first, no select-this) | See [08-catalog-reshape.md](08-catalog-reshape.md). |
| **Validation indicator dot** | Inspector header | Subsumed into audit-readiness panel | Single richer surface. |
| **Version selector (v{N})** | Inspector header | Header, next to session name; **renamed** to "Composition history" | Operates on the session, not the inspector. |
| **Templates / starting cards** | First-empty-chat screen | **REPLACED** with audit-domain exemplars from README's `Example Use Cases` table | Same shape, better content. |
| **Switch-to-guided / Exit-to-freeform** | Chat panel header | Same place — visible button when not in default mode | Mode toggle is per-session; preference is per-account. |
| **Save-for-review** | Does not exist | **NEW** completion verb in side rail | Linda's primary path. See [09-completion-gestures.md](09-completion-gestures.md). |
| **Run / Execute** | Inspector header | Completion bar; named "Run pipeline →" | Same action, persona-aware framing. |
| **Export YAML** | Inside YAML tab | Top-level side-rail button | First-class action. |
| **Hello-world tutorial** | Does not exist | **NEW**, first-session-only | See [04-first-run-tutorial.md](04-first-run-tutorial.md). |
| **Command palette (Ctrl+K)** | Global keyboard shortcut | Same | Audit its inventory after IA change (see [11-open-questions.md](11-open-questions.md)). |
| **Keyboard shortcuts** | App.tsx:86-179 | Audit + retain useful ones; drop those tied to removed surfaces (e.g. Ctrl+Shift+P for Catalog stays but the shortcut neighborhood is different) | See [11-open-questions.md](11-open-questions.md). |
| **Blob manager** | Toggle in chat input toolbar | Same | Out of scope; flag for separate review re: provenance treatment. |

## Three behaviour modes

The composer operates in three top-level modes:

1. **First-run tutorial** — new account, never composed. Forced sequential
   tutorial; specified in [04-first-run-tutorial.md](04-first-run-tutorial.md).
2. **Guided** — default for new sessions (unless the user has opted out).
   Step-by-step wizard with the workflow stepper visible. Detailed in
   [05-modes-and-opt-out.md](05-modes-and-opt-out.md).
3. **Freeform** — for users who opted out of guided default, or who opted
   in per-session. Chat-driven authoring with the same audit-readiness and
   graph surfaces visible.

The audit-readiness panel, graph mini-view, completion bar, and Catalog
button are **mode-independent** — they look the same and behave the same
in all three modes.

## What's persistent vs on-demand

| Visibility | Surface |
|---|---|
| **Persistent** (always on-screen when composing) | Header, audit-readiness panel, graph mini-view, completion bar (when pipeline is composable), chat input, Catalog button, Export YAML button |
| **Mode-conditional** | Guided stepper (guided only), per-step chat sidecar (guided only), turn widgets (guided only), freeform chat stream (freeform only), switch-mode affordance |
| **Action-triggered** | Catalog drawer (on click), YAML export modal (on click), graph full view (on click of mini), run results (after Execute fires) |
| **First-run only** | Tutorial |

## Density and breakpoints

The target layout assumes a desktop authoring surface (>1024px). Behaviour
at narrower widths is out-of-scope for this redesign except: the
audit-readiness panel and graph mini-view should remain visible at
≥1280px and collapse to a single combined "Status" tab below that. This
is intentionally conservative — the composer is not optimized for narrow
screens and forcing it to be so would punish the desktop case for an
audience that doesn't exist in the personas.

## What's NOT in the IA

These are deliberate omissions and are flagged so future reviewers don't
"helpfully add them back":

- **A persistent inspector tab strip.** The Spec / Graph / YAML / Runs
  arrangement is gone. Each former tab has a different home (deleted /
  side rail / button / inline after Execute).
- **A separate "saved pipelines" library.** Sessions are the unit of
  persistence; the session switcher is the only library surface.
- **An admin / org settings panel inside the composer.** Org-level config
  lives elsewhere; the composer's settings menu is user-level only.
- **A "running pipelines" dashboard.** Ad-hoc runs surface their results
  inline; longer batch runs are an operator concern outside this UI.
- **A debug mode toggle.** No documented persona needs it. If engine
  diagnostics are useful, surface them in the audit-readiness panel's
  "Explain" view, not as a separate mode.

## Companion documents

- [04-first-run-tutorial.md](04-first-run-tutorial.md)
- [05-modes-and-opt-out.md](05-modes-and-opt-out.md)
- [06-chat-as-data-entry.md](06-chat-as-data-entry.md)
- [07-audit-readiness-panel.md](07-audit-readiness-panel.md)
- [08-catalog-reshape.md](08-catalog-reshape.md)
- [09-completion-gestures.md](09-completion-gestures.md)
- [10-implementation-phasing.md](10-implementation-phasing.md)
