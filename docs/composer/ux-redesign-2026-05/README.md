# Composer UX Redesign — Planning Document Set (2026-05-15)

This directory contains the design rationale and target-state plans produced
during a first-principles review of the ELSPETH Web Composer UI on branch
RC5.2. The review was triggered by repeated reviews "waving through" legacy
decisions (notably the Catalog button) instead of relitigating them, and was
grounded in the README's two-audience model and the four documented composer
personas in `evals/`.

The early numbered documents are the result of that review. They are target
state, not a description of the live UI. Later phase documents in this same
directory record the RC5.2 implementation plan and follow-up work; treat the
current phase documents and Filigree as the source of execution truth.

## How to read this set

Documents are numbered for reading order. The recommended sequence:

| # | Document | What it covers |
|---|---|---|
| 01 | [Design rationale](01-design-rationale.md) | Why this redesign exists; what's wrong with the current UI; what changed during the review |
| 02 | [Personas and audiences](02-personas-and-audiences.md) | The README's two-audience model and the four composer personas (Linda / Sarah / Marcus / Dev) |
| 03 | [Target information architecture](03-target-information-architecture.md) | Surface-by-surface inventory of what stays, what changes, and what's removed |
| 04 | [First-run tutorial](04-first-run-tutorial.md) | Hello-world tutorial spec — turn-by-turn arc using the canonical test prompt |
| 05 | [Modes and opt-out](05-modes-and-opt-out.md) | Default-guided with persistent per-user opt-out; three-scope model |
| 06 | [Chat as data entry](06-chat-as-data-entry.md) | Dynamic-source-from-chat + the "surface the LLM's interpretation" affordance |
| 07 | [Audit-readiness panel](07-audit-readiness-panel.md) | Persistent audit-visibility surface written in Linda-vocabulary |
| 08 | [Catalog reshape](08-catalog-reshape.md) | Catalog as searchable reference, not interactive toolkit |
| 09 | [Completion gestures](09-completion-gestures.md) | Run / Save-for-review / Export YAML — persona-aware completion verbs |
| 10 | [Implementation phasing](10-implementation-phasing.md) | Recommended sequence; what can ship independently vs together |
| 11 | [Open questions](11-open-questions.md) | Product decisions that need adjudication before or during implementation |

## Scope

**In scope:** the Web Composer authoring surface in `src/elspeth/web/frontend/`
and its backing API in `src/elspeth/web/composer/`. The hand-edited YAML path,
the CLI, and the engine itself are out of scope except where the composer's
target state requires small additions (e.g., recording the user's accepted
interpretation of subjective LLM terms in the audit trail).

**Not in scope:** wholesale redesign of the inspector, the chat protocol, or
the session-state model. Where these touch the recommendations, the docs note
the touch-point and leave the implementation strategy to a follow-up.

## Audiences for this document set

- **Operator / product owner** — adjudicating the recommendations and choosing
  what to commit to. Read 01 → 02 → 03 → 11 first.
- **Implementer (frontend or backend)** — translating committed recommendations
  into work. Read 03 → the relevant feature doc → 10.
- **Reviewer / auditor of this design work** — sanity-checking the
  first-principles reasoning. Read 01 → 02 → 11.

## Status

As of 2026-05-15, this set captures the closing state of a multi-turn design
conversation between the operator and the reviewing assistant. The
recommendations have been adjudicated by the operator on several points
(catalog reshape, two-audience model, default-guided, hello-world tutorial,
dynamic-source-from-chat) and remain open for adjudication on the rest.

See [11-open-questions.md](11-open-questions.md) for the explicit decisions
still pending.

Generated review sidecars for these plans were removed from active docs during
the 2026-05-19 cleanout. Findings that still matter are folded into the paired
plan files.

## Memory persistence

This work has seven persistent memory entries in
`~/.claude/projects/-home-john-elspeth/memory/`:

1. `feedback_catalog_is_reference_not_toolkit`
2. `project_composer_two_audiences`
3. `project_composer_personas`
4. `project_composer_default_guided_with_opt_out`
5. `project_composer_first_run_tutorial`
6. `project_composer_dynamic_source_from_chat`
7. `project_composer_canonical_test_case`

Future review sessions inherit these and should start from them rather than
re-deriving the framing.
