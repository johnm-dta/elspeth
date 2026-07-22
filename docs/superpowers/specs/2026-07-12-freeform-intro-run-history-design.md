# Freeform Introduction and Run History Design

**Date:** 2026-07-12
**Status:** Approved
**Selected direction:** Lyra UX mockup A, “Calm guide card”

## Goal

Replace the interactive freeform starter-example splash with a short, quiet introduction that an account can permanently dismiss. Improve run-history identity so people recognise runs by sequence and time before relying on a UUID.

## Freeform empty state

The empty freeform conversation renders one static, scannable operator primer.
It introduces ELSPETH's vocabulary without adding disclosures, carousels,
template selection, or other interaction:

- Heading: **How pipelines work**
- Opening: “A pipeline is a controlled route for information. You choose what
  enters, what happens to it, and where the result goes. ELSPETH records each
  step so you can review how every output was produced.”
- **The three building blocks**:
  - **Sources** bring records into the pipeline from files, databases, APIs, or
    text. ELSPETH tracks each incoming record through the run.
  - **Transforms** examine or change records. They can clean fields, enrich
    content, apply an LLM, or prepare data for the next step.
  - **Sinks** receive records at the end of a route. They can write results to
    files, data stores, or other configured destinations; records requiring
    attention can follow a separate route.
- **Wiring the flow**: “Wiring is the set of connections between these
  components. A simple pipeline runs from source to transforms to sink. For a
  more involved flow, think of each record as a case moving through a
  controlled workplace:”
  - **Gate** is a sorting desk. It sends each case along the appropriate route
    according to a stated condition.
  - **Fork** sends controlled copies of one case to several specialist teams.
    ELSPETH tracks each parallel path independently.
  - **Coalesce** waits for the required specialist responses, then combines
    their findings into one case that can continue.
  - **Aggregate** brings a group of cases together for batch work, such as
    producing totals, statistics, or a report.
  - **Queue** is a shared inbox. It accepts cases from several upstream teams
    and feeds one next step while keeping every case separate.
  - **Expand** opens a bundled case into several independently tracked cases.
- Closing: “Describe the outcome you need in ordinary language. ELSPETH will
  propose the components and their wiring; review the graph and details before
  you run it.”
- Secondary action: **Don’t show this again**

The content uses headings and a definition list so an operator can scan it
without reading every sentence. The card remains visually quiet and bounded;
it may grow wider than the original brief card but must not become a full-page
welcome splash.

The template-card grid, starter-examples disclosure, and all example-selection interaction are removed from this surface. After dismissal, an empty freeform conversation renders no replacement artwork or placeholder; the conversation background remains blank and the composer stays available.

The card appears only when all of these are true:

1. the active surface is freeform;
2. the conversation has no messages;
3. account preferences loaded successfully; and
4. `freeform_intro_dismissed_at` is null.

If preferences cannot be loaded, the introduction stays hidden to avoid a flash for accounts that may already have dismissed it. Existing preference-error UI remains the failure explanation.

## Account-wide dismissal

Add nullable `freeform_intro_dismissed_at` to the account-level composer-preferences contract and `user_preferences` table. It follows the existing partial-PATCH semantics used by `banner_dismissed_at`: absent means unchanged, an ISO-8601 timestamp records dismissal, and explicit null remains a supported API reset even though this change adds no user-facing reset control.

On activation, the dismissal control:

1. becomes disabled and changes its label to **Hiding…**;
2. PATCHes the server with the current timestamp;
3. removes the card only after the server confirms the write; and
4. leaves the card visible and relies on the existing preferences error surface if the write fails.

The preference is account-scoped, so dismissal follows the user across sessions, browsers, and devices. Store bootstrap and cross-tab reconciliation treat the new field the same way as other account preferences.

## Run history

Replace the oversized **Runs** header and standalone Close button with a compact header:

- label: **Run history · N**;
- an accessible close control at the trailing edge; and
- no UUID in the heading.

Each row uses this hierarchy:

1. primary label: `Run <ordinal> · <local date and time>`;
2. secondary metadata: the full run UUID in muted monospace text;
3. status badge; and
4. existing Cancel or Show/Hide detail actions.

Ordinals are deterministic within the current session. Runs are sorted by `started_at` ascending with `id` as the tie-breaker to assign `Run 1…N`, then displayed newest first. Date/time uses the user’s browser locale while retaining the complete date, year, and time. Accessible names for detail and cancel actions use the human label plus UUID discriminator.

On narrow containers, metadata and actions wrap below the primary label without horizontal overflow. Status remains text-and-glyph based rather than colour-only.

## Components and data flow

- Replace `TemplateCards` in `ChatPanel` with a focused freeform-introduction component controlled by `preferencesStore`.
- Extend the backend preference model, table schema, service read/write projection, API payloads, and frontend preference types/store with `freeform_intro_dismissed_at`.
- Remove template-card code and data if no other runtime consumer remains.
- Add run-label formatting and deterministic ordinal derivation beside `RunsHistoryDrawer`; keep diagnostics, outputs, cancellation, focus trapping, and Escape-to-close behaviour unchanged.
- Reuse existing design tokens, Button primitives, status badges, preference error handling, and responsive CSS conventions.

## Verification

- Backend model, schema, service, and route tests cover default null, timestamp persistence, absent-field preservation, explicit-null reset, and account isolation.
- Frontend store tests cover bootstrap, successful dismissal, failed dismissal, and persistence across a fresh render.
- Empty-state component and `ChatPanel` tests confirm the intro gating, operator
  primer headings and terms, disabled saving state, removal after success,
  blank dismissed state, and absence of template cards.
- Run-history tests confirm deterministic ordinals, newest-first display, locale-safe complete timestamps, UUID secondary metadata, accessible action names, empty history, and live-run cancellation.
- Accessibility tests cover the introduction and updated run-history dialog.
- Playwright verifies desktop and narrow layouts, persistent dismissal after reload, blank dismissed freeform state, readable run rows, keyboard focus, and zero console errors.

## Out of scope

- A new template gallery elsewhere in the product.
- A user-facing “show introduction again” setting.
- Changes to run storage, run IDs, diagnostics, cancellation, or output retrieval.
- Relative-only timestamps such as “Today” that lose screenshot and audit context.
