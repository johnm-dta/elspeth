// ============================================================================
// InlineSourceDisambiguationTurn.tsx — Phase 5a Task 4
//
// Pre-success interactive widget rendered when the composer LLM has
// proposed an inline-blob source whose row count is ambiguous — e.g., the
// user wrote "check these URLs: a.com, b.com, c.com" and the LLM
// interpreted it as 3 rows rather than 1. The widget surfaces both the
// user's original input and the LLM's parsed row breakdown, then offers
// four explicit actions: confirm the multi-row interpretation, fall back
// to a single-row interpretation, edit the rows directly, or escape
// entirely ("this isn't source data").
//
// Cousin of InlineSourceCreatedTurn (Phase 5a Task 3). KEY DIFFERENCE:
// the created-turn widget is post-success informational (the inline_blob
// has already been attached to the composition state); this widget is
// pre-success interactive (the proposal is pending and will not commit
// until the user picks one of the four actions). For that reason the
// disambiguation widget never writes to inlineSourceStore.summariesBySession
// — that population happens after a "Yes — N rows" lands the proposal
// through the standard accept flow. What this widget DOES drive is the
// disambiguation re-fire guards in inlineSourceStore:
//   * F-10 "this isn't source data" → addNonSourceMessage(messageId)
//   * F-11 "treat as 1 row"         → addUserRequestedSingleRow(messageId)
//
// Design constraints (load-bearing — do not change without re-reading the
// Phase 5a Task 4 spec):
//
//   * Root element is `<section role="region" aria-label="...row count
//     interpretation (N rows)">`. AT users navigate to the
//     disambiguation surface by region role + the substring "row count"
//     in the accessible name. The aria-label substring "row count" is
//     matched by the widget test and by the ChatPanel wiring tests —
//     keep it stable.
//
//   * Action button accessible names are a CLOSED LIST (see ACTIONS
//     comment below). The names are the contract between the widget,
//     the ChatPanel wiring, and the test suite. Renaming a button
//     ("Edit the rows" → "Edit list") is a spec amendment, not a
//     refactor.
//
//   * Focus management (F-19): on mount, move focus to the primary
//     action button via useEffect + ref. This satisfies the "new turn
//     receives focus so keyboard users don't have to Tab from the top
//     of the chat panel" contract that the same wiring solves for
//     guided-mode turn advances in ChatPanel.tsx.
// ============================================================================

import { useEffect, useRef } from "react";

/**
 * Action surface for this widget.
 *
 * CLOSED LIST — four actions, each wired to one accessible-name pattern
 * tested in InlineSourceDisambiguationTurn.test.tsx. Adding a fifth
 * action would mean a spec amendment; widening one button's label is
 * also a spec amendment because the ChatPanel wiring tests query by
 * the literal substring.
 *
 *   1. "Yes — N rows"            (primary; confirms multi-row interpretation)
 *   2. "No — treat as 1 row"     (F-11: re-fire guard via store)
 *   3. "Edit the rows"           (opens chat-mediated edit path)
 *   4. "This isn't source data"  (F-10: escape via store)
 *
 * The em-dash "—" is intentional (matches Phase 5a Task 4 spec copy);
 * tests query by case-insensitive regex so we don't trip on hyphen
 * normalisation, but the rendered text MUST keep the em-dash.
 */

export interface InlineSourceDisambiguationTurnProps {
  /**
   * The user's original input text — rendered verbatim in a <blockquote>
   * so the user can confirm what the LLM heard. Pass the exact ChatMessage
   * content; do NOT pre-clip or pre-format. The widget does not enforce a
   * length budget here because the user authored the text themselves and
   * has already seen it in the chat scrollback.
   */
  userInput: string;
  /**
   * The LLM's parsed row breakdown — one string per proposed row. Rendered
   * as an ordered list so the user can see how their input was split. The
   * widget treats this as display data only; the actual inline_blob
   * content lives in the proposal's arguments_redacted_json and is
   * authoritative there.
   */
  proposedRows: ReadonlyArray<string>;
  /**
   * Proposal ID — round-tripped to the three proposal-scoped handlers
   * (confirm / single-row / edit). Lets the caller dispatch through the
   * existing acceptCompositionProposal / rejectCompositionProposal
   * surfaces by ID without the widget needing to know about the
   * sessionStore wiring.
   */
  proposalId: string;
  /**
   * Originating message ID — the user message whose interpretation is
   * being disambiguated. The F-10 escape handler keys on this (NOT
   * on proposalId) because the guard's purpose is "the user said this
   * specific message isn't source data; don't re-prompt about it",
   * which is a per-message claim, not a per-proposal one. Same shape
   * for F-11 (single-row) — keyed on message so a follow-up LLM turn
   * doesn't redundantly re-prompt.
   */
  messageId: string;

  // ── Action handlers ────────────────────────────────────────────────────
  /** "Yes — N rows" → confirm the multi-row interpretation. */
  onConfirmMultiRow: (proposalId: string) => void;
  /** "No — treat as 1 row" → F-11 fallback. */
  onTreatAsOneRow: (proposalId: string) => void;
  /** "Edit the rows" → open the inline editor (or chat-mediated edit). */
  onEditRows: (proposalId: string) => void;
  /** "This isn't source data" → F-10 escape. */
  onNotSourceData: (messageId: string) => void;
}

export function InlineSourceDisambiguationTurn({
  userInput,
  proposedRows,
  proposalId,
  messageId,
  onConfirmMultiRow,
  onTreatAsOneRow,
  onEditRows,
  onNotSourceData,
}: InlineSourceDisambiguationTurnProps): JSX.Element {
  const rowCount = proposedRows.length;
  const primaryButtonRef = useRef<HTMLButtonElement>(null);

  // F-19: move focus to the primary action on mount so keyboard users
  // don't have to Tab from the top of the chat panel to reach the new
  // turn widget. Mirrors the focus-handoff pattern in ChatPanel's
  // guided step-advance effect (search for "spec §7.4" there).
  useEffect(() => {
    primaryButtonRef.current?.focus();
  }, []);

  return (
    <section
      role="region"
      aria-label={`Confirm row count interpretation (${rowCount} rows)`}
      data-testid="inline-source-disambiguation-turn"
      className="inline-source-disambiguation-turn"
    >
      <header className="inline-source-disambiguation-turn-header">
        <h3 className="inline-source-disambiguation-turn-title">
          Confirm row count
        </h3>
        <p className="inline-source-disambiguation-turn-explainer">
          You wrote:
        </p>
        <blockquote className="inline-source-disambiguation-turn-input">
          {userInput}
        </blockquote>
        <p className="inline-source-disambiguation-turn-explainer">
          I read this as <strong>{rowCount}</strong>{" "}
          {rowCount === 1 ? "row" : "rows"}:
        </p>
      </header>

      <ol className="inline-source-disambiguation-turn-rows">
        {proposedRows.map((row, index) => (
          // Index keys are acceptable here: the list is render-only display
          // of the LLM's parse; the widget never re-orders or splices it,
          // and the same list is re-rendered only when the proposal itself
          // changes (which remounts the widget anyway).
          <li
            key={index}
            className="inline-source-disambiguation-turn-row"
          >
            {row}
          </li>
        ))}
      </ol>

      <div className="inline-source-disambiguation-turn-actions">
        {/*
          Primary action. Receives focus on mount (F-19).
          Label "Yes — N rows" is load-bearing — the widget test
          and the ChatPanel wiring test both query by /yes.*N rows/i.
        */}
        <button
          ref={primaryButtonRef}
          type="button"
          className="btn btn-primary inline-source-disambiguation-turn-confirm"
          onClick={() => onConfirmMultiRow(proposalId)}
        >
          {`Yes — ${rowCount} ${rowCount === 1 ? "row" : "rows"}`}
        </button>
        <button
          type="button"
          className="btn inline-source-disambiguation-turn-single"
          onClick={() => onTreatAsOneRow(proposalId)}
        >
          {"No — treat as 1 row"}
        </button>
        <button
          type="button"
          className="btn inline-source-disambiguation-turn-edit"
          onClick={() => onEditRows(proposalId)}
        >
          Edit the rows
        </button>
      </div>

      {/*
        Escape action (F-10) is rendered as a link-style button so it
        reads as a less-emphasised "this is the wrong frame" exit,
        distinct from the three in-frame disambiguation choices above.
        The accessible name MUST contain the substring "this isn't
        source data" for both the widget test and the ChatPanel
        wiring test to find it; the surrounding wrapper is presentational.
      */}
      <div className="inline-source-disambiguation-turn-escape">
        <button
          type="button"
          className="inline-source-disambiguation-turn-not-source"
          onClick={() => onNotSourceData(messageId)}
        >
          {"This isn’t source data"}
        </button>
      </div>
    </section>
  );
}
