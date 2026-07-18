// src/components/chat/guided/SingleSelectTurn.tsx
//
// Template widget for the guided-mode protocol (Task 7.2).
// Conventions established here will be replicated by Tasks 7.3-7.7:
//   - Props: { payload: <WidgetPayload>; onSubmit: (body: GuidedRespondAction) => void }
//   - onSubmit is SYNC — the widget constructs the body; the store awaits the round-trip
//   - Every GuidedRespondAction field is explicit; unused ones are null
//   - <fieldset>+<legend> for chip groups; <button type="button"> (never <div onClick>)
//   - aria-describedby only wired when hint is non-null (no dangling IDs)
//   - DOM IDs prefixed with React 18 useId() — multiple turn instances coexist in
//     GuidedHistory (Task 7.9) and option IDs recur across turns; document-global
//     IDs would collide
//   - Visible labels (htmlFor / button text) ARE the accessible name; do not add
//     redundant aria-label that overrides what sighted users see
//   - CSS via components/chat/guided/guided.css class names with design tokens; no hardcoded colours
//
// SHAPE NOTE for Tasks 7.3-7.7:
// The chip-group structure (<fieldset>+<legend>+chip-button-group) applies to
// single_select and multi_select_with_custom. Schema-form and
// pipeline-proposal turns establish their own structures.

import { useId, useState } from "react";
import type { GuidedRespondAction, SingleSelectPayload } from "@/types/guided";

interface SingleSelectTurnProps {
  payload: SingleSelectPayload;
  onSubmit: (body: GuidedRespondAction) => void;
  disabled?: boolean;
  /**
   * Tutorial mode is passive — the per-stage prompt is built by pressing Send,
   * so the learner does not have to pick an option here (the chips still work
   * if clicked). Suppresses the "Choosing an option continues" subtext that
   * otherwise contradicts the "press Send" coaching note above the widget.
   */
  isTutorial?: boolean;
}

export function SingleSelectTurn({
  payload,
  onSubmit,
  disabled = false,
  isTutorial = false,
}: SingleSelectTurnProps) {
  const [customText, setCustomText] = useState("");

  // useId scopes DOM IDs per-instance so multiple SingleSelectTurns rendered
  // simultaneously (e.g. active turn + GuidedHistory replay in Task 7.9) don't
  // produce id collisions when option IDs ("csv", "api") recur across turns.
  const reactId = useId();
  const customInputId = `${reactId}-custom-input`;
  const instructionId = `${reactId}-instruction`;
  const hintIdFor = (optionId: string) => `${reactId}-hint-${optionId}`;

  function handleOptionClick(optionId: string) {
    onSubmit({
      chosen: [optionId],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });
  }

  function handleCustomSubmit() {
    const trimmed = customText.trim();
    if (!trimmed) return;
    onSubmit({
      chosen: null,
      edited_values: null,
      custom_inputs: [trimmed],
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });
  }

  return (
    <div className="guided-turn guided-single-select">
      <fieldset
        className="guided-chip-fieldset"
        // Tutorial is passive: pressing Send builds the step. Suppress the
        // "Choosing an option continues" subtext (it contradicts the coaching
        // note) and drop its aria-describedby so the fieldset carries no
        // dangling IDREF. The chips remain interactive for anyone who does pick.
        aria-describedby={isTutorial ? undefined : instructionId}
      >
        <legend className="guided-chip-legend">{payload.question}</legend>
        {!isTutorial && (
          <p id={instructionId} className="guided-chip-instruction">
            Select one. Choosing an option continues to the next step.
          </p>
        )}
        {/* No role="group" on the inner div — <fieldset> already provides
            group semantics; duplicating creates two nested groups in the
            accessibility tree. */}
        <div className="guided-chip-group">
          {payload.options.map((option) => {
            const hintId =
              option.hint !== null ? hintIdFor(option.id) : undefined;
            return (
              <div key={option.id} className="guided-chip-item">
                <button
                  type="button"
                  className="guided-chip-btn"
                  onClick={() => handleOptionClick(option.id)}
                  aria-describedby={hintId}
                  disabled={disabled}
                >
                  {option.label}
                </button>
                {option.hint !== null && (
                  <p id={hintId} className="guided-chip-hint">
                    {option.hint}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </fieldset>

      {payload.allow_custom && (
        <div className="guided-custom-row">
          <label htmlFor={customInputId} className="guided-custom-label">
            Custom
          </label>
          <input
            id={customInputId}
            type="text"
            className="guided-custom-input"
            value={customText}
            disabled={disabled}
            onChange={(e) => setCustomText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                // preventDefault so a future <form> wrapper (e.g. SchemaFormTurn
                // composition) doesn't double-submit. handleCustomSubmit
                // already no-ops on empty — disabled-button + no-op
                // early-return are the same predicate, deduplicated.
                e.preventDefault();
                handleCustomSubmit();
              }
            }}
            placeholder="Describe your custom option..."
          />
          <button
            type="button"
            className="guided-custom-submit-btn"
            onClick={handleCustomSubmit}
            disabled={disabled || !customText.trim()}
          >
            Submit custom
          </button>
        </div>
      )}
    </div>
  );
}
