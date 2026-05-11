// src/components/chat/guided/SingleSelectTurn.tsx
//
// Template widget for the guided-mode protocol (Task 7.2).
// Conventions established here will be replicated by Tasks 7.3-7.7:
//   - Props: { payload: <WidgetPayload>; onSubmit: (body: GuidedRespondRequest) => void }
//   - onSubmit is SYNC — the widget constructs the body; the store awaits the round-trip
//   - All 6 GuidedRespondRequest fields set explicitly; unused ones = null (no omission)
//   - <fieldset>+<legend> for chip groups; <button type="button"> (never <div onClick>)
//   - aria-describedby only wired when hint is non-null (no dangling IDs)
//   - CSS via App.css class names with design tokens; no hardcoded colours

import { useState } from "react";
import type { GuidedRespondRequest, SingleSelectPayload } from "@/types/guided";

interface SingleSelectTurnProps {
  payload: SingleSelectPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
}

export function SingleSelectTurn({ payload, onSubmit }: SingleSelectTurnProps) {
  const [customText, setCustomText] = useState("");

  function handleOptionClick(optionId: string) {
    onSubmit({
      chosen: [optionId],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
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
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  return (
    <div className="guided-turn guided-single-select">
      <fieldset className="guided-chip-fieldset">
        <legend className="guided-chip-legend">{payload.question}</legend>
        <div className="guided-chip-group" role="group">
          {payload.options.map((option) => {
            const hintId =
              option.hint !== null ? `hint-${option.id}` : undefined;
            return (
              <div key={option.id} className="guided-chip-item">
                <button
                  type="button"
                  className="guided-chip-btn"
                  onClick={() => handleOptionClick(option.id)}
                  aria-describedby={hintId}
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
          <label htmlFor="guided-custom-input" className="guided-custom-label">
            Custom
          </label>
          <input
            id="guided-custom-input"
            type="text"
            aria-label="Custom option"
            className="guided-custom-input"
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && customText.trim()) {
                handleCustomSubmit();
              }
            }}
            placeholder="Describe your custom option…"
          />
          <button
            type="button"
            className="guided-custom-submit-btn"
            aria-label="Submit custom option"
            onClick={handleCustomSubmit}
            disabled={!customText.trim()}
          >
            Submit custom
          </button>
        </div>
      )}
    </div>
  );
}
