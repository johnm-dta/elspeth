// src/components/chat/guided/MultiSelectWithCustomTurn.tsx
//
// Guided-mode widget for the multi_select_with_custom turn type (Task 7.4).
// Conventions inherited from SingleSelectTurn (Task 7.2 template) and
// InspectAndConfirmTurn (Task 7.3):
//   - Props: { payload: MultiSelectWithCustomPayload; onSubmit: (body: GuidedRespondRequest) => void }
//   - onSubmit is SYNC — the widget constructs the body; the store awaits the round-trip
//   - All 6 GuidedRespondRequest fields set explicitly; unused ones = null (no omission)
//   - <fieldset>+<legend> for the chip group (per Task 7.2 SHAPE NOTE — chip-group
//     family widgets share the structure)
//   - <button type="button"> (never <div onClick>)
//   - DOM IDs prefixed with React 18 useId() — multiple turn instances coexist in
//     GuidedHistory (Task 7.9) and option IDs recur across turns
//   - Visible labels (htmlFor / button text) ARE the accessible name; do not add
//     redundant aria-label that overrides what sighted users see
//   - CSS via App.css class names with design tokens; no hardcoded colours
//   - Reuses guided-chip-btn / guided-chip-group styles per the App.css convention
//
// State encoding (single nullable-free struct per Task 7.3 convention):
//   selection: { chosen: Set<string>; customs: string[]; pending: string }
//   - chosen: currently-pressed option IDs (initialised from payload.default_chosen)
//   - customs: free-added custom strings, in addition order
//   - pending: controlled value of the custom-input box
//   The struct lives even when "empty" (no Set/list view toggle to encode);
//   updaters return new objects to keep React's referential equality contract.
//
// CHIP TOGGLE PATTERN:
//   Chips use aria-pressed (toggle-button semantics), NOT aria-selected
//   (which is for listbox/option semantics) and NOT <input type="checkbox">
//   styled as buttons (less consistent across screen readers). aria-pressed
//   is the canonical accessibility pattern for two-state toggle buttons.
//
// CONTINUE INVARIANT:
//   The widget enforces "user must assert at least one field" by disabling
//   the Continue button while chosen is empty AND customs is empty. This
//   prevents an empty submit that the backend would reject (Step 2's
//   required-field reads cannot be satisfied by a zero-output sink). The
//   negative branch is pinned in the test suite — clicking the disabled
//   button does NOT fire onSubmit.
//
// CUSTOM-ADD DUPLICATE GUARD:
//   The Add button is disabled when the trimmed input is empty/whitespace
//   OR matches an existing option ID OR matches an already-added custom.
//   The disable predicate is the same as the silent-no-op early-return on
//   Enter — a single source of truth for "can this be added?".
//
// NOTE: payload.escape_label is intentionally NOT rendered in this version.
// The wire shape for the escape submission requires a cross-layer protocol
// decision (frontend wire shape + backend handler branch in
// state_machine.py:_advance_step_2 — which currently reads
// edited_values["outputs"] unconditionally). The plan describes
// `{edited_values: {schema_mode: "observed", required_fields: []}}` but
// that contradicts the only backend read site, and this widget owns
// neither the plugin name nor the options needed to construct the full
// outputs[] array. Tracked as a follow-up; do NOT add the escape button
// to this widget without resolving the contract first. The deferral is
// pinned by tests (escape-button-not-rendered for both null and non-null
// escape_label) so a future contributor can't quietly re-add it.

import { useId, useState } from "react";
import type {
  GuidedRespondRequest,
  MultiSelectWithCustomPayload,
} from "@/types/guided";

interface MultiSelectWithCustomTurnProps {
  payload: MultiSelectWithCustomPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
}

interface Selection {
  chosen: Set<string>;
  customs: string[];
  pending: string;
}

export function MultiSelectWithCustomTurn({
  payload,
  onSubmit,
}: MultiSelectWithCustomTurnProps) {
  const [selection, setSelection] = useState<Selection>(() => ({
    chosen: new Set(payload.default_chosen),
    customs: [],
    pending: "",
  }));

  // useId scopes DOM IDs per-instance so multiple MultiSelectWithCustomTurns
  // rendered simultaneously (e.g. active turn + GuidedHistory replay in
  // Task 7.9) don't produce id collisions when option IDs ("name", "price")
  // recur across turns.
  const reactId = useId();
  const customInputId = `${reactId}-custom-input`;
  const hintIdFor = (optionId: string) => `${reactId}-hint-${optionId}`;

  // Stable order: option-array order. The wire-response test pins this so a
  // future "sort alphabetically" refactor would visibly fail.
  const chosenInOptionOrder = (chosen: Set<string>): string[] =>
    payload.options.filter((opt) => chosen.has(opt.id)).map((opt) => opt.id);

  function toggleOption(optionId: string) {
    setSelection((prev) => {
      const next = new Set(prev.chosen);
      if (next.has(optionId)) {
        next.delete(optionId);
      } else {
        next.add(optionId);
      }
      return { ...prev, chosen: next };
    });
  }

  // Single source of truth for "can the current pending value be added?".
  // Used both as the Add button's disabled predicate AND as the early-return
  // guard inside handleAddCustom — they MUST agree.
  function canAddPending(s: Selection): boolean {
    const trimmed = s.pending.trim();
    if (!trimmed) return false;
    if (s.customs.includes(trimmed)) return false;
    if (payload.options.some((opt) => opt.id === trimmed)) return false;
    return true;
  }

  function handleAddCustom() {
    setSelection((prev) => {
      if (!canAddPending(prev)) return prev;
      return {
        ...prev,
        customs: [...prev.customs, prev.pending.trim()],
        pending: "",
      };
    });
  }

  function handleRemoveCustom(value: string) {
    setSelection((prev) => ({
      ...prev,
      customs: prev.customs.filter((c) => c !== value),
    }));
  }

  function handleContinue() {
    // Disabled-button guard: this handler is only reachable when the Continue
    // button is enabled, but the type system can't prove that across the
    // render boundary. Narrow because TS doesn't see the cross-handler
    // invariant — illegal state would be a code bug worth catching, not user
    // data to coerce. Returning silently here matches the same predicate as
    // the disabled attribute, deduplicated.
    if (selection.chosen.size === 0 && selection.customs.length === 0) return;
    onSubmit({
      chosen: chosenInOptionOrder(selection.chosen),
      custom_inputs: [...selection.customs],
      edited_values: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  const continueDisabled =
    selection.chosen.size === 0 && selection.customs.length === 0;
  const addDisabled = !canAddPending(selection);

  return (
    <div className="guided-turn guided-multi-select">
      <fieldset className="guided-chip-fieldset">
        <legend className="guided-chip-legend">{payload.question}</legend>
        {/* No role="group" on the inner div — <fieldset> already provides
            group semantics; duplicating creates two nested groups in the
            accessibility tree. */}
        <div className="guided-chip-group">
          {payload.options.map((option) => {
            const hintId =
              option.hint !== null ? hintIdFor(option.id) : undefined;
            const pressed = selection.chosen.has(option.id);
            return (
              <div key={option.id} className="guided-chip-item">
                <button
                  type="button"
                  className="guided-chip-btn"
                  aria-pressed={pressed}
                  aria-describedby={hintId}
                  onClick={() => toggleOption(option.id)}
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

      <div className="guided-multi-custom-row">
        <label htmlFor={customInputId} className="guided-custom-label">
          Custom field
        </label>
        <input
          id={customInputId}
          type="text"
          className="guided-custom-input"
          value={selection.pending}
          onChange={(e) =>
            setSelection((prev) => ({ ...prev, pending: e.target.value }))
          }
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              // preventDefault so a future <form> wrapper doesn't double-submit.
              // handleAddCustom already no-ops on a non-addable pending —
              // disabled-button + no-op early-return are the same predicate,
              // deduplicated.
              e.preventDefault();
              handleAddCustom();
            }
          }}
          placeholder="Add a field name..."
        />
        <button
          type="button"
          className="guided-custom-submit-btn"
          onClick={handleAddCustom}
          disabled={addDisabled}
        >
          Add
        </button>
      </div>

      {selection.customs.length > 0 && (
        <ul className="guided-multi-custom-list">
          {selection.customs.map((value) => (
            <li key={value} className="guided-multi-custom-chip">
              <span className="guided-multi-custom-chip-label">{value}</span>
              <button
                type="button"
                className="guided-multi-custom-remove-btn"
                onClick={() => handleRemoveCustom(value)}
                aria-label={`Remove ${value}`}
              >
                {/* ASCII multiplication-style X; visible glyph for sighted
                    users, accessible name from aria-label since the glyph
                    alone wouldn't read meaningfully. This is the documented
                    exception to the "no aria-label on visible-text controls"
                    rule — there is no useful visible text. */}
                x
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="guided-multi-actions">
        <button
          type="button"
          className="guided-multi-continue-btn"
          onClick={handleContinue}
          disabled={continueDisabled}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
