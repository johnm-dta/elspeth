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
//   - CSS via components/chat/guided/guided.css class names with design tokens; no hardcoded colours
//   - Reuses guided-chip-btn / guided-chip-group styles per the guided.css convention
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
// FOCUS MANAGEMENT ON CUSTOM-CHIP REMOVAL (WCAG 2.4.3 Focus Order):
//   Removing a custom chip via Enter on the X button unmounts the focused
//   element; the browser falls back to <body> and the keyboard user loses
//   their place. We restore focus explicitly using the same ref + effect +
//   firstRunRef pattern as InspectAndConfirmTurn (Task 7.3):
//     - Removing chip at index i, with N customs remaining post-removal:
//         * N > 0 and i < N  → focus the X button now at index i (the next
//           chip in the original order takes the removed slot).
//         * N > 0 and i == N → focus the X button at index N-1 (we removed
//           the last chip; fall back one slot).
//         * N == 0           → focus the custom-input field (list is now
//           empty; the input is the entry point for adding more, and
//           unlike the Add button it is always focusable — the Add button
//           is disabled while the input is empty/whitespace, and focusing
//           a disabled button is a no-op in HTML).
//   The focus target is decided synchronously inside the click handler
//   (BEFORE the state update fires) and stored in a ref. The effect, keyed
//   on customs.length, reads the ref after the post-removal render commits.
//   firstRunRef skips the initial mount so we don't steal focus from
//   wherever the user actually was when the widget first appeared.
//
// SUBMIT WIRE SHAPE:
//   handleContinue emits:
//     { chosen: [required field names in option order],
//       custom_inputs: [...custom field names],
//       edited_values: null,
//       accepted_step_index: null, edit_step_index: null, control_signal: null }
//   The backend's _advance_step_2 reads chosen + custom_inputs and combines them
//   with the sink plugin + options stored in GuidedSession.step_2_sink_intent
//   (persisted by the preceding SCHEMA_FORM dispatcher) to construct
//   SinkOutputResolved. edited_values is null — the widget does not own the
//   plugin context. Resolved by elspeth-5e905f3c9d.
//
// ESCAPE PATH:
//   payload.escape_label renders as a first-class "let source decide" action.
//   It submits chosen=[] and custom_inputs=[] with control_signal="passthrough"
//   (C-3a — the explicit, unambiguous signal; a bare empty chosen/custom_inputs
//   pair is otherwise indistinguishable from a stale client submitting nothing,
//   so the backend fail-closes it). The backend treats passthrough as observed
//   schema mode with no fixed required fields, using the persisted sink intent
//   for plugin/options. NOT gated by continueDisabled — the Continue button's
//   "must assert at least one field" invariant applies to Continue only; the
//   escape hatch is deliberately available even when nothing is chosen.

import { useEffect, useId, useRef, useState } from "react";
import type {
  GuidedRespondRequest,
  MultiSelectWithCustomPayload,
} from "@/types/guided";

interface MultiSelectWithCustomTurnProps {
  payload: MultiSelectWithCustomPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
  disabled?: boolean;
  /**
   * Tutorial passive mode: suppress the "Select all that apply, then press
   * Continue" subtext that contradicts the "press Send" coaching note (mirrors
   * SingleSelectTurn). The chips remain interactive.
   */
  isTutorial?: boolean;
}

interface Selection {
  chosen: Set<string>;
  customs: string[];
  pending: string;
}

export function MultiSelectWithCustomTurn({
  payload,
  onSubmit,
  disabled = false,
  isTutorial = false,
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
  const instructionId = `${reactId}-instruction`;
  const hintIdFor = (optionId: string) => `${reactId}-hint-${optionId}`;

  // Focus-restoration refs for the custom-chip remove path (WCAG 2.4.3).
  // The X buttons live inside .map() so we collect them into a ref-Map keyed
  // by the custom value (which is unique within the list — the duplicate
  // guard in canAddPending enforces it).  The custom-input ref is captured
  // separately and used as the fallback when the list becomes empty (the
  // Add button is disabled while pending is empty, so it can't be focused).
  const removeBtnRefs = useRef<Map<string, HTMLButtonElement | null>>(
    new Map(),
  );
  const customInputRef = useRef<HTMLInputElement | null>(null);

  // After a remove, this ref holds either:
  //   - a string (the value of the surviving chip whose X should receive focus)
  //   - "__input__" (focus the custom-input; list is now empty)
  //   - null        (no pending focus restoration; do nothing)
  // The effect below consumes and clears it on each customs.length change.
  const pendingFocusTarget = useRef<string | "__input__" | null>(null);

  // Skip the first effect run: on initial widget mount the user did NOT
  // remove anything — the widget appeared because a new turn arrived.
  // Auto-focusing on mount would steal focus from wherever the user
  // actually was. Same convention as InspectAndConfirmTurn (Task 7.3).
  const firstRunRef = useRef(true);
  useEffect(() => {
    if (firstRunRef.current) {
      firstRunRef.current = false;
      return;
    }
    const target = pendingFocusTarget.current;
    pendingFocusTarget.current = null;
    if (target === null) return;
    if (target === "__input__") {
      customInputRef.current?.focus();
      return;
    }
    removeBtnRefs.current.get(target)?.focus();
  }, [selection.customs.length]);

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
    // Decide focus target BEFORE the state update fires so the decision
    // sees the pre-removal indices. After the removal:
    //   - If anything remains and `value` was NOT the last chip, the chip
    //     that was at index+1 takes the removed slot — focus its X button
    //     (look up by the value of the surviving chip at the same index).
    //   - If `value` WAS the last chip and others remain, focus the X
    //     button of the new last chip (one slot back).
    //   - If nothing remains, focus the Add button.
    setSelection((prev) => {
      const idx = prev.customs.indexOf(value);
      // idx === -1 should be unreachable — the X button only renders for
      // values currently in customs, and React unmounts the button when
      // the value disappears. If it ever happens, no-op (safer than
      // crashing a UI thread on a stale event).
      if (idx === -1) return prev;
      const next = prev.customs.filter((_, i) => i !== idx);
      if (next.length === 0) {
        pendingFocusTarget.current = "__input__";
      } else if (idx < next.length) {
        // The chip formerly at idx+1 now sits at idx.
        pendingFocusTarget.current = next[idx];
      } else {
        // Removed the last chip; fall back to the new last chip.
        pendingFocusTarget.current = next[next.length - 1];
      }
      return { ...prev, customs: next };
    });
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

  function handleEscape() {
    // C-3a (composer first-principles review 2026-07-04): an empty
    // chosen/custom_inputs pair is otherwise indistinguishable from a
    // stale/buggy client submitting nothing, so the backend fail-closes it
    // (guided_step2_no_fields_selected) — even from this legitimate escape
    // hatch. control_signal: "passthrough" is the explicit, unambiguous
    // signal the backend's STEP_2_SINK MULTI_SELECT_WITH_CUSTOM dispatcher
    // requires before it will accept the empty required-fields set (see
    // ControlSignal.PASSTHROUGH in protocol.py).
    onSubmit({
      chosen: [],
      custom_inputs: [],
      edited_values: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: "passthrough",
    });
  }

  const continueDisabled =
    selection.chosen.size === 0 && selection.customs.length === 0;
  const addDisabled = !canAddPending(selection);

  return (
    <div className="guided-turn guided-multi-select">
      <fieldset
        className="guided-chip-fieldset"
        // Tutorial is passive: pressing Send builds the step. Suppress the
        // "Select all that apply" subtext (it contradicts the coaching note) and
        // drop its aria-describedby so the fieldset carries no dangling IDREF.
        aria-describedby={isTutorial ? undefined : instructionId}
      >
        <legend className="guided-chip-legend">{payload.question}</legend>
        {!isTutorial && (
          <p id={instructionId} className="guided-chip-instruction">
            Select all that apply, then press Continue.
          </p>
        )}
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

      <div className="guided-multi-custom-row">
        <label htmlFor={customInputId} className="guided-custom-label">
          Custom field
        </label>
        <input
          ref={customInputRef}
          id={customInputId}
          type="text"
          className="guided-custom-input"
          value={selection.pending}
          disabled={disabled}
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
          disabled={disabled || addDisabled}
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
                ref={(el) => {
                  // Keep the ref-Map in sync with mount/unmount lifecycle.
                  // React invokes the callback with `el` on mount and `null`
                  // on unmount; treating null as a delete prevents stale
                  // entries pointing at detached nodes.
                  if (el === null) {
                    removeBtnRefs.current.delete(value);
                  } else {
                    removeBtnRefs.current.set(value, el);
                  }
                }}
                type="button"
                className="guided-multi-custom-remove-btn"
                onClick={() => handleRemoveCustom(value)}
                aria-label={`Remove ${value}`}
                disabled={disabled}
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
        {payload.escape_label !== null && (
          <button
            type="button"
            className="guided-multi-escape-btn"
            onClick={handleEscape}
            disabled={disabled}
          >
            {payload.escape_label}
          </button>
        )}
        <button
          type="button"
          className="guided-multi-continue-btn"
          onClick={handleContinue}
          disabled={disabled || continueDisabled}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
