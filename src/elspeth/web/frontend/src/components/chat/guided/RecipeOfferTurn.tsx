// src/components/chat/guided/RecipeOfferTurn.tsx
//
// Guided-mode widget for the recipe_offer turn type (Task 7.7, extended for
// Gap 6 / Task 10.0 — editable inputs for unsatisfied required slots).
// Conventions inherited from SingleSelectTurn (Task 7.2 template),
// InspectAndConfirmTurn (Task 7.3), MultiSelectWithCustomTurn (Task 7.4),
// SchemaFormTurn (Task 7.5), and ProposeChainTurn (Task 7.6).
//
//   - Props: { payload: RecipeOfferPayload; onSubmit: (body: GuidedRespondRequest) => void }
//   - onSubmit is SYNC -- the widget constructs the body; the store awaits the round-trip
//   - All 6 GuidedRespondRequest fields set explicitly; unused ones = null (no omission)
//   - <button type="button"> (never <div onClick>)
//   - DOM IDs prefixed with React 18 useId() -- multiple turn instances coexist in
//     GuidedHistory (Task 7.9) and element IDs would collide without per-instance scoping
//   - Visible labels (button text) ARE the accessible name; no redundant aria-label
//   - CSS via App.css class names with design tokens; no hardcoded colours
//
// SHAPE NOTE (card family):
// recipe_offer is listed alongside single_select and multi_select_with_custom in
// SingleSelectTurn's SHAPE NOTE as a "chip-group family" member.  However, the
// actual layout ("single card with recipe name + slot summary + action buttons")
// is structurally a card, not a chip group.  This widget uses the card layout.
//
// CARD-STYLE DECISION (a) -- Reuse guided-propose-step-card directly:
// The .guided-propose-step-card rule (App.css:4285-4290) provides:
//   background-color: surface-elevated, border: border-strong, border-radius: radius-lg,
//   padding: space-md.
// This is pure card chrome with zero step-list-specific styling.  New rules added
// for this widget live in the guided-recipe-* namespace.
//
// SCOPE -- submit paths:
//
//   WIRED: "Apply recipe" -> chosen: ["accept"]
//          The backend reconstructs the RecipeMatch from edited_values; the
//          widget MUST send the merged slot map (pre-filled + operator-supplied).
//          Disabled until every required unsatisfied slot has a non-empty
//          trimmed value (offensive UX: the same constraint the backend will
//          enforce, surfaced earlier).
//
//   WIRED: "Build manually" -> chosen: ["build_manually"]
//          Backend only reads chosen on this path; edited_values: null is correct.
//
//   NOT WIRED: alternatives-based alternative recipe selection.  The current
//          backend only accepts ["accept"] or ["build_manually"]; any other chosen
//          value returns HTTP 400.  The alternatives list is displayed for user
//          information only.
//
// WIRE-SHAPE:
//   Apply recipe:   { chosen: ["accept"],
//                     edited_values: { recipe_name: payload.recipe_name,
//                                      slots: { ...payload.slots, ...inputs } },
//                     custom_inputs: null, accepted_step_index: null,
//                     edit_step_index: null, control_signal: null }
//   Build manually: { chosen: ["build_manually"],
//                     edited_values: null, custom_inputs: null,
//                     accepted_step_index: null, edit_step_index: null,
//                     control_signal: null }
//
// SECURITY -- api_key_secret and other slot_type="str" inputs:
//   "api_key_secret" carries the NAME of an inventory secret_ref, not a literal
//   credential.  All str slots render with <input type="text">.  Using
//   type="password" would suggest pasting a raw key, which the audit trail would
//   then record verbatim through edited_values — that is the exact thing the
//   secret_ref indirection exists to prevent.
//
// NUMERIC slots (slot_type="int" / "float"):
//   Rendered with <input type="number">.  The backend recipe coercion accepts
//   string forms ("42" → 42), so the input value is submitted as the raw string;
//   no per-type coercion in the widget.
//
// STATE:
//   Local state holds one string per unsatisfied slot, keyed by slot name.
//   Initialised to "" for each entry.  Inputs are controlled.  Typing into
//   inputs is not a "click" for the demo SLA budget.

import { useId, useMemo, useState } from "react";
import type { GuidedRespondRequest, RecipeOfferPayload, RecipeSlotInput } from "@/types/guided";

interface RecipeOfferTurnProps {
  payload: RecipeOfferPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
}

/**
 * Format a single slot value for display in the key-value definition list.
 *
 * Scalars (string, number, boolean) are coerced to string.  Null/undefined
 * render as "null".  Arrays and objects are JSON-stringified so the user sees
 * the structure without a full JSON editor component.
 */
function formatSlotValue(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

/**
 * Map a slot_type to the appropriate <input type="..."> value.
 *
 * "str" / "blob_id" / "str_list" all render as type="text".  Critically,
 * type="password" is NEVER returned — api_key_secret carries the NAME of an
 * inventory secret_ref, not the credential itself; the audit trail records
 * edited_values verbatim, so masking the input would mislead the operator
 * into pasting raw credentials.
 */
function inputTypeForSlot(slotType: RecipeSlotInput["slot_type"]): "text" | "number" {
  if (slotType === "int" || slotType === "float") return "number";
  return "text";
}

function isSecretLikeSlotName(name: string): boolean {
  return /_(secret|password|token|key)$/i.test(name);
}

function SecretAuditWarning({ id }: { id?: string }) {
  return (
    <p id={id} className="guided-recipe-input-warning">
      <span className="guided-recipe-input-warning-icon" aria-hidden="true">
        <svg viewBox="0 0 16 16" focusable="false" aria-hidden="true">
          <path d="M4.5 7V5.5a3.5 3.5 0 0 1 7 0V7h.5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1h.5Zm1.5 0h4V5.5a2 2 0 0 0-4 0V7Z" />
        </svg>
        <span className="sr-only">Lock</span>
      </span>
      Secret values are written to the audit trail exactly as typed. They will appear
      in operator logs.
    </p>
  );
}

export function RecipeOfferTurn({ payload, onSubmit }: RecipeOfferTurnProps) {
  // useId scopes DOM IDs per-instance so multiple RecipeOfferTurns rendered
  // simultaneously (e.g. active turn + GuidedHistory replay in Task 7.9) don't
  // produce id collisions when element IDs recur across turns.
  const reactId = useId();
  const slotsId = `${reactId}-slots`;
  const alternativesId = `${reactId}-alts`;
  const inputsId = `${reactId}-inputs`;

  // Controlled inputs for each unsatisfied slot, keyed by slot name.
  // Initial value is "" — the disabled-state check below treats trimmed empty
  // as "not filled", so the Apply button stays disabled until the operator
  // supplies every required value.
  const initialInputs = useMemo<Record<string, string>>(() => {
    const seed: Record<string, string> = {};
    for (const slot of payload.unsatisfied_slots) {
      seed[slot.name] = "";
    }
    return seed;
  }, [payload.unsatisfied_slots]);
  const [slotInputs, setSlotInputs] = useState<Record<string, string>>(initialInputs);

  function setSlotInput(name: string, value: string) {
    setSlotInputs((prev) => ({ ...prev, [name]: value }));
  }

  // Apply is disabled until every unsatisfied slot has a non-empty trimmed
  // value.  RecipeSlotInput.required is absent from the wire shape because the
  // RecipeMatch invariant guarantees every entry is required — treat all as
  // required here.  Mirrors the backend's validate_slots check: fail fast at
  // the UI rather than burn a round-trip on an HTTP 400.
  const applyDisabled = payload.unsatisfied_slots.some((slot) => {
    const value = slotInputs[slot.name] ?? "";
    return value.trim() === "";
  });

  function handleApply() {
    // Merge pre-filled slots with operator inputs.  Operator inputs take
    // precedence in collisions, though the backend resolver only writes the
    // pre-filled slots it can derive and the unsatisfied list is by definition
    // disjoint from those — collision is not expected in practice.
    const mergedSlots: Record<string, unknown> = {
      ...payload.slots,
      ...slotInputs,
    };
    onSubmit({
      chosen: ["accept"],
      edited_values: {
        recipe_name: payload.recipe_name,
        slots: mergedSlots,
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  function handleBuildManually() {
    onSubmit({
      chosen: ["build_manually"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  const slotKeys = Object.keys(payload.slots);

  return (
    <div className="guided-turn guided-recipe-offer">
      <div className="guided-propose-step-card guided-recipe-card">
        <h3 className="guided-recipe-name">{payload.recipe_name}</h3>

        {slotKeys.length > 0 && (
          <dl id={slotsId} className="guided-recipe-slots">
            {slotKeys.map((key) => (
              <div key={key} className="guided-recipe-slot-row">
                <dt className="guided-recipe-slot-key">{key}</dt>
                <dd className="guided-recipe-slot-val">
                  {formatSlotValue(payload.slots[key])}
                </dd>
                {isSecretLikeSlotName(key) && <SecretAuditWarning />}
              </div>
            ))}
          </dl>
        )}

        {/* Unsatisfied required slots: inline editable form fields.
            Rendered between the pre-filled slot summary and the action row so
            the operator sees the partial state then fills the gaps. */}
        {payload.unsatisfied_slots.length > 0 && (
          <fieldset id={inputsId} className="guided-recipe-inputs">
            <legend className="guided-recipe-inputs-legend">
              Additional values required
            </legend>
            {payload.unsatisfied_slots.map((slot) => {
              const inputId = `${reactId}-input-${slot.name}`;
              const descriptionId = `${reactId}-desc-${slot.name}`;
              const warningId = `${reactId}-warning-${slot.name}`;
              const inputType = inputTypeForSlot(slot.slot_type);
              const showAuditWarning = isSecretLikeSlotName(slot.name);
              const describedBy = [
                slot.description ? descriptionId : null,
                showAuditWarning ? warningId : null,
              ]
                .filter(Boolean)
                .join(" ");
              return (
                <div key={slot.name} className="guided-recipe-input-row">
                  <label htmlFor={inputId} className="guided-recipe-input-label">
                    {slot.name}
                    {/* Every unsatisfied slot is required by the RecipeMatch invariant;
                        the asterisk is always shown rather than gated on slot.required. */}
                    <span className="guided-recipe-input-required" aria-hidden="true">
                      {" *"}
                    </span>
                  </label>
                  <input
                    id={inputId}
                    type={inputType}
                    className="guided-recipe-input-field"
                    value={slotInputs[slot.name] ?? ""}
                    onChange={(event) => setSlotInput(slot.name, event.target.value)}
                    required
                    aria-describedby={describedBy || undefined}
                    aria-required="true"
                  />
                  {slot.description && (
                    <p id={descriptionId} className="guided-recipe-input-hint">
                      {slot.description}
                    </p>
                  )}
                  {showAuditWarning && <SecretAuditWarning id={warningId} />}
                </div>
              );
            })}
          </fieldset>
        )}

        {payload.alternatives.length > 0 && (
          <div className="guided-recipe-alternatives">
            <p className="guided-recipe-alternatives-heading">Alternatives:</p>
            <ul id={alternativesId} className="guided-recipe-alternatives-list">
              {payload.alternatives.map((alt, idx) => (
                <li key={idx} className="guided-recipe-alternative-item">
                  {alt}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="guided-recipe-actions">
          <button
            type="button"
            className="guided-recipe-apply-btn"
            onClick={handleApply}
            disabled={applyDisabled}
          >
            Apply recipe
          </button>
          <button
            type="button"
            className="guided-recipe-build-btn"
            onClick={handleBuildManually}
          >
            Build manually
          </button>
        </div>
      </div>
    </div>
  );
}
