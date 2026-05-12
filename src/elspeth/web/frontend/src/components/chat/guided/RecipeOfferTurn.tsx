// src/components/chat/guided/RecipeOfferTurn.tsx
//
// Guided-mode widget for the recipe_offer turn type (Task 7.7).
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
// This is pure card chrome with zero step-list-specific styling.  The step-list
// scaffolding (guided-propose-steps <ol>, guided-propose-step-header badge row)
// is step-list-specific and is NOT applied here.  RecipeOfferTurn uses a single
// <div class="guided-propose-step-card"> as the card shell.  New rules added
// for this widget live in the guided-recipe-* namespace.  See App.css section
// header at ~line 4420 for the reuse documentation.
//
// SCOPE -- submit paths (verified against routes.py:1943-2023 and
//          state_machine.py:_advance_step_2_5):
//
//   WIRED: "Apply recipe" -> chosen: ["accept"]
//          IMPORTANT: The backend (routes.py:1965-1967) reads edited_values to
//          reconstruct the RecipeMatch.  This widget MUST echo payload.recipe_name
//          and payload.slots in edited_values.  Sending edited_values: null causes
//          the backend to fall back to RecipeMatch(recipe_name="", slots={}) and
//          fail.  This differs from the plan-body spec which stated null -- the
//          plan body was incomplete.
//
//   WIRED: "Build manually" -> chosen: ["build_manually"]
//          Backend only reads chosen on this path; edited_values: null is correct.
//
//   NOT WIRED: alternatives-based alternative recipe selection.  The current
//          backend only accepts ["accept"] or ["build_manually"]; any other chosen
//          value returns HTTP 400.  The alternatives list is displayed for user
//          information only.  If the user wants an alternative, Build Manually
//          advances to Step 3 for manual construction.
//
// WIRE-SHAPE:
//   Apply recipe:   { chosen: ["accept"],
//                     edited_values: { recipe_name: payload.recipe_name,
//                                      slots: payload.slots },
//                     custom_inputs: null, accepted_step_index: null,
//                     edit_step_index: null, control_signal: null }
//   Build manually: { chosen: ["build_manually"],
//                     edited_values: null, custom_inputs: null,
//                     accepted_step_index: null, edit_step_index: null,
//                     control_signal: null }
//
// STATE:
//   Zero widget-side state.  Each button is a pure click -> submit; no
//   intermediate state is accumulated.  The firstRunRef focus-skip pattern is
//   not needed (no view transitions).
//
// FOCUS MANAGEMENT:
//   No programmatic focus on mount or interaction.  No view transitions or
//   collapsible regions that would trigger the focus management patterns used in
//   InspectAndConfirmTurn and SchemaFormTurn.

import { useId } from "react";
import type { GuidedRespondRequest, RecipeOfferPayload } from "@/types/guided";

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
 *
 * Mirrors formatOptionValue in ProposeChainTurn.  Not extracted to a shared
 * helper: two call sites do not meet the rule-of-three extraction threshold.
 */
function formatSlotValue(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

export function RecipeOfferTurn({ payload, onSubmit }: RecipeOfferTurnProps) {
  // useId scopes DOM IDs per-instance so multiple RecipeOfferTurns rendered
  // simultaneously (e.g. active turn + GuidedHistory replay in Task 7.9) don't
  // produce id collisions when element IDs recur across turns.
  const reactId = useId();
  const slotsId = `${reactId}-slots`;
  const alternativesId = `${reactId}-alts`;

  function handleApply() {
    // MUST echo recipe_name and slots in edited_values.
    // Backend (routes.py:1965-1967) reads these to reconstruct the RecipeMatch.
    // edited_values: null would silently produce RecipeMatch("", {}) and fail.
    onSubmit({
      chosen: ["accept"],
      edited_values: {
        recipe_name: payload.recipe_name,
        slots: payload.slots,
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  function handleBuildManually() {
    // Backend only reads chosen on this path (routes.py:1988-2018).
    // edited_values: null is correct here.
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
      {/* Single card: recipe name + slot summary + alternatives + actions */}
      <div className="guided-propose-step-card guided-recipe-card">
        {/* Recipe name as <h3> for screen-reader landmark navigation.
            Same convention as ProposeChainTurn's step plugin headings (Task 7.6 M3).
            Font-size is pinned to --font-size-base via .guided-recipe-name in App.css
            so the browser's default <h3> sizing does not bleed through. */}
        <h3 className="guided-recipe-name">{payload.recipe_name}</h3>

        {/* Slot summary as a key-value definition list.
            <dl> is used because (key, value) pairs form a description/definition
            relationship -- <dt> for key, <dd> for value.  Mirrors ProposeChainTurn's
            options rendering pattern. */}
        {slotKeys.length > 0 && (
          <dl id={slotsId} className="guided-recipe-slots">
            {slotKeys.map((key) => (
              <div key={key} className="guided-recipe-slot-row">
                <dt className="guided-recipe-slot-key">{key}</dt>
                <dd className="guided-recipe-slot-val">
                  {formatSlotValue(payload.slots[key])}
                </dd>
              </div>
            ))}
          </dl>
        )}

        {/* Alternatives list -- only rendered when non-empty (negative-space pin).
            Displayed for user information; no submit path for alternatives exists
            in the current backend (only ["accept"] and ["build_manually"] are
            valid).  See SCOPE NOTE in the file header. */}
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

        {/* Action row: Apply recipe + Build manually.
            Two separate handlers; each constructs its own body literal.
            No "construct chosen body" helper extracted: rule of three requires
            a third widget before justifying extraction (ProposeChainTurn + this
            widget = 2 sites). */}
        <div className="guided-recipe-actions">
          <button
            type="button"
            className="guided-recipe-apply-btn"
            onClick={handleApply}
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
