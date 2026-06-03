// src/components/chat/guided/ProposeChainTurn.tsx
//
// Guided-mode widget for the propose_chain turn type (Task 7.6).
// Conventions inherited from SingleSelectTurn (Task 7.2 template),
// InspectAndConfirmTurn (Task 7.3), MultiSelectWithCustomTurn (Task 7.4),
// and SchemaFormTurn (Task 7.5).
//
//   - Props: { payload: ProposeChainPayload; onSubmit: (body: GuidedRespondRequest) => void }
//   - onSubmit is SYNC -- the widget constructs the body; the store awaits the round-trip
//   - All 6 GuidedRespondRequest fields set explicitly; unused ones = null (no omission)
//   - <button type="button"> (never <div onClick>)
//   - DOM IDs prefixed with React 18 useId() -- multiple turn instances coexist in
//     GuidedHistory (Task 7.9) and element IDs would collide without per-instance scoping
//   - Visible labels (button text) ARE the accessible name; no redundant aria-label
//   - CSS via components/chat/guided/guided.css class names with design tokens; no hardcoded colours
//
// SHAPE NOTE (differs from chip-group widgets):
// propose_chain uses a card-list layout (per Task 7.2 SHAPE NOTE which explicitly
// excludes propose_chain from the chip-group family).  Each step is a card.
// No <fieldset>/<legend>.  The accept button is at the bottom of the card list.
//
// WIRE-SHAPES:
//   Accept all steps: chosen ["accept"].
//   Edit step N:      edit_step_index = N.
//   Reject:           control_signal = "reject".
//   Ask advisor:      control_signal = "request_advisor".
// All GuidedRespondRequest fields are explicit on every handler.
//
// OPTIONS RENDERING:
//   step.options is an arbitrary mapping (Mapping[str, Any] on the wire).  Each
//   key-value pair is rendered as a <dl> definition list: <dt> for the key,
//   <dd> for the value.  Non-scalar values (arrays, objects) are stringified via
//   JSON.stringify so they remain human-readable without a full JSON viewer.
//   This is the simplest renderable shape for the demo path; a richer diff view
//   (collapsible JSON tree etc.) is deferred to a future task.
//
// STATE:
//   Near-zero widget-side state.  Accept / Edit / Ask-advisor each construct and
//   emit the body literal immediately on click.  The single exception is Reject,
//   which gates the emit on a ConfirmDialog (rejecting a multi-step plan is a
//   destructive operation that deserves an explicit confirm step per the
//   button-audit S3.5 finding).  No view transitions, so the firstRunRef
//   focus-skip pattern from InspectAndConfirmTurn and SchemaFormTurn is not
//   needed.
//
// FOCUS MANAGEMENT:
//   No programmatic focus on mount or on interaction.  All four prior widgets
//   that use focus management do so because a view transition reveals new content
//   (InspectAndConfirmTurn, SchemaFormTurn) or a destructive interaction removes
//   an element (MultiSelectWithCustomTurn).  ProposeChainTurn renders statically;
//   no equivalent trigger exists.

import { useId, useState } from "react";
import type { GuidedRespondRequest, ProposeChainPayload } from "@/types/guided";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";

interface ProposeChainTurnProps {
  payload: ProposeChainPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
  disabled?: boolean;
}

/**
 * Format a single option value for display in the key-value definition list.
 *
 * Scalars (string, number, boolean) are coerced to string.  Null/undefined
 * render as "null".  Arrays and objects are JSON-stringified so the user sees
 * the structure without a full JSON editor component.
 */
function formatOptionValue(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

export function ProposeChainTurn({
  payload,
  onSubmit,
  disabled = false,
}: ProposeChainTurnProps) {
  // useId scopes DOM IDs per-instance so multiple ProposeChainTurns rendered
  // simultaneously (e.g. active turn + GuidedHistory replay in Task 7.9) don't
  // produce id collisions when per-card IDs recur across turns.
  const reactId = useId();
  const cardId = (index: number) => `${reactId}-step-${index}`;
  const optionsId = (index: number) => `${reactId}-opts-${index}`;
  const [rejectConfirmOpen, setRejectConfirmOpen] = useState(false);

  function handleAccept() {
    onSubmit({
      chosen: ["accept"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  function handleEdit(index: number) {
    onSubmit({
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: index,
      control_signal: null,
    });
  }

  function handleReject() {
    onSubmit({
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: "reject",
    });
  }

  function handleAskAdvisor() {
    onSubmit({
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: "request_advisor",
    });
  }

  return (
    <div className="guided-turn guided-propose-chain">
      {/* Overall rationale paragraph */}
      <p className="guided-propose-why">{payload.why}</p>

      {/* Blockers list -- only rendered when non-empty (negative-space pin #3) */}
      {payload.blockers.length > 0 && (
        <div className="guided-propose-blockers">
          <p className="guided-propose-blockers-heading">Blockers identified:</p>
          <ul className="guided-propose-blockers-list">
            {payload.blockers.map((blocker, idx) => (
              <li key={idx} className="guided-propose-blocker-item">
                {blocker}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Step cards -- card-list layout, one card per step */}
      <ol className="guided-propose-steps">
        {payload.steps.map((step, idx) => {
          const optKeys = Object.keys(step.options as Record<string, unknown>);
          return (
            <li
              key={idx}
              id={cardId(idx)}
              className="guided-propose-step-card"
              aria-label={`Step ${idx + 1}: ${step.plugin}`}
            >
              <div className="guided-propose-step-header">
                <span className="guided-propose-step-number">
                  {idx + 1}
                </span>
                {/* <h3> (not <span>) so screen-reader users can navigate the
                    proposed chain by heading landmarks. The CSS rule on
                    .guided-propose-step-plugin already pins font-size to
                    --font-size-base, so the visual rank does not jump to the
                    browser-default <h3> size. Same convention as Task 7.3 M10
                    (edit-mode heading). */}
                <h3 className="guided-propose-step-plugin">{step.plugin}</h3>
              </div>

              {/* Options as a key-value definition list.
                  <dl> is used because (key, value) pairs semantically form a
                  description/definition relationship -- <dt> for key, <dd> for
                  value.  A <table> would be heavier; a plain div grid would
                  lose the semantic association between keys and their values. */}
              {optKeys.length > 0 && (
                <dl id={optionsId(idx)} className="guided-propose-options">
                  {optKeys.map((key) => (
                    <div key={key} className="guided-propose-option-row">
                      <dt className="guided-propose-option-key">{key}</dt>
                      <dd className="guided-propose-option-val">
                        {formatOptionValue(
                          (step.options as Record<string, unknown>)[key],
                        )}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}

              <p className="guided-propose-step-rationale">{step.rationale}</p>
              <div className="guided-propose-step-actions">
                <button
                  type="button"
                  className="guided-propose-edit-btn"
                  onClick={() => handleEdit(idx)}
                  disabled={disabled}
                >
                  Edit step {idx + 1}
                </button>
              </div>
            </li>
          );
        })}
      </ol>

      <div className="guided-propose-actions">
        <button
          type="button"
          className="guided-propose-secondary-btn"
          onClick={() => setRejectConfirmOpen(true)}
          disabled={disabled}
        >
          Reject
        </button>
        <button
          type="button"
          className="guided-propose-secondary-btn"
          onClick={handleAskAdvisor}
          disabled={disabled}
        >
          Ask advisor
        </button>
        <button
          type="button"
          className="guided-propose-accept-btn"
          onClick={handleAccept}
          disabled={disabled}
        >
          Accept all steps
        </button>
      </div>
      {rejectConfirmOpen && (
        <ConfirmDialog
          title="Reject this plan?"
          message="The composer's multi-step plan will be discarded. You can ask the composer to revise the plan or propose a different approach afterwards."
          confirmLabel="Reject plan"
          cancelLabel="Keep open"
          variant="danger"
          onConfirm={() => {
            setRejectConfirmOpen(false);
            handleReject();
          }}
          onCancel={() => setRejectConfirmOpen(false)}
        />
      )}
    </div>
  );
}
