// src/components/chat/guided/InspectAndConfirmTurn.tsx
//
// Guided-mode widget for the inspect_and_confirm turn type (Task 7.3).
// Conventions inherited from SingleSelectTurn (Task 7.2 template):
//   - Props: { payload: InspectAndConfirmPayload; onSubmit: (body: GuidedRespondRequest) => void }
//   - onSubmit is SYNC — the widget constructs the body; the store awaits the round-trip
//   - All 6 GuidedRespondRequest fields set explicitly; unused ones = null (no omission)
//   - <button type="button"> (never <div onClick>)
//   - DOM IDs prefixed with React 18 useId() — multiple turn instances coexist in
//     GuidedHistory (Task 7.9) and element IDs would collide without per-instance scoping
//   - Visible labels (htmlFor / button text) ARE the accessible name; do not add
//     redundant aria-label that overrides what sighted users see
//   - CSS via components/chat/guided/guided.css class names with design tokens; no hardcoded colours
//
// SHAPE NOTE (differs from chip-group widgets):
// This widget does NOT use <fieldset>+<legend> or chip buttons.
// It has fundamentally different semantic structure:
//   inspect-view: <table> of columns + samples, optional warnings <aside>, two actions
//   edit-view:    per-column rename inputs + remove buttons, cancel/apply actions
//
// State encoding (load-bearing for 7.5 / 7.6 multi-view widgets):
//   editorState: { columns: string[] } | null
//   null     → inspect view; non-null → edit view, with the edited columns array.
//   A single nullable struct makes illegal states unrepresentable: there is no
//   way to be in "edit mode without edited columns" or "inspect mode while
//   holding stale edits". TS narrowing inside `editorState !== null` proves
//   `.columns` is set, removing the need for any `?? fallback` at submit time.
//
// Focus management (convention for 7.5 / 7.6):
//   Toggling between inspect <-> edit unmounts the action buttons that received
//   the click, dumping keyboard focus to <body>. We restore focus explicitly:
//     - Entering edit mode  → focus the first column input.
//     - Returning to inspect view → focus the "Edit columns..." button.
//   Refs + an effect on `editorState !== null` perform the restore after the
//   new view mounts. The effect skips its first run (initial widget mount)
//   because the user hasn't requested a view change — they've just received
//   a new turn from the protocol.
//
// Warnings accessibility (convention for 7.4-7.7 widgets with passive regions):
//   The warnings <aside> does NOT declare its own aria-live region. The parent
//   ChatPanel's guided-active branch wraps the turn surface in a
//   `<div className="chat-panel-guided-log" role="log" aria-live="polite">`
//   region (see `ChatPanel.tsx` — search for `chat-panel-guided-log`), which
//   announces the warnings on widget mount.
//   ComposingIndicator follows the same "don't nest live regions" convention
//   (see `ComposingIndicator.test.tsx` — the "ComposingIndicator live region
//   scope" describe block). If a future maintainer removes
//   the parent live region, warnings will be silent — the contract is documented
//   here so the dependency is discoverable.
//
// Wire-response shapes (narrow contract — state_machine.SourceIntent holds the rest server-side):
//   "Looks right": edited_values = { columns: payload.observed.columns }
//   "Apply edits": edited_values = { columns: <edited> }
//   plugin, options, and sample_rows are held in step_1_source_intent on the server;
//   the widget never sends them back. The server recovers them from intent on advance.
//   warnings come from SourceInspectionFacts at emit time (advisory only, not stored in intent).

import { useEffect, useId, useRef, useState } from "react";
import type { GuidedRespondRequest, InspectAndConfirmPayload } from "@/types/guided";

interface InspectAndConfirmTurnProps {
  payload: InspectAndConfirmPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
  disabled?: boolean;
}

/** Edit-mode state. `null` = inspect view; non-null = edit view. */
interface EditorState {
  columns: string[];
}

export function InspectAndConfirmTurn({
  payload,
  onSubmit,
  disabled = false,
}: InspectAndConfirmTurnProps) {
  const [editorState, setEditorState] = useState<EditorState | null>(null);

  // useId scopes DOM IDs per-instance so multiple InspectAndConfirmTurns rendered
  // simultaneously (e.g. active turn + GuidedHistory replay in Task 7.9) don't
  // produce id collisions when edit-mode input IDs recur across turns.
  const reactId = useId();
  const warningsId = `${reactId}-warnings`;
  const columnInputId = (index: number) => `${reactId}-col-${index}`;

  // Focus-restoration refs. Attached only in the view that owns each element.
  const firstEditInputRef = useRef<HTMLInputElement | null>(null);
  const editButtonRef = useRef<HTMLButtonElement | null>(null);

  // Skip the first effect run: on initial widget mount the user did NOT toggle
  // the view — the widget appeared because a new turn arrived. Auto-focusing
  // the "Edit columns..." button on mount would steal focus from wherever the
  // user actually was (e.g. the chat input). Only restore focus on subsequent
  // toggles, which ARE user-initiated.
  const firstRunRef = useRef(true);
  const isEditing = editorState !== null;
  useEffect(() => {
    if (firstRunRef.current) {
      firstRunRef.current = false;
      return;
    }
    if (isEditing) {
      firstEditInputRef.current?.focus();
    } else {
      editButtonRef.current?.focus();
    }
  }, [isEditing]);

  function handleLooksRight() {
    onSubmit({
      chosen: null,
      edited_values: {
        columns: payload.observed.columns,
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  function handleOpenEditor() {
    // Copy columns array so edits don't mutate the original payload.
    setEditorState({ columns: [...payload.observed.columns] });
  }

  function handleCancelEdit() {
    setEditorState(null);
  }

  function handleRenameColumn(index: number, newName: string) {
    setEditorState((prev) => {
      if (prev === null) return prev;
      const next = [...prev.columns];
      next[index] = newName;
      return { columns: next };
    });
  }

  function handleRemoveColumn(index: number) {
    setEditorState((prev) => {
      if (prev === null) return prev;
      return { columns: prev.columns.filter((_, i) => i !== index) };
    });
  }

  function handleApplyEdits() {
    // TS narrowing: this handler is only reachable from the edit-view branch
    // (which guards `editorState !== null`), but the type system can't prove
    // that across the render boundary. The early-return is an offensive
    // invariant check — illegal state would be a code bug worth catching, not
    // user data to coerce. Returning silently here matches the convention used
    // in handleRenameColumn / handleRemoveColumn (state updaters fast-out when
    // prev is null) and keeps the function total under TS narrowing.
    if (editorState === null) return;
    onSubmit({
      chosen: null,
      edited_values: {
        columns: editorState.columns,
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  // ── Edit view ────────────────────────────────────────────────────────────────
  if (editorState !== null) {
    return (
      <div className="guided-turn guided-inspect-turn">
        <h3 className="guided-inspect-edit-heading">Edit columns</h3>
        <ul className="guided-inspect-editor-list">
          {editorState.columns.map((col, index) => (
            // Positional key: stable across renames (input keeps DOM identity ->
            // focus and IME state survive typing). On removal, surviving keys
            // shift and React reconciles the moved inputs as the same nodes with
            // new value props — behaviourally correct under controlled inputs.
            // Do NOT switch to `${reactId}-${col}` (content-based): renames
            // would remount the input mid-typing and lose focus.
            <li key={`${reactId}-${index}`} className="guided-inspect-editor-row">
              <label
                htmlFor={columnInputId(index)}
                className="guided-inspect-editor-label"
              >
                Column {index + 1}
              </label>
              <input
                id={columnInputId(index)}
                ref={index === 0 ? firstEditInputRef : null}
                type="text"
                className="guided-inspect-editor-input"
                value={col}
                disabled={disabled}
                onChange={(e) => handleRenameColumn(index, e.target.value)}
              />
              <button
                type="button"
                className="guided-inspect-remove-btn"
                onClick={() => handleRemoveColumn(index)}
                disabled={disabled}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
        <div className="guided-inspect-editor-actions">
          <button
            type="button"
            className="guided-inspect-cancel-btn"
            onClick={handleCancelEdit}
            disabled={disabled}
          >
            Cancel
          </button>
          <button
            type="button"
            className="guided-inspect-apply-btn"
            onClick={handleApplyEdits}
            disabled={disabled}
          >
            Apply edits
          </button>
        </div>
      </div>
    );
  }

  // ── Inspect view (default) ───────────────────────────────────────────────────
  return (
    <div className="guided-turn guided-inspect-turn">
      <table className="guided-inspect-table">
        <thead>
          <tr>
            {payload.observed.columns.map((col) => (
              <th key={col} className="guided-inspect-th" scope="col">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {payload.observed.samples.map((sample, rowIndex) => (
            <tr key={rowIndex} className="guided-inspect-tr">
              {payload.observed.columns.map((col) => (
                <td key={col} className="guided-inspect-td">
                  {String(sample[col] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {payload.observed.warnings.length > 0 && (
        <aside
          id={warningsId}
          className="guided-inspect-warnings"
          aria-label="Data warnings"
        >
          <ul className="guided-inspect-warnings-list">
            {payload.observed.warnings.map((warning, i) => (
              <li key={i} className="guided-inspect-warning-item">
                {warning}
              </li>
            ))}
          </ul>
        </aside>
      )}

      <div className="guided-inspect-actions">
        <button
          type="button"
          className="guided-inspect-confirm-btn"
          onClick={handleLooksRight}
          disabled={disabled}
        >
          Looks right
        </button>
        <button
          ref={editButtonRef}
          type="button"
          className="guided-inspect-edit-btn"
          onClick={handleOpenEditor}
          disabled={disabled}
        >
          Edit columns...
        </button>
      </div>
    </div>
  );
}
