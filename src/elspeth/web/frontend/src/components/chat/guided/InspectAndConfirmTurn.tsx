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
//   - CSS via App.css class names with design tokens; no hardcoded colours
//
// SHAPE NOTE (differs from chip-group widgets):
// This widget does NOT use <fieldset>+<legend> or chip buttons.
// It has fundamentally different semantic structure:
//   inspect-view: <table> of columns + samples, optional warnings <aside>, two actions
//   edit-view:    per-column rename inputs + remove buttons, cancel/apply actions
//
// Edit-view state:
//   editedColumns: string[] — starts as a copy of payload.observed.columns at
//   edit-mode entry; mutated by renames and removals; null signals not-yet-entered.
//   Rows (samples) and warnings pass through unchanged on both submit paths.
//
// Wire-response shapes:
//   "Looks right": edited_values = { columns, samples, warnings } verbatim from payload.observed
//   "Apply edits": edited_values = { columns: <edited>, samples: payload.observed.samples, warnings: payload.observed.warnings }

import { useId, useState } from "react";
import type { GuidedRespondRequest, InspectAndConfirmPayload } from "@/types/guided";

interface InspectAndConfirmTurnProps {
  payload: InspectAndConfirmPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
}

export function InspectAndConfirmTurn({ payload, onSubmit }: InspectAndConfirmTurnProps) {
  const [editing, setEditing] = useState(false);
  // null = not yet entered edit mode; non-null = user is editing (may equal original)
  const [editedColumns, setEditedColumns] = useState<string[] | null>(null);

  // useId scopes DOM IDs per-instance so multiple InspectAndConfirmTurns rendered
  // simultaneously (e.g. active turn + GuidedHistory replay in Task 7.9) don't
  // produce id collisions when edit-mode input IDs recur across turns.
  const reactId = useId();
  const warningsId = `${reactId}-warnings`;
  const columnInputId = (index: number) => `${reactId}-col-${index}`;

  function handleLooksRight() {
    onSubmit({
      chosen: null,
      edited_values: {
        columns: payload.observed.columns,
        samples: payload.observed.samples,
        warnings: payload.observed.warnings,
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  function handleOpenEditor() {
    // Copy columns array so edits don't mutate the original payload
    setEditedColumns([...payload.observed.columns]);
    setEditing(true);
  }

  function handleCancelEdit() {
    setEditing(false);
    setEditedColumns(null);
  }

  function handleRenameColumn(index: number, newName: string) {
    setEditedColumns((prev) => {
      if (prev === null) return prev;
      const next = [...prev];
      next[index] = newName;
      return next;
    });
  }

  function handleRemoveColumn(index: number) {
    setEditedColumns((prev) => {
      if (prev === null) return prev;
      return prev.filter((_, i) => i !== index);
    });
  }

  function handleApplyEdits() {
    // editedColumns is non-null when editing is true; assert to satisfy TS
    const cols = editedColumns ?? payload.observed.columns;
    onSubmit({
      chosen: null,
      edited_values: {
        columns: cols,
        samples: payload.observed.samples,
        warnings: payload.observed.warnings,
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  // ── Edit view ────────────────────────────────────────────────────────────────
  if (editing) {
    const cols = editedColumns ?? payload.observed.columns;
    return (
      <div className="guided-turn guided-inspect-turn">
        <p className="guided-inspect-edit-heading">Edit columns</p>
        <ul className="guided-inspect-editor-list">
          {cols.map((col, index) => (
            <li key={`${reactId}-${index}`} className="guided-inspect-editor-row">
              <label
                htmlFor={columnInputId(index)}
                className="guided-inspect-editor-label"
              >
                Column {index + 1}
              </label>
              <input
                id={columnInputId(index)}
                type="text"
                className="guided-inspect-editor-input"
                value={col}
                onChange={(e) => handleRenameColumn(index, e.target.value)}
              />
              <button
                type="button"
                className="guided-inspect-remove-btn"
                onClick={() => handleRemoveColumn(index)}
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
          >
            Cancel
          </button>
          <button
            type="button"
            className="guided-inspect-apply-btn"
            onClick={handleApplyEdits}
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
        >
          Looks right
        </button>
        <button
          type="button"
          className="guided-inspect-edit-btn"
          onClick={handleOpenEditor}
        >
          Edit columns...
        </button>
      </div>
    </div>
  );
}
