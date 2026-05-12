// src/components/chat/guided/GuidedHistory.tsx
//
// Guided-mode collapsible read-only list of completed wizard steps (Task 7.9).
// Conventions inherited from the 7.x widget family:
//   - useId() per-instance scoping (Task 7.4 I2): the toggle↔region
//     aria-controls pair uses a useId() prefix so multiple GuidedHistory
//     instances coexist without DOM id collisions.
//   - aria-controls + hidden attribute (Task 7.5 I2): the collapsible region is
//     ALWAYS rendered (hidden attribute toggles) so aria-controls resolves in
//     both expanded and collapsed states. The conditional {expanded && children}
//     pattern is BROKEN for aria-controls — a screen reader reaching the toggle
//     while collapsed cannot resolve "expanded/collapsed WHAT" if the id is a
//     dangling reference. Children remain gated on `expanded` to skip rendering
//     work when collapsed.
//   - <button type="button"> — never <div onClick>.
//   - CSS via App.css class names with design tokens; no hardcoded colours.
//   - Read-only: no onSubmit, no event handlers other than the toggle.
//   - No auto-focus on mount (firstRunRef pattern inherited from
//     InspectAndConfirmTurn.tsx): the widget appears because a turn was
//     completed, not because the user requested focus.
//
// SCOPE REDUCTION (wire-shape gap):
//   The plan body described rich summaries like "Step 1: CSV with cols [price, qty]".
//   The actual TurnRecordResponse wire shape (schemas.py:213-220) carries only
//   step, turn_type, payload_hash, response_hash, and emitter — no summary field.
//   GuidedHistory therefore renders only step label + turn type + emitter.
//   To enable rich summaries, the backend would need a cross-layer protocol
//   change (adding a denormalized summary string to TurnRecordResponse), which
//   requires operator adjudication before implementation.
//   Tracker: elspeth-obs-cc8fa78524.

import { useId, useState } from "react";
import type { GuidedStep, TurnRecord, TurnType } from "@/types/guided";

// ── Display mappings ─────────────────────────────────────────────────────────

/**
 * Human-readable labels for each GuidedStep enum value.
 * CLOSED LIST — mirrors GuidedStep in types/guided.ts:28-32.
 * Update here whenever protocol.py adds a new step.
 */
const STEP_LABELS: Record<GuidedStep, string> = {
  step_1_source: "Source",
  step_2_sink: "Sink",
  step_2_5_recipe_match: "Recipe match",
  step_3_transforms: "Transforms",
};

/**
 * Human-readable labels for turn types.
 * CLOSED LIST — mirrors TurnType in types/guided.ts:15-21.
 */
const TURN_TYPE_LABELS: Record<TurnType, string> = {
  inspect_and_confirm: "inspect_and_confirm",
  single_select: "single_select",
  multi_select_with_custom: "multi_select_with_custom",
  schema_form: "schema_form",
  propose_chain: "propose_chain",
  recipe_offer: "recipe_offer",
};

// ── Component ────────────────────────────────────────────────────────────────

interface Props {
  /** Completed turn records from GuidedSession.history. */
  history: TurnRecord[];
  /** When true, the list starts expanded. Defaults to false. */
  initiallyExpanded?: boolean;
}

/**
 * Collapsible read-only list of completed guided-mode wizard steps.
 *
 * Returns null when history is empty — the parent should not render this
 * widget unless there are completed steps to show.
 */
export function GuidedHistory({ history, initiallyExpanded = false }: Props): React.ReactElement | null {
  const [expanded, setExpanded] = useState(initiallyExpanded);
  const reactId = useId();
  const regionId = `${reactId}-history-region`;

  // Empty history: nothing to show.
  if (history.length === 0) {
    return null;
  }

  const count = history.length;
  const toggleLabel = expanded
    ? `Hide steps (${count})`
    : `Show steps (${count})`;

  return (
    <div className="guided-history">
      {/* Toggle button — aria-controls always resolves because the region is
          always rendered (hidden toggles, not conditional mounting). */}
      <button
        type="button"
        className="guided-history-toggle"
        aria-expanded={expanded}
        aria-controls={regionId}
        onClick={() => setExpanded((prev) => !prev)}
      >
        {toggleLabel}
      </button>

      {/* Region container: ALWAYS rendered so aria-controls resolves in both
          expanded and collapsed states (Task 7.5 I2 contract).
          The `hidden` attribute removes this from the AT tree when collapsed —
          no spurious "empty region" announcement — while keeping the id
          resolvable. Children are gated on `expanded` to skip render work
          while collapsed. */}
      <div
        id={regionId}
        role="region"
        aria-label="Wizard step history"
        className="guided-history-region"
        hidden={!expanded}
      >
        {expanded && (
          <ol className="guided-history-list">
            {history.map((turn, idx) => (
              <li key={`${turn.step}-${idx}`} className="guided-history-item">
                <span className="guided-history-step-number">
                  Step {idx + 1}
                </span>
                <span className="guided-history-step-name">
                  {STEP_LABELS[turn.step]}
                </span>
                <span className="guided-history-separator" aria-hidden="true">
                  ·
                </span>
                <span className="guided-history-turn-type">
                  {TURN_TYPE_LABELS[turn.turn_type]}
                </span>
                <span className="guided-history-separator" aria-hidden="true">
                  ·
                </span>
                <span className="guided-history-emitter">{turn.emitter}</span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}
