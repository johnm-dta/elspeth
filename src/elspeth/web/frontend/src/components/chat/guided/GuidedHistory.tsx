// src/components/chat/guided/GuidedHistory.tsx
//
// Guided-mode decision summary. This is deliberately not a protocol log: the
// operator needs a visible recap of choices made so far, while low-level
// emitter/type/hash details stay out of the default workflow surface.
import type { GuidedStep, TurnRecord, TurnType } from "@/types/guided";

// ── Display mappings ─────────────────────────────────────────────────────────

/**
 * Human-readable labels for each GuidedStep enum value.
 * CLOSED LIST — mirrors GuidedStep in types/guided.ts:28-32.
 * Update here whenever protocol.py adds a new step.
 */
const STEP_LABELS: Record<GuidedStep, string> = {
  step_1_source: "Source",
  step_2_sink: "Output",
  step_2_5_recipe_match: "Recipe",
  step_3_transforms: "Transforms",
  step_4_wire: "Wire",
};

/**
 * Human-readable labels for turn types.
 * CLOSED LIST — mirrors TurnType in types/guided.ts:15-21.
 */
const TURN_TYPE_LABELS: Record<TurnType, string> = {
  inspect_and_confirm: "Inspect and confirm",
  single_select: "Single select",
  multi_select_with_custom: "Multi-select with custom fields",
  schema_form: "Schema form",
  propose_chain: "Proposed chain",
  recipe_offer: "Recipe offer",
  // Phase 5b — interpretation-review history label.  The widget itself lands
  // in Task 4 of 18b-phase-5b-frontend.md; this label entry exists so the
  // Record<TurnType, string> exhaustive-keys contract holds the moment the
  // TurnType union widens.
  interpretation_review: "Interpretation review",
  confirm_wiring: "Confirm wiring",
};

// ── Component ────────────────────────────────────────────────────────────────

interface Props {
  /** Completed turn records from GuidedSession.history. */
  history: TurnRecord[];
}

/**
 * Read-only plain-language list of completed guided-mode decisions.
 *
 * Returns null when history is empty — the parent should not render this
 * widget unless there are completed steps to show.
 */
export function GuidedHistory({ history }: Props): React.ReactElement | null {
  // Empty history: nothing to show.
  if (history.length === 0) {
    return null;
  }

  return (
    <section
      className="guided-history"
      aria-labelledby="guided-history-heading"
    >
      <h2 id="guided-history-heading" className="guided-history-heading">
        Decisions so far
      </h2>
      <ol className="guided-history-list">
        {history.map((turn, idx) => (
          <li key={`${turn.step}-${idx}`} className="guided-history-item">
            <span className="guided-history-step-name">
              {STEP_LABELS[turn.step]}
            </span>
            <span className="guided-history-summary">
              {turn.summary ?? TURN_TYPE_LABELS[turn.turn_type]}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
