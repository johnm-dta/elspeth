// src/components/chat/guided/GuidedHistory.tsx
//
// Guided-mode decision summary. This is deliberately not a protocol log: the
// operator needs a visible recap of choices made so far, while low-level
// emitter/type/hash details stay out of the default workflow surface.
import type { GuidedStep, TurnRecord } from "@/types/guided";

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

// ── Component ────────────────────────────────────────────────────────────────

interface Props {
  /** Completed turn records from GuidedSession.history. */
  history: TurnRecord[];
  /**
   * The step the learner is currently on. The in-progress step is NOT a
   * completed decision, so its turns are excluded even when an already-answered
   * sub-turn (e.g. the source single_select before the source schema_form) has
   * recorded a summary.
   */
  currentStep: GuidedStep;
}

/**
 * Read-only plain-language list of completed guided-mode decisions.
 *
 * Returns null when history is empty — the parent should not render this
 * widget unless there are completed steps to show.
 */
export function GuidedHistory({ history, currentStep }: Props): React.ReactElement | null {
  // Empty history: nothing to show.
  if (history.length === 0) {
    return null;
  }

  // One row per COMPLETED step — a step the learner has moved PAST. The current
  // step is in progress (its answered single_select may already carry a summary
  // while a schema_form sub-turn is still pending), so it is never "decided" yet
  // and is excluded outright. Among past steps, keep the most-recent summarised
  // turn (Map.set on the chronological history wins last); a step that produced
  // no summary at all contributes nothing. The Map preserves first-seen order.
  const latestByStep = new Map<GuidedStep, TurnRecord>();
  for (const turn of history) {
    if (turn.step === currentStep) continue;
    if (turn.summary === null) continue;
    latestByStep.set(turn.step, turn);
  }
  const rows = [...latestByStep.values()];

  // No completed decisions yet (e.g. still on the first step before any Send):
  // render nothing rather than an empty "Decisions so far" card.
  if (rows.length === 0) {
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
        {rows.map((turn) => (
          <li key={turn.step} className="guided-history-item">
            <span className="guided-history-step-name">
              {STEP_LABELS[turn.step]}
            </span>
            {/* Only past, summarised steps reach here (the current step and
                summary-null turns are filtered above), so every row has a real
                summary — no fallback needed. */}
            <span className="guided-history-summary">
              {turn.summary}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
