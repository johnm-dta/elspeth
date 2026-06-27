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

  // One row per step: a step can emit several turns (e.g. a single_select then a
  // schema_form), which previously rendered as duplicate "Source" rows. Collapse
  // to the most-recent turn that carries a SUMMARY, falling back to the most
  // recent turn only when the step has produced no summary yet. Preferring the
  // summarised record stops a freshly-emitted, not-yet-answered next-turn record
  // (summary === null) from masking the step's actual decision. The Map preserves
  // first-seen step order.
  const latestByStep = new Map<GuidedStep, TurnRecord>();
  for (const turn of history) {
    const existing = latestByStep.get(turn.step);
    if (existing === undefined) {
      latestByStep.set(turn.step, turn);
    } else if (turn.summary !== null) {
      // A summarised record always supersedes (most-recent summary wins).
      latestByStep.set(turn.step, turn);
    } else if (existing.summary === null) {
      // Neither is summarised: keep the most recent as the fallback.
      latestByStep.set(turn.step, turn);
    }
    // else: existing is summarised and this one is not — keep the summary.
  }
  const rows = [...latestByStep.values()];

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
            {/* Show the decision summary only. The prior code fell back to the
                widget-type label ("Single select") when summary was null, which
                read as a meaningless "decision"; a step with no summary now
                shows a muted "Decided" instead of leaking the widget type. */}
            <span className="guided-history-summary">
              {turn.summary ?? "Decided"}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
