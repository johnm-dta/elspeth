import { useMemo } from "react";
import { InterpretationReviewTurn } from "./InterpretationReviewTurn";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { InterpretationEvent } from "@/types/interpretation";

interface GuidedInterpretationReviewsProps {
  sessionId: string;
}

function byCreatedAt(a: InterpretationEvent, b: InterpretationEvent): number {
  const order = a.created_at.localeCompare(b.created_at);
  return order !== 0 ? order : a.id.localeCompare(b.id);
}

/**
 * Projects pending interpretation-review events for the guided session and
 * renders each via InterpretationReviewTurn. The guided ChatPanel branch
 * blocks advancement while any pending remains (D12). The GuidedTurn
 * interpretation_review case is dead code; this store projection is the path.
 *
 * Returns null when there is nothing to review (mirrors the
 * TutorialTurn2bShowBuilt.tsx projection, minus its empty-state paragraph —
 * the guided branch keeps its own copy/affordances).
 */
export function GuidedInterpretationReviews({
  sessionId,
}: GuidedInterpretationReviewsProps): JSX.Element | null {
  const pendingBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  const pending = useMemo(() => {
    const events = Object.values(pendingBySession[sessionId] ?? {});
    return events
      .filter(
        (event) =>
          event.choice === "pending" &&
          event.interpretation_source === "user_approved",
      )
      .sort(byCreatedAt);
  }, [pendingBySession, sessionId]);

  if (pending.length === 0) return null;

  return (
    <section
      className="guided-interpretation-reviews"
      aria-label="Assumptions to review"
    >
      <p className="guided-interpretation-count" role="status">
        {pending.length}{" "}
        {pending.length === 1 ? "assumption" : "assumptions"} to review
      </p>
      {pending.map((event, index) => (
        <InterpretationReviewTurn
          key={event.id}
          event={event}
          sessionId={sessionId}
          showOptOut={false}
          showAmend={event.kind === "vague_term"}
          autoFocusOnMount={index === 0}
          onResolved={(newState) => {
            if (newState !== null) {
              useSessionStore.setState({ compositionState: newState });
            }
          }}
        />
      ))}
    </section>
  );
}

/**
 * True when the guided session has at least one pending user_approved
 * interpretation card — the predicate the ChatPanel guided branch uses to
 * disable the wizard turn while reviews are outstanding (D12).
 */
export function useHasPendingGuidedInterpretations(sessionId: string): boolean {
  const pendingBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  return useMemo(() => {
    const events = Object.values(pendingBySession[sessionId] ?? {});
    return events.some(
      (event) =>
        event.choice === "pending" &&
        event.interpretation_source === "user_approved",
    );
  }, [pendingBySession, sessionId]);
}
