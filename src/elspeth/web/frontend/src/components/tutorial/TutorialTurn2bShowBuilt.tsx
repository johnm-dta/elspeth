import { useEffect, useMemo, useRef } from "react";
import { InterpretationReviewTurn } from "@/components/chat/guided/InterpretationReviewTurn";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { InterpretationEvent } from "@/types/interpretation";
import {
  TURN_2B_ASSUMPTIONS_BLOCKED_STATUS,
  TURN_2B_ASSUMPTIONS_READY_STATUS,
  TURN_2B_PRIMARY_BUTTON,
} from "./copy";
import type { TutorialBuiltSummary } from "./tutorialMachine";

interface TutorialTurn2bShowBuiltProps {
  sessionId: string;
  summary: TutorialBuiltSummary;
  onContinue: () => void;
  onBack: () => void;
}

export function TutorialTurn2bShowBuilt({
  sessionId,
  summary,
  onContinue,
  onBack,
}: TutorialTurn2bShowBuiltProps): JSX.Element {
  const pendingBySession = useInterpretationEventsStore(
    (state) => state.pendingBySession,
  );
  const headingRef = useRef<HTMLHeadingElement | null>(null);

  useEffect(() => {
    headingRef.current?.focus();
  }, []);
  const pendingInterpretations = useMemo(() => {
    const events = Object.values(pendingBySession[sessionId] ?? {});
    return events
      .filter(
        (event) =>
          event.choice === "pending" &&
          event.interpretation_source === "user_approved",
      )
      .sort(compareInterpretationEventsByCreatedAt);
  }, [pendingBySession, sessionId]);
  const pendingCount = pendingInterpretations.length;
  const hasPendingInterpretations = pendingCount > 0;
  const assumptionStatus = hasPendingInterpretations
    ? TURN_2B_ASSUMPTIONS_BLOCKED_STATUS
    : TURN_2B_ASSUMPTIONS_READY_STATUS;

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-built-title">
      <p className="tutorial-kicker">Draft</p>
      <h2 id="tutorial-built-title" ref={headingRef} tabIndex={-1}>
        Here is what the composer drafted - review its assumptions.
      </h2>
      <ol className="tutorial-summary-grid" aria-label="Pipeline layers">
        <li>
          <h3>Source</h3>
          <p>{summary.sourceLabel}</p>
          {summary.urls.length > 0 ? (
            <ul>
              {summary.urls.map((url) => (
                <li key={url}>{url}</li>
              ))}
            </ul>
          ) : (
            <p className="tutorial-muted">
              The generated source is attached to the session state.
            </p>
          )}
        </li>
        <li>
          <h3>Transform</h3>
          {summary.transforms.length > 0 ? (
            <ul>
              {summary.transforms.map((transform) => (
                <li key={transform}>{transform}</li>
              ))}
            </ul>
          ) : (
            <p className="tutorial-muted">No transform nodes were returned.</p>
          )}
        </li>
        <li>
          <h3>Sink</h3>
          <p>{summary.sinkLabel}</p>
        </li>
      </ol>

      <p role="status" className="sr-only">
        {assumptionStatus}
      </p>

      {hasPendingInterpretations ? (
        <div className="tutorial-interpretation">
          <p className="tutorial-assumption-count">
            {pendingCount} {pendingCount === 1 ? "assumption" : "assumptions"}{" "}
            to review
          </p>
          {pendingInterpretations.map((event, index) => (
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
        </div>
      ) : (
        <p className="tutorial-muted">
          If the draft surfaced an interpretation to review, it appears here.
          Otherwise, the run will still record the prompt and model calls.
        </p>
      )}

      <div className="tutorial-actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={onContinue}
          disabled={hasPendingInterpretations}
          aria-disabled={hasPendingInterpretations ? "true" : undefined}
          title={
            hasPendingInterpretations
              ? TURN_2B_ASSUMPTIONS_BLOCKED_STATUS
              : undefined
          }
        >
          {TURN_2B_PRIMARY_BUTTON}
        </button>
        <button
          type="button"
          className="tutorial-link-button"
          onClick={onBack}
        >
          Edit prompt
        </button>
      </div>
    </section>
  );
}

function compareInterpretationEventsByCreatedAt(
  left: InterpretationEvent,
  right: InterpretationEvent,
): number {
  const createdAtOrder = left.created_at.localeCompare(right.created_at);
  if (createdAtOrder !== 0) return createdAtOrder;
  return left.id.localeCompare(right.id);
}
