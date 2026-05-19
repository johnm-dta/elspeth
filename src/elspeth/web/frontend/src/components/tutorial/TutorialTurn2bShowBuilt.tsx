import { useMemo } from "react";
import { InterpretationReviewTurn } from "@/components/chat/guided/InterpretationReviewTurn";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import { TURN_2B_PRIMARY_BUTTON } from "./copy";
import type { TutorialBuiltSummary } from "./tutorialMachine";

interface TutorialTurn2bShowBuiltProps {
  sessionId: string;
  summary: TutorialBuiltSummary;
  onContinue: () => void;
}

export function TutorialTurn2bShowBuilt({
  sessionId,
  summary,
  onContinue,
}: TutorialTurn2bShowBuiltProps): JSX.Element {
  const pendingBySession = useInterpretationEventsStore(
    (state) => state.pendingBySession,
  );
  const pendingInterpretation = useMemo(() => {
    const events = Object.values(pendingBySession[sessionId] ?? {});
    return events.find(
      (event) =>
        event.choice === "pending" &&
        event.interpretation_source === "user_approved",
    );
  }, [pendingBySession, sessionId]);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-built-title">
      <p className="tutorial-kicker">Draft</p>
      <h2 id="tutorial-built-title">Here is what the composer drafted.</h2>
      <div className="tutorial-summary-grid">
        <div>
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
        </div>
        <div>
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
        </div>
        <div>
          <h3>Sink</h3>
          <p>{summary.sinkLabel}</p>
        </div>
      </div>

      {pendingInterpretation ? (
        <div className="tutorial-interpretation">
          <InterpretationReviewTurn
            event={pendingInterpretation}
            sessionId={sessionId}
            onResolved={(newState) => {
              if (newState !== null) {
                useSessionStore.setState({ compositionState: newState });
              }
            }}
          />
        </div>
      ) : (
        <p className="tutorial-muted">
          If the draft surfaced an interpretation to review, it appears here.
          Otherwise, the run will still record the prompt and model calls.
        </p>
      )}

      <div className="tutorial-actions">
        <button type="button" className="btn btn-primary" onClick={onContinue}>
          {TURN_2B_PRIMARY_BUTTON}
        </button>
      </div>
    </section>
  );
}
