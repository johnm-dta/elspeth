import { useEffect, useReducer } from "react";
import { deleteTutorialOrphans } from "@/api/client";
import { TutorialTurn1Welcome } from "./TutorialTurn1Welcome";
import { TutorialTurn2Describe } from "./TutorialTurn2Describe";
import { TutorialTurn2bShowBuilt } from "./TutorialTurn2bShowBuilt";
import { TutorialTurn3Graph } from "./TutorialTurn3Graph";
import { TutorialTurn4Run } from "./TutorialTurn4Run";
import { TutorialTurn5AuditStory } from "./TutorialTurn5AuditStory";
import { TutorialTurn6ModeChoice } from "./TutorialTurn6ModeChoice";
import { TutorialTurn7Graduation } from "./TutorialTurn7Graduation";
import {
  initialTutorialState,
  tutorialReducer,
} from "./tutorialMachine";

export function HelloWorldTutorial(): JSX.Element {
  const [state, dispatch] = useReducer(
    tutorialReducer,
    initialTutorialState,
  );

  useEffect(() => {
    void deleteTutorialOrphans().catch((err) => {
      console.error("[tutorial] orphan cleanup failed:", err);
    });
  }, []);

  const goBack = (): void => dispatch({ type: "back" });
  const stepLabels = TUTORIAL_STEP_LABELS;
  const currentIndex = stepIndex(state.step);
  const totalSteps = stepLabels.length;
  const currentLabel = stepLabels[currentIndex]?.label ?? "Step";

  return (
    <main className="tutorial-shell" aria-label="First-run tutorial">
      <nav
        className="tutorial-progress"
        role="group"
        aria-label="Tutorial progress"
      >
        <p className="sr-only">
          Step {currentIndex + 1} of {totalSteps}: {currentLabel}
        </p>
        {stepLabels.map(({ key, label }, index) => {
          const isActive = index === currentIndex;
          const isComplete = index < currentIndex;
          // The dots are purely decorative — the sr-only "Step N of M:
          // label" paragraph above carries the step-progress signal for
          // assistive tech. Don't add aria-current here: it would be
          // ignored anyway (aria-hidden removes the element from the AT
          // tree) and the dual-encoding is misleading to future readers.
          return (
            <span
              key={key}
              aria-hidden="true"
              className={
                isActive || isComplete
                  ? "tutorial-progress-dot tutorial-progress-dot--active"
                  : "tutorial-progress-dot"
              }
              title={label}
            />
          );
        })}
      </nav>
      {state.step === "welcome" && (
        <TutorialTurn1Welcome
          onStart={() => dispatch({ type: "start" })}
          onSkip={() => dispatch({ type: "skipToMode" })}
        />
      )}
      {state.step === "describe" && (
        <TutorialTurn2Describe
          initialPrompt={state.prompt}
          onBuilt={(result) => dispatch({ type: "built", result })}
          onBack={goBack}
        />
      )}
      {state.step === "showBuilt" && state.sessionId !== null && state.builtSummary !== null && (
        <TutorialTurn2bShowBuilt
          sessionId={state.sessionId}
          summary={state.builtSummary}
          onContinue={() => dispatch({ type: "showGraph" })}
          onBack={goBack}
        />
      )}
      {state.step === "graph" && state.builtSummary !== null && (
        <TutorialTurn3Graph
          summary={state.builtSummary}
          onContinue={() => dispatch({ type: "startRun" })}
          onBack={goBack}
        />
      )}
      {state.step === "run" && state.sessionId !== null && (
        <TutorialTurn4Run
          sessionId={state.sessionId}
          prompt={state.prompt}
          onCompleted={(result) => dispatch({ type: "runCompleted", result })}
          onCancelled={() => dispatch({ type: "cancelRun" })}
          onBack={goBack}
        />
      )}
      {state.step === "audit" &&
        state.sessionId !== null &&
        state.runId !== null &&
        state.sourceDataHash !== null && (
          <TutorialTurn5AuditStory
            sessionId={state.sessionId}
            runId={state.runId}
            sourceDataHash={state.sourceDataHash}
            onContinue={() => dispatch({ type: "continueToMode" })}
            onBack={goBack}
          />
        )}
      {state.step === "mode" && (
        <TutorialTurn6ModeChoice
          sessionId={state.sessionId}
          skipped={state.skipped}
          cancelled={state.cancelled}
          onBack={goBack}
          onFinished={() => dispatch({ type: "finishMode" })}
        />
      )}
      {state.step === "graduation" && (
        <TutorialTurn7Graduation onBack={goBack} />
      )}
    </main>
  );
}

/**
 * Display labels for the progress dots and the sr-only "Step N of M" hint.
 * The key matches the `TutorialStep` union but `showBuilt` is shown as
 * "Draft" for the user-facing label.
 */
const TUTORIAL_STEP_LABELS: ReadonlyArray<{ key: string; label: string }> = [
  { key: "welcome", label: "Welcome" },
  { key: "describe", label: "Describe" },
  { key: "showBuilt", label: "Draft" },
  { key: "graph", label: "Graph" },
  { key: "run", label: "Run" },
  { key: "audit", label: "Audit" },
  { key: "mode", label: "Mode" },
  { key: "graduation", label: "Ready" },
];

function stepIndex(step: string): number {
  switch (step) {
    case "welcome":
      return 0;
    case "describe":
      return 1;
    case "showBuilt":
      return 2;
    case "graph":
      return 3;
    case "run":
      return 4;
    case "audit":
      return 5;
    case "mode":
      return 6;
    case "graduation":
      return 7;
    default:
      return 0;
  }
}
