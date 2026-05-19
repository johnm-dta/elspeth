import { useEffect, useReducer } from "react";
import { deleteTutorialOrphans } from "@/api/client";
import { TutorialTurn1Welcome } from "./TutorialTurn1Welcome";
import { TutorialTurn2Describe } from "./TutorialTurn2Describe";
import { TutorialTurn2bShowBuilt } from "./TutorialTurn2bShowBuilt";
import { TutorialTurn3Graph } from "./TutorialTurn3Graph";
import { TutorialTurn4Run } from "./TutorialTurn4Run";
import { TutorialTurn5AuditStory } from "./TutorialTurn5AuditStory";
import { TutorialTurn6ModeChoice } from "./TutorialTurn6ModeChoice";
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

  return (
    <main className="tutorial-shell" aria-label="First-run tutorial">
      <div className="tutorial-progress" aria-hidden="true">
        {["welcome", "describe", "draft", "graph", "run", "audit", "mode"].map(
          (label, index) => (
            <span
              key={label}
              className={
                index <= stepIndex(state.step)
                  ? "tutorial-progress-dot tutorial-progress-dot--active"
                  : "tutorial-progress-dot"
              }
            />
          ),
        )}
      </div>
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
        />
      )}
      {state.step === "showBuilt" && state.sessionId !== null && state.builtSummary !== null && (
        <TutorialTurn2bShowBuilt
          sessionId={state.sessionId}
          summary={state.builtSummary}
          onContinue={() => dispatch({ type: "showGraph" })}
        />
      )}
      {state.step === "graph" && state.builtSummary !== null && (
        <TutorialTurn3Graph
          summary={state.builtSummary}
          onContinue={() => dispatch({ type: "startRun" })}
        />
      )}
      {state.step === "run" && state.sessionId !== null && (
        <TutorialTurn4Run
          sessionId={state.sessionId}
          prompt={state.prompt}
          onCompleted={(result) => dispatch({ type: "runCompleted", result })}
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
          />
        )}
      {state.step === "mode" && (
        <TutorialTurn6ModeChoice
          sessionId={state.sessionId}
          skipped={state.skipped}
        />
      )}
    </main>
  );
}

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
    default:
      return 0;
  }
}
