import { useEffect, useReducer, useRef, useState } from "react";
import {
  createSession,
  deleteTutorialOrphans,
  renameSession,
  sendTutorialAbandonBeacon,
} from "@/api/client";
import { TutorialTurn1Welcome } from "./TutorialTurn1Welcome";
import { TutorialGuidedShell } from "./TutorialGuidedShell";
import { TutorialTurn4Run } from "./TutorialTurn4Run";
import { TutorialTurn5AuditStory } from "./TutorialTurn5AuditStory";
import { TutorialTurn7Graduation } from "./TutorialTurn7Graduation";
import { initialTutorialState, isAbandonOnPageHide, tutorialReducer } from "./tutorialMachine";
import { HELLO_WORLD_PENDING_SESSION_TITLE } from "./copy";

export function HelloWorldTutorial(): JSX.Element {
  const [state, dispatch] = useReducer(tutorialReducer, initialTutorialState);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  useEffect(() => {
    void deleteTutorialOrphans().catch((err) => {
      console.error("[tutorial] orphan cleanup failed:", err);
    });
  }, []);

  // Fire the best-effort tutorial-abandoned telemetry beacon (composer.
  // tutorial.abandon_total) when the tab/page is torn down while a tutorial
  // is genuinely in progress — started (past the welcome bookend) but not
  // yet at its terminal step. `graduation` is reached by EVERY finishing
  // path (completed, skipped, and cancelled all dispatch into it — see
  // tutorialReducer), so landing there before teardown means the learner
  // saw the tutorial through, not abandoned it. A ref (not a `step` dep on
  // the listener effect) keeps the same `pagehide` listener attached for the
  // component's whole lifetime; `pagehide` only fires on real page teardown
  // (tab close, external navigation, refresh) — never on this component's
  // own in-app step transitions or its eventual unmount when the tutorial
  // completes and `showTutorial` flips false in App.tsx, so a normal
  // graduation can never be misread as an abandon. Graduation LATCHES
  // (`hasGraduatedRef`): stepping Back from graduation to re-view the audit
  // story or run results and then closing the tab is still a completed
  // tutorial, not an abandon — the gate itself is the pure
  // `isAbandonOnPageHide` in tutorialMachine.ts.
  const stepRef = useRef(state.step);
  const hasGraduatedRef = useRef(false);
  useEffect(() => {
    stepRef.current = state.step;
    if (state.step === "graduation") {
      hasGraduatedRef.current = true;
    }
  }, [state.step]);
  useEffect(() => {
    function handlePageHide(): void {
      if (isAbandonOnPageHide(stepRef.current, hasGraduatedRef.current)) {
        sendTutorialAbandonBeacon();
      }
    }
    window.addEventListener("pagehide", handlePageHide);
    return () => window.removeEventListener("pagehide", handlePageHide);
  }, []);

  // Create the tutorial session on Start so TutorialGuidedShell has a
  // sessionId. Tag it with the pending title BEFORE the shell's external
  // POST /guided/start so the backend orphan-cleanup scan (which filters by
  // the "hello-world (" prefix) catches sessions abandoned mid-tutorial.
  const onStart = async (): Promise<void> => {
    setStarting(true);
    setStartError(null);
    try {
      const session = await createSession();
      await renameSession(session.id, HELLO_WORLD_PENDING_SESSION_TITLE);
      setSessionId(session.id);
      dispatch({ type: "start" });
    } catch (err) {
      setStartError(formatError(err));
    } finally {
      setStarting(false);
    }
  };

  const goBack = (): void => dispatch({ type: "back" });
  const stepLabels = TUTORIAL_STEP_LABELS;
  const currentIndex = stepIndex(state.step);
  const totalSteps = stepLabels.length;
  const currentLabel = stepLabels[currentIndex]?.label ?? "Step";

  // Back from graduation lands on audit (previousStep). The audit turn only
  // renders content when a real run exists (sessionId + runId +
  // sourceDataHash). On the skipped path all three are null; on the cancelled
  // path runId/sourceDataHash are null. In both cases an unconditional Back
  // would render an empty audit, so suppress the Back affordance unless audit
  // would render real content — the same predicate that guards the audit
  // branch below.
  const auditHasContent =
    state.sessionId !== null &&
    state.runId !== null &&
    state.sourceDataHash !== null;

  return (
    <main
      className={
        // The guided step embeds the full-height ChatPanel, whose composer docks
        // at the bottom; that dock needs the wrapper to be a growing flex column
        // (see `.tutorial-shell--guided` in tutorial.css). The bookend turns are
        // short centred cards and keep the base scrolling-column layout.
        state.step === "guided"
          ? "tutorial-shell tutorial-shell--guided"
          : "tutorial-shell"
      }
      aria-label="First-run tutorial"
    >
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
          //
          // Current and completed get DISTINCT classes (ringed vs plain fill):
          // collapsing both to `--active` made "you are here" indistinguishable
          // from "already done".
          return (
            <span
              key={key}
              aria-hidden="true"
              className={
                isActive
                  ? "tutorial-progress-dot tutorial-progress-dot--active"
                  : isComplete
                    ? "tutorial-progress-dot tutorial-progress-dot--complete"
                    : "tutorial-progress-dot"
              }
              title={label}
            />
          );
        })}
      </nav>
      {state.step === "welcome" && (
        <>
          <p role="status" className="sr-only">
            {starting ? "Creating tutorial session" : ""}
          </p>
          {startError !== null && (
            <p role="alert" className="tutorial-error">
              {startError}
            </p>
          )}
          <TutorialTurn1Welcome
            onStart={() => void onStart()}
            onSkip={() => dispatch({ type: "skipToGraduation" })}
          />
        </>
      )}
      {state.step === "guided" && sessionId !== null && (
        // One TutorialGuidedShell per tutorial session. The shell is
        // mount-once (startedRef); keying it on sessionId guarantees a new
        // session remounts a fresh shell rather than reusing a started one.
        <TutorialGuidedShell
          key={sessionId}
          sessionId={sessionId}
          onCompleted={(id) =>
            dispatch({ type: "guidedCompleted", sessionId: id })
          }
        />
      )}
      {state.step === "run" && state.sessionId !== null && (
        // No onBack: the guided wizard is terminal once completed
        // (previousStep(run) is null), so the run turn has no prior step to
        // return to and renders no Back affordance.
        <TutorialTurn4Run
          sessionId={state.sessionId}
          onCompleted={(result) => dispatch({ type: "runCompleted", result })}
          onCancelled={() => dispatch({ type: "cancelRun" })}
        />
      )}
      {state.step === "audit" &&
        state.sessionId !== null &&
        state.runId !== null &&
        state.sourceDataHash !== null && (
          <TutorialTurn5AuditStory
            sessionId={state.sessionId}
            runId={state.runId}
            onContinue={() => dispatch({ type: "continueToGraduation" })}
            onBack={goBack}
          />
        )}
      {state.step === "graduation" && (
        <TutorialTurn7Graduation
          sessionId={state.sessionId}
          skipped={state.skipped}
          cancelled={state.cancelled}
          onBack={auditHasContent ? goBack : undefined}
        />
      )}
    </main>
  );
}

/**
 * Display labels for the progress dots and the sr-only "Step N of M" hint.
 * The staged guided flow is welcome -> guided -> run -> audit -> graduation;
 * the guided surface owns its own internal stages (source/sink/transform/wire).
 */
const TUTORIAL_STEP_LABELS: ReadonlyArray<{ key: string; label: string }> = [
  { key: "welcome", label: "Welcome" },
  { key: "guided", label: "Build" },
  { key: "run", label: "Run" },
  { key: "audit", label: "Audit" },
  // "Graduate" (was "Ready"): this macro phase IS the graduation turn. The inner
  // guided stepper's terminal step is also labelled "Ready" (an assembled,
  // ready-to-run pipeline); two different "Ready"s in nested progress trackers
  // read as a collision. Rename the macro one — "Ready" stays the product term
  // for a finished pipeline on the stepper.
  { key: "graduation", label: "Graduate" },
];

function stepIndex(step: string): number {
  switch (step) {
    case "welcome":
      return 0;
    case "guided":
      return 1;
    case "run":
      return 2;
    case "audit":
      return 3;
    case "graduation":
      return 4;
    default:
      return 0;
  }
}

function formatError(err: unknown): string {
  if (
    typeof err === "object" &&
    err !== null &&
    "detail" in err &&
    typeof (err as { detail?: unknown }).detail === "string"
  ) {
    return (err as { detail: string }).detail;
  }
  if (err instanceof Error) {
    return err.message;
  }
  return "The tutorial session could not be created.";
}
