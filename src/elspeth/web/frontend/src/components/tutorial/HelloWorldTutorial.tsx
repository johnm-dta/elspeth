import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import {
  createSession,
  deleteTutorialOrphans,
  fetchSessions,
  renameSession,
  sendTutorialAbandonBeacon,
} from "@/api/client";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import { TutorialTurn1Welcome } from "./TutorialTurn1Welcome";
import { TutorialGuidedShell } from "./TutorialGuidedShell";
import { abandonTutorialRun, TutorialTurn4Run } from "./TutorialTurn4Run";
import { TutorialTurn5AuditStory } from "./TutorialTurn5AuditStory";
import { TutorialTurn7Graduation } from "./TutorialTurn7Graduation";
import {
  isAbandonOnPageHide,
  progressForTutorialState,
  resumeTutorialState,
  tutorialReducer,
} from "./tutorialMachine";
import { HELLO_WORLD_PENDING_SESSION_TITLE } from "./copy";

interface HelloWorldTutorialProps {
  composerAvailable?: boolean;
  composerUnavailableReason?: string | null;
}

/**
 * Reducer lazy-initialiser: reconstruct the mount state from the
 * server-persisted resume fields (elspeth-918f4434b3). App.tsx only mounts
 * the tutorial after the preferences bootstrap completes (`showTutorial`
 * gates on `preferencesLoaded`), so the store fields are authoritative here.
 */
function initTutorialStateFromPreferences(): ReturnType<typeof resumeTutorialState> {
  const prefs = usePreferencesStore.getState();
  return resumeTutorialState({
    stage: prefs.tutorialStage,
    sessionId: prefs.tutorialSessionId,
    runId: prefs.tutorialRunId,
    sourceDataHash: prefs.tutorialSourceDataHash,
  });
}

export function HelloWorldTutorial({
  composerAvailable = true,
  composerUnavailableReason = null,
}: HelloWorldTutorialProps): JSX.Element {
  const [state, dispatch] = useReducer(
    tutorialReducer,
    null,
    initTutorialStateFromPreferences,
  );
  const [sessionId, setSessionId] = useState<string | null>(state.sessionId);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  // Orphan cleanup runs ONLY on a fresh tutorial entry. On a resume the
  // persisted tutorial session still carries the pending title — sweeping it
  // here is exactly the "abandoned-<title>-<timestamp>" rename
  // the resume path exists to prevent. (The backend also refuses to sweep
  // the session recorded in preferences.tutorial_session_id — defence in
  // depth.)
  const resumedAtMountRef = useRef(state.resumed);
  useEffect(() => {
    if (resumedAtMountRef.current) {
      return;
    }
    void deleteTutorialOrphans().catch((err) => {
      console.error("[tutorial] orphan cleanup failed:", err);
    });
  }, []);

  // The persisted resume session can outlive its session row (orphan sweep,
  // archive, a prerelease DB wipe). A dead session dead-ends EVERY resumed
  // stage — guided renders "Session not found" with no forward affordance
  // (the tutorial suppresses skip/exit past Welcome), run/audit 404 into a
  // Retry that can never succeed. Recovery: fall back to a fresh Welcome;
  // the stage-persist effect above then clears the stale server-side resume
  // fields (welcome maps to all-null in progressForTutorialState).
  // Idempotent per dead id: the mount-time membership check below and
  // TutorialGuidedShell's guided/start 404 handler can BOTH detect the same
  // dead resume (they race), and the recovery must run once — otherwise the
  // warning double-logs and the reducer resets twice.
  const recoveredSessionIdRef = useRef<string | null>(null);
  const onSessionMissing = useCallback((deadSessionId: string): void => {
    if (recoveredSessionIdRef.current === deadSessionId) {
      return;
    }
    recoveredSessionIdRef.current = deadSessionId;
    console.warn(
      "[tutorial] persisted resume session no longer exists — restarting at Welcome",
    );
    // Release the app-level binding too: resetForTutorialSession bound
    // activeSessionId to the (dead) resume id before the server could 404
    // it, and consumers keyed on activeSessionId (run list, composer
    // progress) would otherwise keep polling the dead session while the
    // user sits at Welcome.
    useSessionStore.getState().unbindMissingSession(deadSessionId);
    setSessionId(null);
    dispatch({ type: "reset" });
  }, []);

  // Mount-time validation of the resumed session (covers the run/audit
  // stages, which have no equivalent of TutorialGuidedShell's 404 recovery).
  // Best-effort: a failed LIST keeps the resume — the shell's own 404
  // recovery still applies for the guided stage, and a transient list error
  // must not throw away a healthy resume.
  const initialResumeSessionIdRef = useRef(state.resumed ? state.sessionId : null);
  useEffect(() => {
    const resumedSessionId = initialResumeSessionIdRef.current;
    if (resumedSessionId === null) {
      return;
    }
    let active = true;
    void fetchSessions(false)
      .then((sessions) => {
        if (!active) {
          return;
        }
        if (!sessions.some((session) => session.id === resumedSessionId)) {
          onSessionMissing(resumedSessionId);
        }
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [onSessionMissing]);

  // Persist the tutorial stage server-side on every stage transition so a
  // reload resumes instead of restarting. Best-effort: a failed persist
  // must not interrupt the in-page tutorial (and deliberately does NOT set
  // the store's writeError — that would unmount the tutorial via App.tsx's
  // showTutorial gate); the residual cost of a failure is only that a
  // reload resumes one stage earlier. The skipped path persists the
  // completion opt-out instead (see onSkip), which clears these fields
  // server-side — so no stage write happens for it.
  useEffect(() => {
    if (state.skipped) {
      return;
    }
    const target = progressForTutorialState(state, state.sessionId ?? sessionId);
    const prefs = usePreferencesStore.getState();
    if (
      prefs.tutorialStage === target.stage &&
      prefs.tutorialSessionId === target.sessionId &&
      prefs.tutorialRunId === target.runId &&
      prefs.tutorialSourceDataHash === target.sourceDataHash
    ) {
      return;
    }
    void prefs.saveTutorialProgress(target).catch((err) => {
      console.error("[tutorial] progress persist failed:", err);
    });
  }, [state, sessionId]);

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
  // POST /guided/start so the backend orphan-cleanup scan (which matches the
  // exact pending title) catches sessions abandoned mid-tutorial.
  const onStart = async (): Promise<void> => {
    if (!composerAvailable) {
      setStartError(tutorialComposerUnavailableMessage(composerUnavailableReason));
      return;
    }
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

  // "Skip the tutorial" persists the opt-out IMMEDIATELY on this first
  // click, not on the follow-up "Take me to the composer" click — a closed
  // tab between the two must not restart the tutorial. publishLocally=false
  // keeps the graduation card mounted (flipping tutorialCompleted here
  // would unmount the whole tutorial mid-farewell); the graduation card's
  // finish click re-persists idempotently and then publishes. Best-effort:
  // on failure the finish click is still the second chance to persist.
  const onSkip = (): void => {
    dispatch({ type: "skipToGraduation" });
    void usePreferencesStore
      .getState()
      .markTutorialGraduated({ publishLocally: false })
      .catch((err) => {
        console.error("[tutorial] skip opt-out persist failed:", err);
      });
  };

  // Exit (unlike skip) leaves the tutorial for a usable freeform composer
  // NOW: persist the opt-out AND publish it locally, so App's showTutorial
  // gate unmounts the whole shell and the learner lands in the freeform
  // composer on the same session (elspeth-61591e64bb). Fired by (a) the
  // guided wizard's exited_to_freeform terminal — the wire-stage "Exit to
  // freeform" button is reachable in tutorial mode and on blocked outcomes
  // is the ONLY affordance — and (b) the persistent "Exit tutorial" chrome
  // control below. Resilient even on failure: a rejected PATCH sets the
  // store's writeError, which ALSO flips showTutorial false — the exit can
  // never strand the learner in the shell.
  const onExitTutorial = useCallback((): void => {
    // Two guided surfaces survive the shell unmount and would keep the learner
    // OFF freeform, so both must be handed off through exitToFreeform (which
    // POSTs control_signal=exit_to_freeform, backend-recorded as
    // user_pressed_exit so guided stays re-enterable):
    //   * a LIVE (terminal == null) build — ChatPanel's discriminator re-renders
    //     the guided workspace;
    //   * a COMPLETED build — the discriminator checks `completed` FIRST and
    //     re-renders CompletionSummary, whose own "Open freeform editor" button
    //     just calls exitToFreeform (elspeth-e2c3dba6b5 review P2). Firing it
    //     here up front lands the learner in freeform NOW instead of on the
    //     summary with an extra click. The backend exempts kind=COMPLETED from
    //     the terminal-rejection for exactly this transition (guided.py:1222).
    // An already-exited_to_freeform terminal (the wizard-path onExited hand-off
    // reaches this handler with the terminal already set) is left alone: it
    // already falls through to freeform, and re-firing would be a duplicate
    // respond POST the backend 409s. Best-effort like the persist below; the
    // duplicate markTutorialGraduated onExited can trigger (the shell observes
    // the terminal and hands off) is absorbed by the store's landed-completion
    // guard.
    const { guidedSession, exitToFreeform } = useSessionStore.getState();
    const terminalKind = guidedSession?.terminal?.kind ?? null;
    if (
      guidedSession !== null &&
      (terminalKind === null || terminalKind === "completed")
    ) {
      void exitToFreeform().catch((err) => {
        console.error("[tutorial] exit-to-freeform hand-off failed:", err);
      });
    }
    // Exit during an in-flight run: the run turn's effect cleanup
    // deliberately never aborts (StrictMode), and its Cancel button is the
    // only other abort path — without this the backend run (LLM spend, sink
    // writes) outlives the tutorial. runId stays null until the run's result
    // lands, so this fires only while the run is genuinely still executing.
    if (
      state.step === "run" &&
      state.runId === null &&
      state.sessionId !== null
    ) {
      abandonTutorialRun(state.sessionId);
    }
    void usePreferencesStore
      .getState()
      .markTutorialGraduated({ via: "exit" })
      .catch((err) => {
        console.error("[tutorial] exit opt-out persist failed:", err);
      });
  }, [state.step, state.runId, state.sessionId]);
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
        {/* Visible counterpart of the sr-only line (elspeth-d75756fa2c): the
            unlabeled 5-dot row sat directly above the 5-chip build stepper and
            read as a second, broken copy of it. Naming this row "Tutorial ·
            <stage>" makes the two indicators read as different hierarchies.
            aria-hidden — the sr-only paragraph above already carries the same
            signal for AT, so exposing this too would double-announce. */}
        <span className="tutorial-progress-label" aria-hidden="true">
          Tutorial · {currentLabel} — step {currentIndex + 1} of {totalSteps}
        </span>
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
        {/* Persistent in-context exit (elspeth-61591e64bb): past the Welcome
            bookend the tutorial previously offered NO discoverable way out —
            the only escape was the buried Account menu → Composer preferences
            → "Reset tutorial" two-step. Welcome keeps its own "Skip the
            tutorial"; graduation IS the exit (its finish CTA persists the
            same opt-out); every step between gets this control. */}
        {(state.step === "guided" ||
          state.step === "run" ||
          state.step === "audit") && (
          <button
            type="button"
            className="tutorial-link-button tutorial-exit-button"
            onClick={onExitTutorial}
          >
            Exit tutorial
          </button>
        )}
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
            onSkip={onSkip}
            startDisabled={!composerAvailable}
            startDisabledReason={
              composerAvailable
                ? null
                : tutorialComposerUnavailableMessage(composerUnavailableReason)
            }
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
          onExited={onExitTutorial}
          onSessionMissing={onSessionMissing}
        />
      )}
      {state.step === "run" && state.sessionId !== null && (
        // No onBack: the guided wizard is terminal once completed
        // (previousStep(run) is null), so the run turn has no prior step to
        // return to and renders no Back affordance.
        <TutorialTurn4Run
          sessionId={state.sessionId}
          onResult={(result) => dispatch({ type: "runResultReady", result })}
          onCompleted={(result) => dispatch({ type: "runCompleted", result })}
          onCancelled={() => dispatch({ type: "cancelRun" })}
        />
      )}
      {state.step === "audit" &&
        state.sessionId !== null &&
        state.runId !== null &&
        state.sourceDataHash !== null && (
          // A RESUMED audit has no in-memory run cache (it was rebuilt from
          // the persisted resume fields after a reload), so Back into the
          // run turn would silently re-fire the tutorial pipeline — real
          // LLM spend. Suppress the Back affordance on the resumed flow;
          // the in-page flow keeps it (the run result stays cache-backed).
          <TutorialTurn5AuditStory
            sessionId={state.sessionId}
            runId={state.runId}
            onContinue={() => dispatch({ type: "continueToGraduation" })}
            onBack={state.resumed ? undefined : goBack}
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

function tutorialComposerUnavailableMessage(reason: string | null): string {
  return reason ?? "The guided tutorial needs the composer model, but it is not available. Configure the model provider or skip the tutorial for now.";
}
