import { useEffect, useRef, useState, type MutableRefObject } from "react";
import {
  getTutorialSample,
  respondGuided,
  startGuidedSession,
} from "@/api/client";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import type {
  GuidedRespondRequest,
  GuidedRespondResponse,
  GuidedStep,
} from "@/types/guided";
import {
  TUTORIAL_SINK_PROMPT,
  TUTORIAL_SOURCE_PROMPT,
  TUTORIAL_TRANSFORMS_PROMPT,
} from "./tutorialMachine";

/**
 * The per-stage prelocked prompts the passive learner Sends verbatim, one per
 * LLM-driven phase. The composer is a STAGED orchestrator: each phase gets ONLY
 * its stage's intent so the light model can focus. The SOURCE prompt carries NO
 * URLs in its constant — the LLM cannot guess the runtime-served addresses, so
 * the resolved synthetic URLs (fetched per-session from the 8a GET surface) are
 * appended to the source prompt only (the sink/transforms phases don't need
 * them). Recipe and Wire are confirm-only (no chat prompt).
 */
function buildLockedPrompts(
  sampleUrls: string[],
): Partial<Record<GuidedStep, string>> {
  return {
    step_1_source: `${TUTORIAL_SOURCE_PROMPT}\n${sampleUrls.join("\n")}`,
    step_2_sink: TUTORIAL_SINK_PROMPT,
    step_3_transforms: TUTORIAL_TRANSFORMS_PROMPT,
  };
}

interface TutorialGuidedShellProps {
  sessionId: string;
  onCompleted: (sessionId: string) => void;
  /**
   * An observed live wizard terminated with `exited_to_freeform` (the
   * wire-stage "Exit to freeform" button is reachable in tutorial mode and
   * on blocked outcomes is the ONLY affordance). Without this hand-off the
   * shell stays mounted over a terminal session and the learner dead-ends
   * on ChatPanel's "Preparing your guided pipeline…" placeholder
   * (elspeth-61591e64bb). The parent persists the tutorial opt-out and
   * lets the shell unmount into the freeform composer.
   */
  onExited?: (sessionId: string) => void;
  /**
   * The persisted resume session no longer exists server-side (404 from the
   * start chain). Without this the shell dead-ends on a "Session not found"
   * error with NO forward affordance — the tutorial suppresses skip/exit, so
   * the learner is stranded on an empty page. The parent resets to a fresh
   * Welcome and clears the stale resume fields.
   */
  onSessionMissing?: (deadSessionId: string) => void;
  /**
   * Set by the tutorial chrome when the learner exits during the startup
   * window before GET /guided has populated guidedSession. The shell still
   * owns the in-flight /guided/start promise, so it is the only layer that can
   * reliably send the server-side exit once that start has actually landed.
   */
  exitRequestedRef?: MutableRefObject<boolean>;
}

const EXIT_TO_FREEFORM_REQUEST = {
  chosen: null,
  edited_values: null,
  custom_inputs: null,
  accepted_step_index: null,
  edit_step_index: null,
  control_signal: "exit_to_freeform",
} satisfies GuidedRespondRequest;

/** The start chain 404s when the persisted resume session was swept/archived. */
function isSessionMissingError(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "status" in err &&
    (err as { status?: unknown }).status === 404
  );
}

/**
 * Tutorial bridge (D9): renders the welcome bookend, starts a TUTORIAL-profile
 * guided session, EMBEDS the real ChatPanel guided surface (the truest "use
 * the real thing"), and on guided terminal=completed hands the session back to
 * the surviving tutorialMachine run/audit/graduation tail. Per-stage
 * interpretation review + the wire confirm are owned by the ChatPanel guided
 * branch, which already projects interpretationEventsStore.pendingBySession and
 * blocks advancement while pending (P4.T2). Coaching/bookend copy reads off the
 * wire GuidedSession.profile; the welcome framing text comes from the frontend
 * copy.ts.
 */
export function TutorialGuidedShell({
  sessionId,
  onCompleted,
  onExited,
  onSessionMissing,
  exitRequestedRef,
}: TutorialGuidedShellProps): JSX.Element {
  const guidedSession = useSessionStore((s) => s.guidedSession);
  const startGuided = useSessionStore((s) => s.startGuided);
  const resetForTutorialSession = useSessionStore(
    (s) => s.resetForTutorialSession,
  );
  const startedRef = useRef(false);
  // A shell hands off exactly once per mount, whether the terminal it
  // observed was a completion (onCompleted) or an exit (onExited).
  const handedOffRef = useRef(false);
  // True once this mount has OBSERVED a live (non-null, non-terminal)
  // guidedSession. The terminal hand-offs may fire only for a terminal this
  // shell saw transition to while mounted — never when it mounts directly
  // onto an already-completed (or already-exited) session.
  const sawActiveRef = useRef(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The runtime-resolved synthetic sample URLs for THIS tutorial session
  // (8a GET surface). null until they resolve; the ChatPanel is gated on them
  // so the box is never an editable/empty placeholder and the learner can never
  // Send the URL-less canonical prompt before the source driver has addresses.
  const [sampleUrls, setSampleUrls] = useState<string[] | null>(null);

  // Start the TUTORIAL-profile guided session exactly once. The start
  // endpoint is idempotent server-side (P7.1): a second POST for a session
  // that already has a persisted GuidedSession returns it unchanged. The
  // startedRef guard avoids a redundant round-trip under StrictMode's
  // double-invoke.
  useEffect(() => {
    if (startedRef.current) {
      return;
    }
    startedRef.current = true;
    void (async () => {
      setStarting(true);
      setError(null);
      // Bind the store's activeSessionId to this tutorial session BEFORE
      // startGuided. startGuided (sessionStore.ts) DISCARDS its fetched guided
      // payload unless get().activeSessionId === the requested id, and ChatPanel
      // renders the empty-session surface (chat-panel--empty) whenever
      // activeSessionId is null. resetForTutorialSession clears the same
      // session/guided payload that selectSession clears before loading,
      // otherwise a completed guided session from the previous active session
      // can make ChatPanel render the completed surface and fire onCompleted
      // before the new tutorial session has loaded.
      resetForTutorialSession(sessionId);
      const exitIfRequested = async (): Promise<boolean> => {
        if (exitRequestedRef?.current !== true) {
          return false;
        }
        try {
          await exitStartedGuidedSession(sessionId);
        } catch (err) {
          console.error("[tutorial] startup exit-to-freeform failed:", err);
        }
        return true;
      };
      try {
        await startGuidedSession(sessionId, "tutorial");
        if (await exitIfRequested()) {
          return;
        }
        // Fetch the runtime-resolved synthetic URLs BEFORE entering the wizard.
        // The GET requires the TUTORIAL profile to be persisted (done by the
        // start above). Appended to the locked STEP_1 prompt; the box stays
        // gated (never editable) until they arrive.
        const sample = await getTutorialSample(sessionId);
        if (await exitIfRequested()) {
          return;
        }
        // Only sample_urls are consumed client-side, appended to the locked
        // STEP_1 prompt. The synthetic pages are publicly hosted, so the
        // tutorial's web_scrape node carries no SSRF allowlist — it uses the
        // plugin default `allowed_hosts="public_only"`. The client must never
        // set an allowlist (a client-set allowlist is an SSRF widening vector).
        setSampleUrls(sample.sample_urls);
        await startGuided(sessionId);
        if (await exitIfRequested()) {
          return;
        }
        // Rehydrate the interpretation-event projection for THIS session.
        // Every other route into a session goes through selectSession, which
        // does this (Phase 5b Task 3) — the tutorial bridge bypasses it, so
        // without this a mid-Build reload resumed with pendingBySession
        // EMPTY: no acknowledgement cards rendered, the wire-stage Confirm
        // was not blocked, and the run then failed server-side with
        // UnresolvedInterpretationPlaceholderError (the backend run gate is
        // the final guard; the ack-before-advance UX gate lives client-side).
        // Awaited, not fire-and-forget: a resume can land DIRECTLY on the
        // wire stage, where the gate must be up before the first paint of an
        // enabled Confirm.
        await useInterpretationEventsStore.getState().refreshAll(sessionId);
      } catch (err) {
        if (isSessionMissingError(err) && onSessionMissing !== undefined) {
          onSessionMissing(sessionId);
          return;
        }
        setError(formatError(err));
      } finally {
        setStarting(false);
      }
    })();
  }, [
    sessionId,
    startGuided,
    resetForTutorialSession,
    onSessionMissing,
    exitRequestedRef,
  ]);

  // Hand off when guided reaches a terminal — but ONLY on a terminal this
  // mount OBSERVED transition to. The back-nav GET path remounts this shell
  // against the PERSISTED terminal guided session (startGuided clears
  // guidedSession to null, then sets it to the terminal payload), so the
  // shell mounts onto a terminal without ever seeing a live wizard. Firing
  // there bounces the user straight back to run (completed) or re-PATCHes
  // the opt-out on every remount (exited). Gate on sawActiveRef: a terminal
  // session that was never preceded by a live, non-terminal session during
  // this mount must NOT hand off.
  //
  // Two distinct hand-offs: `completed` graduates into the run/audit/
  // graduation tail (onCompleted); `exited_to_freeform` is NOT a graduation
  // — it leaves the tutorial for the freeform composer (onExited persists
  // the opt-out so the shell unmounts, elspeth-61591e64bb).
  useEffect(() => {
    if (handedOffRef.current) {
      return;
    }
    const current = useSessionStore.getState();
    if (
      current.activeSessionId !== sessionId ||
      current.guidedSession !== guidedSession
    ) {
      return;
    }
    const kind = guidedSession?.terminal?.kind;
    // Record that we observed a live wizard: a non-null session with no
    // terminal yet. The mount-effect's clear-to-null step leaves guidedSession
    // null (not "active"), so requiring non-null here keeps the back-nav path
    // from spuriously marking the wizard observed. Requiring terminal == null
    // (not just "not completed") keeps a mount onto an already-exited session
    // from arming the exit hand-off.
    if (guidedSession !== undefined && guidedSession !== null && kind == null) {
      sawActiveRef.current = true;
    }
    if (kind === "completed" && sawActiveRef.current) {
      handedOffRef.current = true;
      onCompleted(sessionId);
    } else if (kind === "exited_to_freeform" && sawActiveRef.current) {
      handedOffRef.current = true;
      onExited?.(sessionId);
    }
  }, [guidedSession, onCompleted, onExited, sessionId]);

  const bookends = guidedSession?.profile?.bookends ?? true;

  return (
    <section
      className="tutorial-guided-shell"
      aria-label="Guided pipeline composer"
    >
      {bookends && (
        <p className="tutorial-kicker">
          Let's build your first pipeline one stage at a time.
        </p>
      )}
      <p role="status" className="sr-only">
        {starting ? "Starting guided composer" : ""}
      </p>
      {error !== null && (
        <p role="alert" className="tutorial-error">
          {error}
        </p>
      )}
      {sampleUrls !== null ? (
        // Gate the wizard on the resolved URLs: the locked STEP_1 prompt must
        // carry them so the source driver can parse the runtime-served
        // addresses. The box is read-only in tutorial mode (ChatPanel's
        // lockedChatPrompt), so the only learner action — Send — is never
        // exposed with a URL-less prompt.
        <ChatPanel
          isTutorial
          lockedChatPrompt={buildLockedPrompts(sampleUrls)}
        />
      ) : (
        error === null && (
          // Plain text, NOT role="status": the sr-only "Starting guided
          // composer" status above already announces this loading phase;
          // a second live region would double-announce.
          <p className="tutorial-sample-loading">
            Preparing the tutorial's sample pages…
          </p>
        )
      )}
    </section>
  );
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
  return "The guided tutorial could not be started.";
}

async function exitStartedGuidedSession(sessionId: string): Promise<void> {
  const current = useSessionStore.getState();
  if (
    current.activeSessionId === sessionId &&
    current.guidedSession?.terminal?.kind === "exited_to_freeform"
  ) {
    return;
  }
  if (current.activeSessionId === sessionId && current.guidedSession !== null) {
    await current.exitToFreeform();
    return;
  }
  const response = await respondGuided(sessionId, EXIT_TO_FREEFORM_REQUEST);
  applyGuidedExitResponse(sessionId, response);
}

function applyGuidedExitResponse(
  sessionId: string,
  response: GuidedRespondResponse,
): void {
  if (useSessionStore.getState().activeSessionId !== sessionId) {
    return;
  }
  useSessionStore.setState({
    guidedSession: response.guided_session,
    guidedNextTurn: response.next_turn,
    guidedTerminal: response.terminal,
    compositionState: response.composition_state,
    guidedResponsePending: false,
    error: null,
    errorDetails: null,
    guidedSelfHealNotice: null,
  });
}
