import { useEffect, useRef, useState } from "react";
import { startGuidedSession } from "@/api/client";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { useSessionStore } from "@/stores/sessionStore";

interface TutorialGuidedShellProps {
  sessionId: string;
  onCompleted: (sessionId: string) => void;
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
 * copy.ts, never from a server entry_seed (which is server-side only and NOT a
 * field on the TS WorkflowProfile — P6.4 security carry-note).
 */
export function TutorialGuidedShell({
  sessionId,
  onCompleted,
}: TutorialGuidedShellProps): JSX.Element {
  const guidedSession = useSessionStore((s) => s.guidedSession);
  const startGuided = useSessionStore((s) => s.startGuided);
  const startedRef = useRef(false);
  const completedRef = useRef(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      // activeSessionId is null. Clear the same session/guided payload that
      // selectSession clears before loading (mirrors selectSession +
      // cleared{Guided,Recovery}State), otherwise a completed guided session
      // from the previous active session can make ChatPanel render the completed
      // surface and fire onCompleted before the new tutorial session has loaded.
      useSessionStore.setState({
        activeSessionId: sessionId,
        messages: [],
        compositionState: null,
        compositionProposals: [],
        composerPreferences: null,
        staleProposalIds: [],
        proposalActionPendingIds: [],
        composerProgress: null,
        stateVersions: [],
        isComposing: false,
        error: null,
        selectedNodeId: null,
        guidedSession: null,
        guidedNextTurn: null,
        guidedTerminal: null,
        guidedChatPending: false,
        guidedResponsePending: false,
        recoveryError: null,
        recoveryStartedCompositionVersion: null,
      });
      try {
        await startGuidedSession(sessionId, "tutorial");
        await startGuided(sessionId);
      } catch (err) {
        setError(formatError(err));
      } finally {
        setStarting(false);
      }
    })();
  }, [sessionId, startGuided]);

  // Hand off to the run/audit/graduation tail when guided reaches completion.
  useEffect(() => {
    if (completedRef.current) {
      return;
    }
    const current = useSessionStore.getState();
    if (
      current.activeSessionId !== sessionId ||
      current.guidedSession !== guidedSession
    ) {
      return;
    }
    if (guidedSession?.terminal?.kind === "completed") {
      completedRef.current = true;
      onCompleted(sessionId);
    }
  }, [guidedSession, onCompleted, sessionId]);

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
      <ChatPanel />
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
