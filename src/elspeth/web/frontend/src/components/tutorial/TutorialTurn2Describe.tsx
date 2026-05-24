import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "@/api/client";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import type {
  ChatMessage,
  ComposerPreferences,
  ComposerProgressSnapshot,
  CompositionProposal,
  CompositionState,
  Session,
} from "@/types/index";
import {
  CANONICAL_TUTORIAL_PROMPT,
  HELLO_WORLD_PENDING_SESSION_TITLE,
  TURN_2_PRIMARY_BUTTON,
  TURN_2_RESTORE_BUTTON,
} from "./copy";
import {
  summariseCompositionState,
  type TutorialBuildResult,
} from "./tutorialMachine";

const TUTORIAL_DRAFT_PROGRESS_POLL_INTERVAL_MS = 1_500;

interface TutorialTurn2DescribeProps {
  initialPrompt: string;
  onBuilt: (result: TutorialBuildResult) => void;
  onBack: () => void;
}

interface BuildTutorialDraftOptions {
  onSessionReady?: (sessionId: string) => void;
}

export function TutorialTurn2Describe({
  initialPrompt,
  onBuilt,
  onBack,
}: TutorialTurn2DescribeProps): JSX.Element {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [pending, setPending] = useState(false);
  const [draftSessionId, setDraftSessionId] = useState<string | null>(null);
  const [composerProgress, setComposerProgress] =
    useState<ComposerProgressSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const headingRef = useRef<HTMLHeadingElement | null>(null);

  // Move focus to this turn's heading on mount so screen-reader users
  // hear the transition. tabIndex={-1} on the h2 keeps it out of the
  // tab order while still programmatically focusable.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!pending || draftSessionId === null) {
      return;
    }

    let active = true;
    const pollProgress = async (): Promise<void> => {
      try {
        const progress = await api.fetchComposerProgress(draftSessionId);
        if (!active) return;
        setComposerProgress(progress.phase === "idle" ? null : progress);
      } catch {
        // Advisory only: keep the visible local fallback while the draft call
        // continues. The final build request still owns user-facing failures.
      }
    };

    void pollProgress();
    const timer = window.setInterval(
      () => void pollProgress(),
      TUTORIAL_DRAFT_PROGRESS_POLL_INTERVAL_MS,
    );

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [draftSessionId, pending]);

  const onSubmit = useCallback(async () => {
    setPending(true);
    setDraftSessionId(null);
    setComposerProgress(null);
    setError(null);
    try {
      const result = await buildTutorialDraft(prompt, {
        onSessionReady: setDraftSessionId,
      });
      onBuilt(result);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setPending(false);
      setDraftSessionId(null);
      setComposerProgress(null);
    }
  }, [onBuilt, prompt]);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-describe-title">
      <p className="tutorial-kicker">Describe</p>
      <h2 id="tutorial-describe-title" ref={headingRef} tabIndex={-1}>
        Describe your pipeline in one sentence.
      </h2>
      <p>
        You do not have to build the layers one at a time. Start with this
        prompt, or edit it before we ask the composer to draft the pipeline.
      </p>
      <label className="tutorial-textarea-label" htmlFor="tutorial-prompt">
        Pipeline description
      </label>
      <textarea
        id="tutorial-prompt"
        className="tutorial-prompt-input"
        value={prompt}
        rows={4}
        disabled={pending}
        onChange={(event) => setPrompt(event.target.value)}
      />
      <div className="tutorial-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={pending || prompt.trim().length === 0}
          onClick={() => void onSubmit()}
        >
          {pending ? "Building..." : TURN_2_PRIMARY_BUTTON}
        </button>
        <button
          type="button"
          className="btn"
          disabled={pending || prompt === CANONICAL_TUTORIAL_PROMPT}
          onClick={() => setPrompt(CANONICAL_TUTORIAL_PROMPT)}
        >
          {TURN_2_RESTORE_BUTTON}
        </button>
        <button
          type="button"
          className="tutorial-link-button"
          disabled={pending}
          onClick={onBack}
        >
          Back
        </button>
      </div>
      {pending && <TutorialDraftProgress progress={composerProgress} />}
      {error !== null && (
        <p role="alert" className="tutorial-error">
          {error}
        </p>
      )}
    </section>
  );
}

export async function buildTutorialDraft(
  prompt: string,
  options: BuildTutorialDraftOptions = {},
): Promise<TutorialBuildResult> {
  const effectivePrompt = prompt.trim() || CANONICAL_TUTORIAL_PROMPT;
  const session = await api.createSession();
  // Tag the session as a tutorial session BEFORE any further work so the
  // backend orphan-cleanup scan (filters by "hello-world (" title prefix)
  // catches sessions abandoned anywhere between createSession and Turn 6's
  // final rename. Without this, a tab close at any point in buildTutorialDraft
  // would leave a "New session" titled session that cleanup never matches.
  await api.renameSession(session.id, HELLO_WORLD_PENDING_SESSION_TITLE);
  options.onSessionReady?.(session.id);
  const response = await api.sendMessage(session.id, effectivePrompt);
  const pendingProposals = (response.proposals ?? []).filter(
    (proposal) => proposal.status === "pending",
  );

  for (const proposal of pendingProposals) {
    await api.acceptCompositionProposal(session.id, proposal.id);
  }

  let compositionState =
    pendingProposals.length > 0 || response.state === null
      ? await api.fetchCompositionState(session.id)
      : response.state;
  if (compositionState === null) {
    throw new Error("The composer did not return a pipeline draft.");
  }

  // The three post-build fetches and the interpretation-events store refresh
  // form a *single* atomic state-publish step: every consumer (chat history,
  // proposals list, composer-preferences trust banner, interpretation-events
  // store) must be wired to the same backend snapshot before the user is
  // shown the built tutorial. Any failure here — auth expiry (401), upstream
  // degradation (5xx), schema drift, or store-refresh rejection — is
  // surfaced to ``onSubmit``'s error region. The prior implementation wrapped
  // each fetch in an ``OrFallback`` helper that substituted materially
  // different state on error, hiding auth and 5xx failures from the user.
  const [messages, proposals, composerPreferences] = await Promise.all([
    api.fetchMessages(session.id),
    api.fetchCompositionProposals(session.id),
    api.fetchComposerPreferences(session.id),
  ]);
  if (typeof composerPreferences.interpretation_review_disabled !== "boolean") {
    throw new Error(
      "composer preferences response missing interpretation_review_disabled",
    );
  }
  if (composerPreferences.interpretation_review_disabled) {
    throw new Error(
      "tutorial sessions must not have interpretation review disabled",
    );
  }

  publishTutorialSession(session, {
    messages,
    compositionState,
    proposals,
    composerPreferences,
  });
  await useInterpretationEventsStore.getState().refreshAll(session.id);

  return {
    sessionId: session.id,
    prompt: effectivePrompt,
    summary: summariseCompositionState(compositionState),
  };
}

function publishTutorialSession(
  session: Session,
  payload: {
    messages: ChatMessage[];
    compositionState: CompositionState;
    proposals: CompositionProposal[];
    composerPreferences: ComposerPreferences | null;
  },
): void {
  useSessionStore.setState((state) => ({
    sessions: [
      session,
      ...state.sessions.filter((existing) => existing.id !== session.id),
    ],
    activeSessionId: session.id,
    messages: payload.messages,
    compositionState: payload.compositionState,
    compositionProposals: payload.proposals,
    composerPreferences: payload.composerPreferences,
    staleProposalIds: [],
    proposalActionPendingIds: [],
    composerProgress: null,
    stateVersions: [],
    isComposing: false,
    error: null,
    errorDetails: null,
    selectedNodeId: null,
  }));
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
  return "The tutorial could not build the draft pipeline.";
}

function TutorialDraftProgress({
  progress,
}: {
  progress: ComposerProgressSnapshot | null;
}): JSX.Element {
  const headline =
    progress?.headline || "Asking the composer to draft the pipeline.";
  const evidence =
    progress !== null && progress.evidence.length > 0
      ? progress.evidence
      : ["Waiting for the composer to report its current step."];
  const likelyNext =
    progress?.likely_next ??
    "ELSPETH will draft the pipeline, accept safe proposals, and prepare it for review.";

  return (
    <div
      role="status"
      aria-busy="true"
      className="tutorial-running tutorial-draft-progress"
    >
      <span className="tutorial-progress-bar" aria-hidden="true" />
      <div className="tutorial-draft-progress-copy">
        <span className="tutorial-draft-progress-headline">{headline}</span>
        <ul className="tutorial-draft-progress-evidence">
          {evidence.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        <span className="tutorial-draft-progress-next">{likelyNext}</span>
      </div>
    </div>
  );
}
