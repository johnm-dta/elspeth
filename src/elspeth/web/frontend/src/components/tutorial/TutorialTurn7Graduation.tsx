import { useCallback, useEffect, useRef, useState } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import {
  GRADUATION_CANCELLED_NOTE,
  HELLO_WORLD_SESSION_TITLE,
  TURN_7_LEARNING_BULLETS,
  TURN_7_LEARNING_BULLETS_SKIPPED,
  TURN_7_PRIMARY_BUTTON,
} from "./copy";

interface TutorialTurn7GraduationProps {
  sessionId: string | null;
  skipped: boolean;
  cancelled: boolean;
  /**
   * Back affordance. Omitted (undefined) when there is no real prior step to
   * return to — e.g. a skipped or cancelled tutorial whose audit step has no
   * run. When undefined the Back button is not rendered so the user can never
   * navigate back into an empty audit.
   */
  onBack?: () => void;
}

export function TutorialTurn7Graduation({
  sessionId,
  skipped,
  cancelled,
  onBack,
}: TutorialTurn7GraduationProps): JSX.Element {
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const writing = usePreferencesStore((state) => state.writing);

  useEffect(() => {
    headingRef.current?.focus();
    window.dispatchEvent(new CustomEvent("tutorial_graduation_shown"));
  }, []);

  const onFinish = useCallback(async () => {
    setPending(true);
    setError(null);
    try {
      // Promote the tutorial session to its final title and save Guided as
      // the default composer mode. A skipped tutorial has no session to
      // rename (and may not have created one), so only rename a real,
      // non-skipped session. Both calls run BEFORE the landing + graduation
      // publish; if either fails we surface the error and do not transition
      // (fail-closed).
      if (sessionId !== null && !skipped) {
        await useSessionStore
          .getState()
          .renameSession(sessionId, HELLO_WORLD_SESSION_TITLE);
      }
      await usePreferencesStore.getState().saveTutorialMode("guided");

      const completedAt = await usePreferencesStore
        .getState()
        .markTutorialGraduated({ publishLocally: false });

      if (sessionId !== null && !skipped) {
        // Land the user ON the pipeline they just built so they can click Run
        // for real and revisit it later — graduation used to drop them into a
        // fresh empty composer, abandoning the worked example. Refresh the
        // session list FIRST so the completed tutorial session appears in the
        // switcher: it was created outside the store (raw API on Start), so
        // renameSession alone never adds it to the in-memory list the switcher
        // renders. selectSession loads its composition state, which the
        // composer auto-validates so the Run button enables.
        await useSessionStore.getState().loadSessions();
        await useSessionStore.getState().selectSession(sessionId);
        const landed = useSessionStore.getState();
        if (landed.activeSessionId !== sessionId) {
          throw new Error(
            landed.error ?? "The composer could not open your pipeline.",
          );
        }
      } else {
        // Skipped (no built pipeline to land on): drop into a fresh composer
        // session so the user still lands somewhere usable.
        const previousActiveSessionId =
          useSessionStore.getState().activeSessionId;
        await useSessionStore.getState().createSession();
        const sessionState = useSessionStore.getState();
        if (sessionState.activeSessionId === previousActiveSessionId) {
          throw new Error(
            sessionState.error ?? "The composer session could not be created.",
          );
        }
      }

      usePreferencesStore.getState().publishTutorialGraduation(completedAt);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setPending(false);
    }
  }, [sessionId, skipped]);

  const busy = pending || writing;

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-graduation-title">
      <p className="tutorial-kicker">Graduation</p>
      <h2 id="tutorial-graduation-title" ref={headingRef} tabIndex={-1}>
        You're ready to use the composer.
      </h2>
      {cancelled && (
        <p role="status" className="tutorial-cancelled-note">
          {GRADUATION_CANCELLED_NOTE}
        </p>
      )}
      <ul className="tutorial-graduation-list">
        {/* Honest skip variant (elspeth-918f4434b3): a user who skipped
            never built or ran anything, so the "pipeline you just ran" /
            "gestures you just practised" bullets would be false. The first
            two bullets swap for future-tense equivalents; the last two are
            shared (they make no just-ran claims). */}
        {(skipped
          ? [...TURN_7_LEARNING_BULLETS_SKIPPED, ...TURN_7_LEARNING_BULLETS.slice(2)]
          : TURN_7_LEARNING_BULLETS
        ).map((bullet) => (
          <li key={bullet.title}>
            <strong>{bullet.title}</strong>
            <span>{bullet.body}</span>
          </li>
        ))}
      </ul>
      <div className="tutorial-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy}
          onClick={() => void onFinish()}
        >
          {busy ? "Saving..." : TURN_7_PRIMARY_BUTTON}
        </button>
        {onBack !== undefined && (
          <button
            type="button"
            className="tutorial-link-button"
            disabled={busy}
            onClick={onBack}
          >
            Back
          </button>
        )}
      </div>
      <p role="status" className="sr-only">
        {busy ? "Saving tutorial completion" : ""}
      </p>
      {error !== null && (
        <p role="alert" className="tutorial-error">
          {error}
        </p>
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
  return "The tutorial could not be completed.";
}
