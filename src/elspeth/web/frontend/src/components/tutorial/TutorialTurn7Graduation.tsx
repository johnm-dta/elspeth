import { useCallback, useEffect, useRef, useState } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import {
  TURN_7_LEARNING_BULLETS,
  TURN_7_PRIMARY_BUTTON,
} from "./copy";

interface TutorialTurn7GraduationProps {
  onBack: () => void;
}

export function TutorialTurn7Graduation({
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
      const completedAt = await usePreferencesStore
        .getState()
        .markTutorialGraduated({ publishLocally: false });
      const previousActiveSessionId = useSessionStore.getState().activeSessionId;
      await useSessionStore.getState().createSession();
      const sessionState = useSessionStore.getState();
      if (sessionState.activeSessionId === previousActiveSessionId) {
        throw new Error(
          sessionState.error ?? "The composer session could not be created.",
        );
      }
      usePreferencesStore.getState().publishTutorialGraduation(completedAt);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setPending(false);
    }
  }, []);

  const busy = pending || writing;

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-graduation-title">
      <p className="tutorial-kicker">Graduation</p>
      <h2 id="tutorial-graduation-title" ref={headingRef} tabIndex={-1}>
        You're ready to use the composer.
      </h2>
      <ul className="tutorial-graduation-list">
        {TURN_7_LEARNING_BULLETS.map((bullet) => (
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
        <button
          type="button"
          className="tutorial-link-button"
          disabled={busy}
          onClick={onBack}
        >
          Back
        </button>
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
