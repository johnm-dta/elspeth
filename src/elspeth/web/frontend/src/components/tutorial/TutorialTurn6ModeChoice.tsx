import { useCallback, useState } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { ComposerMode } from "@/types/api";
import { HELLO_WORLD_SESSION_TITLE, TURN_6_PRIMARY_BUTTON } from "./copy";

interface TutorialTurn6ModeChoiceProps {
  sessionId: string | null;
  skipped: boolean;
  onFinished?: () => void;
}

export function TutorialTurn6ModeChoice({
  sessionId,
  skipped,
  onFinished,
}: TutorialTurn6ModeChoiceProps): JSX.Element {
  const [mode, setMode] = useState<ComposerMode>("guided");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const writing = usePreferencesStore((state) => state.writing);

  const onSave = useCallback(async () => {
    setPending(true);
    setError(null);
    try {
      if (sessionId !== null && !skipped) {
        await useSessionStore
          .getState()
          .renameSession(sessionId, HELLO_WORLD_SESSION_TITLE);
      }
      await usePreferencesStore.getState().markTutorialCompleted(mode);
      await useSessionStore.getState().createSession();
      onFinished?.();
    } catch (err) {
      setError(formatError(err));
    } finally {
      setPending(false);
    }
  }, [mode, onFinished, sessionId, skipped]);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-mode-title">
      <p className="tutorial-kicker">Choose</p>
      <h2 id="tutorial-mode-title">Choose your default composer mode.</h2>
      <p>
        You can switch any time from the chat panel. This only decides how new
        sessions start.
      </p>
      <fieldset className="tutorial-mode-fieldset" disabled={pending || writing}>
        <legend>Default for new sessions</legend>
        <label>
          <input
            type="radio"
            name="tutorial-mode"
            value="guided"
            checked={mode === "guided"}
            onChange={() => setMode("guided")}
          />
          <span>
            <strong>Guided</strong>
            Step-by-step composition with validation and audit checks.
          </span>
        </label>
        <label>
          <input
            type="radio"
            name="tutorial-mode"
            value="freeform"
            checked={mode === "freeform"}
            onChange={() => setMode("freeform")}
          />
          <span>
            <strong>Freeform</strong>
            Describe what you want in chat when you already know the shape.
          </span>
        </label>
      </fieldset>
      <div className="tutorial-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={pending || writing}
          onClick={() => void onSave()}
        >
          {pending || writing ? "Saving..." : TURN_6_PRIMARY_BUTTON}
        </button>
      </div>
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
