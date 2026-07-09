import { useCallback, useEffect, useRef } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import type { ComposerMode } from "@/types/api";

/**
 * Inner radio-group form. Exported standalone so component tests can render
 * it without the modal chrome; the full panel embeds it.
 *
 * Returns null before bootstrap completes — defaultMode is null until then.
 *
 * Surfaces a role="alert" region (Panel a11y F2) for failed PATCH results
 * so the write failure is announced rather than silently logging to
 * console only. Also forwards activeSessionId to setDefaultMode so the
 * banner's timing watermark is set if the user opts out from settings
 * while a session is active.
 */
interface ComposerPreferencesFormProps {
  onResetTutorialComplete?: () => void;
}

export function ComposerPreferencesForm({
  onResetTutorialComplete,
}: ComposerPreferencesFormProps = {}): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const loaded = usePreferencesStore((s) => s.loaded);
  const writing = usePreferencesStore((s) => s.writing);
  const writeError = usePreferencesStore((s) => s.writeError);
  const setDefaultMode = usePreferencesStore((s) => s.setDefaultMode);
  const resetTutorial = usePreferencesStore((s) => s.resetTutorial);

  // TODO(hidden-jobs-settings): Add a user-settings view for hidden jobs
  // (run-bearing sessions archived from the switcher). The session switcher
  // can hide/show archived rows locally, but settings should become the
  // durable management surface for review/restore/delete policy.

  // useCallback must be unconditional (React rules of hooks); the early-return
  // for !loaded sits after the hook calls.
  const onChange = useCallback(
    async (mode: ComposerMode) => {
      const activeSessionId = useSessionStore.getState().activeSessionId;
      try {
        await setDefaultMode(mode, activeSessionId);
      } catch (err) {
        // Surfaced via writeError -> role="alert" region below.
        console.error("[preferences] setDefaultMode failed:", err);
      }
    },
    [setDefaultMode],
  );

  const onResetTutorial = useCallback(async () => {
    try {
      await resetTutorial();
      onResetTutorialComplete?.();
    } catch (err) {
      console.error("[preferences] resetTutorial failed:", err);
    }
  }, [onResetTutorialComplete, resetTutorial]);

  if (!loaded || defaultMode === null) return null;

  return (
    <>
      <fieldset disabled={writing} aria-busy={writing}>
        <legend>Default mode for new sessions</legend>
        <label>
          <input
            type="radio"
            name="composer-default-mode"
            value="guided"
            checked={defaultMode === "guided"}
            disabled={writing}
            onChange={() => void onChange("guided")}
          />
          <span>Guided (recommended)</span>
        </label>
        <label>
          <input
            type="radio"
            name="composer-default-mode"
            value="freeform"
            checked={defaultMode === "freeform"}
            disabled={writing}
            onChange={() => void onChange("freeform")}
          />
          <span>Freeform</span>
        </label>
      </fieldset>
      {writeError !== null && (
        <div
          role="alert"
          className="composer-preferences-error"
          style={{
            marginTop: 8,
            color: "var(--color-error)",
            fontSize: 13,
          }}
        >
          {writeError}
        </div>
      )}
      {/* ALWAYS offered (operator requirement: restart the tutorial from
          preferences at any time). The button was originally gated on
          tutorialCompleted, which hid it from mid-tutorial users (the wedged-
          resume escape-hatch case) and from fresh/reset users — reading as
          "the button disappeared". resetTutorial clears completion AND the
          resume fields server-side, so the next load starts a fresh Welcome;
          for a user who never started, it is a harmless no-op PATCH. */}
      <button
        type="button"
        className="btn btn-compact"
        disabled={writing}
        onClick={() => void onResetTutorial()}
        style={{ marginTop: 16 }}
      >
        Reset tutorial
      </button>
    </>
  );
}

interface ComposerPreferencesPanelProps {
  onClose: () => void;
}

/**
 * Modal wrapper around ComposerPreferencesForm. Backdrop + focus-trap +
 * Escape-close + role=dialog/aria-modal, matching the SecretsPanel pattern
 * in this codebase (src/components/settings/SecretsPanel.tsx). The project
 * does not have a generic Dialog component — this layout is the convention.
 */
export function ComposerPreferencesPanel({
  onClose,
}: ComposerPreferencesPanelProps): JSX.Element {
  const modalRef = useRef<HTMLDivElement>(null);
  useFocusTrap(
    modalRef,
    true,
    "input[name='composer-default-mode'][value='guided']",
  );

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <>
      {/* Backdrop */}
      <div
        role="presentation"
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          backgroundColor: "rgba(0,0,0,0.45)",
          zIndex: 100,
        }}
      />
      {/* Modal */}
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="composer-preferences-title"
        style={{
          position: "fixed",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          zIndex: 101,
          width: 480,
          maxWidth: "calc(100vw - 32px)",
          maxHeight: "calc(100vh - 64px)",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "var(--color-surface, #fff)",
          borderRadius: 8,
          boxShadow: "0 8px 32px rgba(0,0,0,0.25)",
          border: "1px solid var(--color-border)",
          fontSize: 13,
          overflow: "hidden",
        }}
      >
        <div className="secrets-panel-header">
          <h2 id="composer-preferences-title" className="secrets-panel-title">
            Composer preferences
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close composer preferences panel"
            className="secrets-panel-close"
            style={{
              minWidth: 32,
              minHeight: 32,
              padding: 4,
              fontSize: 18,
              lineHeight: 1,
              cursor: "pointer",
            }}
          >
            ×
          </button>
        </div>
        <div className="secrets-panel-body">
          <ComposerPreferencesForm onResetTutorialComplete={onClose} />
        </div>
      </div>
    </>
  );
}
