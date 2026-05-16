import { useCallback, useEffect, useRef } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import type { ComposerMode } from "@/types/api";

/**
 * Inner radio-group form. Exported standalone so component tests can render
 * it without the modal chrome; the full panel embeds it.
 *
 * Returns null before bootstrap completes — defaultMode is null until then.
 */
export function ComposerPreferencesForm(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const loaded = usePreferencesStore((s) => s.loaded);
  const writing = usePreferencesStore((s) => s.writing);
  const setDefaultMode = usePreferencesStore((s) => s.setDefaultMode);

  // useCallback must be unconditional (React rules of hooks); the early-return
  // for !loaded sits after the hook calls.
  const onChange = useCallback(
    async (mode: ComposerMode) => {
      try {
        await setDefaultMode(mode);
      } catch (err) {
        console.error("[preferences] setDefaultMode failed:", err);
      }
    },
    [setDefaultMode],
  );

  if (!loaded || defaultMode === null) return null;

  return (
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
        aria-label="Composer preferences"
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
          <h2 className="secrets-panel-title">Composer preferences</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close composer preferences panel"
            className="secrets-panel-close"
          >
            ×
          </button>
        </div>
        <div className="secrets-panel-body">
          <ComposerPreferencesForm />
        </div>
      </div>
    </>
  );
}
