import { useCallback } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";

/**
 * Small in-line opt-out affordance for users currently in guided mode. Ticked
 * means "always start new sessions in freeform mode" (i.e. the user has
 * opted out of the new guided default). Bound to the same setDefaultMode()
 * the settings panel uses — both surfaces write to the same backend row.
 *
 * Returns null until preferences load so the chrome doesn't flash an
 * unchecked-then-checked state.
 *
 * Passes the current activeSessionId as the timing-watermark argument so
 * the DefaultModeChangedBanner suppresses itself in the session of
 * opt-out (banner cluster fix, see DefaultModeChangedBanner.tsx).
 *
 * Visual: meets WCAG 2.5.8 target size via padding (the label hit-area
 * is the row, not just the checkbox glyph). 12px font-size is below the
 * 1.4.3 contrast spec on a low-opacity element; opacity removed so the
 * label is full-contrast.
 */
export function InlineOptOutCheckbox(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const loaded = usePreferencesStore((s) => s.loaded);
  const writing = usePreferencesStore((s) => s.writing);
  const writeError = usePreferencesStore((s) => s.writeError);
  const setDefaultMode = usePreferencesStore((s) => s.setDefaultMode);

  // Read the live value at toggle time (rather than closing over `defaultMode`)
  // so two quick clicks don't both target the same starting state.
  const onToggle = useCallback(async () => {
    const current = usePreferencesStore.getState().defaultMode;
    const activeSessionId = useSessionStore.getState().activeSessionId;
    try {
      await setDefaultMode(
        current === "freeform" ? "guided" : "freeform",
        activeSessionId,
      );
    } catch (err) {
      // Surfaced via writeError -> role="alert" region below; keep this
      // as an operator-side console breadcrumb.
      console.error("[preferences] inline opt-out failed:", err);
    }
  }, [setDefaultMode]);

  if (!loaded || defaultMode === null) return null;

  const checked = defaultMode === "freeform";

  return (
    <div className="inline-opt-out-wrapper">
      <label
        className="inline-opt-out"
        style={{
          fontSize: 13,
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          padding: "6px 4px",
          minHeight: 24,
          cursor: writing ? "wait" : "pointer",
        }}
      >
        <input
          type="checkbox"
          checked={checked}
          disabled={writing}
          onChange={() => void onToggle()}
        />
        <span>Always start new sessions in freeform mode</span>
      </label>
      {writeError !== null && (
        <div role="alert" className="inline-opt-out-error" style={{ color: "var(--color-error)", fontSize: 12 }}>
          {writeError}
        </div>
      )}
    </div>
  );
}
