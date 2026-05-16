import { useCallback } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";

/**
 * Small in-line opt-out affordance for users currently in guided mode. Ticked
 * means "always start new sessions in freeform mode" (i.e. the user has
 * opted out of the new guided default). Bound to the same setDefaultMode()
 * the settings panel uses — both surfaces write to the same backend row.
 *
 * Returns null until preferences load so the chrome doesn't flash an
 * unchecked-then-checked state.
 */
export function InlineOptOutCheckbox(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const loaded = usePreferencesStore((s) => s.loaded);
  const writing = usePreferencesStore((s) => s.writing);
  const setDefaultMode = usePreferencesStore((s) => s.setDefaultMode);

  // Read the live value at toggle time (rather than closing over `defaultMode`)
  // so two quick clicks don't both target the same starting state.
  const onToggle = useCallback(async () => {
    const current = usePreferencesStore.getState().defaultMode;
    try {
      await setDefaultMode(current === "freeform" ? "guided" : "freeform");
    } catch (err) {
      console.error("[preferences] inline opt-out failed:", err);
    }
  }, [setDefaultMode]);

  if (!loaded || defaultMode === null) return null;

  const checked = defaultMode === "freeform";

  return (
    <label className="inline-opt-out" style={{ fontSize: 12, opacity: 0.8 }}>
      <input
        type="checkbox"
        checked={checked}
        disabled={writing}
        onChange={() => void onToggle()}
      />{" "}
      <span>Always start new sessions in freeform mode</span>
    </label>
  );
}
