import { useCallback } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";

/**
 * One-time informational banner that surfaces to users currently on freeform
 * mode (defaultMode === 'freeform') who haven't dismissed yet. Since the only
 * path to defaultMode === 'freeform' is now explicit opt-out (the original
 * session-count heuristic was retired in Phase 1A rev-5), the banner serves
 * as the soft-confirmation surface that an opt-out is active.
 *
 * role="status" — informational, not interrupting. The plan's earlier
 * role="alert" was downgraded after the 2026-05-15 panel review.
 */
export function DefaultModeChangedBanner(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const bannerDismissedAt = usePreferencesStore((s) => s.bannerDismissedAt);
  const loaded = usePreferencesStore((s) => s.loaded);
  const dismiss = usePreferencesStore((s) => s.dismissDefaultChangedBanner);

  const onDismiss = useCallback(() => {
    void dismiss().catch((err) => {
      console.error("[preferences] banner dismiss failed:", err);
    });
  }, [dismiss]);

  const visible =
    loaded && defaultMode === "freeform" && bannerDismissedAt === null;

  if (!visible) return null;

  return (
    <div role="status" className="banner banner-info">
      <p>
        We changed the default for new sessions to{" "}
        <strong>guided mode</strong>. Your account is currently set to start
        new sessions in <strong>freeform</strong>. You can switch from the
        chat panel any time, or change your default in Settings.
      </p>
      <button type="button" onClick={onDismiss}>
        Got it
      </button>
    </div>
  );
}
