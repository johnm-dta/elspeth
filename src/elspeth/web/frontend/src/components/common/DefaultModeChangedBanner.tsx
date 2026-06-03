import { useCallback } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";

/**
 * One-time informational confirmation that surfaces AFTER a user has
 * opted out of guided as their default — the next session they open
 * (or the next page reload) sees this message acknowledging their
 * choice. Dismiss is persisted via dismissDefaultChangedBanner() and
 * cross-tab-broadcast via localStorage (see preferencesStore).
 *
 * Panel cluster fixes (Phase 1B accessibility + UX panel):
 *   - Copy: spec 05 user-choice acknowledgment, not system migration
 *     narration.
 *   - Timing: suppressed while the user is still in the session they
 *     opted out from. The visible-first-time is the NEXT session.
 *   - Cross-tab race: dismiss persists to localStorage; peer tabs
 *     reflect via storage event without making a second PATCH.
 *   - Focus management (WCAG 2.4.3): on dismiss, focus moves to the
 *     chat input rather than stranding on the unmounted dismiss
 *     button (which would revert focus to <body>).
 *
 * role="status" — informational, not interrupting.
 */
export function DefaultModeChangedBanner(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const bannerDismissedAt = usePreferencesStore((s) => s.bannerDismissedAt);
  const loaded = usePreferencesStore((s) => s.loaded);
  const optedOutAtSessionId = usePreferencesStore(
    (s) => s.optedOutAtSessionId,
  );
  const writeError = usePreferencesStore((s) => s.writeError);
  const writing = usePreferencesStore((s) => s.writing);
  const dismiss = usePreferencesStore((s) => s.dismissDefaultChangedBanner);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);

  const onDismiss = useCallback(() => {
    void dismiss()
      .catch((err) => {
        // Error is surfaced through preferencesStore.writeError +
        // role="alert" region (Panel a11y F2); keep console.error as
        // operator-side breadcrumb only.
        console.error("[preferences] banner dismiss failed:", err);
      })
      .finally(() => {
        // WCAG 2.4.3: do not strand focus on an unmounted element.
        // The chat input is the natural next interactive target since
        // the banner sits above the chat panel. Fallback to <main>
        // (skip-link target on the chat surface) if the input is not
        // present (e.g. user is on a non-chat page when the banner
        // dismisses asynchronously).
        const target =
          (document.querySelector("[data-chat-input]") as HTMLElement | null) ??
          (document.getElementById("chat-main") as HTMLElement | null);
        if (target) {
          // chat-main is a non-focusable div; mark it programmatically
          // focusable for this transfer. Removing tabindex on blur is
          // not strictly necessary (it's still excluded from natural Tab
          // traversal at -1), but we leave it set so subsequent banner
          // dismissals don't need to re-set it.
          if (
            target.id === "chat-main" &&
            !target.hasAttribute("tabindex")
          ) {
            target.setAttribute("tabindex", "-1");
          }
          target.focus();
        }
      });
  }, [dismiss]);

  // Timing predicate (Panel banner cluster, item 2):
  //   loaded — bootstrap completed
  //   defaultMode === "freeform" — only opt-out users see this banner
  //   bannerDismissedAt === null — not yet dismissed
  //   optedOutAtSessionId === null OR activeSessionId !== optedOutAtSessionId
  //     — suppress in the SAME session the user opted out from; surface
  //       in the NEXT session (or after a reload that loses the
  //       in-memory watermark — matches "I opted out, refreshed, see
  //       confirmation" intuition).
  const sessionAfterOptOut =
    optedOutAtSessionId === null || activeSessionId !== optedOutAtSessionId;
  const visible =
    loaded &&
    defaultMode === "freeform" &&
    bannerDismissedAt === null &&
    sessionAfterOptOut;

  if (!visible) return null;

  return (
    <div role="status" className="banner banner-info">
      <p>
        Future sessions will start in <strong>freeform mode</strong>. You can
        switch to guided anytime from the chat panel header, or re-enable as
        your default in Composer preferences.
      </p>
      {writeError ? (
        <p role="alert" className="composer-preferences-error">
          {writeError}
        </p>
      ) : null}
      <button
        type="button"
        onClick={onDismiss}
        disabled={writing}
        className="banner-dismiss-btn"
      >
        Got it
      </button>
    </div>
  );
}
