// ============================================================================
// RunsHistoryDrawer
//
// Slide-over drawer listing every run for the current session. Opened from
// InlineRunResults' "Past runs" button. Preserves audit-trail access to old
// runs after the inspector Runs tab is removed.
// ============================================================================

import { useEffect, useRef } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";

interface RunsHistoryDrawerProps {
  onClose: () => void;
}

const FOCUSABLE_SELECTOR =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function RunsHistoryDrawer({ onClose }: RunsHistoryDrawerProps): JSX.Element {
  const runs = useExecutionStore((s) => s.runs);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const drawerRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeBtnRef.current?.focus();
  }, []);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab") {
        return;
      }

      const drawer = drawerRef.current;
      if (!drawer) {
        return;
      }
      const focusables = drawer.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR);
      if (focusables.length === 0) {
        return;
      }

      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && (active === first || !drawer.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !drawer.contains(active))) {
        e.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div
      ref={drawerRef}
      role="dialog"
      aria-modal="true"
      aria-label="Past pipeline runs"
      className="runs-history-drawer"
    >
      <header className="runs-history-drawer-header">
        <h2>Past runs</h2>
        <button
          ref={closeBtnRef}
          type="button"
          aria-label="Close past runs"
          onClick={onClose}
          className="btn"
        >
          Close
        </button>
      </header>
      <div className="runs-history-drawer-body">
        {runs.length === 0 ? (
          <p>No prior runs for session {activeSessionId ?? "(none)"}.</p>
        ) : (
          <ul className="runs-history-list">
            {runs.map((run) => (
              <li key={run.id} className="runs-history-item">
                <span className="runs-history-item-id">{run.id}</span>
                <span className="runs-history-item-status">
                  {run.status.replace(/_/g, " ")}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
