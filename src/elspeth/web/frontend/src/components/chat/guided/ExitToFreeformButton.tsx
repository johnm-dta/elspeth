// ============================================================================
// ExitToFreeformButton -- persistent guided-mode exit control.
//
// Delegates to `exitToFreeform()` in useSessionStore.  The store action posts
// `{control_signal: "exit_to_freeform"}` to the backend; this component owns
// no wire-body construction.
//
// Design decision:
//   Exit is a two-click confirmation.  The first click exposes explicit
//   Confirm/Cancel controls; only Confirm calls the store action.
//
// Placement: rendered alongside every guided turn by ChatPanel (Phase 8).
// The component mounts without stealing focus.
// ============================================================================

import { useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";

// ── Component ─────────────────────────────────────────────────────────────────

export function ExitToFreeformButton() {
  const [confirming, setConfirming] = useState(false);
  const exitToFreeform = useSessionStore((s) => s.exitToFreeform);

  function handleConfirm(): void {
    void exitToFreeform();
  }

  if (confirming) {
    return (
      <div
        className="guided-exit-confirmation"
        role="group"
        aria-label="Confirm exit to freeform"
      >
        <button
          type="button"
          className="guided-exit-button guided-exit-button--danger"
          onClick={handleConfirm}
        >
          Confirm exit to freeform
        </button>
        <button
          type="button"
          className="guided-exit-button"
          onClick={() => setConfirming(false)}
        >
          Cancel exit
        </button>
      </div>
    );
  }

  return (
    <button
      type="button"
      className="guided-exit-button"
      onClick={() => setConfirming(true)}
    >
      Exit to freeform
    </button>
  );
}
