// ============================================================================
// ExitToFreeformButton -- persistent guided-mode exit control.
//
// A single `<button type="button">` that delegates to `exitToFreeform()` in
// useSessionStore.  The store action posts `{control_signal: "exit_to_freeform"}`
// to the backend; this component owns no wire-body construction.
//
// Design decisions:
//   NO confirmation dialog -- the button fires immediately on click.  The demo
//   path prioritises minimal friction; a confirming-mode variant can be added
//   later by wrapping this component rather than modifying it.
//
//   NO props -- the component is self-contained.  It reads exitToFreeform from
//   the store directly.  There is nothing to parameterise for the demo.
//
// Placement: rendered alongside every guided turn by ChatPanel (Phase 8).
// The component has no local state and mounts without stealing focus.
// ============================================================================

import { useSessionStore } from "@/stores/sessionStore";

// ── Component ─────────────────────────────────────────────────────────────────

export function ExitToFreeformButton() {
  const exitToFreeform = useSessionStore((s) => s.exitToFreeform);

  function handleClick(): void {
    void exitToFreeform();
  }

  return (
    <button
      type="button"
      className="guided-exit-button"
      onClick={handleClick}
    >
      Exit to freeform
    </button>
  );
}
