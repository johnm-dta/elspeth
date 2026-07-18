// ============================================================================
// ModeSwitchButton — the symmetric guided<->freeform mode toggle.
//
// One component for BOTH directions:
//   target="guided"   -> "Switch to guided"  (freeform body header)
//   target="freeform" -> "Exit to freeform"  (guided body header)
//
// Switching converts the CURRENT session in place (enterGuided / exitToFreeform);
// it is non-destructive — the server retains both modes' state on the session.
// Because a stray click still yanks the user out of an in-progress chat, the
// switch is gated by a light two-step confirm WHEN the chat has work; an empty
// chat switches on a single click. The two directions are symmetric.
//
// `hasWork` is computed once by ChatPanel (messages / guided turns / a non-empty
// composition) and passed in, so this component needs no store-shape knowledge
// beyond the two switch actions — and avoids importing ChatPanel's helpers back.
// ============================================================================

import { useId, useState } from "react";

import { useSessionStore } from "@/stores/sessionStore";

interface ModeSwitchButtonProps {
  target: "guided" | "freeform";
  hasWork: boolean;
  /** Optional caller-owned explanation for a disabled mode transition. */
  disabledReason?: string;
}

export function ModeSwitchButton({
  target,
  hasWork,
  disabledReason,
}: ModeSwitchButtonProps): JSX.Element {
  const [confirming, setConfirming] = useState(false);
  const enterGuided = useSessionStore((s) => s.enterGuided);
  const exitToFreeform = useSessionStore((s) => s.exitToFreeform);
  const reactId = useId();
  const disabledReasonId = `${reactId}-mode-switch-disabled-reason`;
  const confirmDescriptionId = `${reactId}-mode-switch-confirm-description`;

  const label = target === "guided" ? "Switch to guided" : "Exit to freeform";
  const confirmLabel =
    target === "guided"
      ? "Confirm switch to guided"
      : "Confirm exit to freeform";
  const confirmTitle =
    target === "guided" ? "Switch to guided mode?" : "Exit to freeform mode?";
  const confirmNote =
    target === "guided"
      ? "Guided mode starts a fresh pipeline. Your current pipeline is saved to version history and can be restored."
      : "Your guided progress remains saved. You can continue in the freeform composer with the current pipeline context.";

  function doSwitch(): void {
    void (target === "guided" ? enterGuided() : exitToFreeform());
  }

  if (disabledReason !== undefined) {
    return (
      <div className="mode-switch-disabled">
        <button
          type="button"
          className="mode-switch-btn"
          disabled
          aria-describedby={disabledReasonId}
        >
          {label}
        </button>
        <span id={disabledReasonId} className="mode-switch-disabled-reason">
          {disabledReason}
        </span>
      </div>
    );
  }

  if (confirming) {
    return (
      <div
        className="mode-switch-confirm mode-switch-confirm-card"
        role="group"
        aria-label={confirmLabel}
        aria-describedby={confirmDescriptionId}
      >
        <span className="mode-switch-confirm-title">{confirmTitle}</span>
        <span id={confirmDescriptionId} className="mode-switch-confirm-note">
          {confirmNote}
        </span>
        <button
          type="button"
          className="mode-switch-btn mode-switch-btn--confirm"
          aria-describedby={confirmDescriptionId}
          onClick={() => {
            setConfirming(false);
            doSwitch();
          }}
        >
          {confirmLabel}
        </button>
        <button
          type="button"
          className="mode-switch-btn"
          onClick={() => setConfirming(false)}
        >
          Cancel
        </button>
      </div>
    );
  }

  return (
    <button
      type="button"
      className="mode-switch-btn"
      onClick={() => (hasWork ? setConfirming(true) : doSwitch())}
    >
      {label}
    </button>
  );
}
