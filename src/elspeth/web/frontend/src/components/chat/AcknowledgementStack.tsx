// ============================================================================
// AcknowledgementStack.tsx — pinned, top-of-chat stack of LLM-authored
// decisions awaiting acknowledgement.
//
// Unifies BOTH guided and freeform modes onto one surface (the surfaces can
// no longer drift).  Driven by the existing `pendingBySession[sessionId]`
// projection; renders nothing when empty so the conversation is unobstructed
// ("clear it to proceed").
//
//   * Header: "N decisions the LLM made — acknowledge each".
//   * Cards ordered by pipeline step then created_at (stable).
//   * ONE foot-of-stack session opt-out link (reuses the store's optOut
//     action + the shared error mapping + the verbatim ConfirmDialog copy).
//   * role="status" live region announces the count (announce-don't-steal —
//     the persistent stack must NOT yank focus from someone typing).
//
// Behaviour (resolve / amend / 8 KB cap / error mapping) is reused verbatim
// via `useInterpretationResolver` inside each AcknowledgementCard.
// ============================================================================

import { useEffect, useMemo, useRef, useState } from "react";
import type { CompositionState } from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import {
  AcknowledgementCard,
  supportsAmendment,
} from "./AcknowledgementCard";
import { buildStepOrder, humaniseStepLabel } from "./interpretationStepLabel";
import {
  describeError,
  type DisplayedError,
} from "@/hooks/useInterpretationResolver";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";

/**
 * The single predicate that defines a "pending acknowledgement": a user-approved
 * interpretation still awaiting the operator's decision.  Shared by the stack
 * (which cards to render), the announcer (the count it reads aloud), and the
 * guided advancement gate so the three can never drift — an announced N that
 * disagreed with the rendered card count would be a fresh defect.
 */
export function isPendingAcknowledgement(event: InterpretationEvent): boolean {
  return (
    event.choice === "pending" &&
    event.interpretation_source === "user_approved"
  );
}

/** Compose the live-region announce text for a given pending count ("" when 0). */
function announceTextFor(count: number): string {
  if (count === 0) return "";
  return count === 1
    ? "1 decision to acknowledge"
    : `${count} decisions to acknowledge`;
}

export interface AcknowledgementLiveRegionProps {
  sessionId: string;
}

/**
 * Persistent, ALWAYS-mounted `role="status"` live region for the acknowledgement
 * stack's count.
 *
 * The stack itself returns null when empty, so a live region rendered inside it
 * would be inserted into the DOM *with its content already present* on the 0→1
 * transition — the WAI-ARIA / MDN-documented unreliable pattern (a polite live
 * region must pre-exist its content for the change to be announced).  This
 * companion region is mounted by ChatPanel regardless of pending count and only
 * the text mutates, so "announce on appearance" (0→1) and "announce on count
 * change" (N→M) are both reliable content mutations inside a stable node.
 */
export function AcknowledgementLiveRegion({
  sessionId,
}: AcknowledgementLiveRegionProps): JSX.Element {
  const pendingBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  const count = useMemo(
    () =>
      Object.values(pendingBySession[sessionId] ?? {}).filter(
        isPendingAcknowledgement,
      ).length,
    [pendingBySession, sessionId],
  );

  return (
    <div
      role="status"
      className="visually-hidden"
      data-testid="acknowledgement-live-region"
    >
      {announceTextFor(count)}
    </div>
  );
}

export interface AcknowledgementStackProps {
  sessionId: string;
  /** Tutorial passive mode: hide the inline amend escape hatch + opt-out. */
  isTutorial?: boolean;
  /**
   * Fired after a successful per-card resolve (with the resolved event) or a
   * session opt-out (event = null).  The parent uses the event for its own
   * post-resolve UI (e.g. the freeform "Got it…" confirmation bubble).
   */
  onResolved?: (
    newState: CompositionState | null,
    event: InterpretationEvent | null,
  ) => void;
}

export function AcknowledgementStack({
  sessionId,
  isTutorial = false,
  onResolved,
}: AcknowledgementStackProps): JSX.Element | null {
  const pendingBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  const compositionState = useSessionStore((s) => s.compositionState);
  const optOut = useInterpretationEventsStore((s) => s.optOut);

  const pending = useMemo(() => {
    const events = Object.values(pendingBySession[sessionId] ?? {}).filter(
      isPendingAcknowledgement,
    );
    const stepOrder = buildStepOrder(compositionState);
    const stepIndexOf = (event: InterpretationEvent): number => {
      const id = event.affected_node_id;
      if (id === null) return Number.POSITIVE_INFINITY;
      const index = stepOrder.get(id);
      return index === undefined ? Number.POSITIVE_INFINITY : index;
    };
    return [...events].sort((a, b) => {
      // Compare step indices directly (subtraction would yield NaN when both
      // are POSITIVE_INFINITY — e.g. no composition loaded yet).
      const stepA = stepIndexOf(a);
      const stepB = stepIndexOf(b);
      if (stepA !== stepB) return stepA < stepB ? -1 : 1;
      const createdDelta = a.created_at.localeCompare(b.created_at);
      return createdDelta !== 0 ? createdDelta : a.id.localeCompare(b.id);
    });
  }, [pendingBySession, sessionId, compositionState]);

  const [showOptOutConfirm, setShowOptOutConfirm] = useState(false);
  const [optOutInFlight, setOptOutInFlight] = useState(false);
  const [optOutError, setOptOutError] = useState<DisplayedError | null>(null);

  // ── Focus restoration on per-card resolve ─────────────────────────────────
  //
  // When a card resolves it is dropped from `pending` and unmounts; the
  // browser would otherwise send focus to document.body, stranding a keyboard
  // / SR user at the top of the page.  We capture each card's Acknowledge
  // button and labelled <section> by id, and after a resolve move focus to the
  // NEXT remaining card (its Acknowledge button when enabled, else its section
  // as a fallback for a still-gated prompt-template card).  Focus is moved
  // ONLY on resolve — never on mount/appearance (announce-don't-steal).
  const acceptRefs = useRef(new Map<string, HTMLButtonElement | null>());
  const sectionRefs = useRef(new Map<string, HTMLElement | null>());
  const [focusTargetId, setFocusTargetId] = useState<string | null>(null);

  useEffect(() => {
    if (focusTargetId === null) return;
    const button = acceptRefs.current.get(focusTargetId);
    const section = sectionRefs.current.get(focusTargetId);
    if (button != null && !button.disabled) {
      button.focus();
    } else if (section != null) {
      section.focus();
    }
    // Clear regardless: a single resolve should restore focus exactly once
    // (last-card → no target → body is the accepted terminal case).
    setFocusTargetId(null);
  }, [pending, focusTargetId]);

  async function handleConfirmOptOut(): Promise<void> {
    setShowOptOutConfirm(false);
    setOptOutInFlight(true);
    try {
      await optOut(sessionId);
      onResolved?.(null, null);
    } catch (err) {
      setOptOutError(describeError(err));
    } finally {
      setOptOutInFlight(false);
    }
  }

  if (pending.length === 0) return null;

  const count = pending.length;
  const headerText =
    count === 1
      ? "1 decision the LLM made — acknowledge it"
      : `${count} decisions the LLM made — acknowledge each`;

  return (
    <section
      className="ack-stack"
      aria-label="Decisions to acknowledge"
      data-testid="acknowledgement-stack"
    >
      {/*
        The count's role="status" live region is NOT rendered here: the stack
        returns null when empty, so a region mounted inside it would be inserted
        WITH its content on the 0→1 transition (the documented-unreliable
        announce pattern).  ChatPanel mounts <AcknowledgementLiveRegion>
        persistently instead, so appearance + count changes are reliable
        content mutations inside a pre-existing node.

        Announce-don't-steal still holds: focus is moved only on a per-card
        resolve (to the next card), never on mount/appearance.
      */}
      <p className="ack-stack-header">{headerText}</p>

      {pending.map((event, index) => (
        <AcknowledgementCard
          key={event.id}
          event={event}
          sessionId={sessionId}
          stepLabel={humaniseStepLabel(compositionState, event.affected_node_id)}
          showAmend={!isTutorial && supportsAmendment(event.kind)}
          acceptButtonRef={(el) => {
            if (el != null) acceptRefs.current.set(event.id, el);
            else acceptRefs.current.delete(event.id);
          }}
          sectionRef={(el) => {
            if (el != null) sectionRefs.current.set(event.id, el);
            else sectionRefs.current.delete(event.id);
          }}
          onResolved={(newState) => {
            // Restore focus to the next remaining card (if any) once this one
            // unmounts.  `pending` is the live ordered list at resolve time,
            // so the next card is simply the following entry.
            const next = pending[index + 1];
            setFocusTargetId(next?.id ?? null);
            onResolved?.(newState, event);
          }}
        />
      ))}

      {!isTutorial && (
        <div className="ack-stack-opt-out">
          {optOutError !== null && (
            <div role="alert" className="ack-stack-error">
              <strong className="ack-stack-error-heading">
                {optOutError.heading}
              </strong>
              <span className="ack-stack-error-body">{optOutError.body}</span>
            </div>
          )}
          <button
            type="button"
            className="ack-stack-opt-out-link"
            onClick={() => setShowOptOutConfirm(true)}
            disabled={optOutInFlight}
          >
            Stop reviewing interpretations this session
          </button>
        </div>
      )}

      {showOptOutConfirm && (
        <ConfirmDialog
          title="Stop reviewing interpretations for this session?"
          message={
            "For the rest of this session, I'll bake interpretations in " +
            "automatically without asking you to review each one.  You can " +
            "audit what was baked from the session's audit-readiness panel."
          }
          confirmLabel="Stop reviewing for this session"
          cancelLabel="Keep reviewing"
          variant="default"
          onConfirm={() => void handleConfirmOptOut()}
          onCancel={() => setShowOptOutConfirm(false)}
        />
      )}
    </section>
  );
}

/**
 * True when the session has at least one pending user_approved interpretation
 * — the predicate the ChatPanel guided branch uses to block wizard
 * advancement while acknowledgements remain (D12).  Relocated here from the
 * retired GuidedInterpretationReviews module.
 */
export function useHasPendingGuidedInterpretations(sessionId: string): boolean {
  const pendingBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  return useMemo(() => {
    const events = Object.values(pendingBySession[sessionId] ?? {});
    return events.some(isPendingAcknowledgement);
  }, [pendingBySession, sessionId]);
}
