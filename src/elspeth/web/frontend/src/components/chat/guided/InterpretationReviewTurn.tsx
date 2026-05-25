// ============================================================================
// InterpretationReviewTurn.tsx — Phase 5b Task 4 of 18b-phase-5b-frontend.md
//
// Guided-mode widget that surfaces an LLM-authored assumption for review:
// vague-term meaning, invented source data, or an LLM prompt template.  Vague
// terms can be accepted, amended, or session-opted-out; source data and prompt
// templates are accept-only surfaces with session opt-out still available.
//
// Pattern follows the convention established by InspectAndConfirmTurn
// (sibling file) and InlineSourceDisambiguationTurn (Phase 5a Task 4):
//
//   - Real <button> elements (no <div onClick>); aria-label on each primary
//     button names the reviewed surface so screen-reader users hear what
//     they are approving.
//   - On mount, focus moves to the accept button unless the caller opts out
//     (parity with InlineSourceDisambiguationTurn's F-19 focus-handoff contract).
//     When the user toggles into amend-mode the focus shifts to the
//     textarea (focus-restoration on view toggle — same convention used by
//     InspectAndConfirmTurn).
//   - role="region" with aria-labelledby pointing at the header so AT users
//     navigate to the widget by region role + a kind-aware accessible name.
//   - role="status" live-region announces that the current surface needs
//     review on mount; this is independent of (and not nested inside) any
//     parent live-region — the parent ChatPanel wraps GuidedTurn in a role="log"
//     region, but the status announcement here is structural to the widget's
//     own contract and must survive ChatPanel refactors.
//
// Behavioural state machine, error-mapping, and resolve/opt-out wiring are
// shared with the freeform-mode counterpart (InterpretationReviewInlineMessage)
// via the `useInterpretationResolver` hook in hooks/useInterpretationResolver.ts.
// This component owns only its rendering, ARIA wiring, and focus management;
// the wire/store logic and 8 KB byte cap live in the hook.
// ============================================================================

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import type { CompositionState } from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import {
  INTERPRETATION_AMENDMENT_MAX_BYTES,
  useInterpretationResolver,
} from "@/hooks/useInterpretationResolver";

// Re-export the byte-cap constant so existing callers / tests that imported
// it from this module continue to compile after the hook extraction.
export { INTERPRETATION_AMENDMENT_MAX_BYTES };

const SCROLL_END_TOLERANCE_PX = 1;

const REVIEW_SURFACE_STYLE: CSSProperties = {
  maxHeight: "16rem",
  overflow: "auto",
  border: "1px solid var(--color-border)",
  borderRadius: "var(--radius-md)",
  padding: "var(--space-sm)",
  background: "var(--color-surface)",
};

const PREFORMATTED_DRAFT_STYLE: CSSProperties = {
  margin: 0,
  whiteSpace: "pre-wrap",
  overflowWrap: "anywhere",
  fontFamily: "var(--font-mono, monospace)",
  fontSize: "var(--font-size-sm)",
  lineHeight: 1.5,
};

interface ReviewPresentation {
  heading: string;
  status: string;
  body: ReactNode;
  acceptLabel: string;
  acceptAriaLabel: string;
}

function supportsAmendment(kind: InterpretationEvent["kind"]): boolean {
  return kind === null || kind === "vague_term";
}

function assertNever(value: never): never {
  throw new Error(`Unhandled interpretation kind: ${String(value)}`);
}

function getReviewPresentation(event: InterpretationEvent): ReviewPresentation {
  const userTerm = event.user_term ?? "this term";
  const llmDraft = event.llm_draft ?? "";
  switch (event.kind) {
    case "invented_source":
      return {
        heading: "Invented source data",
        status: "Invented source data needs review",
        body:
          "You did not provide this source data. Review it before the pipeline fetches anything.",
        acceptLabel: "Use source data",
        acceptAriaLabel: "Accept invented source data",
      };
    case "llm_prompt_template":
      return {
        heading: "LLM prompt template",
        status: "LLM prompt template needs review",
        body: (
          <>
            This is the instruction written for{" "}
            <em>{event.affected_node_id ?? "this transform"}</em>.
          </>
        ),
        acceptLabel: "Use prompt template",
        acceptAriaLabel: "Accept LLM prompt template",
      };
    case "pipeline_decision":
      return {
        heading: "Pipeline decision",
        status: "Pipeline decision needs review",
        body: (
          <>
            The composer made a pipeline-shaping decision for{" "}
            <em>{event.affected_node_id ?? "this transform"}</em>.
          </>
        ),
        acceptLabel: "Use pipeline decision",
        acceptAriaLabel: "Accept pipeline decision",
      };
    case "llm_model_choice":
      return {
        heading: "LLM model choice",
        status: "LLM model choice needs review",
        body: (
          <>
            I picked the model <em>{llmDraft || "(unspecified)"}</em> for{" "}
            <em>{event.affected_node_id ?? "this transform"}</em>. Accept it, or
            change the model before running.
          </>
        ),
        acceptLabel: "Use this model",
        acceptAriaLabel: "Accept LLM model choice",
      };
    case "vague_term":
    case null:
      return {
        heading: "Interpretation review",
        status: "Your input needs review",
        body: (
          <>
            Before we finalise: when you said{" "}
            <em className="guided-interpretation-review-user-term">{userTerm}</em>,
            I read that as roughly{" "}
            <em className="guided-interpretation-review-llm-draft">{llmDraft}</em>.
          </>
        ),
        acceptLabel: "Use my interpretation",
        acceptAriaLabel: `Accept the LLM's interpretation of ${userTerm}`,
      };
    default:
      return assertNever(event.kind);
  }
}

function hasScrolledToEnd(element: HTMLElement): boolean {
  const overflows =
    element.scrollHeight > element.clientHeight + SCROLL_END_TOLERANCE_PX;
  if (!overflows) return true;
  return (
    element.scrollTop + element.clientHeight >=
    element.scrollHeight - SCROLL_END_TOLERANCE_PX
  );
}

function InventedSourceDraft({ value }: { value: string }) {
  return (
    <div
      role="group"
      aria-label="Source data draft"
      tabIndex={0}
      className="guided-interpretation-review-draft guided-interpretation-review-source-draft"
      style={REVIEW_SURFACE_STYLE}
    >
      <pre style={PREFORMATTED_DRAFT_STYLE}>{value}</pre>
    </div>
  );
}

function PromptTemplateDraft({
  value,
  onReviewStateChange,
  surfaceRef,
}: {
  value: string;
  onReviewStateChange: (reviewed: boolean) => void;
  surfaceRef: { current: HTMLDivElement | null };
}) {
  const reportReviewState = useCallback(() => {
    const surface = surfaceRef.current;
    if (surface === null) return;
    onReviewStateChange(hasScrolledToEnd(surface));
  }, [onReviewStateChange, surfaceRef]);

  useEffect(() => {
    reportReviewState();
  }, [reportReviewState, value]);

  return (
    <div
      ref={surfaceRef}
      role="region"
      aria-label="Prompt template review"
      tabIndex={0}
      className="guided-interpretation-review-draft guided-interpretation-review-prompt-template"
      style={REVIEW_SURFACE_STYLE}
      onScroll={reportReviewState}
    >
      <pre style={PREFORMATTED_DRAFT_STYLE}>{value}</pre>
    </div>
  );
}

export interface InterpretationReviewTurnProps {
  /** The pending interpretation event to review. */
  event: InterpretationEvent;
  /** Owning session id; round-tripped to the store actions. */
  sessionId: string;
  /** Whether to render the session-scope opt-out control. */
  showOptOut?: boolean;
  /** Whether to render the amendment affordance when the kind supports it. */
  showAmend?: boolean;
  /** Whether to focus the accept button when the widget mounts. */
  autoFocusOnMount?: boolean;
  /**
   * Optional callback fired after a successful resolve OR opt-out so the
   * parent can advance its own surface (e.g., scroll to the next turn,
   * dismiss the widget).  Errors do NOT fire onResolved — the widget
   * stays mounted with an error banner.
   */
  onResolved?: (newState: CompositionState | null) => void;
}

export function InterpretationReviewTurn({
  event,
  sessionId,
  showOptOut = true,
  showAmend = true,
  autoFocusOnMount = true,
  onResolved,
}: InterpretationReviewTurnProps) {
  const {
    mode,
    amendText,
    setAmendText,
    resolveInFlight,
    showOptOutConfirm,
    displayedError,
    amendByteLength,
    amendIsTooLong,
    submitDisabled,
    primaryButtonsDisabled,
    handleUseMine,
    handleOpenAmend,
    handleCancelAmend,
    handleSubmitAmend,
    handleRequestOptOut,
    handleCancelOptOut,
    handleConfirmOptOut,
  } = useInterpretationResolver({ event, sessionId, onResolved });

  // useId scopes DOM IDs per-instance so multiple widgets coexisting in
  // GuidedHistory (or two open tabs of the same session in development)
  // don't produce colliding header IDs.
  const reactId = useId();
  const headerId = `${reactId}-header`;
  const statusId = `${reactId}-status`;
  const amendInputId = `${reactId}-amend`;
  const errorId = `${reactId}-error`;
  const promptGateId = `${reactId}-prompt-gate`;

  // Refs for focus management.
  const useMineButtonRef = useRef<HTMLButtonElement | null>(null);
  const amendTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const changeItButtonRef = useRef<HTMLButtonElement | null>(null);
  const promptTemplateSurfaceRef = useRef<HTMLDivElement | null>(null);
  const requiresPromptTemplateScroll = event.kind === "llm_prompt_template";
  const [promptTemplateScrolledToEnd, setPromptTemplateScrolledToEnd] =
    useState<boolean | null>(requiresPromptTemplateScroll ? null : true);
  const promptTemplateReviewKey = `${event.id}:${event.kind ?? ""}:${
    event.llm_draft ?? ""
  }`;
  const promptTemplateReviewKeyRef = useRef(promptTemplateReviewKey);

  useEffect(() => {
    if (promptTemplateReviewKeyRef.current === promptTemplateReviewKey) return;
    promptTemplateReviewKeyRef.current = promptTemplateReviewKey;
    setPromptTemplateScrolledToEnd(requiresPromptTemplateScroll ? null : true);
  }, [promptTemplateReviewKey, requiresPromptTemplateScroll]);

  const handlePromptTemplateReviewChange = useCallback((reviewed: boolean) => {
    setPromptTemplateScrolledToEnd(reviewed);
  }, []);

  const presentation = getReviewPresentation(event);
  const amendmentSupported = supportsAmendment(event.kind);
  const shouldShowAmend = showAmend && amendmentSupported;
  const chooseMode = mode === "choose" || !shouldShowAmend;
  const promptTemplateReadyForAccept =
    !requiresPromptTemplateScroll || promptTemplateScrolledToEnd === true;
  const acceptDisabled =
    primaryButtonsDisabled || !promptTemplateReadyForAccept;

  function handleAccept(): void {
    if (acceptDisabled) return;
    void handleUseMine();
  }

  // Focus on mount: the accept button for immediately actionable reviews,
  // or the prompt-template review surface while its scroll gate is closed.
  // Mirrors InlineSourceDisambiguationTurn's F-19 focus-handoff so keyboard
  // users don't tab from the top of the chat panel.
  //
  // This is a guided-mode-only contract: mounting the widget IS the AT
  // user's entry to the surface.  The freeform-mode inline-message
  // counterpart intentionally does NOT focus on mount because it lives
  // inside the chat log and would yank focus from a user typing in the
  // chat input below.
  const initialFocusHandledRef = useRef(false);
  useEffect(() => {
    if (!autoFocusOnMount || initialFocusHandledRef.current) return;
    if (requiresPromptTemplateScroll) {
      if (promptTemplateScrolledToEnd === null) return;
      if (!promptTemplateScrolledToEnd) {
        promptTemplateSurfaceRef.current?.focus();
        initialFocusHandledRef.current = true;
        return;
      }
    }
    useMineButtonRef.current?.focus();
    initialFocusHandledRef.current = true;
  }, [
    autoFocusOnMount,
    promptTemplateScrolledToEnd,
    requiresPromptTemplateScroll,
  ]);

  // Focus on view toggle.  Skip the first run (handled by the mount effect
  // above); on subsequent toggles, move focus to the new view's primary
  // control.
  const firstRunRef = useRef(true);
  useEffect(() => {
    if (firstRunRef.current) {
      firstRunRef.current = false;
      return;
    }
    if (mode === "amend") {
      amendTextareaRef.current?.focus();
    } else {
      changeItButtonRef.current?.focus();
    }
  }, [mode]);

  const userTerm = event.user_term ?? "this term";
  const llmDraft = event.llm_draft ?? "";

  return (
    <section
      role="region"
      aria-labelledby={headerId}
      className="guided-turn guided-interpretation-review-turn"
    >
      {/*
        Live-region announcement on mount.  role="status" implies
        aria-live="polite"; AT users focused elsewhere (e.g. the chat
        input) hear "Your input needs review" when the widget appears.
      */}
      <div id={statusId} role="status" className="visually-hidden">
        {presentation.status}
      </div>

      <h3
        id={headerId}
        className="guided-interpretation-review-heading"
      >
        {presentation.heading}
      </h3>
      <p className="guided-interpretation-review-body">
        {presentation.body}
      </p>

      {event.kind === "invented_source" && (
        <InventedSourceDraft value={llmDraft} />
      )}
      {event.kind === "pipeline_decision" && (
        <InventedSourceDraft value={llmDraft} />
      )}
      {event.kind === "llm_model_choice" && (
        <InventedSourceDraft value={llmDraft} />
      )}
      {event.kind === "llm_prompt_template" && (
        <>
          <PromptTemplateDraft
            value={llmDraft}
            onReviewStateChange={handlePromptTemplateReviewChange}
            surfaceRef={promptTemplateSurfaceRef}
          />
          <span id={promptGateId} className="visually-hidden">
            Review the full prompt template before accepting.
          </span>
        </>
      )}

      {displayedError !== null && (
        <div
          id={errorId}
          role="alert"
          className="guided-interpretation-review-error"
        >
          <strong className="guided-interpretation-review-error-heading">
            {displayedError.heading}
          </strong>
          <span className="guided-interpretation-review-error-body">
            {displayedError.body}
          </span>
        </div>
      )}

      {chooseMode ? (
        <div className="guided-interpretation-review-actions">
          <button
            ref={useMineButtonRef}
            type="button"
            className="btn btn-primary guided-interpretation-review-accept-btn"
            aria-label={presentation.acceptAriaLabel}
            aria-describedby={
              requiresPromptTemplateScroll && !promptTemplateReadyForAccept
                ? promptGateId
                : undefined
            }
            onClick={handleAccept}
            disabled={acceptDisabled}
          >
            {resolveInFlight ? (
              <>
                <span
                  className="guided-interpretation-review-spinner"
                  aria-hidden="true"
                />
                Saving…
              </>
            ) : (
              presentation.acceptLabel
            )}
          </button>
          {shouldShowAmend && (
            <button
              ref={changeItButtonRef}
              type="button"
              className="btn guided-interpretation-review-amend-btn"
              aria-label={`Edit the interpretation of ${userTerm}`}
              onClick={handleOpenAmend}
              disabled={primaryButtonsDisabled}
            >
              Change it: I meant…
            </button>
          )}
        </div>
      ) : (
        <div className="guided-interpretation-review-amend">
          <label
            htmlFor={amendInputId}
            className="guided-interpretation-review-amend-label"
          >
            What did you mean by{" "}
            <em>{userTerm}</em>?
          </label>
          <textarea
            ref={amendTextareaRef}
            id={amendInputId}
            className="guided-interpretation-review-amend-input"
            value={amendText}
            onChange={(e) => setAmendText(e.target.value)}
            rows={4}
            disabled={resolveInFlight}
          />
          {amendIsTooLong && (
            <p
              className="guided-interpretation-review-amend-cap-warning"
              role="status"
            >
              Amendment is {amendByteLength} bytes; the maximum is{" "}
              {INTERPRETATION_AMENDMENT_MAX_BYTES} bytes.
            </p>
          )}
          <div className="guided-interpretation-review-amend-actions">
            <button
              type="button"
              className="btn guided-interpretation-review-cancel-btn"
              onClick={handleCancelAmend}
              disabled={resolveInFlight}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary guided-interpretation-review-submit-btn"
              onClick={() => void handleSubmitAmend()}
              disabled={submitDisabled}
            >
              {resolveInFlight ? (
                <>
                  <span
                    className="guided-interpretation-review-spinner"
                    aria-hidden="true"
                  />
                  Saving…
                </>
              ) : (
                "Submit"
              )}
            </button>
          </div>
        </div>
      )}

      {/*
        Session-scope opt-out.  Rendered as a de-emphasised but Tab-reachable
        text button so the most common per-term action (accepting the draft)
        remains visually distinct from the session-wide opt-out.  Must NOT
        have tabIndex="-1" — keyboard users reach it by Tab; Enter/Space
        activates it via the native <button> contract.
      */}
      {showOptOut && (
        <div className="guided-interpretation-review-opt-out">
          <button
            type="button"
            className="guided-interpretation-review-opt-out-link"
            onClick={handleRequestOptOut}
            disabled={primaryButtonsDisabled}
          >
            Stop reviewing interpretations this session
          </button>
        </div>
      )}

      {showOptOut && showOptOutConfirm && (
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
          onCancel={handleCancelOptOut}
        />
      )}
    </section>
  );
}
