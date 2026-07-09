// ============================================================================
// AcknowledgementCard.tsx — one LLM-authored decision, reframed as an
// acknowledgement (not an approval gate).
//
// Extracted from the retired InterpretationReviewTurn / InterpretationReview-
// InlineMessage presentation.  The behaviour — resolve / amend / error
// mapping / 8 KB cap — is reused VERBATIM via `useInterpretationResolver`;
// this component owns only the compact card rendering, the per-kind copy, the
// value rendering (shared CodeBlock for JSON; monospace prompt template
// behind the two-stage View→Approve primary button), ARIA wiring, and focus
// on amend toggle.
//
// Acknowledge == today's accept (`accepted_as_drafted`).  The card NEVER
// auto-steals focus on mount (the stack is persistent and must not yank focus
// from someone typing in the chat input); the stack announces instead.
// ============================================================================

import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import type { CompositionState } from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import { CodeBlock } from "./CodeBlock";
import {
  INTERPRETATION_AMENDMENT_MAX_BYTES,
  useInterpretationResolver,
} from "@/hooks/useInterpretationResolver";

const PROMPT_TEMPLATE_STYLE: CSSProperties = {
  maxHeight: "16rem",
  overflow: "auto",
};

/** Kinds the operator can amend inline (vague_term / legacy null). */
export function supportsAmendment(kind: InterpretationEvent["kind"]): boolean {
  return kind === null || kind === "vague_term";
}

/**
 * Stable DOM id for a card's labelled <section>. The wire-stage named-blocker
 * links (WireStageTurn) target this id to scroll to + focus the blocking card
 * from the other column (elspeth-3b35abf148 variant 1).
 */
export function acknowledgementCardDomId(eventId: string): string {
  return `ack-card-${eventId}`;
}

/**
 * Scroll the card into view and move focus to its section (tabIndex=-1).
 * Deliberate focus-steal: unlike mount-time announce-don't-steal, this runs
 * only on an explicit user click of a "go to blocker" link.
 */
export function focusAcknowledgementCard(eventId: string): void {
  const element = document.getElementById(acknowledgementCardDomId(eventId));
  if (element === null) return;
  element.scrollIntoView({ behavior: "smooth", block: "center" });
  element.focus({ preventScroll: true });
}

function assertNever(value: never): never {
  throw new Error(`Unhandled interpretation kind: ${String(value)}`);
}

interface CardPresentation {
  /** Title row: humanised step label · kind (e.g. "Summarise step · model"). */
  title: string;
  /** One punchy, LLM-attributed line. */
  line: ReactNode;
  /** Accessible label for the Acknowledge button (names the decision). */
  acceptAriaLabel: string;
}

function getCardPresentation(
  event: InterpretationEvent,
  stepLabel: string,
): CardPresentation {
  const userTerm = event.user_term ?? "this term";
  const llmDraft = event.llm_draft ?? "";
  switch (event.kind) {
    case "llm_prompt_template":
      return {
        title: `${stepLabel} step · prompt`,
        line: "The LLM wrote the instruction for this step.",
        // Prompt cards use the two-stage View→Approve button, so the accept
        // action is named "Approve" (visible label and accessible name must
        // agree — WCAG 2.5.3 label-in-name).
        acceptAriaLabel: "Approve the LLM prompt template",
      };
    case "pipeline_decision":
      return {
        title: `${stepLabel} step · decision`,
        line: (
          <span className="ack-card-decision">
            {llmDraft || "(no decision recorded)"}
          </span>
        ),
        acceptAriaLabel: "Acknowledge the pipeline decision",
      };
    case "llm_model_choice":
      return {
        title: `${stepLabel} step · model`,
        line: (
          <>
            The LLM picked{" "}
            <code className="ack-card-model">{llmDraft || "(unspecified)"}</code>
            .
          </>
        ),
        acceptAriaLabel: "Acknowledge the LLM model choice",
      };
    case "invented_source":
      return {
        title: "Source data",
        line: "The LLM invented this source data — review before fetching.",
        acceptAriaLabel: "Acknowledge the invented source data",
      };
    case "vague_term":
    case null:
      return {
        title: "Interpretation",
        line: (
          <>
            You said{" "}
            <em className="ack-card-user-term">{userTerm}</em>; the LLM read it
            as{" "}
            <em className="ack-card-llm-draft">{llmDraft}</em>.
          </>
        ),
        acceptAriaLabel: `Acknowledge the LLM's interpretation of ${userTerm}`,
      };
    default:
      return assertNever(event.kind);
  }
}

/**
 * The card's plain-string title ("Summarise step · prompt"). Exported so the
 * wire-stage blocker list can name the pending card it links to using the
 * exact same wording the card renders — a blocker label that disagreed with
 * the card title would be a fresh way to get lost.
 */
export function acknowledgementCardTitle(
  event: InterpretationEvent,
  stepLabel: string,
): string {
  return getCardPresentation(event, stepLabel).title;
}

export interface AcknowledgementCardProps {
  /** The pending interpretation event to acknowledge. */
  event: InterpretationEvent;
  /** Owning session id; round-tripped to the store actions. */
  sessionId: string;
  /** Humanised step label resolved from the composition (e.g. "Summarise"). */
  stepLabel: string;
  /** Render the inline amend affordance (vague_term only; off in tutorial). */
  showAmend?: boolean;
  /**
   * Fired after a successful resolve so the parent can advance its surface.
   * Errors do NOT fire onResolved — the card stays mounted with an error
   * banner.
   */
  onResolved?: (newState: CompositionState | null) => void;
  /**
   * Callback ref for the Acknowledge button.  The stack uses it to restore
   * focus to the NEXT card's primary action after this card resolves and
   * unmounts (so a keyboard / SR user is not stranded at document.body).
   * Null while the card is in amend mode (no Acknowledge button rendered).
   */
  acceptButtonRef?: (el: HTMLButtonElement | null) => void;
  /**
   * Callback ref for the card's labelled <section> (tabIndex=-1).  Used by the
   * stack as a focus fallback when the next card's primary button is absent
   * or disabled (e.g. amend mode, or a resolve in flight).
   */
  sectionRef?: (el: HTMLElement | null) => void;
}

export function AcknowledgementCard({
  event,
  sessionId,
  stepLabel,
  showAmend = false,
  onResolved,
  acceptButtonRef,
  sectionRef,
}: AcknowledgementCardProps) {
  const {
    mode,
    amendText,
    setAmendText,
    resolveInFlight,
    displayedError,
    amendByteLength,
    amendIsTooLong,
    submitDisabled,
    primaryButtonsDisabled,
    handleUseMine,
    handleOpenAmend,
    handleCancelAmend,
    handleSubmitAmend,
  } = useInterpretationResolver({ event, sessionId, onResolved });

  const reactId = useId();
  const titleId = `${reactId}-title`;
  const amendInputId = `${reactId}-amend`;
  const errorId = `${reactId}-error`;
  const promptGateId = `${reactId}-prompt-gate`;
  const valueRegionId = `${reactId}-value`;

  const llmDraft = event.llm_draft ?? "";
  const userTerm = event.user_term ?? "this term";

  const requiresPromptView = event.kind === "llm_prompt_template";
  const valueIsLong =
    llmDraft.length > 140 || llmDraft.split("\n").length > 4;
  // invented_source shows pretty-printed JSON; short values inline, long
  // values behind the View expander.  The prompt template is ALWAYS behind
  // an expander — opened by the primary button's first stage.
  const hasViewExpander =
    requiresPromptView ||
    (event.kind === "invented_source" && valueIsLong);
  const hasInlineValue = event.kind === "invented_source" && !valueIsLong;

  const [expanded, setExpanded] = useState(false);
  // Two-stage primary button for prompt cards (operator ask 2026-07-03): the
  // big green button does double duty — click 1 reveals the prompt ("View
  // prompt"), the label flips to "Approve", click 2 accepts. Replaces the
  // old scroll-to-end gate, whose disabled-Acknowledge-beside-a-small-View
  // arrangement read as a dead end. Once viewed, always viewed — collapsing
  // the prompt afterwards does not demote the button back to stage 1.
  const [promptViewed, setPromptViewed] = useState(!requiresPromptView);

  const acceptDisabled = primaryButtonsDisabled;

  function handlePrimaryAction(): void {
    if (acceptDisabled) return;
    if (!promptViewed) {
      setExpanded(true);
      setPromptViewed(true);
      return;
    }
    void handleUseMine();
  }

  // Focus on amend-mode toggle ONLY (never on mount — see file header).  Skip
  // the first run so mounting the card does not move focus.
  const amendTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const changeButtonRef = useRef<HTMLButtonElement | null>(null);
  const firstRunRef = useRef(true);
  useEffect(() => {
    if (firstRunRef.current) {
      firstRunRef.current = false;
      return;
    }
    if (mode === "amend") {
      amendTextareaRef.current?.focus();
    } else {
      changeButtonRef.current?.focus();
    }
  }, [mode]);

  const presentation = getCardPresentation(event, stepLabel);
  const chooseMode = mode === "choose" || !showAmend;

  const spinner = (
    <>
      <span className="ack-card-spinner" aria-hidden="true" />
      Saving…
    </>
  );

  return (
    <section
      ref={sectionRef}
      id={acknowledgementCardDomId(event.id)}
      tabIndex={-1}
      className="ack-card"
      aria-labelledby={titleId}
      data-testid="acknowledgement-card"
    >
      <h3 id={titleId} className="ack-card-title">
        {presentation.title}
      </h3>

      <div className="ack-card-main">
        <p className="ack-card-line">{presentation.line}</p>
        {/* Two-stage note: names what the first click does so the primary
            button is never a mystery (successor of the old scroll-gate note,
            elspeth-3b35abf148 variant 2). */}
        {chooseMode && requiresPromptView && !promptViewed && (
          <p id={promptGateId} className="ack-card-gate-note">
            <strong>View prompt</strong> shows the LLM's instruction; the same
            button then approves it.
          </p>
        )}
        {chooseMode && (
          <div className="ack-card-actions">
            <button
              ref={acceptButtonRef}
              type="button"
              className="btn btn-primary ack-card-accept-btn"
              // Stage 1 is a disclosure (visible label is its own accessible
              // name + expander semantics); stage 2 carries the decision-
              // naming accept label.
              aria-label={promptViewed ? presentation.acceptAriaLabel : undefined}
              aria-expanded={!promptViewed ? expanded : undefined}
              aria-controls={!promptViewed ? valueRegionId : undefined}
              aria-describedby={
                requiresPromptView && !promptViewed ? promptGateId : undefined
              }
              onClick={handlePrimaryAction}
              disabled={acceptDisabled}
            >
              {!promptViewed
                ? "View prompt"
                : resolveInFlight
                  ? spinner
                  : requiresPromptView
                    ? "Approve"
                    : "Acknowledge"}
            </button>
            {showAmend && (
              <button
                ref={changeButtonRef}
                type="button"
                className="btn ack-card-amend-btn"
                aria-label={`Edit the interpretation of ${userTerm}`}
                onClick={handleOpenAmend}
                disabled={primaryButtonsDisabled}
              >
                Change…
              </button>
            )}
          </div>
        )}
      </div>

      {hasInlineValue && (
        <CodeBlock
          code={llmDraft}
          prettyJson
          ariaLabel="Invented source data"
        />
      )}

      {hasViewExpander && (
        <div className="ack-card-value">
          {/* Prompt cards: the PRIMARY button owns the first reveal (two-stage
              View→Approve), so this small toggle only appears once viewed —
              two side-by-side "View prompt" buttons would be a duplicate-name
              trap. It then offers collapse/re-open without demoting the
              primary back to stage 1. */}
          {(!requiresPromptView || promptViewed) && (
            <button
              type="button"
              className="ack-card-view-toggle"
              aria-expanded={expanded}
              aria-controls={valueRegionId}
              onClick={() => setExpanded((prev) => !prev)}
            >
              {requiresPromptView
                ? expanded
                  ? "Hide prompt"
                  : "View prompt"
                : expanded
                  ? "Hide"
                  : "View"}
            </button>
          )}
          {expanded &&
            (requiresPromptView ? (
              <div
                id={valueRegionId}
                role="region"
                aria-label="Prompt template review"
                tabIndex={0}
                className="ack-card-prompt-template"
                style={PROMPT_TEMPLATE_STYLE}
              >
                <pre className="ack-card-prompt-pre">{llmDraft}</pre>
              </div>
            ) : (
              <div id={valueRegionId}>
                <CodeBlock
                  code={llmDraft}
                  prettyJson
                  ariaLabel="Invented source data"
                />
              </div>
            ))}
        </div>
      )}

      {displayedError !== null && (
        <div id={errorId} role="alert" className="ack-card-error">
          <strong className="ack-card-error-heading">
            {displayedError.heading}
          </strong>
          <span className="ack-card-error-body">{displayedError.body}</span>
        </div>
      )}

      {!chooseMode && (
        <div className="ack-card-amend">
          <label htmlFor={amendInputId} className="ack-card-amend-label">
            What did you mean by <em>{userTerm}</em>?
          </label>
          <textarea
            ref={amendTextareaRef}
            id={amendInputId}
            className="ack-card-amend-input"
            value={amendText}
            onChange={(e) => setAmendText(e.target.value)}
            rows={4}
            disabled={resolveInFlight}
          />
          {amendIsTooLong && (
            <p className="ack-card-amend-cap-warning" role="status">
              Amendment is {amendByteLength} bytes; the maximum is{" "}
              {INTERPRETATION_AMENDMENT_MAX_BYTES} bytes.
            </p>
          )}
          <div className="ack-card-amend-actions">
            <button
              type="button"
              className="btn ack-card-cancel-btn"
              onClick={handleCancelAmend}
              disabled={resolveInFlight}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary ack-card-submit-btn"
              onClick={() => void handleSubmitAmend()}
              disabled={submitDisabled}
            >
              {resolveInFlight ? spinner : "Submit"}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
