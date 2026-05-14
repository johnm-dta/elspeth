// src/components/chat/ChatPanel.tsx
import { useEffect, useMemo, useRef, useCallback, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useComposer } from "@/hooks/useComposer";
import { FOCUSABLE_SELECTOR } from "@/hooks/useFocusTrap";
import { MessageBubble } from "./MessageBubble";
import { ComposingIndicator } from "./ComposingIndicator";
import { ChatInput } from "./ChatInput";
import { TemplateCards } from "./TemplateCards";
import { BlobManager } from "@/components/blobs/BlobManager";
import { CompletionSummary } from "./guided/CompletionSummary";
import { ExitToFreeformButton } from "./guided/ExitToFreeformButton";
import { GuidedChatHistory } from "./guided/GuidedChatHistory";
import { GuidedHistory } from "./guided/GuidedHistory";
import { GuidedTurn } from "./guided/GuidedTurn";
import type { BlobMetadata, ChatMessage } from "@/types/api";
import type { GuidedStep } from "@/types/guided";

/**
 * Per-step placeholder text for the chat input in guided mode (Phase A slice 4).
 *
 * The wording frames what's *useful* to ask at each wizard step.  This is a
 * UX nudge, not a server-enforced scope — the backend still validates
 * step_index against the live session.step and the per-step skill briefing
 * shapes what the LLM will engage with.  Mirrors the playbook fragments in
 * src/elspeth/web/composer/guided/skills/step_*.md.
 *
 * CLOSED LIST — must cover every GuidedStep member.  Adding a new step
 * member without extending this map produces a TypeScript exhaustiveness
 * error at the lookup site (see assertion in the lookup below).
 */
const GUIDED_CHAT_PLACEHOLDERS: Record<GuidedStep, string> = {
  step_1_source: "Ask about source options, columns, or paste a sample row…",
  step_2_sink: "Ask about sink config, outputs, or schema mode…",
  step_2_5_recipe_match: "Ask about the suggested recipe or alternatives…",
  step_3_transforms: "Ask about the proposed transform chain…",
};

interface ChatPanelProps {
  onOpenSecrets?: () => void;
}

/**
 * Main chat panel combining the message list, composing indicator, and input.
 *
 * Auto-scrolls to the bottom on new messages unless the user has scrolled up.
 * Focus returns to the ChatInput textarea after the assistant response arrives.
 */
export function ChatPanel({ onOpenSecrets }: ChatPanelProps) {
  const messages = useSessionStore((s) => s.messages);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const sessions = useSessionStore((s) => s.sessions);
  const compositionState = useSessionStore((s) => s.compositionState);
  const compositionProposals = useSessionStore((s) => s.compositionProposals);
  const staleProposalIds = useSessionStore((s) => s.staleProposalIds);
  const proposalActionPendingIds = useSessionStore(
    (s) => s.proposalActionPendingIds,
  );
  const acceptProposal = useSessionStore((s) => s.acceptProposal);
  const rejectProposal = useSessionStore((s) => s.rejectProposal);
  const composerProgress = useSessionStore((s) => s.composerProgress);
  const clearError = useSessionStore((s) => s.clearError);
  const forkFromMessage = useSessionStore((s) => s.forkFromMessage);
  // Guided-mode discriminator state.  Selectors are hoisted here (not inside a
  // branch) to comply with React's Rules of Hooks; the discriminator early
  // returns below decide which surface to render based on these values.
  const guidedSession = useSessionStore((s) => s.guidedSession);
  const guidedNextTurn = useSessionStore((s) => s.guidedNextTurn);
  const respondGuided = useSessionStore((s) => s.respondGuided);
  const chatGuided = useSessionStore((s) => s.chatGuided);
  const guidedChatPending = useSessionStore((s) => s.guidedChatPending);
  const guidedResponsePending = useSessionStore((s) => s.guidedResponsePending);

  const activeSessionTitle = sessions.find((s) => s.id === activeSessionId)?.title;
  const { sendMessage, retryMessage, isComposing, error } = useComposer();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const guidedLogRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [showBlobManager, setShowBlobManager] = useState(false);
  const [inputText, setInputText] = useState("");
  const activeComposerMessage = findActiveComposerMessage(messages);
  const proposalsByToolCallId = useMemo(
    () =>
      new Map(
        compositionProposals.map((proposal) => [
          proposal.tool_call_id,
          proposal,
        ]),
      ),
    [compositionProposals],
  );

  function scrollToBottom() {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    setShowScrollButton(false);
  }

  // Track whether the user has scrolled up from the bottom
  function handleScroll() {
    const container = scrollContainerRef.current;
    if (!container) return;
    const threshold = 40; // pixels from bottom
    const atBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight <
      threshold;
    setShowScrollButton(!atBottom);
  }

  // Auto-scroll to bottom when new messages arrive (unless user scrolled up)
  useEffect(() => {
    if (!showScrollButton) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isComposing, showScrollButton]);

  // Return focus to input when composing ends (assistant response arrived)
  useEffect(() => {
    if (!isComposing) {
      inputRef.current?.focus();
    }
  }, [isComposing]);

  // Reset scroll state when switching sessions
  useEffect(() => {
    setShowScrollButton(false);
  }, [activeSessionId]);

  // Spec §7.4 — maintain focus on the first interactive element of the new turn
  // after step advance.  Without this, a step-advancing button click unmounts
  // the button before the browser can return focus elsewhere, so focus falls to
  // <body>.  Keyboard users then have to Tab from the very top to reach the new
  // turn widget — unacceptable for general a11y.
  //
  // Keyed on step_index: fires only when the guided wizard advances to a new
  // step, not on every store mutation that produces a new TurnPayload object
  // with the same step_index.  The ref-null short-circuit handles all non-guided
  // branches implicitly — guidedLogRef.current is null whenever the
  // chat-panel-guided-log div is not mounted (completed surface, freeform
  // surface, no session).  Observation elspeth-obs-5ea21f94af documents the
  // original defect and the chosen Option (c) implementation.
  useEffect(() => {
    if (!guidedLogRef.current) return;
    guidedLogRef.current.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
    const first = guidedLogRef.current.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
    first?.focus();
  }, [guidedNextTurn?.step_index]);

  const handleSend = useCallback(
    (content: string) => {
      sendMessage(content);
      // Explicit send means user has returned to live conversation —
      // force-scroll to bottom and resume auto-scroll.
      setShowScrollButton(false);
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    },
    [sendMessage],
  );

  const handleFork = useCallback(
    (messageId: string, newContent: string) => {
      forkFromMessage(messageId, newContent);
    },
    [forkFromMessage],
  );

  const handleUseAsInput = useCallback(
    (blob: BlobMetadata) => {
      // Insert a helper message referencing the blob by filename.
      // The assistant/composer will use blob tools to wire it as source.
      const prompt = `Please use the file "${blob.filename}" as the pipeline input.`;
      sendMessage(prompt);
      setShowBlobManager(false);
    },
    [sendMessage],
  );

  const handleSelectTemplate = useCallback(
    (prompt: string) => {
      setInputText(prompt);
      // Focus the input so user can edit or press Enter to send
      inputRef.current?.focus();
    },
    [],
  );

  // No active session: show prompt to select or create one
  if (!activeSessionId) {
    return (
      <div
        id="chat-main"
        className="chat-panel chat-panel--empty"
        aria-label="Chat panel"
      >
        Select a session from the sidebar, or create a new one to get
        started.
      </div>
    );
  }

  // ── Guided-mode discriminator ────────────────────────────────────────────────
  //
  // Precedence (intentional):
  //   1. terminal.kind === "completed"  → CompletionSummary surface.
  //   2. active guided session + non-null next turn  → GuidedTurn surface.
  //   3. anything else (no guidedSession, exited_to_freeform terminal, or a
  //      transient state where guidedSession is set but guidedNextTurn is null)
  //      → fall through to the freeform body below.
  //
  // The completed branch is checked FIRST so that a stale `guidedNextTurn`
  // alongside a completed terminal still surfaces the summary (correct UX)
  // rather than dispatching a widget.
  //
  // When `terminal.kind === "exited_to_freeform"`, branch 1 does not match
  // (kind !== "completed") and branch 2 does not match (`!guidedSession.terminal`
  // is false because `terminal` is set). Execution falls through to the existing
  // freeform body — which is the correct outcome (the user has exited; show
  // them the chat surface).
  //
  // Both branches preserve `id="chat-main"` so the skip-link target is honoured;
  // the modifier class (`--guided` / `--completed`) provides a per-branch hook
  // for future CSS without coupling layout to the freeform surface.
  if (guidedSession?.terminal?.kind === "completed") {
    return (
      <div
        id="chat-main"
        className="chat-panel chat-panel--completed"
        aria-label="Pipeline summary"
      >
        <GuidedWorkflowStepper activeStep="ready" />
        {error && (
          <GuidedErrorBanner error={error} onDismiss={clearError} />
        )}
        <CompletionSummary terminal={guidedSession.terminal} />
      </div>
    );
  }

  if (guidedSession && !guidedSession.terminal && guidedNextTurn) {
    return (
      <div
        id="chat-main"
        className="chat-panel chat-panel--guided"
        aria-label="Guided composer"
      >
        <GuidedWorkflowStepper activeStep={guidedSession.step} />
        {error && (
          <GuidedErrorBanner error={error} onDismiss={clearError} />
        )}
        {/*
          aria-live region scope (mirrors the freeform body's
          `<div className="chat-panel-messages">` region below).

          Only the live turn surface (<GuidedTurn>) lives inside the role="log"
          region.  Rationale:

          * GuidedHistory is historical context — already-resolved turns that
            were announced when they first arrived.  Replaying them through the
            live region on every step transition would create redundant SR
            chatter; keep it outside.
          * ExitToFreeformButton is a persistent affordance (always present
            in guided mode).  It is not "new content" on turn change, so it
            also lives outside the log region.
          * GuidedTurn replaces in place when a new step's payload arrives.
            That replacement IS the "new content" event that SRs need to hear
            about — hence the wrapping log region.

          Load-bearing for `InspectAndConfirmTurn.tsx` — search for the
          "Warnings accessibility" comment block (the widget's warnings <aside>
          deliberately omits its own aria-live region under the convention that
          the parent ChatPanel wraps turn content in one).
        */}
        <GuidedHistory history={guidedSession.history} />
        {/*
          Per-step chat log (Phase A slice 6).  Placed ABOVE the wizard
          turn's role="log" region per handover guidance — the user
          reads the chat above their current control surface, and the
          ChatInput at the bottom of the branch is where they reply.
          GuidedChatHistory carries its OWN role="log" + aria-live so
          new chat turns are announced independently of wizard turn
          advances.  Empty-state returns null; no DOM contribution
          before the first chat exchange.
        */}
        <GuidedChatHistory chatHistory={guidedSession.chat_history} />
        <section
          className="guided-current-decision"
          aria-labelledby="guided-current-decision-heading"
        >
          <div className="guided-current-decision-copy">
            <h2 id="guided-current-decision-heading">
              Current decision
            </h2>
            <p>{GUIDED_STEP_PURPOSES[guidedSession.step]}</p>
          </div>
          <div
            ref={guidedLogRef}
            className="chat-panel-guided-log"
            role="log"
            aria-label="Guided wizard step"
            aria-live="polite"
            aria-relevant="additions"
          >
            <GuidedTurn
              turn={guidedNextTurn}
              onSubmit={(body) => void respondGuided(body)}
              disabled={guidedResponsePending}
            />
          </div>
          {guidedResponsePending && (
            <p className="guided-current-decision-pending" role="status">
              Saving decision...
            </p>
          )}
        </section>
        <ExitToFreeformButton />
        {/*
          Per-step conversational chat input (Phase A slice 4).

          Lives below the active wizard turn widget so the widget remains the
          primary control surface; chat is a sidecar.  The textarea is its
          own ChatInput instance separate from the freeform composer's
          ChatInput at the bottom of the freeform branch — they have
          independent uncontrolled state.

          `placeholder` is keyed on the live `guidedSession.step` via the
          GUIDED_CHAT_PLACEHOLDERS map (closed list at module top).  The
          per-step skill briefing on the backend already scopes what the
          LLM will engage with; the placeholder text is a UX nudge that
          mirrors the playbook framing.

          `disabled={guidedChatPending}` blocks rapid double-submits while
          a chat round-trip is in flight.  The store's chatGuided action
          flips the flag back on response (or error).
        */}
        <section
          className="guided-step-chat"
          role="region"
          aria-label="Ask about this step"
        >
          <h2 className="guided-step-chat-heading">Ask about this step</h2>
          <ChatInput
            onSend={(content) => void chatGuided(content)}
            disabled={guidedChatPending}
            inputRef={inputRef}
            placeholder={GUIDED_CHAT_PLACEHOLDERS[guidedSession.step]}
          />
        </section>
      </div>
    );
  }

  return (
    <div
      id="chat-main"
      className="chat-panel"
      aria-label="Chat panel"
    >
      {/* Session title header */}
      {activeSessionTitle && (
        <div className="chat-panel-header">
          <h2 className="chat-panel-header-title">{activeSessionTitle}</h2>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div role="alert" className="chat-panel-error">
          <span>{error}</span>
          <button
            onClick={clearError}
            className="chat-panel-error-dismiss"
            aria-label="Dismiss error"
          >
            {"\u00D7"}
          </button>
        </div>
      )}

      {/* Message list */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="chat-panel-messages"
        role="log"
        aria-label="Chat messages"
        aria-live="polite"
        aria-relevant="additions"
      >
        {messages.length === 0 ? (
          <TemplateCards onSelectTemplate={handleSelectTemplate} />
        ) : (
          messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              isComposing={isComposing}
              onRetry={msg.role === "user" ? retryMessage : undefined}
              onFork={msg.role === "user" ? handleFork : undefined}
              proposalsByToolCallId={proposalsByToolCallId}
              staleProposalIds={staleProposalIds}
              proposalActionPendingIds={proposalActionPendingIds}
              onAcceptProposal={acceptProposal}
              onRejectProposal={rejectProposal}
            />
          ))
        )}
        {isComposing && (
          <ComposingIndicator
            latestRequest={activeComposerMessage?.content ?? null}
            compositionState={compositionState}
            composerProgress={composerProgress}
          />
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Scroll-to-bottom button */}
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          aria-label="Scroll to bottom"
          className="btn scroll-to-bottom-btn"
        >
          {"\u2193"} Jump to latest
        </button>
      )}

      {/* Blob manager drawer */}
      {showBlobManager && <BlobManager onUseAsInput={handleUseAsInput} />}

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        disabled={isComposing}
        inputRef={inputRef}
        onToggleBlobManager={() => setShowBlobManager((v) => !v)}
        showBlobManager={showBlobManager}
        onOpenSecrets={onOpenSecrets}
        value={inputText}
        onChange={setInputText}
      />
    </div>
  );
}

type WorkflowStepId = GuidedStep | "ready";

const GUIDED_STEP_PURPOSES: Record<GuidedStep, string> = {
  step_1_source: "Choose the input and confirm what ELSPETH can read.",
  step_2_sink: "Choose the output shape and the fields the pipeline should produce.",
  step_2_5_recipe_match: "Review the suggested recipe before ELSPETH builds the transforms.",
  step_3_transforms: "Review the transform chain that turns source data into the output.",
};

const GUIDED_WORKFLOW_STEPS: ReadonlyArray<{
  id: WorkflowStepId;
  label: string;
}> = [
  { id: "step_1_source", label: "Source" },
  { id: "step_2_sink", label: "Output" },
  { id: "step_2_5_recipe_match", label: "Recipe" },
  { id: "step_3_transforms", label: "Transforms" },
  { id: "ready", label: "Ready" },
];

function GuidedWorkflowStepper({ activeStep }: { activeStep: WorkflowStepId }) {
  const activeIndex = GUIDED_WORKFLOW_STEPS.findIndex((step) => step.id === activeStep);
  return (
    <nav className="guided-workflow" aria-label="Guided workflow progress">
      <ol className="guided-workflow-list" aria-label="Guided workflow">
        {GUIDED_WORKFLOW_STEPS.map((step, index) => {
          const state =
            index < activeIndex
              ? "complete"
              : index === activeIndex
                ? "current"
                : "upcoming";
          return (
            <li
              key={step.id}
              className={`guided-workflow-step guided-workflow-step--${state}`}
              aria-current={state === "current" ? "step" : undefined}
            >
              <span className="guided-workflow-index">{index + 1}</span>
              <span className="guided-workflow-label">{step.label}</span>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function GuidedErrorBanner({
  error,
  onDismiss,
}: {
  error: string;
  onDismiss: () => void;
}) {
  return (
    <div role="alert" className="chat-panel-error">
      <span>{error}</span>
      <button
        onClick={onDismiss}
        className="chat-panel-error-dismiss"
        aria-label="Dismiss error"
      >
        {"\u00D7"}
      </button>
    </div>
  );
}

function findActiveComposerMessage(messages: ChatMessage[]): ChatMessage | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user" && message.local_status === "pending") {
      return message;
    }
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user") {
      return message;
    }
  }
  return null;
}
