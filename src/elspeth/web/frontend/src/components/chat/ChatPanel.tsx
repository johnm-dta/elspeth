// src/components/chat/ChatPanel.tsx
import { useEffect, useRef, useCallback, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useComposer } from "@/hooks/useComposer";
import { MessageBubble } from "./MessageBubble";
import { ComposingIndicator } from "./ComposingIndicator";
import { ChatInput } from "./ChatInput";
import { TemplateCards } from "./TemplateCards";
import { BlobManager } from "@/components/blobs/BlobManager";
import { CompletionSummary } from "./guided/CompletionSummary";
import { ExitToFreeformButton } from "./guided/ExitToFreeformButton";
import { GuidedHistory } from "./guided/GuidedHistory";
import { GuidedTurn } from "./guided/GuidedTurn";
import type { BlobMetadata, ChatMessage } from "@/types/api";

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
  const composerProgress = useSessionStore((s) => s.composerProgress);
  const clearError = useSessionStore((s) => s.clearError);
  const forkFromMessage = useSessionStore((s) => s.forkFromMessage);
  // Guided-mode discriminator state.  Selectors are hoisted here (not inside a
  // branch) to comply with React's Rules of Hooks; the discriminator early
  // returns below decide which surface to render based on these values.
  const guidedSession = useSessionStore((s) => s.guidedSession);
  const guidedNextTurn = useSessionStore((s) => s.guidedNextTurn);
  const respondGuided = useSessionStore((s) => s.respondGuided);

  const activeSessionTitle = sessions.find((s) => s.id === activeSessionId)?.title;
  const { sendMessage, retryMessage, isComposing, error } = useComposer();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [showBlobManager, setShowBlobManager] = useState(false);
  const [inputText, setInputText] = useState("");
  const activeComposerMessage = findActiveComposerMessage(messages);

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
        <div
          className="chat-panel-guided-log"
          role="log"
          aria-label="Guided wizard step"
          aria-live="polite"
          aria-relevant="additions"
        >
          <GuidedTurn
            turn={guidedNextTurn}
            onSubmit={(body) => void respondGuided(body)}
          />
        </div>
        <ExitToFreeformButton />
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
