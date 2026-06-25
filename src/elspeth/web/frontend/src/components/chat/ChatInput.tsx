// src/components/chat/ChatInput.tsx
import {
  useId,
  useState,
  useCallback,
  useEffect,
  useRef,
  type KeyboardEvent,
  type ChangeEvent,
} from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useBlobStore } from "@/stores/blobStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { PREFILL_CHAT_INPUT_EVENT } from "@/components/catalog/PluginCard";

interface ChatInputProps {
  onSend: (content: string) => void;
  disabled: boolean;
  onCancel?: () => void;
  inputRef: React.RefObject<HTMLTextAreaElement>;
  onToggleBlobManager?: () => void;
  showBlobManager?: boolean;
  onOpenSecrets?: () => void;
  /** Controlled mode: external value (use with onChange) */
  value?: string;
  /** Controlled mode: callback when value changes */
  onChange?: (value: string) => void;
  /** Optional native textarea maxLength, used by guided chat to mirror backend validation. */
  maxLength?: number;
  /**
   * Optional placeholder override.  Used by the guided-mode chat input
   * (Phase A slice 4) to surface a per-step nudge.  Defaults to the
   * freeform composer wording when absent.
   */
  placeholder?: string;
  /**
   * Tutorial lock: when true the textarea is read-only (the prepopulated
   * prompt cannot be edited) and the source-composition affordances
   * (file upload, blob manager, secrets) are hidden. The learner still
   * presses Send to submit the locked value. Used by the guided tutorial
   * so the worked-example prompt is prepopulated-and-locked — the learner
   * steps through the normal flow but types nothing.
   */
  readOnly?: boolean;
}

type ChatInputIconName = "folder" | "upload" | "key";

function ChatInputIcon({ name }: { name: ChatInputIconName }): JSX.Element {
  const path =
    name === "folder"
      ? "M3 6.5h6l1.5 2H21v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6.5Zm0 3h18"
      : name === "upload"
        ? "M12 16V4m0 0 4 4m-4-4-4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"
        : "M14.5 9.5a4 4 0 1 0-3.2 3.9L9 15.7V18H6.7L5 19.7 3.3 18l6.3-6.3a4 4 0 0 0 4.9-2.2Zm.5-2h.01";
  return (
    <svg
      aria-hidden="true"
      className="chat-input-icon"
      viewBox="0 0 24 24"
      focusable="false"
    >
      <path d={path} />
    </svg>
  );
}

export function ChatInput({
  onSend,
  disabled,
  onCancel,
  inputRef,
  onToggleBlobManager,
  showBlobManager,
  onOpenSecrets,
  value: controlledValue,
  onChange: controlledOnChange,
  maxLength,
  placeholder,
  readOnly = false,
}: ChatInputProps) {
  // Stable id for the keyboard-hint element, wired into the textarea's
  // aria-describedby so screen readers announce "Shift+Enter for new line"
  // alongside the textarea label. useId keeps it stable across renders and
  // unique per ChatInput instance.
  const hintId = useId();
  // Support both controlled and uncontrolled modes
  const [internalText, setInternalText] = useState("");
  const isControlled = controlledValue !== undefined;
  const text = isControlled ? controlledValue : internalText;
  const setText = isControlled
    ? (v: string) => controlledOnChange?.(v)
    : setInternalText;
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Track current text in a ref to avoid stale closures during async operations
  const textRef = useRef(text);
  textRef.current = text;
  // Track current setText in a ref to avoid re-registering the prefill listener
  // when setText identity changes (in controlled mode) on each render
  const setTextRef = useRef(setText);
  setTextRef.current = setText;
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  // Phase 5a Task 1 — empty-state placeholder primes the user to type data
  // directly into chat (URL / a few rows / a short brief).  Reads two
  // singleton fields on sessionStore (verified: sessionStore.ts:154-155):
  //   - messages: ChatMessage[]                  → message count
  //   - compositionState: CompositionState | null → version (0 when null)
  // Both must read as "empty" to keep the data-priming wording; either
  // signal flipping reverts to the canonical placeholder.
  const messageCount = useSessionStore((s) => s.messages.length);
  const compositionVersion = useSessionStore(
    (s) => s.compositionState?.version ?? 0,
  );
  const uploadBlob = useBlobStore((s) => s.uploadBlob);
  // Phase 5b Task 8 — when a pending interpretation event exists for the
  // active session AND it has a non-null `user_term`, the chat-input
  // placeholder briefly cues the user that the InterpretationReviewTurn
  // widget above is waiting on a decision.  Auto-baked rows
  // (interpretation_source = auto_interpreted_opt_out / no_surfaces) have
  // user_term=null per types/interpretation.ts:101-106 — there is no term
  // to echo, so the cue falls through to the next placeholder layer.
  // Selector returns just the string|null so this component only re-renders
  // when the *displayed term* changes, not on every pending-map mutation.
  const pendingInterpretationUserTerm = useInterpretationEventsStore((s) => {
    if (!activeSessionId) return null;
    const pending = s.pendingBySession[activeSessionId];
    if (!pending) return null;
    for (const event of Object.values(pending)) {
      if (event.user_term !== null) return event.user_term;
    }
    return null;
  });

  // Listen for prefill events dispatched by InlineChatSourceEntry (catalog
  // Sources tab, Phase 7C). After Phase 7B the PluginCard no longer dispatches
  // this event; after Phase 7C InlineChatSourceEntry is the sole dispatcher.
  // Uses setText (the controlled/uncontrolled abstraction) to set the value,
  // then focuses the textarea via the external inputRef.
  useEffect(() => {
    function handlePrefill(e: Event) {
      const detail = (e as CustomEvent<string>).detail;
      if (typeof detail !== "string") {
        // System-to-system contract violation (the dispatcher always sends a
        // string).  Per CLAUDE.md trust-tier model: internal contract
        // violations crash, not log-and-continue.  This surfaces immediately
        // in dev / tests / DevTools rather than producing a silent no-op
        // that a future contributor wouldn't notice.
        throw new TypeError(
          `[ChatInput] PREFILL_CHAT_INPUT_EVENT: expected string detail, got ${typeof detail}`,
        );
      }
      setTextRef.current(detail);
      // Defer focus + caret placement to a microtask so React flushes the
      // controlled-value re-render first; otherwise focus() runs against a
      // stale textarea value and setSelectionRange uses the wrong length.
      // queueMicrotask (not requestAnimationFrame) keeps the prefill
      // synchronous from the user's perspective — no visible 16ms gap.
      queueMicrotask(() => {
        const ta = inputRef.current;
        if (!ta) return;
        ta.focus();
        const len = detail.length;
        ta.setSelectionRange(len, len);
      });
    }
    window.addEventListener(PREFILL_CHAT_INPUT_EVENT, handlePrefill);
    return () => window.removeEventListener(PREFILL_CHAT_INPUT_EVENT, handlePrefill);
  }, [inputRef]);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    // Clear input after send
    if (isControlled) {
      controlledOnChange?.("");
    } else {
      setInternalText("");
    }
  }, [text, disabled, onSend, isControlled, controlledOnChange]);

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleFileSelect(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || !activeSessionId) return;

    setUploadStatus(null);
    try {
      const blob = await uploadBlob(activeSessionId, file);
      // Use ref to get current text (user may have typed during async upload)
      const currentText = textRef.current;
      const newText =
        currentText +
        (currentText ? "\n" : "") +
        `I've uploaded "${blob.filename}"; please use it as the pipeline input.`;
      if (isControlled) {
        controlledOnChange?.(newText);
      } else {
        setInternalText(newText);
      }
    } catch {
      // Error is shown in the blob store / blob manager
      setUploadStatus("Upload failed. Check the file manager for details.");
    } finally {
      // Reset the file input so the same file can be re-selected
      const input = e.target;
      input.value = "";
    }
  }

  const canSend = !disabled && text.trim().length > 0;

  // Phase 5b Task 8 (extends Phase 5a Task 1) — derive the effective
  // placeholder.  Precedence (highest wins):
  //   1. explicit `placeholder` prop (Phase A slice 4 guided-mode nudge)
  //   2. pending-interpretation cue (Phase 5b Task 8 — when an interpretation
  //      review widget is awaiting the user's decision and has a non-null
  //      user_term to echo)
  //   3. empty-state data-priming wording (Phase 5a — no messages, no
  //      composition)
  //   4. canonical "describe the pipeline" wording
  //
  // The pending-interpretation cue sits above empty-state because it
  // describes a *concrete pending decision* the user must address;
  // empty-state is only a generic prime for first-typed-input.  Both
  // continue to sit below an explicit prop so guided-mode per-step nudges
  // (Phase A slice 4) remain authoritative.
  const isEmptyState = messageCount === 0 && compositionVersion === 0;
  const defaultPlaceholder = isEmptyState
    ? "Describe your pipeline, paste a URL, or type a few rows of data to start..."
    : "Describe the pipeline you want to build...";
  const interpretationCuePlaceholder =
    pendingInterpretationUserTerm !== null
      ? `Reviewing your interpretation of "${pendingInterpretationUserTerm}" above — pick Use mine or Change it to continue.`
      : null;
  const effectivePlaceholder =
    placeholder ?? interpretationCuePlaceholder ?? defaultPlaceholder;

  return (
    <div className="chat-input">
      {uploadStatus && (
        <div role="alert" className="chat-input-upload-alert">
          {uploadStatus}
        </div>
      )}
      <div className="chat-input-row" role="group" aria-label="Message composition">
        <textarea
          ref={inputRef}
          data-chat-input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={effectivePlaceholder}
          maxLength={maxLength}
          readOnly={readOnly}
          aria-label="Message input"
          aria-describedby={hintId}
          rows={2}
          className="chat-input-textarea"
        />

        {/* File manager toggle */}
          {!readOnly && onToggleBlobManager && (
            <button
              type="button"
              onClick={onToggleBlobManager}
              title={showBlobManager ? "Hide file manager" : "Show file manager"}
              aria-label={showBlobManager ? "Hide file manager" : "Show file manager"}
              className={`chat-input-icon-btn${showBlobManager ? " chat-input-icon-btn--active" : ""}`}
            >
              <ChatInputIcon name="folder" />
            </button>
          )}

        {/* File upload button — using a visible button that clicks a hidden input */}
        {!readOnly && (
          <>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={!activeSessionId}
              className="chat-input-icon-btn"
              title="Upload file"
              aria-label="Upload file"
            >
              <ChatInputIcon name="upload" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileSelect}
              disabled={!activeSessionId}
              style={{ display: "none" }}
              aria-hidden="true"
              tabIndex={-1}
            />
          </>
        )}

        {/* Secrets button — key icon, co-located with file actions (A5) */}
        {!readOnly && onOpenSecrets && (
          <button
            type="button"
            onClick={onOpenSecrets}
            className="chat-input-icon-btn"
            title="API Keys & Secrets"
            aria-label="Open secrets settings"
          >
            <ChatInputIcon name="key" />
          </button>
        )}

        {disabled && onCancel && (
          <button
            type="button"
            onClick={onCancel}
            aria-label="Stop composing"
            className="chat-input-cancel-btn"
          >
            Stop
          </button>
        )}

        {/* Send button */}
        <button
          type="button"
          onClick={handleSend}
          disabled={!canSend}
          aria-label="Send message"
          aria-keyshortcuts="Enter"
          className="chat-input-send-btn"
        >
          Send
        </button>
      </div>
      {/* Hint is the textarea's aria-describedby target — DO NOT mark it
          aria-hidden, that masks it for some screen readers despite the
          describedby reference. Visible to sighted users, announced after the
          textarea's label by AT. */}
      <div id={hintId} className="chat-input-hint">
        Shift+Enter for new line
      </div>
    </div>
  );
}
