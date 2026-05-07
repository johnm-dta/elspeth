// src/components/chat/ChatInput.tsx
import {
  useState,
  useCallback,
  useRef,
  type KeyboardEvent,
  type ChangeEvent,
} from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useBlobStore } from "@/stores/blobStore";

interface ChatInputProps {
  onSend: (content: string) => void;
  disabled: boolean;
  inputRef: React.RefObject<HTMLTextAreaElement>;
  onToggleBlobManager?: () => void;
  showBlobManager?: boolean;
  onOpenSecrets?: () => void;
  /** Controlled mode: external value (use with onChange) */
  value?: string;
  /** Controlled mode: callback when value changes */
  onChange?: (value: string) => void;
}

export function ChatInput({
  onSend,
  disabled,
  inputRef,
  onToggleBlobManager,
  showBlobManager,
  onOpenSecrets,
  value: controlledValue,
  onChange: controlledOnChange,
}: ChatInputProps) {
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
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const uploadBlob = useBlobStore((s) => s.uploadBlob);

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
          placeholder="Describe the pipeline you want to build..."
          aria-label="Message input"
          rows={2}
          className="chat-input-textarea"
        />

        {/* File manager toggle */}
        {onToggleBlobManager && (
          <button
            onClick={onToggleBlobManager}
            title={showBlobManager ? "Hide file manager" : "Show file manager"}
            aria-label={showBlobManager ? "Hide file manager" : "Show file manager"}
            className={`chat-input-icon-btn${showBlobManager ? " chat-input-icon-btn--active" : ""}`}
          >
            <span aria-hidden="true">{"\uD83D\uDCC1"}</span>
          </button>
        )}

        {/* File upload button — using a visible button that clicks a hidden input */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={!activeSessionId}
          className="chat-input-icon-btn"
          title="Upload file"
          aria-label="Upload file"
        >
          <span aria-hidden="true">{"\uD83D\uDCCE"}</span>
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

        {/* Secrets button — key icon, co-located with file actions (A5) */}
        {onOpenSecrets && (
          <button
            type="button"
            onClick={onOpenSecrets}
            className="chat-input-icon-btn"
            title="API Keys & Secrets"
            aria-label="Open secrets settings"
          >
            <span aria-hidden="true">{"\uD83D\uDD11"}</span>
          </button>
        )}

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={!canSend}
          aria-label="Send message"
          className="chat-input-send-btn"
        >
          Send
        </button>
      </div>
      <div className="chat-input-hint" aria-hidden="true">
        Shift+Enter for new line
      </div>
    </div>
  );
}
