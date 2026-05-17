// ============================================================================
// InlineChatSourceEntry
//
// Synthetic catalog entry rendered as the first row of the Sources tab.
// NOT a backend plugin — it represents "type your data directly in chat;
// no plugin required" per design doc 08-§"The 'Inline data from chat'
// entry" and project_composer_dynamic_source_from_chat.
//
// Clicking the "Try it" action prefills the chat input with a suggested
// prompt and closes the catalog drawer. The composer's chat-driven
// flow takes it from there, creating a one-row dynamic source from the
// user's adapted prompt at runtime.
// ============================================================================

import { useCallback } from "react";
import { PREFILL_CHAT_INPUT_EVENT } from "./PluginCard";

const SUGGESTED_PROMPT =
  "Use the LLM to summarise this article in one sentence: " +
  "https://example.com/article";

interface InlineChatSourceEntryProps {
  onCloseDrawer: () => void;
}

export function InlineChatSourceEntry({ onCloseDrawer }: InlineChatSourceEntryProps) {
  const handleClick = useCallback(() => {
    window.dispatchEvent(
      new CustomEvent(PREFILL_CHAT_INPUT_EVENT, { detail: SUGGESTED_PROMPT }),
    );
    onCloseDrawer();
  }, [onCloseDrawer]);

  return (
    <div className="inline-chat-source-entry" role="region" aria-label="Inline data from chat">
      <div className="inline-chat-source-entry-header">
        <span className="inline-chat-source-entry-title">Inline data from chat</span>
        <span className="inline-chat-source-entry-badge">no plugin needed</span>
      </div>
      <div className="inline-chat-source-entry-desc">
        Type your data directly into chat for small inputs — a URL, a sentence, one record.
        The composer creates a one-row dynamic source from your message. Best for ad-hoc
        runs, demos, and exploring; switch to a real source plugin when you have a
        recurring batch.
      </div>
      <div className="inline-chat-source-entry-example">
        <div className="inline-chat-source-entry-example-label">Suggested prompt:</div>
        <pre className="inline-chat-source-entry-example-code">{SUGGESTED_PROMPT}</pre>
      </div>
      <button
        type="button"
        className="btn btn-small inline-chat-source-entry-try"
        onClick={handleClick}
      >
        Try it
      </button>
    </div>
  );
}
