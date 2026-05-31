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
        <span className="inline-chat-source-entry-badge">chat input</span>
      </div>
      <div className="inline-chat-source-entry-desc">
        For a URL, sentence, or single record. Use a real source for recurring batches.
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
