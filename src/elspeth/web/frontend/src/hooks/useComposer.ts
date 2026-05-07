// src/hooks/useComposer.ts
import { useCallback } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { COMPOSE_TIMEOUT_MS } from "@/config/composer";

/**
 * Hook for composing messages. Wraps sessionStore.sendMessage()
 * with an AbortController timeout. Dispatches error messages
 * based on HTTP status and error_type field.
 *
 * The AbortController is wired to abort the underlying fetch when the
 * timeout fires. The sessionStore.sendMessage() call rejects with an
 * AbortError which is then mapped to the timeout user-facing message.
 */
export function useComposer() {
  const storeSendMessage = useSessionStore((s) => s.sendMessage);
  const storeRetryMessage = useSessionStore((s) => s.retryMessage);
  const isComposing = useSessionStore((s) => s.isComposing);
  const compositionState = useSessionStore((s) => s.compositionState);
  const error = useSessionStore((s) => s.error);

  const sendMessage = useCallback(
    async (content: string) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), COMPOSE_TIMEOUT_MS);
      try {
        await storeSendMessage(content, controller.signal);
      } finally {
        clearTimeout(timer);
      }
    },
    [storeSendMessage],
  );

  const retryMessage = useCallback(
    async (messageId: string) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), COMPOSE_TIMEOUT_MS);
      try {
        await storeRetryMessage(messageId, controller.signal);
      } finally {
        clearTimeout(timer);
      }
    },
    [storeRetryMessage],
  );

  return { sendMessage, retryMessage, isComposing, compositionState, error };
}
