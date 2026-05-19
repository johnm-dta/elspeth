// src/hooks/useComposer.ts
import { useCallback, useRef } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import {
  COMPOSE_TIMEOUT_ABORT_REASON,
  COMPOSE_TIMEOUT_MS,
  COMPOSE_USER_CANCEL_ABORT_REASON,
} from "@/config/composer";

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
  const errorDetails = useSessionStore((s) => s.errorDetails);
  const activeControllerRef = useRef<AbortController | null>(null);

  const runWithTimeout = useCallback(
    async (runner: (signal: AbortSignal) => Promise<void>) => {
      const controller = new AbortController();
      activeControllerRef.current = controller;
      const timer = setTimeout(
        () => controller.abort(COMPOSE_TIMEOUT_ABORT_REASON),
        COMPOSE_TIMEOUT_MS,
      );
      try {
        await runner(controller.signal);
      } finally {
        clearTimeout(timer);
        if (activeControllerRef.current === controller) {
          activeControllerRef.current = null;
        }
      }
    },
    [],
  );

  const sendMessage = useCallback(
    async (content: string) => {
      await runWithTimeout((signal) => storeSendMessage(content, signal));
    },
    [runWithTimeout, storeSendMessage],
  );

  const retryMessage = useCallback(
    async (messageId: string) => {
      await runWithTimeout((signal) => storeRetryMessage(messageId, signal));
    },
    [runWithTimeout, storeRetryMessage],
  );

  const cancelComposition = useCallback(() => {
    activeControllerRef.current?.abort(COMPOSE_USER_CANCEL_ABORT_REASON);
  }, []);

  return {
    sendMessage,
    retryMessage,
    cancelComposition,
    isComposing,
    compositionState,
    error,
    errorDetails,
  };
}
