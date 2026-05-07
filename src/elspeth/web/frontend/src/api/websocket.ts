// ============================================================================
// ELSPETH WebSocket Manager
//
// Manages WebSocket connections for pipeline execution progress streaming.
// Connects to /ws/runs/{runId}?token=<jwt> with the JWT appended as a
// query parameter (not a header, since the WebSocket API doesn't support
// custom headers).
//
// Close code discrimination:
//   1000 (normal)   -- run terminal, do NOT reconnect, poll REST
//   1006 (abnormal) -- network drop, auto-reconnect with backoff
//   1011 (internal) -- server error, do NOT reconnect, poll REST
//   4001 (auth)     -- token invalid/expired, do NOT reconnect, trigger logout
//
// Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (capped).
// ============================================================================

import type {
  RunEvent,
  RunEventProgress,
  RunEventError,
  RunEventCompleted,
  RunEventCancelled,
  RunEventFailed,
} from "@/types/index";

/**
 * Callback handlers for WebSocket lifecycle events.
 *
 * - onProgress: Non-terminal row count update.
 * - onError: Non-terminal per-row exception. The pipeline continues;
 *   the frontend appends the error to the exceptions list.
 * - onComplete: Terminal. Pipeline finished successfully.
 * - onCancelled: Terminal. Pipeline was cancelled.
 * - onFailed: Terminal. Pipeline aborted due to unrecoverable error.
 * - onAuthFailure: Close code 4001. Token invalid/expired.
 *   Caller should trigger authStore.logout(). No reconnect attempt.
 */
export interface WebSocketCallbacks {
  onProgress: (event: RunEvent, data: RunEventProgress) => void;
  onError: (event: RunEvent, data: RunEventError) => void;
  onComplete: (event: RunEvent, data: RunEventCompleted) => void;
  onCancelled: (event: RunEvent, data: RunEventCancelled) => void;
  onFailed: (event: RunEvent, data: RunEventFailed) => void;
  onAuthFailure: () => void;
}

/** Handle returned by connectToRun for explicit close. */
export interface WebSocketConnection {
  /** Close the connection and stop reconnect attempts. */
  close: () => void;
}

/** Initial reconnect delay in milliseconds. */
const INITIAL_RECONNECT_DELAY_MS = 1000;

/** Maximum reconnect delay in milliseconds. */
const MAX_RECONNECT_DELAY_MS = 30_000;

/**
 * Connect to the execution progress WebSocket for a given run.
 *
 * The JWT is appended as a query parameter since the WebSocket API
 * does not support custom request headers. The backend validates the
 * token on connection upgrade; close code 4001 indicates auth failure.
 *
 * Returns a connection handle with a close() method. The connection
 * auto-reconnects on abnormal disconnects (code 1006 or unknown codes)
 * until close() is called explicitly or a terminal close code is received.
 */
export function connectToRun(
  runId: string,
  token: string,
  callbacks: WebSocketCallbacks,
): WebSocketConnection {
  let socket: WebSocket | null = null;
  let closed = false;
  let reconnectDelay = INITIAL_RECONNECT_DELAY_MS;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function buildWsUrl(): string {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    return `${protocol}//${host}/ws/runs/${runId}?token=${encodeURIComponent(token)}`;
  }

  function dispatchEvent(event: RunEvent): void {
    switch (event.event_type) {
      case "progress":
        callbacks.onProgress(event, event.data as RunEventProgress);
        break;
      case "error":
        // Non-terminal: per-row exception, pipeline continues.
        // Do NOT close the WebSocket or stop the progress view.
        callbacks.onError(event, event.data as RunEventError);
        break;
      case "completed":
        callbacks.onComplete(event, event.data as RunEventCompleted);
        break;
      case "cancelled":
        callbacks.onCancelled(event, event.data as RunEventCancelled);
        break;
      case "failed":
        callbacks.onFailed(event, event.data as RunEventFailed);
        break;
    }
  }

  function connect(): void {
    if (closed) return;

    socket = new WebSocket(buildWsUrl());

    socket.onopen = (): void => {
      // Reset backoff on successful connection
      reconnectDelay = INITIAL_RECONNECT_DELAY_MS;
    };

    socket.onmessage = (messageEvent: MessageEvent): void => {
      const event: RunEvent = JSON.parse(messageEvent.data as string);
      dispatchEvent(event);

      // Terminal events: stop reconnecting. The server will close
      // the connection with code 1000 after sending a terminal event.
      if (event.event_type === "completed" || event.event_type === "cancelled" || event.event_type === "failed") {
        closed = true;
      }
    };

    socket.onerror = (): void => {
      // The onerror event fires before onclose. Actual handling
      // (reconnect vs. stop) is determined by the close code in onclose.
      // No action needed here beyond the implicit close that follows.
    };

    socket.onclose = (closeEvent: CloseEvent): void => {
      if (closed) return;

      switch (closeEvent.code) {
        case 1000:
          // Normal closure -- run reached terminal state.
          // Do NOT reconnect. The caller should poll GET /api/runs/{id}
          // for the final status if they haven't received a terminal event.
          closed = true;
          return;

        case 1006:
          // Abnormal closure -- network drop or server restart.
          // Auto-reconnect with exponential backoff.
          scheduleReconnect();
          return;

        case 1011:
          // Internal error -- server-side failure.
          // Do NOT reconnect. The caller should poll REST for status.
          closed = true;
          return;

        case 4001:
          // Auth failure -- token invalid or expired.
          // Do NOT reconnect. Trigger logout to redirect to LoginPage.
          closed = true;
          callbacks.onAuthFailure();
          return;

        default:
          // Unknown close code -- treat as abnormal, attempt reconnect.
          scheduleReconnect();
          return;
      }
    };
  }

  function scheduleReconnect(): void {
    if (closed) return;

    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
      // Exponential backoff, capped at MAX_RECONNECT_DELAY_MS
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY_MS);
    }, reconnectDelay);
  }

  // Start the initial connection
  connect();

  return {
    close(): void {
      closed = true;
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (socket !== null) {
        socket.close();
        socket = null;
      }
    },
  };
}
