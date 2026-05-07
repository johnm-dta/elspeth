// ============================================================================
// useWebSocket Hook
//
// Convenience hook for components that need to react to WebSocket state
// (disconnected banner, progress updates, etc.). The actual WebSocket
// management lives in executionStore.connectWebSocket().
// ============================================================================

import { useExecutionStore } from "@/stores/executionStore";

/**
 * Hook that exposes WebSocket-related state from the execution store.
 *
 * Components use this to display disconnect banners, progress indicators,
 * and other UI tied to the live WebSocket connection.
 */
export function useWebSocket() {
  const wsDisconnected = useExecutionStore((s) => s.wsDisconnected);
  const progress = useExecutionStore((s) => s.progress);
  const activeRunId = useExecutionStore((s) => s.activeRunId);

  return { wsDisconnected, progress, activeRunId };
}
