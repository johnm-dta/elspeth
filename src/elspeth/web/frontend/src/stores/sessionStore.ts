// src/stores/sessionStore.ts
import { create } from "zustand";
import type {
  Session,
  ChatMessage,
  CompositionState,
  CompositionStateVersion,
  ComposerProgressSnapshot,
  ApiError,
  ValidationResult,
} from "@/types/api";
import type {
  GuidedSession,
  TurnPayload,
  TerminalState,
  GuidedRespondRequest,
} from "@/types/guided";
import * as api from "@/api/client";
import { COMPOSE_TIMEOUT_MS } from "@/config/composer";
import { useBlobStore } from "./blobStore";
import { useExecutionStore } from "./executionStore";

function getExecutionStore() {
  return useExecutionStore.getState();
}

const COMPOSER_PROGRESS_POLL_INTERVAL_MS = 1500;
const LLM_UNAVAILABLE_MESSAGE =
  "The AI service is temporarily unavailable. Please try again in a moment.";
const LLM_AUTH_ERROR_MESSAGE =
  "The AI service configuration is invalid. Please contact your administrator.";
// Surfaced when the client-side AbortController fires (typically the
// COMPOSE_TIMEOUT_MS guard in useComposer). Distinct from the backend's
// 422/convergence_wall_clock_timeout copy because the cause is different:
// the browser gave up before the server reached its own deadline.
const COMPOSE_TIMEOUT_MESSAGE =
  "ELSPETH took too long to compose a response. Try a smaller request or split it into multiple steps.";

function isAbortError(err: unknown): boolean {
  // DOMException ('AbortError'/'TimeoutError') is not always an Error
  // subclass across runtimes (browsers, jsdom, Node). Match on the
  // structural `name` field — that's the cross-platform contract.
  if (typeof err !== "object" || err === null) {
    return false;
  }
  const name = (err as { name?: unknown }).name;
  return name === "AbortError" || name === "TimeoutError";
}

let composerProgressPollTimer: ReturnType<typeof setInterval> | null = null;
let composerProgressPollSessionId: string | null = null;

function clearComposerProgressPollTimer(): void {
  if (composerProgressPollTimer !== null) {
    clearInterval(composerProgressPollTimer);
    composerProgressPollTimer = null;
  }
  composerProgressPollSessionId = null;
}

function formatProviderDiagnostic(apiErr: ApiError): string {
  const lines: string[] = [];
  if (apiErr.provider_detail) {
    lines.push(apiErr.provider_detail);
  }
  if (apiErr.provider_status_code !== undefined) {
    lines.push(`Provider status: ${apiErr.provider_status_code}`);
  }
  return lines.length > 0 ? `\n\n${lines.join("\n")}` : "";
}

function formatLlmUnavailableError(apiErr: ApiError): string {
  return `${LLM_UNAVAILABLE_MESSAGE}${formatProviderDiagnostic(apiErr)}`;
}

function formatLlmAuthError(apiErr: ApiError): string {
  return `${LLM_AUTH_ERROR_MESSAGE}${formatProviderDiagnostic(apiErr)}`;
}

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: ChatMessage[];
  compositionState: CompositionState | null;
  composerProgress: ComposerProgressSnapshot | null;
  isComposing: boolean;
  stateVersions: CompositionStateVersion[];
  error: string | null;

  // Shared selection state for cross-component sync (GraphView <-> SpecView)
  selectedNodeId: string | null;
  selectNode: (nodeId: string | null) => void;

  loadSessions: () => Promise<void>;
  createSession: () => Promise<void>;
  selectSession: (id: string) => Promise<void>;
  archiveSession: (id: string) => Promise<void>;
  sendMessage: (content: string, signal?: AbortSignal) => Promise<void>;
  loadComposerProgress: (sessionId?: string) => Promise<void>;
  startComposerProgressPolling: (sessionId: string) => void;
  stopComposerProgressPolling: (sessionId?: string) => void;
  sendValidationFeedback: (result: ValidationResult) => Promise<void>;
  retryMessage: (messageId: string, signal?: AbortSignal) => Promise<void>;
  forkFromMessage: (messageId: string, newContent: string) => Promise<void>;
  loadStateVersions: () => Promise<void>;
  isLoadingVersions: boolean;
  revertToVersion: (stateId: string) => Promise<void>;

  // Guided-mode protocol state — all three are null when not in a guided session
  guidedSession: GuidedSession | null;
  guidedNextTurn: TurnPayload | null;
  guidedTerminal: TerminalState | null;
  // Guided-mode actions
  startGuided: (sessionId: string) => Promise<void>;
  respondGuided: (body: GuidedRespondRequest) => Promise<void>;
  exitToFreeform: () => Promise<void>;
  clearError: () => void;
  injectSystemMessage: (content: string, stableId?: string) => void;
  reset: () => void;
}

const initialState = {
  sessions: [] as Session[],
  activeSessionId: null as string | null,
  messages: [] as ChatMessage[],
  compositionState: null as CompositionState | null,
  composerProgress: null as ComposerProgressSnapshot | null,
  isComposing: false,
  stateVersions: [] as CompositionStateVersion[],
  isLoadingVersions: false,
  error: null as string | null,
  selectedNodeId: null as string | null,
  guidedSession: null as GuidedSession | null,
  guidedNextTurn: null as TurnPayload | null,
  guidedTerminal: null as TerminalState | null,
};

export const useSessionStore = create<SessionState>((set, get) => ({
  ...initialState,

  async loadSessions() {
    try {
      const sessions = await api.fetchSessions();
      set({ sessions });
    } catch {
      set({ error: "Failed to load sessions. Please refresh the page." });
    }
  },

  async createSession() {
    try {
      const session = await api.createSession();
      clearComposerProgressPollTimer();
      set((state) => ({
        sessions: [session, ...state.sessions],
        activeSessionId: session.id,
        messages: [],
        compositionState: null,
        composerProgress: null,
        stateVersions: [],
        error: null,
        selectedNodeId: null, // Clear selection for new session
        guidedSession: null,
        guidedNextTurn: null,
        guidedTerminal: null,
      }));
      // Auto-start guided mode.  New sessions are created with
      // GuidedSession.initial() already attached by the backend (spec §5.2,
      // errata C7).  startGuided's own catch block handles failures without
      // clobbering the session we just set above.
      void get().startGuided(session.id);
    } catch {
      set({ error: "Failed to create session. Please try again." });
    }
  },

  async archiveSession(id: string) {
    try {
      await api.archiveSession(id);
      set((state) => {
        const sessions = state.sessions.filter((s) => s.id !== id);
        // If we archived the active session, clear selection
        const wasActive = state.activeSessionId === id;
        if (wasActive) {
          clearComposerProgressPollTimer();
        }
        return {
          sessions,
          ...(wasActive
            ? {
                activeSessionId: null,
                messages: [],
                compositionState: null,
                composerProgress: null,
                stateVersions: [],
                isComposing: false,
                selectedNodeId: null,
                guidedSession: null,
                guidedNextTurn: null,
                guidedTerminal: null,
              }
            : {}),
        };
      });
    } catch {
      set({ error: "Failed to archive session. Please try again." });
    }
  },

  async selectSession(id: string) {
    // R4-H3: Clear validation when switching sessions to prevent
    // stale validation from a previous session being visible
    getExecutionStore().clearValidation();
    clearComposerProgressPollTimer();

    set({
      activeSessionId: id,
      messages: [],
      compositionState: null,
      composerProgress: null,
      stateVersions: [],
      isComposing: false,
      error: null,
      selectedNodeId: null, // Clear selection when switching sessions
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
    });

    try {
      const [messages, compositionState] = await Promise.all([
        api.fetchMessages(id),
        api.fetchCompositionState(id),
      ]);
      set({ messages, compositionState });

      // Auto-start guided mode for the selected session.
      //
      // Per spec §5.2 and errata C7: new sessions are created with
      // GuidedSession.initial() already attached by the backend.  The
      // GET /guided endpoint auto-initialises on first call and returns the
      // current step (or terminal state for sessions that have already
      // exited to freeform).
      //
      // startGuided is fire-and-forget here: its own catch block sets the
      // error field without clobbering the session/messages/compositionState
      // we just loaded.  ChatPanel's guided-mode discriminator falls through
      // to the freeform surface when guidedSession is null (startup race) or
      // when terminal.kind === "exited_to_freeform" — both safe.
      void get().startGuided(id);

      // Fire-and-forget: refresh blob list for the newly selected session
      useBlobStore.getState().loadBlobs(id);
    } catch {
      set({ error: "Failed to load session. Please refresh the page." });
    }
  },

  async sendMessage(content: string, signal?: AbortSignal) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    const optimisticMessage: ChatMessage = {
      id: `local-${crypto.randomUUID()}`,
      session_id: activeSessionId,
      role: "user",
      content,
      tool_calls: null,
      created_at: new Date().toISOString(),
      local_status: "pending",
    };

    set((state) => ({
      isComposing: true,
      error: null,
      composerProgress: null,
      messages: [...state.messages, optimisticMessage],
    }));
    get().startComposerProgressPolling(activeSessionId);

    try {
      const stateId = get().compositionState?.id;
      const result = await api.sendMessage(activeSessionId, content, stateId, signal);
      const { message, state } = result;
      set((s) => {
        const previousVersion = s.compositionState?.version ?? null;
        const newVersion = state?.version ?? null;
        const versionChanged =
          newVersion !== null && newVersion !== previousVersion;

        // R4-H3: Clear validation BEFORE updating compositionState
        // when a new state version arrives from the composer
        if (versionChanged) {
          getExecutionStore().clearValidation();
        }

        // Clear selection if the selected node no longer exists in new state
        const newState = state ?? s.compositionState;
        const nodeStillExists =
          !s.selectedNodeId ||
          newState?.nodes.some((n) => n.id === s.selectedNodeId);

        return {
          messages: s.messages.map((existing) =>
            existing.id === optimisticMessage.id
              ? { ...existing, local_status: undefined, local_error: undefined }
              : existing,
          ).concat(message),
          compositionState: newState,
          isComposing: false,
          ...(nodeStillExists ? {} : { selectedNodeId: null }),
        };
      });

      // Fire-and-forget: refresh blob list in case the LLM created files
      useBlobStore.getState().loadBlobs(activeSessionId);
    } catch (err) {
      let errorMessage: string;
      // Client-side abort (the useComposer COMPOSE_TIMEOUT_MS guard or any
      // user-supplied signal) fires a DOMException with name 'AbortError',
      // not a structured ApiError — the apiErr.detail fallback below would
      // otherwise mask it as a generic send failure.
      if (isAbortError(err)) {
        errorMessage = COMPOSE_TIMEOUT_MESSAGE;
      } else {
        const apiErr = err as ApiError;
        // Error dispatch based on HTTP status + error_type field
        if (apiErr.status === 422 && apiErr.error_type === "convergence") {
          errorMessage =
            "ELSPETH couldn't complete the composition after multiple attempts. Try breaking your request into smaller steps.";
        } else if (
          apiErr.status === 502 &&
          apiErr.error_type === "llm_unavailable"
        ) {
          errorMessage = formatLlmUnavailableError(apiErr);
        } else if (
          apiErr.status === 502 &&
          apiErr.error_type === "llm_auth_error"
        ) {
          errorMessage = formatLlmAuthError(apiErr);
        } else {
          errorMessage =
            apiErr.detail ?? "Failed to send message. Please try again.";
        }
      }
      set((state) => ({
        isComposing: false,
        error: errorMessage,
        messages: state.messages.map((existing) =>
          existing.id === optimisticMessage.id
            ? { ...existing, local_status: "failed", local_error: errorMessage }
            : existing,
        ),
      }));
    } finally {
      get().stopComposerProgressPolling(activeSessionId);
    }
  },

  async loadComposerProgress(sessionId?: string) {
    const targetSessionId = sessionId ?? get().activeSessionId;
    if (!targetSessionId) return;

    try {
      const progress = await api.fetchComposerProgress(targetSessionId);
      const current = get();
      if (
        current.activeSessionId !== targetSessionId ||
        !current.isComposing
      ) {
        return;
      }
      set({ composerProgress: progress.phase === "idle" ? null : progress });
    } catch {
      // Composer progress is advisory. Keep the local heuristic fallback.
    }
  },

  startComposerProgressPolling(sessionId: string) {
    clearComposerProgressPollTimer();
    composerProgressPollSessionId = sessionId;
    set({ composerProgress: null });
    void get().loadComposerProgress(sessionId);
    composerProgressPollTimer = setInterval(() => {
      if (composerProgressPollSessionId !== sessionId) return;
      void useSessionStore.getState().loadComposerProgress(sessionId);
    }, COMPOSER_PROGRESS_POLL_INTERVAL_MS);
  },

  stopComposerProgressPolling(sessionId?: string) {
    if (
      sessionId !== undefined &&
      composerProgressPollSessionId !== null &&
      composerProgressPollSessionId !== sessionId
    ) {
      return;
    }
    clearComposerProgressPollTimer();
    if (sessionId === undefined || get().activeSessionId === sessionId) {
      set({ composerProgress: null });
    }
  },

  async sendValidationFeedback(result: ValidationResult) {
    // Format validation errors into a message the LLM can act on.
    const lines = ["Pipeline validation failed with the following errors:"];
    for (const err of result.errors) {
      lines.push(
        `- [${err.component_type ?? "unknown"}] ${err.component_id ?? "unknown"}: ${err.message}`,
      );
      if (err.suggestion) {
        lines.push(`  Suggestion: ${err.suggestion}`);
      }
    }
    lines.push("", "Please fix these validation errors.");
    const content = lines.join("\n");

    // Use sendMessage with the same timeout as manual sends.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), COMPOSE_TIMEOUT_MS);
    try {
      await get().sendMessage(content, controller.signal);
    } finally {
      clearTimeout(timer);
    }
  },

  async retryMessage(messageId: string, signal?: AbortSignal) {
    const { activeSessionId, messages } = get();
    if (!activeSessionId) return;

    const message = messages.find((entry) => entry.id === messageId);
    if (!message || message.role !== "user") return;

    set((state) => ({
      isComposing: true,
      error: null,
      composerProgress: null,
      messages: state.messages.map((existing) =>
        existing.id === messageId
          ? { ...existing, local_status: "pending" }
          : existing,
      ),
    }));
    get().startComposerProgressPolling(activeSessionId);

    try {
      // Use recompose (not sendMessage) — the user message is already
      // persisted from the original send. Calling sendMessage again
      // would insert a duplicate user message.
      const result = await api.recompose(activeSessionId, signal);
      const { message: assistantMessage, state } = result;
      set((s) => {
        const previousVersion = s.compositionState?.version ?? null;
        const newVersion = state?.version ?? null;
        const versionChanged =
          newVersion !== null && newVersion !== previousVersion;

        if (versionChanged) {
          getExecutionStore().clearValidation();
        }

        // Clear selection if the selected node no longer exists in new state
        const newState = state ?? s.compositionState;
        const nodeStillExists =
          !s.selectedNodeId ||
          newState?.nodes.some((n) => n.id === s.selectedNodeId);

        return {
          messages: s.messages.map((existing) =>
            existing.id === messageId
              ? { ...existing, local_status: undefined, local_error: undefined }
              : existing,
          ).concat(assistantMessage),
          compositionState: newState,
          isComposing: false,
          ...(nodeStillExists ? {} : { selectedNodeId: null }),
        };
      });

      // Fire-and-forget: refresh blob list in case the LLM created files
      useBlobStore.getState().loadBlobs(activeSessionId);
    } catch (err) {
      let errorMessage: string;
      if (isAbortError(err)) {
        errorMessage = COMPOSE_TIMEOUT_MESSAGE;
      } else {
        const apiErr = err as ApiError;
        errorMessage =
          apiErr.status === 502 && apiErr.error_type === "llm_unavailable"
            ? formatLlmUnavailableError(apiErr)
            : apiErr.status === 502 && apiErr.error_type === "llm_auth_error"
              ? formatLlmAuthError(apiErr)
              : apiErr.status === 422 && apiErr.error_type === "convergence"
                ? "ELSPETH couldn't complete the composition after multiple attempts. Try breaking your request into smaller steps."
                : apiErr.detail ?? "Failed to send message. Please try again.";
      }

      set((state) => ({
        isComposing: false,
        error: errorMessage,
        messages: state.messages.map((existing) =>
          existing.id === messageId
            ? { ...existing, local_status: "failed", local_error: errorMessage }
            : existing,
        ),
      }));
    } finally {
      get().stopComposerProgressPolling(activeSessionId);
    }
  },

  async forkFromMessage(messageId: string, newContent: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    clearComposerProgressPollTimer();
    set({ isComposing: true, error: null });
    try {
      const result = await api.forkFromMessage(
        activeSessionId,
        messageId,
        newContent,
      );
      // Clear validation for the new session
      getExecutionStore().clearValidation();

      set((state) => ({
        sessions: [result.session, ...state.sessions],
        activeSessionId: result.session.id,
        messages: result.messages,
        compositionState: result.composition_state,
        composerProgress: null,
        stateVersions: [],
        isComposing: false,
        selectedNodeId: null, // Clear selection for forked session
        // Clear guided state synchronously — the fork is a new session context;
        // the parent's guidedSession must not bleed into the fork's UI before
        // startGuided resolves.  Mirrors selectSession lines 225-228.
        guidedSession: null,
        guidedNextTurn: null,
        guidedTerminal: null,
      }));

      // Fire-and-forget: refresh blob list for the NEW forked session
      useBlobStore.getState().loadBlobs(result.session.id);

      // Auto-start guided mode for the forked session — same pattern as
      // selectSession and createSession.  The backend initialises a fresh
      // GuidedSession on the first GET /guided call (spec §5.2, errata C7).
      // The stale-fetch guard inside startGuided ensures the response is
      // dropped if the user switches away again before it resolves.
      void get().startGuided(result.session.id);
    } catch {
      set({
        isComposing: false,
        composerProgress: null,
        error: "Failed to fork conversation. Please try again.",
      });
    }
  },

  // Guided-mode actions
  async startGuided(sessionId: string) {
    // Capture which session this fetch belongs to before the await.
    // Mirrors the active-session guard in loadComposerProgress (lines 367-372):
    // if the user switches sessions while the request is in flight, the
    // stale response is silently dropped rather than overwriting the newly
    // active session's guided state.
    const requestedSessionId = sessionId;
    try {
      const response = await api.getGuided(sessionId);
      // Stale-fetch guard (Codex #3): drop the response if the active session
      // changed while the request was in flight.
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      // Atomically replace all 4 wire fields — server is authoritative (spec §7.3)
      set({
        guidedSession: response.guided_session,
        guidedNextTurn: response.next_turn,
        guidedTerminal: response.terminal,
        compositionState: response.composition_state,
      });
    } catch {
      // Error path: set error string, leave existing guided state alone.
      // Mirrors selectSession lines 207-209: set error, don't clobber fields
      // that were already loaded. The caller can inspect error to decide whether
      // to surface a retry prompt.
      set({ error: "Failed to load guided session. Please try again." });
    }
  },

  async respondGuided(body: GuidedRespondRequest) {
    const { activeSessionId } = get();
    // Offensive guard — caller must not invoke this without an active session.
    // Per CLAUDE.md: "Proactively detect invalid states and throw meaningful
    // exceptions." Using ?. to silently skip would mask a programmer error.
    if (activeSessionId === null) {
      throw new Error("respondGuided called without active session");
    }
    // Capture the session identity before the await (Codex #4 stale-fetch guard).
    // Mirrors the active-session guard in loadComposerProgress (lines 367-372).
    const requestedSessionId = activeSessionId;
    try {
      const response = await api.respondGuided(activeSessionId, body);
      // Stale-fetch guard (Codex #4): drop the response if the active session
      // changed while the request was in flight.
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      // Atomically replace all 4 wire fields — no optimistic updates (spec §7.3)
      set({
        guidedSession: response.guided_session,
        guidedNextTurn: response.next_turn,
        guidedTerminal: response.terminal,
        compositionState: response.composition_state,
      });
    } catch {
      set({ error: "Failed to submit guided response. Please try again." });
    }
  },

  async exitToFreeform() {
    // Sugar over respondGuided — sets control_signal and nulls all choice fields.
    // All state mutation is handled by respondGuided.
    await get().respondGuided({
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: "exit_to_freeform",
    });
  },

  async loadStateVersions() {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    set({ isLoadingVersions: true });
    try {
      const versions = await api.fetchStateVersions(activeSessionId);
      set({ stateVersions: versions, isLoadingVersions: false });
    } catch {
      // Version history is non-critical -- fail silently
      set({ isLoadingVersions: false });
    }
  },

  async revertToVersion(stateId: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    try {
      // R4-H3: Clear validation BEFORE updating compositionState
      // to prevent a frame where stale validation is visible with the new version
      getExecutionStore().clearValidation();

      const compositionState = await api.revertToVersion(
        activeSessionId,
        stateId,
      );
      // Clear selection — the reverted version may not contain the selected node
      set({ compositionState, selectedNodeId: null });
    } catch {
      set({ error: "Failed to revert to version. Please try again." });
    }
  },

  clearError() {
    set({ error: null });
  },

  selectNode(nodeId: string | null) {
    set({ selectedNodeId: nodeId });
  },

  injectSystemMessage(content: string, stableId?: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    const messageId = stableId ?? `system-${crypto.randomUUID()}`;

    const systemMessage: ChatMessage = {
      id: messageId,
      session_id: activeSessionId,
      role: "system",
      content,
      tool_calls: null,
      created_at: new Date().toISOString(),
    };

    set((state) => {
      // If a stable ID was provided, replace any existing message with
      // that ID instead of appending. This prevents noise accumulation
      // from repeated validation cycles.
      const filtered = stableId
        ? state.messages.filter((m) => m.id !== stableId)
        : state.messages;
      return { messages: [...filtered, systemMessage] };
    });
  },

  reset() {
    clearComposerProgressPollTimer();
    set(initialState);
  },
}));
