// src/stores/sessionStore.ts
import { create } from "zustand";
import type {
  Session,
  ChatMessage,
  CompositionState,
  CompositionStateVersion,
  ComposerPreferences,
  ComposerProgressSnapshot,
  CompositionProposal,
  ApiError,
  ComposerRecoveryError,
  ValidationResult,
} from "@/types/api";
import { isComposerRecoveryError } from "@/types/recovery";
import type {
  GuidedSession,
  TurnPayload,
  TerminalState,
  GuidedRespondRequest,
} from "@/types/guided";
import * as api from "@/api/client";
import {
  COMPOSE_TIMEOUT_ABORT_REASON,
  COMPOSE_TIMEOUT_MS,
  COMPOSE_USER_CANCEL_ABORT_REASON,
} from "@/config/composer";
import { useBlobStore } from "./blobStore";
import { useExecutionStore } from "./executionStore";
import { useInterpretationEventsStore } from "./interpretationEventsStore";
import { usePreferencesStore } from "./preferencesStore";

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
const COMPOSE_CANCELLED_MESSAGE =
  "Composition stopped. You can revise your request and send it again.";

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

function abortReason(signal?: AbortSignal): unknown {
  return signal?.aborted === true ? signal.reason : undefined;
}

function composeAbortMessage(signal?: AbortSignal): string {
  return abortReason(signal) === COMPOSE_USER_CANCEL_ABORT_REASON
    ? COMPOSE_CANCELLED_MESSAGE
    : COMPOSE_TIMEOUT_MESSAGE;
}

function isHttpConflict(err: unknown): boolean {
  if (typeof err !== "object" || err === null) {
    return false;
  }
  return (err as { status?: unknown }).status === 409;
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

// Live-message polling — runs while a send is inflight so the store can sync
// against assistant rows as the backend persists them. ChatPanel intentionally
// keeps incomplete agent turns hidden (see components/chat/turns.ts) and uses
// composerProgress as the live visible affordance until the final assistant
// text lands.
const INFLIGHT_MESSAGES_POLL_INTERVAL_MS = 1500;
let inflightMessagesPollTimer: ReturnType<typeof setInterval> | null = null;
let inflightMessagesPollSessionId: string | null = null;

function clearInflightMessagesPollTimer(): void {
  if (inflightMessagesPollTimer !== null) {
    clearInterval(inflightMessagesPollTimer);
    inflightMessagesPollTimer = null;
  }
  inflightMessagesPollSessionId = null;
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

/**
 * Pull any interpretation events a compose action created into the
 * interpretationEventsStore.
 *
 * INVARIANT: every compose entry point that can mint interpretive decisions
 * (sendMessage, recompose, acceptCompositionProposal) MUST call this after a
 * successful turn. Interpretation events (invented_source / llm_prompt_template
 * / llm_model_choice / pipeline_decision) live in their own store, populated
 * only by listInterpretationEvents. In freeform mode there is no other trigger
 * to surface the inline review widgets — and their sign-off buttons — while the
 * run-gate blocks execution on unresolved pending rows. Guided mode delivers
 * review as a guided turn and the tutorial refreshes explicitly; this is the
 * freeform-surface equivalent. Only selectSession (session load / deep-link)
 * otherwise refreshes, so without this a mid-session compose deadlocks the user.
 *
 * Fire-and-forget and idempotent (the store keys by session_id and reconciles
 * resolved events across surfaces), so it is safe to call on any compose
 * completion. A new compose entry point that omits this call reintroduces the
 * freeform deadlock.
 */
function refreshInterpretationEventsForSession(sessionId: string): void {
  void useInterpretationEventsStore.getState().refreshAll(sessionId);
}

function mergeCompositionProposals(
  existing: CompositionProposal[],
  incoming: CompositionProposal[],
): CompositionProposal[] {
  if (incoming.length === 0) {
    return existing;
  }
  const byId = new Map(existing.map((proposal) => [proposal.id, proposal]));
  for (const proposal of incoming) {
    byId.set(proposal.id, proposal);
  }
  return Array.from(byId.values());
}

// Resetting guided-mode state landed in five places: initialState plus
// the four navigation actions that switch session context (createSession,
// archiveSession's active-session branch, selectSession, forkFromMessage).
// Phase A slice 4 grew this from three fields to four (added
// guidedChatPending); a future per-step opener field would grow it again.
// Pulling the literal into a single helper means adding a future field
// updates one place — TypeScript exhaustiveness over the Pick<> return
// then forces every call site through the type system instead of through
// grep-and-edit discipline.  See elspeth-obs-01f85f94b5.
function clearedGuidedState(): Pick<
  SessionState,
  | "guidedSession"
  | "guidedNextTurn"
  | "guidedTerminal"
  | "guidedChatPending"
  | "guidedResponsePending"
> {
  return {
    guidedSession: null,
    guidedNextTurn: null,
    guidedTerminal: null,
    guidedChatPending: false,
    guidedResponsePending: false,
  };
}

function clearedRecoveryState(): Pick<
  SessionState,
  "recoveryError" | "recoveryStartedCompositionVersion"
> {
  return {
    recoveryError: null,
    recoveryStartedCompositionVersion: null,
  };
}

interface ApplyRecoveredStateOptions {
  confirmed?: boolean;
}

interface ApplyRecoveredStateResult {
  applied: boolean;
  needsConfirmation: boolean;
}

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: ChatMessage[];
  compositionState: CompositionState | null;
  compositionProposals: CompositionProposal[];
  composerPreferences: ComposerPreferences | null;
  staleProposalIds: string[];
  proposalActionPendingIds: string[];
  composerProgress: ComposerProgressSnapshot | null;
  isComposing: boolean;
  stateVersions: CompositionStateVersion[];
  error: string | null;
  /**
   * Optional structured detail rows rendered as bullet points beneath
   * `error` in the error banner. Populated when an ApiError carries
   * structured `validation_errors` — currently set by `acceptProposal`
   * on a 422 `proposal_validation_failed` response. Cleared whenever
   * `error` is cleared.
   */
  errorDetails: string[] | null;
  recoveryError: ComposerRecoveryError | null;
  recoveryStartedCompositionVersion: number | null;

  // Shared selection state for GraphView component focus.
  selectedNodeId: string | null;
  selectNode: (nodeId: string | null) => void;

  loadSessions: () => Promise<void>;
  createSession: () => Promise<void>;
  selectSession: (id: string) => Promise<void>;
  renameSession: (id: string, title: string) => Promise<void>;
  archiveSession: (id: string) => Promise<void>;
  sendMessage: (content: string, signal?: AbortSignal) => Promise<void>;
  loadCompositionProposals: (sessionId?: string) => Promise<void>;
  acceptProposal: (proposalId: string) => Promise<void>;
  rejectProposal: (proposalId: string) => Promise<void>;
  loadComposerProgress: (sessionId?: string) => Promise<void>;
  startComposerProgressPolling: (sessionId: string) => void;
  stopComposerProgressPolling: (sessionId?: string) => void;
  loadInflightMessages: (sessionId: string) => Promise<void>;
  startInflightMessagesPolling: (sessionId: string) => void;
  stopInflightMessagesPolling: (sessionId?: string) => void;
  sendValidationFeedback: (result: ValidationResult) => Promise<void>;
  retryMessage: (messageId: string, signal?: AbortSignal) => Promise<void>;
  forkFromMessage: (messageId: string, newContent: string) => Promise<void>;
  openRecoveryFromError: (
    error: ApiError,
    recoveryStartedCompositionVersion: number | null,
  ) => boolean;
  applyRecoveredState: (
    options?: ApplyRecoveredStateOptions,
  ) => ApplyRecoveredStateResult;
  discardRecovery: () => void;
  loadStateVersions: () => Promise<void>;
  isLoadingVersions: boolean;
  revertToVersion: (stateId: string) => Promise<void>;

  // Guided-mode protocol state — all three are null when not in a guided session
  guidedSession: GuidedSession | null;
  guidedNextTurn: TurnPayload | null;
  guidedTerminal: TerminalState | null;
  // Per-step chat (Phase A slice 5).  The history itself lives on
  // `guidedSession.chat_history` (server-authoritative); only the in-flight
  // pending flag is local state.  Slice 4 carried an in-memory
  // guidedChatHistory array; slice 5 replaced it with the wire field.
  guidedChatPending: boolean;
  // In-flight wizard answer flag. Distinct from guidedChatPending: this blocks
  // turn-answer buttons while the server-authoritative state machine advances.
  guidedResponsePending: boolean;
  // Guided-mode actions
  startGuided: (sessionId: string) => Promise<void>;
  respondGuided: (body: GuidedRespondRequest) => Promise<void>;
  reenterGuided: () => Promise<void>;
  // Unified entry point bound by the "Switch to guided" button in ChatPanel's
  // freeform body.  Branches on the current guidedSession terminal:
  //   * no session or non-terminal session => startGuided (lazy-create / GET)
  //   * terminal.kind === "exited_to_freeform" => reenterGuided
  // The button stays a single affordance with one label regardless of branch.
  enterGuided: () => Promise<void>;
  chatGuided: (message: string) => Promise<void>;
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
  compositionProposals: [] as CompositionProposal[],
  composerPreferences: null as ComposerPreferences | null,
  staleProposalIds: [] as string[],
  proposalActionPendingIds: [] as string[],
  composerProgress: null as ComposerProgressSnapshot | null,
  isComposing: false,
  stateVersions: [] as CompositionStateVersion[],
  isLoadingVersions: false,
  error: null as string | null,
  errorDetails: null as string[] | null,
  selectedNodeId: null as string | null,
  ...clearedGuidedState(),
  ...clearedRecoveryState(),
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
    // Phase 1B Panel M1: the session-create and prefs-resolve calls live in
    // SEPARATE try blocks so a preferences-bootstrap failure does NOT mask
    // a successful session creation. The earlier single-try shape attributed
    // a prefs-bootstrap rejection to "Failed to create session", which lied
    // to the user: the session was created and is now their active session,
    // but they were told it wasn't. We now surface a prefs-specific error
    // for the second branch so the message matches the actual failure.
    let session;
    try {
      session = await api.createSession();
    } catch {
      set({ error: "Failed to create session. Please try again." });
      return;
    }
    clearComposerProgressPollTimer();
    set((state) => ({
      sessions: [session, ...state.sessions],
      activeSessionId: session.id,
      messages: [],
      compositionState: null,
      compositionProposals: [],
      composerPreferences: null,
      staleProposalIds: [],
      proposalActionPendingIds: [],
      composerProgress: null,
      stateVersions: [],
      error: null,
      selectedNodeId: null, // Clear selection for new session
      ...clearedGuidedState(),
      ...clearedRecoveryState(),
    }));
    // Phase 1B — honour the account-level default-mode preference.
    // resolveDefaultMode() awaits the preferences bootstrap if it
    // hasn't completed yet (Ctrl+N race: user hits "new session" before
    // App.tsx's bootstrap effect has resolved). Guided users enter via
    // the same enterGuided() the manual "Switch to guided" button uses.
    // A failure here surfaces a *prefs-specific* error; the session is
    // already live and usable in freeform, so the message reflects only
    // the secondary failure (couldn't honour your guided default).
    try {
      const mode = await usePreferencesStore.getState().resolveDefaultMode();
      if (mode === "guided") {
        await get().enterGuided();
      }
    } catch {
      set({
        error:
          "Session created, but couldn't apply your default mode. " +
          "You're in freeform; switch to guided from the header if you want.",
      });
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
                compositionProposals: [],
                composerPreferences: null,
                staleProposalIds: [],
                proposalActionPendingIds: [],
                composerProgress: null,
                stateVersions: [],
                isComposing: false,
                selectedNodeId: null,
                ...clearedGuidedState(),
                ...clearedRecoveryState(),
              }
            : {}),
        };
      });
    } catch (err) {
      // Preserve the original error for callers that want to surface
      // ``err.message`` inline (HeaderSessionSwitcher renders an inline
      // role="alert" co-located with the trigger).  We also set the
      // composer-level fallback so the global error region stays useful
      // if no inline handler is wired.  Re-throwing lets the component
      // catch keep the diagnostic detail without losing the fallback.
      set({ error: "Failed to archive session. Please try again." });
      throw err;
    }
  },

  async renameSession(id: string, title: string) {
    const trimmed = title.trim();
    if (!trimmed) return;
    try {
      const session = await api.renameSession(id, trimmed);
      set((state) => ({
        sessions: state.sessions.map((existing) =>
          existing.id === id ? session : existing,
        ),
        error: null,
      }));
    } catch (err) {
      // Same pattern as ``archiveSession`` above — re-raise so an inline
      // handler can preserve ``err.message`` while the global error
      // region still receives a friendly fallback.
      set({ error: "Failed to rename session. Please try again." });
      throw err;
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
      compositionProposals: [],
      composerPreferences: null,
      staleProposalIds: [],
      proposalActionPendingIds: [],
      composerProgress: null,
      stateVersions: [],
      isComposing: false,
      error: null,
      selectedNodeId: null, // Clear selection when switching sessions
      ...clearedGuidedState(),
      ...clearedRecoveryState(),
    });

    try {
      const [
        messages,
        compositionState,
        compositionProposals,
        composerPreferences,
      ] = await Promise.all([
        api.fetchMessages(id),
        api.fetchCompositionState(id),
        api.fetchCompositionProposals(id),
        api.fetchComposerPreferences(id),
      ]);
      set({
        messages,
        compositionState,
        compositionProposals: compositionProposals ?? [],
        composerPreferences: composerPreferences ?? null,
      });

      // Default-freeform: do NOT auto-fetch /guided on session select.
      // The freeform surface renders by default; the "Switch to guided"
      // button in ChatPanel's freeform body calls enterGuided() to fetch
      // (and lazy-persist) a guided session on user request.  Sessions that
      // already have a persisted guided_session require the user to click
      // the button to surface it — symmetric with "Exit to freeform".

      // Fire-and-forget: refresh blob list for the newly selected session
      useBlobStore.getState().loadBlobs(id);

      // Phase 5b Task 3 — rehydrate interpretation-event projection for
      // the newly selected session.  Fire-and-forget: a failure on this
      // route must not block session selection (the readiness panel will
      // simply show stale/empty counts until the next refresh).  The
      // store keys by session_id, so navigating back to a previously
      // visited session preserves its prior pending events without a
      // reset.  Tests #6 and #7 in interpretationEventsStore.test.ts
      // cover the session-load and session-change behaviours.
      void useInterpretationEventsStore.getState().refreshAll(id);
    } catch (err) {
      if ((err as ApiError).status === 404 && get().activeSessionId === id) {
        set({
          activeSessionId: null,
          messages: [],
          compositionState: null,
          compositionProposals: [],
          composerPreferences: null,
          staleProposalIds: [],
          proposalActionPendingIds: [],
          composerProgress: null,
          stateVersions: [],
          isComposing: false,
          error: null,
          selectedNodeId: null,
          ...clearedGuidedState(),
          ...clearedRecoveryState(),
        });
        return;
      }
      if (get().activeSessionId === id) {
        set({ error: "Failed to load session. Please refresh the page." });
      }
    }
  },

  async sendMessage(content: string, signal?: AbortSignal) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;
    const recoveryStartedCompositionVersion =
      get().compositionState?.version ?? null;

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
    get().startInflightMessagesPolling(activeSessionId);

    try {
      const stateId = get().compositionState?.id;
      const result = await api.sendMessage(activeSessionId, content, stateId, signal);
      // Sync the chat panel against the durable DB state before applying the
      // POST's metadata. After this await, get().messages contains every
      // assistant row the compose loop persisted (and the canonical user row
      // — the optimistic local-* version is gone). The set() block below
      // then only needs to update derived state (compositionState, proposals,
      // isComposing) without re-appending the final assistant message that
      // the poll has already pulled in.
      await get().loadInflightMessages(activeSessionId);
      const { message, state } = result;
      const proposals = result.proposals ?? [];
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

        // After loadInflightMessages the message list reflects the canonical
        // DB state — the optimistic local-* row has been dropped and every
        // assistant row the compose loop persisted (including the final
        // ``message`` the POST returned) is present. We still defensively
        // backfill in case the poll request failed: clear the optimistic's
        // pending status and append the final message only when neither is
        // already represented in s.messages (dedup by id).
        const seen = new Set(s.messages.map((m) => m.id));
        const repaired = s.messages.map((existing) =>
          existing.id === optimisticMessage.id
            ? { ...existing, local_status: undefined, local_error: undefined }
            : existing,
        );
        const finalMessages = seen.has(message.id)
          ? repaired
          : repaired.concat(message);

        return {
          messages: finalMessages,
          compositionState: newState,
          compositionProposals: mergeCompositionProposals(
            s.compositionProposals,
            proposals,
          ),
          isComposing: false,
          ...(nodeStillExists ? {} : { selectedNodeId: null }),
        };
      });

      // Fire-and-forget: refresh blob list in case the LLM created files
      useBlobStore.getState().loadBlobs(activeSessionId);
      // Fire-and-forget: refresh the session list. The backend send_message
      // route may have auto-titled this session (first-message-of-session
      // generates a 3-6 word title via a side LLM call and writes it
      // before send_message returns). Refreshing keeps the session switcher
      // title in step with the DB without a manual reload.
      void get().loadSessions();
      // Surface any interpretation reviews this compose turn created (see the
      // invariant on refreshInterpretationEventsForSession). Without this the
      // freeform inline review widgets never render mid-session.
      refreshInterpretationEventsForSession(activeSessionId);
    } catch (err) {
      let errorMessage: string;
      // Client-side abort (the useComposer COMPOSE_TIMEOUT_MS guard or any
      // user-supplied signal) fires a DOMException with name 'AbortError',
      // not a structured ApiError — the apiErr.detail fallback below would
      // otherwise mask it as a generic send failure.
      if (isAbortError(err)) {
        errorMessage = composeAbortMessage(signal);
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
      const apiErr = err as ApiError;
      const recoveryPatch = isComposerRecoveryError(apiErr)
        ? {
            recoveryError: apiErr,
            recoveryStartedCompositionVersion,
          }
        : {};
      set((state) => ({
        isComposing: false,
        error: errorMessage,
        messages: state.messages.map((existing) =>
          existing.id === optimisticMessage.id
            ? { ...existing, local_status: "failed", local_error: errorMessage }
            : existing,
        ),
        ...recoveryPatch,
      }));
    } finally {
      get().stopInflightMessagesPolling(activeSessionId);
      get().stopComposerProgressPolling(activeSessionId);
      await get().loadComposerProgress(activeSessionId);
    }
  },

  async loadCompositionProposals(sessionId?: string) {
    const targetSessionId = sessionId ?? get().activeSessionId;
    if (!targetSessionId) return;

    try {
      const proposals = await api.fetchCompositionProposals(targetSessionId);
      if (get().activeSessionId !== targetSessionId) {
        return;
      }
      set({ compositionProposals: proposals ?? [] });
    } catch {
      set({ error: "Failed to load composition proposals. Please try again." });
    }
  },

  async acceptProposal(proposalId: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) {
      throw new Error("acceptProposal called without active session");
    }

    set((state) => ({
      error: null,
      proposalActionPendingIds: Array.from(
        new Set([...state.proposalActionPendingIds, proposalId]),
      ),
      staleProposalIds: state.staleProposalIds.filter((id) => id !== proposalId),
    }));

    try {
      const proposal = await api.acceptCompositionProposal(
        activeSessionId,
        proposalId,
      );
      const [compositionState, proposals] = await Promise.all([
        api.fetchCompositionState(activeSessionId),
        api.fetchCompositionProposals(activeSessionId),
      ]);
      if (get().activeSessionId !== activeSessionId) {
        return;
      }
      getExecutionStore().clearValidation();
      set({
        compositionState,
        compositionProposals: mergeCompositionProposals(
          proposals ?? [],
          [proposal],
        ),
      });
      // Surface any interpretation reviews accepting this proposal created
      // (see the invariant on refreshInterpretationEventsForSession).
      refreshInterpretationEventsForSession(activeSessionId);
    } catch (err) {
      if (isHttpConflict(err)) {
        await get().loadCompositionProposals(activeSessionId);
        if (get().activeSessionId === activeSessionId) {
          set((state) => ({
            staleProposalIds: Array.from(
              new Set([...state.staleProposalIds, proposalId]),
            ),
          }));
        }
      } else {
        const apiErr = err as ApiError;
        // proposal_validation_failed (HTTP 422) carries structured
        // validation entries plus a server-side auto-reject side effect.
        // Reload proposals so the now-rejected one drops off the pending
        // banner; surface the entries as bullet points in the error
        // banner via `errorDetails`. Without this, the toast renders the
        // Pydantic-flattened message as one wall-of-text line and the
        // banner keeps showing the proposal as actionable until refresh.
        const isProposalValidationFailure =
          apiErr.error_type === "proposal_validation_failed";
        if (isProposalValidationFailure) {
          await get().loadCompositionProposals(activeSessionId);
        }
        if (get().activeSessionId === activeSessionId) {
          const validationEntries = apiErr.validation_errors as
            | Array<{ message?: string }>
            | undefined;
          const errorDetails =
            isProposalValidationFailure && Array.isArray(validationEntries)
              ? validationEntries
                  .map((entry) => entry?.message)
                  .filter((msg): msg is string => typeof msg === "string" && msg.length > 0)
              : null;
          set({
            error: apiErr.detail ?? "Failed to accept proposal. Please try again.",
            errorDetails: errorDetails && errorDetails.length > 0 ? errorDetails : null,
          });
        }
      }
    } finally {
      if (get().activeSessionId === activeSessionId) {
        set((state) => ({
          proposalActionPendingIds: state.proposalActionPendingIds.filter(
            (id) => id !== proposalId,
          ),
        }));
      }
    }
  },

  async rejectProposal(proposalId: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) {
      throw new Error("rejectProposal called without active session");
    }

    set((state) => ({
      error: null,
      proposalActionPendingIds: Array.from(
        new Set([...state.proposalActionPendingIds, proposalId]),
      ),
      staleProposalIds: state.staleProposalIds.filter((id) => id !== proposalId),
    }));

    try {
      const proposal = await api.rejectCompositionProposal(
        activeSessionId,
        proposalId,
      );
      const proposals = await api.fetchCompositionProposals(activeSessionId);
      if (get().activeSessionId !== activeSessionId) {
        return;
      }
      set({
        compositionProposals: mergeCompositionProposals(
          proposals ?? [],
          [proposal],
        ),
      });
    } catch (err) {
      if (isHttpConflict(err)) {
        await get().loadCompositionProposals(activeSessionId);
        if (get().activeSessionId === activeSessionId) {
          set((state) => ({
            staleProposalIds: Array.from(
              new Set([...state.staleProposalIds, proposalId]),
            ),
          }));
        }
      } else {
        const apiErr = err as ApiError;
        set({
          error: apiErr.detail ?? "Failed to reject proposal. Please try again.",
        });
      }
    } finally {
      if (get().activeSessionId === activeSessionId) {
        set((state) => ({
          proposalActionPendingIds: state.proposalActionPendingIds.filter(
            (id) => id !== proposalId,
          ),
        }));
      }
    }
  },

  async loadComposerProgress(sessionId?: string) {
    const targetSessionId = sessionId ?? get().activeSessionId;
    if (!targetSessionId) return;

    try {
      const progress = await api.fetchComposerProgress(targetSessionId);
      const current = get();
      if (current.activeSessionId !== targetSessionId) {
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
  },

  async loadInflightMessages(sessionId: string) {
    // Refresh the chat messages from the server so newly-persisted assistant
    // rows from the inflight compose loop are visible immediately. The
    // optimistic local-* user message is preserved when its canonical
    // counterpart hasn't appeared in the fresh list yet (race between the
    // first poll and the route's user-message persist); once the canonical
    // user row arrives, the optimistic one is dropped.
    try {
      const fresh = await api.fetchMessages(sessionId);
      if (get().activeSessionId !== sessionId) return;
      // If polling has been stopped (or rebound to a different session), drop
      // this stale response on the floor.
      if (
        inflightMessagesPollSessionId !== null &&
        inflightMessagesPollSessionId !== sessionId
      ) {
        return;
      }
      set((s) => {
        if (s.activeSessionId !== sessionId) return s;
        const localOptimistic = s.messages.filter((m) =>
          m.id.startsWith("local-"),
        );
        const survivors = localOptimistic.filter(
          (local) =>
            !fresh.some(
              (f) => f.role === local.role && f.content === local.content,
            ),
        );
        return { messages: [...fresh, ...survivors] };
      });
    } catch {
      // Inflight polling is advisory — failures keep the existing UI state
      // until the next poll or the POST completion handler refreshes it.
    }
  },

  startInflightMessagesPolling(sessionId: string) {
    clearInflightMessagesPollTimer();
    inflightMessagesPollSessionId = sessionId;
    inflightMessagesPollTimer = setInterval(() => {
      if (inflightMessagesPollSessionId !== sessionId) return;
      void useSessionStore.getState().loadInflightMessages(sessionId);
    }, INFLIGHT_MESSAGES_POLL_INTERVAL_MS);
  },

  stopInflightMessagesPolling(sessionId?: string) {
    if (
      sessionId !== undefined &&
      inflightMessagesPollSessionId !== null &&
      inflightMessagesPollSessionId !== sessionId
    ) {
      return;
    }
    clearInflightMessagesPollTimer();
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
    const timer = setTimeout(
      () => controller.abort(COMPOSE_TIMEOUT_ABORT_REASON),
      COMPOSE_TIMEOUT_MS,
    );
    try {
      await get().sendMessage(content, controller.signal);
    } finally {
      clearTimeout(timer);
    }
  },

  async retryMessage(messageId: string, signal?: AbortSignal) {
    const { activeSessionId, messages } = get();
    if (!activeSessionId) return;
    const recoveryStartedCompositionVersion =
      get().compositionState?.version ?? null;

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
    get().startInflightMessagesPolling(activeSessionId);

    try {
      // Use recompose (not sendMessage) — the user message is already
      // persisted from the original send. Calling sendMessage again
      // would insert a duplicate user message.
      const result = await api.recompose(activeSessionId, signal);
      // Sync the chat panel against the DB state (see sendMessage for
      // rationale).
      await get().loadInflightMessages(activeSessionId);
      const { message: assistantMessage, state } = result;
      const proposals = result.proposals ?? [];
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

        // Polling has loaded the canonical messages list. Defensive backfill
        // mirrors the sendMessage success branch: clear the retried message's
        // pending status, and only append the recomposed assistant message
        // if it isn't already represented (dedup by id).
        const seen = new Set(s.messages.map((m) => m.id));
        const repaired = s.messages.map((existing) =>
          existing.id === messageId
            ? { ...existing, local_status: undefined, local_error: undefined }
            : existing,
        );
        const finalMessages = seen.has(assistantMessage.id)
          ? repaired
          : repaired.concat(assistantMessage);

        return {
          messages: finalMessages,
          compositionState: newState,
          compositionProposals: mergeCompositionProposals(
            s.compositionProposals,
            proposals,
          ),
          isComposing: false,
          ...(nodeStillExists ? {} : { selectedNodeId: null }),
        };
      });

      // Fire-and-forget: refresh blob list in case the LLM created files
      useBlobStore.getState().loadBlobs(activeSessionId);
      // Surface any interpretation reviews this recompose created (see the
      // invariant on refreshInterpretationEventsForSession).
      refreshInterpretationEventsForSession(activeSessionId);
    } catch (err) {
      let errorMessage: string;
      if (isAbortError(err)) {
        errorMessage = composeAbortMessage(signal);
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
      const apiErr = err as ApiError;
      const recoveryPatch = isComposerRecoveryError(apiErr)
        ? {
            recoveryError: apiErr,
            recoveryStartedCompositionVersion,
          }
        : {};

      set((state) => ({
        isComposing: false,
        error: errorMessage,
        messages: state.messages.map((existing) =>
          existing.id === messageId
            ? { ...existing, local_status: "failed", local_error: errorMessage }
            : existing,
        ),
        ...recoveryPatch,
      }));
    } finally {
      get().stopInflightMessagesPolling(activeSessionId);
      get().stopComposerProgressPolling(activeSessionId);
      await get().loadComposerProgress(activeSessionId);
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
        compositionProposals: [],
        composerPreferences: null,
        staleProposalIds: [],
        proposalActionPendingIds: [],
        composerProgress: null,
        stateVersions: [],
        isComposing: false,
        selectedNodeId: null, // Clear selection for forked session
        // Clear guided state synchronously — the fork is a new session context;
        // the parent's guidedSession must not bleed into the fork's UI before
        // startGuided resolves.  Mirrors selectSession.
        ...clearedGuidedState(),
        ...clearedRecoveryState(),
      }));

      // Fire-and-forget: refresh blob list for the NEW forked session
      useBlobStore.getState().loadBlobs(result.session.id);

      // Default-freeform: forked sessions surface freeform (same contract as
      // selectSession / createSession).  Forks are a new conversation context;
      // the user activates guided explicitly via the "Switch to guided"
      // button in ChatPanel if they want the wizard surface back.
    } catch {
      set({
        isComposing: false,
        composerProgress: null,
        error: "Failed to fork conversation. Please try again.",
      });
    }
  },

  openRecoveryFromError(error, recoveryStartedCompositionVersion) {
    if (!isComposerRecoveryError(error)) {
      return false;
    }
    set({
      recoveryError: error,
      recoveryStartedCompositionVersion,
    });
    return true;
  },

  applyRecoveredState(options) {
    const { recoveryError, recoveryStartedCompositionVersion, compositionState } =
      get();
    if (recoveryError === null) {
      return { applied: false, needsConfirmation: false };
    }

    const currentVersion = compositionState?.version ?? null;
    if (
      options?.confirmed !== true &&
      currentVersion !== recoveryStartedCompositionVersion
    ) {
      return { applied: false, needsConfirmation: true };
    }

    const recoveredState = recoveryError.partial_state;
    getExecutionStore().clearValidation();
    set((state) => {
      const nodeStillExists =
        !state.selectedNodeId ||
        recoveredState.nodes.some((node) => node.id === state.selectedNodeId);
      return {
        compositionState: recoveredState,
        ...(nodeStillExists ? {} : { selectedNodeId: null }),
        ...clearedRecoveryState(),
      };
    });
    return { applied: true, needsConfirmation: false };
  },

  discardRecovery() {
    set(clearedRecoveryState());
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
    set({ guidedResponsePending: true });
    try {
      const response = await api.respondGuided(activeSessionId, body);
      // Stale-fetch guard (Codex #4): drop the response if the active session
      // changed while the request was in flight.
      if (get().activeSessionId !== requestedSessionId) {
        set({ guidedResponsePending: false });
        return;
      }
      // Atomically replace all 4 wire fields — no optimistic updates (spec §7.3)
      set({
        guidedSession: response.guided_session,
        guidedNextTurn: response.next_turn,
        guidedTerminal: response.terminal,
        compositionState: response.composition_state,
        guidedResponsePending: false,
      });
    } catch {
      set({
        error: "Failed to submit guided response. Please try again.",
        guidedResponsePending: false,
      });
    }
  },

  async reenterGuided() {
    const { activeSessionId } = get();
    if (activeSessionId === null) {
      throw new Error("reenterGuided called without active session");
    }
    const requestedSessionId = activeSessionId;
    try {
      const response = await api.reenterGuided(activeSessionId);
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      set({
        guidedSession: response.guided_session,
        guidedNextTurn: response.next_turn,
        guidedTerminal: response.terminal,
        compositionState: response.composition_state,
        error: null,
      });
    } catch {
      set({ error: "Failed to re-enter guided mode. Please try again." });
    }
  },

  async enterGuided() {
    // Unified "Switch to guided" entry point.  Default-freeform sessions
    // (no fetched guidedSession) take the startGuided path — GET /guided
    // builds and persists the initial wizard turn server-side.  Sessions
    // that were previously in guided and exited via the operator's "Exit
    // to freeform" button reach a terminal of kind === "exited_to_freeform";
    // those go through reenterGuided instead, because startGuided would
    // return the terminal state and leave the discriminator on freeform.
    //
    // Other terminal kinds (completed, solver-exhausted, protocol-violation)
    // are NOT re-entrable by design — see the reenter route handler at
    // routes.py:5921 for the closed-list policy.  We still call startGuided
    // for those; the discriminator at ChatPanel handles each terminal kind
    // appropriately (completed → CompletionSummary; others → freeform).
    const { activeSessionId, guidedSession } = get();
    if (activeSessionId === null) {
      throw new Error("enterGuided called without active session");
    }
    if (guidedSession?.terminal?.kind === "exited_to_freeform") {
      await get().reenterGuided();
      return;
    }
    await get().startGuided(activeSessionId);
  },

  async chatGuided(message: string) {
    const { activeSessionId, guidedSession } = get();
    // Offensive guards: caller must not invoke without an active session
    // or before guidedSession is loaded.  Per CLAUDE.md "proactively detect
    // invalid states and throw meaningful exceptions" — silent ?. would
    // mask a UI bug (ChatInput rendered with no guided session attached).
    if (activeSessionId === null) {
      throw new Error("chatGuided called without active session");
    }
    if (guidedSession === null) {
      throw new Error("chatGuided called before guidedSession loaded");
    }
    // Capture session + step identity before the await (stale-fetch guard
    // mirroring respondGuided / startGuided).  If the user switches
    // session or the wizard advances mid-flight, the response is dropped.
    const requestedSessionId = activeSessionId;
    const requestedStep = guidedSession.step;

    // Slice 5: chat history is server-authoritative — no optimistic local
    // append.  The route handler appends both user + assistant turns to
    // `guidedSession.chat_history` and returns the updated session.  Only
    // `guidedChatPending` is local: it blocks rapid double-submits while
    // the round-trip is in flight.  Slice 4's optimistic-append pattern
    // produced visible drift if the server replied with a slightly
    // different ts_iso / seq than the client guessed.
    set({ guidedChatPending: true });

    try {
      const response = await api.chatGuided(activeSessionId, {
        message,
        step_index: requestedStep,
      });
      // Stale-fetch guard: drop the response if session changed mid-flight.
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      set({
        guidedSession: response.guided_session,
        guidedNextTurn: response.next_turn ?? get().guidedNextTurn,
        guidedTerminal: response.terminal ?? get().guidedTerminal,
        compositionState: response.composition_state ?? get().compositionState,
        guidedChatPending: false,
      });
    } catch {
      // HTTP-layer failure (network, 4xx/5xx).  Distinct from the
      // backend's synthetic-message path, which returns 200 with an
      // unavailable assistant message AND appends both turns to
      // chat_history — that path completes the optimistic write
      // server-side, so even a synthetic reply round-trips through this
      // success branch.  This catch fires only when the request itself
      // failed (no response shape at all).
      set({
        error: "Failed to send chat message. Please try again.",
        guidedChatPending: false,
      });
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
    set({ error: null, errorDetails: null });
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
