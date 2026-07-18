// src/stores/sessionStore.ts
import { create } from "zustand";
import type {
  Session,
  ChatMessage,
  CompositionState,
  CompositionStateVersion,
  ComposerPreferences,
  ComposerProgressPhase,
  ComposerProgressSnapshot,
  CompositionProposal,
  ApiError,
  ComposerRecoveryError,
} from "@/types/api";
import { isComposerRecoveryError } from "@/types/recovery";
import type {
  GetGuidedResponse,
  GuidedSession,
  TurnPayload,
  TerminalState,
  GuidedRespondAction,
  GuidedProposalReviewState,
  GuidedProposalRetryAction,
  GuidedRespondRequest,
  GuidedRespondResponse,
} from "@/types/guided";
import * as api from "@/api/client";
import {
  COMPOSE_TIMEOUT_ABORT_REASON,
  COMPOSE_USER_CANCEL_ABORT_REASON,
} from "@/config/composer";
import { useBlobStore } from "./blobStore";
import { useExecutionStore } from "./executionStore";
import { useInterpretationEventsStore } from "./interpretationEventsStore";
import { usePreferencesStore } from "./preferencesStore";
import {
  acquireGuidedRetry,
  clearAllGuidedRetries,
  clearGuidedRetry,
  clearGuidedRetriesForSession,
  isAmbiguousGuidedRetryFailure,
} from "./guidedOperationRetry";

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

function isComposeAbort(err: unknown): boolean {
  // abort() with NO argument rejects the fetch with a DOMException named
  // 'AbortError' — but useComposer aborts with a bare-string reason
  // (compose_timeout / compose_user_cancel), and per WHATWG semantics the
  // fetch then rejects with that RAW string. Classify on the rejection
  // value as well as the structural shape (elspeth-475647c47a).
  return (
    isAbortError(err) ||
    err === COMPOSE_TIMEOUT_ABORT_REASON ||
    err === COMPOSE_USER_CANCEL_ABORT_REASON
  );
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

function proposalReviewForTurn(
  turn: TurnPayload | null,
): GuidedProposalReviewState | null {
  if (turn?.type !== "propose_pipeline") return null;
  return {
    status: "active",
    proposal_id: turn.payload.proposal_id,
    draft_hash: turn.payload.draft_hash,
  };
}

function proposalRetryActionForBody(
  body: GuidedRespondAction,
): GuidedProposalRetryAction | null {
  if (body.proposal_id === null) return null;
  if (body.chosen !== null) return { kind: "accept" };
  if (body.control_signal === "reject") return { kind: "reject" };
  if (body.edit_target !== null) {
    return { kind: "revise", edit_target: body.edit_target };
  }
  return null;
}

let composerProgressPollTimer: ReturnType<typeof setInterval> | null = null;
let composerProgressPollSessionId: string | null = null;
// True once THIS poll session (the span between startComposerProgressPolling
// and stopComposerProgressPolling) has observed a non-terminal snapshot.
// Guards against the stale-terminal-flash race: the immediate poll fired by
// startComposerProgressPolling can win the race against the POST's own
// "starting" publish and return the PRIOR turn's terminal snapshot
// (complete/failed/cancelled) still sitting in the registry. Surfacing that
// stale snapshot at the START of a fresh compose flashed the tutorial step-2
// indicator's LAST substep as current before dropping back to the first —
// exactly the backward-jump class the calling_model/using_tools remap was
// meant to prevent (elspeth-a8eeebb3aa review follow-up).
let composerProgressPollSeenNonTerminal = false;
// Ownership generations for the module-global pollers: each start* bumps
// its counter and returns it; stop* called with a stale generation no-ops.
// Prevents an aborted turn's delayed teardown (its settle wait yields the
// loop) from stopping the pollers a newer same-session turn now owns.
let composerProgressPollGeneration = 0;
let inflightMessagesPollGeneration = 0;
const TERMINAL_COMPOSER_PROGRESS_PHASES = new Set<ComposerProgressPhase>([
  "complete",
  "failed",
  "cancelled",
]);

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
 * Idempotent (the store keys by session_id and reconciles resolved events
 * across surfaces), so it is safe to call on any compose completion. A new
 * compose entry point that omits this call reintroduces the freeform deadlock.
 * Client-side aborts reach it through resyncAfterAbortedComposeTurn — a
 * cancelled turn can mint review cards before the cancel lands, and those
 * must surface too (elspeth-06a23adfcc).
 *
 * Returns the underlying refresh promise. The freeform callers fire it and
 * forget (each `void`s the result); guided `respondGuided` (P3.6/D12) must
 * `await` it so backend-surfaced pending cards land in the store before the
 * guided submit re-enables.
 */
async function refreshInterpretationEventsForSession(
  sessionId: string,
): Promise<void> {
  await useInterpretationEventsStore.getState().refreshAll(sessionId);
}

// Settle-wait pacing for resyncAfterAbortedComposeTurn. Deliberately NO
// wall-clock budget: synchronous tools are cancel-safe by running to
// completion (tool_batch.py — never wrapped in asyncio.wait_for), so the
// shielded window is unbounded by design and any time budget here would
// reintroduce the reconciliation race past its edge. The wait is bounded
// SEMANTICALLY instead — see waitForCancelledComposeToSettle.
const ABORT_RESYNC_SETTLE_POLL_MS = 500;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Block until the server side of a cancelled compose turn has settled.
 *
 * The client's abort rejects the fetch immediately, but the server is
 * still working: the aborted route may be queued on the per-session
 * compose lock with nothing published yet, and once composing, the
 * dispatch+persist critical section is shielded (deferred cancellation —
 * see _await_tool_turn_with_deferred_cancellation in composer/service.py),
 * so the in-flight tool finishes and P4 publishes its results BEFORE the
 * CancelledError resumes. Resyncing before all of that lands misses
 * durable writes (user row, state advances) with no later refresh.
 *
 * Settlement is QUIESCENCE, not phase: `inflight_requests === 0` on the
 * progress snapshot — the count of compose requests currently inside the
 * route for this session, maintained by the server across the whole
 * request lifecycle (see _track_compose_inflight). The narrative phase
 * cannot carry this signal: after an immediate Stop or for a request
 * queued behind another turn, the registry still holds the PREVIOUS
 * turn's terminal snapshot, which is indistinguishable from real
 * settlement by phase alone.
 *
 * The wait ends on SEMANTIC conditions only, matching the server's own
 * unboundedness (a wall-clock budget would silently reopen the race for
 * any tool that outruns it):
 * - quiescence: zero in-flight compose requests — the normal exit;
 * - the user navigated to a different session — the resync is moot;
 * - a newer compose turn claimed the progress poller (`ownerGeneration`
 *   went stale) — its completion handler owns the state sync now. The
 *   generation is the mode-AGNOSTIC supersession signal: every compose
 *   entry point (sendMessage, retryMessage, chatGuided) claims the poller
 *   via startComposerProgressPolling, whereas store flags cannot see a
 *   guided turn (chatGuided sets guidedChatPending, not isComposing);
 * - the progress endpoint fails or predates the count field — progress is
 *   advisory, degrade to an immediate best-effort resync.
 */
async function waitForCancelledComposeToSettle(
  sessionId: string,
  ownerGeneration: number,
): Promise<void> {
  for (;;) {
    const current = useSessionStore.getState();
    if (
      current.activeSessionId !== sessionId ||
      composerProgressPollGeneration !== ownerGeneration
    ) {
      return;
    }
    let inflightRequests: number;
    try {
      const snapshot = await api.fetchComposerProgress(sessionId);
      inflightRequests = snapshot.inflight_requests ?? 0;
    } catch {
      // Progress is advisory — settle best-effort and resync now.
      return;
    }
    if (inflightRequests === 0) {
      return;
    }
    await sleep(ABORT_RESYNC_SETTLE_POLL_MS);
  }
}

/**
 * INVARIANT (elspeth-06a23adfcc): every freeform compose entry point that
 * can be aborted client-side (sendMessage, retryMessage) MUST call this from
 * its abort branch. A client-side abort (Stop button / COMPOSE_TIMEOUT_MS
 * guard) only rejects the local fetch — the server turn keeps mutating the
 * session until the disconnect watcher cancels it, and every step it
 * completed before the cancel (canonical user row, assistant rows,
 * composition-state advances, proposals, interpretation reviews, blobs) is
 * committed by per-operation transactions, with the shielded in-flight
 * tool's P4 publish landing shortly AFTER the client's fetch has rejected
 * (see waitForCancelledComposeToSettle). The route's cancelled unwind then
 * persists only LLM-call telemetry, so waiting for terminal progress and
 * resyncing once observes everything renderable. Without this the
 * transcript/side rail keep the pre-send snapshot until a manual reload
 * (stale "No pipeline yet" at v1 while the server head is v3 with pending
 * review cards).
 *
 * Best-effort: the abort copy is already on screen, so a refetch failure
 * keeps the stale snapshot rather than stacking a second error on top.
 */
async function resyncAfterAbortedComposeTurn(
  sessionId: string,
  ownerGeneration: number,
): Promise<void> {
  // ownerGeneration is the aborted turn's progress-poller claim (returned
  // by its startComposerProgressPolling). It fences every stage of the
  // resync: a newer compose turn in ANY mode — freeform or guided — claims
  // the poller and thereby invalidates this resync, so a snapshot fetched
  // here can never overwrite state a newer turn has since produced.
  const superseded = () =>
    useSessionStore.getState().activeSessionId !== sessionId ||
    composerProgressPollGeneration !== ownerGeneration;
  await waitForCancelledComposeToSettle(sessionId, ownerGeneration);
  if (superseded()) {
    // The wait exited because the resync became moot (the user navigated
    // away) or because a newer turn owns the state sync now — abandon
    // instead of fetching results only to discard them.
    return;
  }
  // Reuse the inflight reconciler: it drops the optimistic local-* row only
  // when its canonical counterpart was actually persisted, so a request
  // that never reached the route keeps its failed row + retry affordance.
  await useSessionStore.getState().loadInflightMessages(sessionId);
  let state: CompositionState | null | undefined;
  let proposals: CompositionProposal[] | null | undefined;
  try {
    [state, proposals] = await Promise.all([
      api.fetchCompositionState(sessionId),
      api.fetchCompositionProposals(sessionId),
    ]);
  } catch {
    return;
  }
  if (superseded()) {
    // A newer turn started (and possibly finished) while the GETs were in
    // flight — this snapshot is stale against its results. Drop it.
    return;
  }
  useSessionStore.setState((s) => {
    const previousVersion = s.compositionState?.version ?? null;
    const newVersion = state?.version ?? null;
    const versionChanged =
      newVersion !== null && newVersion !== previousVersion;
    // R4-H3 mirror of the success branches: a new state version invalidates
    // any validation verdict rendered against the old one.
    if (versionChanged) {
      getExecutionStore().clearValidation();
    }
    const newState = state ?? s.compositionState;
    const nodeStillExists =
      !s.selectedNodeId ||
      newState?.nodes.some((n) => n.id === s.selectedNodeId);
    return {
      compositionState: newState,
      compositionProposals: proposals ?? s.compositionProposals,
      ...(nodeStillExists ? {} : { selectedNodeId: null }),
    };
  });
  // Same fire-and-forget refreshes as the success branches: the cancelled
  // turn may have created blobs, auto-titled the session, and minted
  // interpretation reviews before the cancel landed.
  useBlobStore.getState().loadBlobs(sessionId);
  void useSessionStore.getState().loadSessions();
  void refreshInterpretationEventsForSession(sessionId);
}

/**
 * Guided sibling of resyncAfterAbortedComposeTurn (elspeth-b2d9e4d084): an
 * aborted guided chat turn keeps running server-side until the disconnect
 * watcher cancels it, and everything it committed before the cancel
 * (chat_history turns, step results, composition-state advances from the
 * step-1 upload commit path) is durable. Guided state is
 * server-authoritative, so the resync is a single GET /guided refetch —
 * applied under the same quiescence wait and poller-generation fence as
 * the freeform resync, and best-effort for the same reasons.
 */
async function resyncAfterAbortedGuidedTurn(
  sessionId: string,
  ownerGeneration: number,
): Promise<void> {
  const superseded = () =>
    useSessionStore.getState().activeSessionId !== sessionId ||
    composerProgressPollGeneration !== ownerGeneration;
  await waitForCancelledComposeToSettle(sessionId, ownerGeneration);
  if (superseded()) {
    return;
  }
  let guided: GetGuidedResponse | undefined;
  try {
    guided = await api.getGuided(sessionId);
  } catch {
    // Best-effort: the abort copy is already on screen.
    return;
  }
  const resynced = guided;
  if (resynced == null || superseded()) {
    return;
  }
  useSessionStore.setState((s) => {
    const previousVersion = s.compositionState?.version ?? null;
    const newVersion = resynced.composition_state?.version ?? null;
    // R4-H3 mirror of the compose branches: a new state version
    // invalidates any validation verdict rendered against the old one.
    if (newVersion !== null && newVersion !== previousVersion) {
      getExecutionStore().clearValidation();
    }
    return {
      guidedSession: resynced.guided_session,
      guidedNextTurn: resynced.next_turn,
      guidedProposalReview: proposalReviewForTurn(resynced.next_turn),
      guidedTerminal: resynced.terminal ?? s.guidedTerminal,
      compositionState: resynced.composition_state ?? s.compositionState,
    };
  });
  // The cancelled turn may have created blobs (step-1 upload path) or
  // minted interpretation reviews before the cancel landed.
  useBlobStore.getState().loadBlobs(sessionId);
  void refreshInterpretationEventsForSession(sessionId);
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

// turn_not_emitted self-heal bookkeeping (C-3, composer first-principles
// review 2026-07-04): counts consecutive turn_not_emitted rejections per
// session so respondGuided's self-heal (refetch GET /guided, re-render the
// current turn) cannot loop forever if the refetch doesn't actually resolve
// the staleness. Deliberately module-scope, NOT store state — it is retry
// bookkeeping the UI never reads, not reactive data; keeping it off `set`/
// `get` keeps the public store surface unchanged. Reset on a successful
// respond (see respondGuided's success branch) so a session that heals once
// and later hits a genuinely new staleness gets a fresh budget.
const turnNotEmittedSelfHealCounts = new Map<string, number>();
const MAX_TURN_NOT_EMITTED_SELF_HEALS = 1;

/**
 * The exit-to-freeform guided-respond body: sets control_signal and nulls
 * every choice field. Exported so it is a single literal shared by
 * exitToFreeform (below, sugar over respondGuided for the already-active
 * session) and TutorialGuidedShell's startup exit path, which cannot use
 * respondGuided/exitToFreeform for a session that is not yet the store's
 * active session (see applyGuidedResponse) and so must call the raw
 * api.respondGuided directly with the same body.
 */
export const EXIT_TO_FREEFORM_ACTION = Object.freeze({
  chosen: null,
  edited_values: null,
  custom_inputs: null,
  proposal_id: null,
  draft_hash: null,
  edit_target: null,
  control_signal: "exit_to_freeform",
} satisfies GuidedRespondAction);

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
  | "guidedProposalReview"
  | "guidedChatPending"
  | "guidedResponsePending"
  | "guidedSelfHealNotice"
> {
  return {
    guidedSession: null,
    guidedNextTurn: null,
    guidedTerminal: null,
    guidedProposalReview: null,
    guidedChatPending: false,
    guidedResponsePending: false,
    guidedSelfHealNotice: null,
  };
}

// C-4a (composer first-principles review 2026-07-04): resume guided state on
// session select/reload, so a browser reload mid-guided-build no longer
// strands the user in freeform with the stepper and current decision gone.
//
// There is no cheap client-side signal for "has this session ever touched
// guided mode" — the session-list summary (types/index.ts Session) carries
// no guided marker, and CompositionStateResponse's composer_meta (which DOES
// carry a server-internal guided_session key — schemas.py:252) is never
// exposed on the wire to the frontend (types/index.ts CompositionState has
// no composer_meta field at all). So this is a fetch-and-tolerate probe, not
// a conditional skip: GET /guided 400s with exactly one shape when the
// session has no guided_session attached (a plain freeform session) — see
// get_guided's single 400 raise in routes/composer/guided.py — so any
// caught error here is treated as "this session is freeform-only", not
// surfaced as a selectSession failure. Select-session callers tolerate a
// non-400 failure because guided restoration is not load-bearing there;
// mutation callers can request propagation so a multi-request action retains
// its operation custody until every authoritative read has succeeded.
async function fetchGuidedStateForSelect(
  sessionId: string,
  unexpectedFailure: "tolerate" | "throw" = "tolerate",
): Promise<GetGuidedResponse | null> {
  try {
    const response = await api.getGuided(sessionId);
    // GET /guided is non-mutating on a session with NO persisted
    // CompositionState record yet: get_guided's docstring documents that it
    // returns a lazy in-memory stub GuidedSession + first turn with
    // composition_state: null, so a user who deliberately clicks "Switch to
    // guided" on a genuinely blank session gets an initial turn without
    // writing a spurious empty version. That stub is NOT evidence this
    // session was ever actually in guided mode — auto-adopting it here would
    // flip a brand-new, freeform-preferring session straight into the guided
    // surface on its very first load, for no reason the user asked for. Only
    // a response with a non-null composition_state confirms a REAL,
    // persisted guided_session (get_guided 400s before reaching this success
    // path whenever the persisted state's guided_session key is unset, so a
    // real composition_state here means it was genuinely set).
    return response.composition_state !== null ? response : null;
  } catch (err) {
    // Only the documented 400 (session has no guided_session — a plain
    // freeform session) is an expected, silent "freeform-only" outcome.
    // Anything else (500 on corrupt guided state, 502 during a backend
    // restart, a network blip) still degrades to freeform because guided
    // restore is best-effort — but it is NOT the same as "never used guided",
    // so surface it in the console so a genuinely stranded mid-build session
    // is at least diagnosable rather than silently indistinguishable.
    const status = (err as ApiError | undefined)?.status;
    if (status !== 400) {
      if (unexpectedFailure === "throw") {
        throw err;
      }
      console.warn(
        `[sessionStore] guided-state probe for session ${sessionId} failed (status ${status ?? "unknown"}); ` +
          "falling back to freeform. If this session was mid-guided-build, its state was not restored.",
      );
    }
    return null;
  }
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

/**
 * The `source_blob_ids` sidecar from the most recent YAML export fetch,
 * paired with the exact YAML it describes and the session it belongs to.
 *
 * Blob refs are session-scoped (the import endpoint 404s a foreign-session
 * blob), and the import handler 400s a sidecar entry naming a source absent
 * from the pasted YAML. So ImportYamlModal replays this ONLY when both guards
 * hold: `sessionId` still matches the active session AND `yaml` matches the
 * pasted text verbatim. Those two checks make a stale binding inert, which is
 * why it is not threaded through every session-reset site — a mismatch simply
 * declines to replay, and the backend then asks the user to re-provide.
 */
export interface ExportedYamlBlobBinding {
  sessionId: string;
  yaml: string;
  sourceBlobIds: Record<string, string>;
}

interface SessionState {
  sessions: Session[];
  /**
   * True once loadSessions has resolved successfully at least once.
   * Consumers (returning-user auto-resume, the no-sessions empty landing)
   * must not act on an EMPTY sessions array before the list has actually
   * loaded — an unfetched list and a genuinely empty account look identical
   * without this flag.
   */
  sessionsLoaded: boolean;
  activeSessionId: string | null;
  messages: ChatMessage[];
  compositionState: CompositionState | null;
  /**
   * True once the active session's composition state is KNOWN — i.e. the
   * selectSession fetch settled (success, 404, or failure), or the session
   * was just created/forked (fresh state is known by construction).
   * `compositionState === null` alone is ambiguous: it means both "still
   * fetching" and "loaded, and this session has no pipeline yet". The
   * #/{id}/yaml hash route gates the Export-YAML modal on content
   * (elspeth-bff8043d33) and needs the disambiguation to avoid either
   * breaking the deep link or opening the modal on an empty pipeline.
   */
  compositionStateLoaded: boolean;
  compositionProposals: CompositionProposal[];
  /**
   * source_blob_ids sidecar captured on the last export fetch, so a
   * same-session verbatim re-import can rebind blob-backed sources. Null
   * until an export is fetched (and reset to null by an export with no
   * blob-backed source). See ExportedYamlBlobBinding for the replay guards.
   */
  exportedYamlBlobBinding: ExportedYamlBlobBinding | null;
  setExportedYamlBlobBinding: (binding: ExportedYamlBlobBinding | null) => void;
  composerPreferences: ComposerPreferences | null;
  staleProposalIds: string[];
  proposalActionPendingIds: string[];
  composerProgress: ComposerProgressSnapshot | null;
  isComposing: boolean;
  /**
   * The single reactive source of truth for "a known-good compose abort ceiling
   * exists". Set true by App.checkHealth once GET /api/system/status supplies a
   * valid backend compose wall clock (applyServerComposerTimeout applied). FALSE
   * until then, gating every Send affordance (freeform, guided, side-rail Apply)
   * so no compose request is scheduled against the stale default abort ceiling
   * during the boot window (bootstrap race).
   */
  composeTimeoutReady: boolean;
  setComposeTimeoutReady: (ready: boolean) => void;
  /**
   * TRUE when the backend is reachable (GET /api/system/status returned) but
   * did NOT supply a usable composer_timeout_seconds, so composeTimeoutReady
   * can never latch. Distinguishes "still booting" (both false) from "up but
   * misconfigured" (this true) so the Send affordances can show a distinct
   * diagnostic instead of a perpetual "Connecting…". Reset to false whenever a
   * valid ceiling lands or the backend goes unreachable.
   */
  composerTimeoutUnavailable: boolean;
  setComposerTimeoutUnavailable: (unavailable: boolean) => void;
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
  /**
   * Bind `activeSessionId` to `sessionId` and clear every field a stale
   * previous session (including a completed guided/tutorial one) could
   * leave behind — the same "start clean" fields `selectSession` clears,
   * but WITHOUT its subsequent fetch-and-populate tail. Used by
   * TutorialGuidedShell's mount effect, which must bind the session BEFORE
   * calling startGuided/getTutorialSample and has no composition state of
   * its own to fetch. Previously that mount effect hand-mirrored this
   * field list in a raw `useSessionStore.setState()` call — a silent-drift
   * seam where a new SessionState field added elsewhere would never be
   * reflected there without someone remembering to touch the shell too.
   * Routing it through this one action means the field list lives here.
   */
  resetForTutorialSession: (sessionId: string) => void;
  /**
   * Release the active-session binding when `sessionId` turns out not to
   * exist server-side (dead tutorial resume). Guarded: a no-op unless
   * `activeSessionId` still equals `sessionId`, so a recovery that races a
   * legitimate re-bind can never blank the new session. Without this, the
   * tutorial's dead-resume recovery resets the tutorial machine but leaves
   * the store bound to the dead id — and every consumer keyed on
   * `activeSessionId` (InlineRunResults' run list, composer progress)
   * keeps 404-ing against a session that will never come back.
   */
  unbindMissingSession: (sessionId: string) => void;
  renameSession: (id: string, title: string) => Promise<void>;
  archiveSession: (id: string) => Promise<void>;
  sendMessage: (content: string, signal?: AbortSignal) => Promise<void>;
  loadCompositionProposals: (sessionId?: string) => Promise<void>;
  acceptProposal: (proposalId: string) => Promise<void>;
  rejectProposal: (proposalId: string) => Promise<void>;
  loadComposerProgress: (
    sessionId?: string,
    options?: { discardStaleTerminal?: boolean },
  ) => Promise<void>;
  /**
   * The pollers are MODULE-GLOBAL singletons, so ownership must be
   * explicit: start* returns a generation token, and stop* with that token
   * no-ops when a newer turn has since claimed the poller. Without the
   * token, an aborted turn whose settle wait outlived it would tear down
   * the pollers of the turn the user started in the meantime.
   */
  startComposerProgressPolling: (sessionId: string) => number;
  stopComposerProgressPolling: (sessionId?: string, generation?: number) => void;
  loadInflightMessages: (sessionId: string) => Promise<void>;
  startInflightMessagesPolling: (sessionId: string) => number;
  stopInflightMessagesPolling: (sessionId?: string, generation?: number) => void;
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
  applyResolvedInterpretation: (newState: CompositionState | null) => void;

  // Guided-mode protocol state — all three are null when not in a guided session
  guidedSession: GuidedSession | null;
  guidedNextTurn: TurnPayload | null;
  guidedTerminal: TerminalState | null;
  /** Exact proposal/hash-bound lifecycle for the current proposal controls. */
  guidedProposalReview: GuidedProposalReviewState | null;
  // Per-step chat (Phase A slice 5).  The history itself lives on
  // `guidedSession.chat_history` (server-authoritative); only the in-flight
  // pending flag is local state.  Slice 4 carried an in-memory
  // guidedChatHistory array; slice 5 replaced it with the wire field.
  guidedChatPending: boolean;
  // In-flight wizard answer flag. Distinct from guidedChatPending: this blocks
  // turn-answer buttons while the server-authoritative state machine advances.
  guidedResponsePending: boolean;
  /**
   * Transient, non-alarming notice shown after a turn_not_emitted self-heal
   * (C-3, composer first-principles review 2026-07-04): respondGuided's
   * catch detected the client's view of the current turn was stale, silently
   * refetched GET /guided, and re-rendered the (possibly new) current turn
   * instead of leaving the raw protocol instruction ("Fetch GET
   * /api/sessions/{id}/guided first") in the user's alert banner. Kept
   * SEPARATE from `error` — ChatPanel renders this via role="status"
   * (polite), not role="alert" like `error`/`errorDetails`: a resync is not
   * a failure. Cleared at the start of the next guided respond/chat attempt
   * and by clearError().
   */
  guidedSelfHealNotice: string | null;
  // Guided-mode actions
  startGuided: (sessionId: string) => Promise<void>;
  seedGuided: (
    sessionId: string,
    profileKind: "live" | "tutorial",
  ) => Promise<void>;
  respondGuided: (body: GuidedRespondAction) => Promise<void>;
  /**
   * Apply a GuidedRespondResponse to the store: atomically replace the 4 wire
   * fields, await the B1/D12 interpretation-event refresh, and clear the C-3
   * turn_not_emitted self-heal bookkeeping. Extracted from respondGuided's
   * success path so TutorialGuidedShell's not-yet-active exit path — which
   * must call the raw api.respondGuided directly with an explicit sessionId,
   * since respondGuided/exitToFreeform both key off get().activeSessionId —
   * applies the response through the SAME bookkeeping instead of a forked
   * copy that silently drifts as this evolves. Takes sessionId explicitly
   * (not get().activeSessionId) so it can no-op correctly if the session it
   * belongs to is no longer active by the time it lands.
   */
  applyGuidedResponse: (
    sessionId: string,
    response: GuidedRespondResponse,
  ) => Promise<boolean>;
  reenterGuided: () => Promise<void>;
  // Convert a freeform session into guided mode (POST /guided/convert). Unlike
  // startGuided's GET, this works for a session that has done freeform work
  // (whose persisted state carries no guided_session and which GET rejects with
  // 400): it seeds a fresh wizard as a new version, leaving the freeform
  // pipeline recoverable via version history. Idempotent for already-guided
  // sessions.
  convertToGuided: (sessionId: string) => Promise<void>;
  // Unified entry point bound by the "Switch to guided" button in ChatPanel's
  // freeform body.  Branches on the current guidedSession terminal:
  //   * terminal.kind === "exited_to_freeform" => reenterGuided
  //   * otherwise => convertToGuided (POST). It is idempotent for empty and
  //     already-guided sessions and does the fresh-wizard conversion for a
  //     worked freeform session — the one case GET /guided cannot serve.
  // The button stays a single affordance with one label regardless of branch.
  enterGuided: () => Promise<void>;
  // `signal` aborts the underlying fetch (Stop button / client timeout) — the
  // guided mirror of sendMessage's AbortController plumbing (useComposer).
  chatGuided: (message: string, signal?: AbortSignal) => Promise<void>;
  exitToFreeform: () => Promise<void>;
  clearError: () => void;
  injectSystemMessage: (content: string, stableId?: string) => void;
  reset: () => void;
}

const initialState = {
  sessions: [] as Session[],
  sessionsLoaded: false,
  activeSessionId: null as string | null,
  messages: [] as ChatMessage[],
  compositionState: null as CompositionState | null,
  compositionStateLoaded: false,
  compositionProposals: [] as CompositionProposal[],
  exportedYamlBlobBinding: null as ExportedYamlBlobBinding | null,
  composerPreferences: null as ComposerPreferences | null,
  staleProposalIds: [] as string[],
  proposalActionPendingIds: [] as string[],
  composerProgress: null as ComposerProgressSnapshot | null,
  isComposing: false,
  composeTimeoutReady: false,
  composerTimeoutUnavailable: false,
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

  setExportedYamlBlobBinding(binding) {
    set({ exportedYamlBlobBinding: binding });
  },

  setComposeTimeoutReady(ready) {
    set({ composeTimeoutReady: ready });
  },

  setComposerTimeoutUnavailable(unavailable) {
    set({ composerTimeoutUnavailable: unavailable });
  },

  async loadSessions() {
    // Fetch-generation guard (elspeth-4d5b0e634a): every store action that
    // mutates `sessions` (createSession, renameSession, archiveSession,
    // forkFromMessage, reset) — and any external
    // `useSessionStore.setState()` caller, e.g. HelloWorldTutorial's
    // post-rename merge — replaces the array with a brand-new reference;
    // none of them mutate it in place. That makes the
    // reference itself a cheap monotonic generation marker: capture it
    // before the fetch, and if it has changed by the time the fetch
    // resolves, something newer already landed while this request was in
    // flight. Applying the snapshot we just fetched would silently clobber
    // that newer state — e.g. the app-start loadSessions racing the
    // tutorial's createSession+renameSession and overwriting the renamed
    // session with its own pre-rename snapshot. Bail without touching
    // `sessions` in that case.
    const sessionsBeforeFetch = get().sessions;
    try {
      const sessions = await api.fetchSessions();
      if (get().sessions !== sessionsBeforeFetch) {
        // Stale response — the list already reflects newer state. The
        // fetch itself still succeeded, so the loaded-ness flag is honest
        // to flip even though we're discarding this particular snapshot.
        set({ sessionsLoaded: true });
        return;
      }
      set({ sessions, sessionsLoaded: true });
    } catch {
      if (get().sessions !== sessionsBeforeFetch) {
        // A newer, non-fetch mutation already superseded this snapshot;
        // a failed background refresh shouldn't surface an alarming error
        // banner for state the user's own actions have already moved past.
        return;
      }
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
    clearInflightMessagesPollTimer();
    set((state) => ({
      sessions: [session, ...state.sessions],
      activeSessionId: session.id,
      messages: [],
      // A freshly created session is KNOWN to have no composition state.
      compositionState: null,
      compositionStateLoaded: true,
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
        if (get().activeSessionId !== session.id) {
          return;
        }
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
          clearInflightMessagesPollTimer();
        }
        return {
          sessions,
          ...(wasActive
            ? {
                activeSessionId: null,
                messages: [],
                compositionState: null,
                compositionStateLoaded: false,
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
    clearInflightMessagesPollTimer();

    set({
      activeSessionId: id,
      messages: [],
      compositionState: null,
      compositionStateLoaded: false,
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
        guided,
      ] = await Promise.all([
        api.fetchMessages(id),
        api.fetchCompositionState(id),
        api.fetchCompositionProposals(id),
        api.fetchComposerPreferences(id),
        fetchGuidedStateForSelect(id),
      ]);
      // The user may switch sessions while these requests are in flight.
      // Drop stale payloads so an older selection cannot overwrite the
      // newly active session's messages or composition state.
      if (get().activeSessionId !== id) {
        return;
      }
      set({
        messages,
        compositionState,
        compositionStateLoaded: true,
        compositionProposals: compositionProposals ?? [],
        composerPreferences: composerPreferences ?? null,
        // C-4a (fp-review 2026-07-04, elspeth-04d2757bf1): resume guided
        // state when the session has one — a live in-progress build, a
        // completed build, or an exited-to-freeform terminal all restore
        // here; ChatPanel's discriminator decides the surface from there
        // (falling through to freeform for the terminal case exactly as it
        // already does mid-session). REPLACES the earlier "default-freeform,
        // never auto-fetch /guided on select" decision — that left a browser
        // reload mid-guided-build stranded in freeform with the stepper and
        // current decision gone (guidedSession stayed null until the user
        // manually clicked "Switch to guided", which then mis-routed:
        // enterGuided() saw no terminal on a null session and took the
        // startGuided/GET branch, which just re-observed the same terminal
        // and landed back in freeform with zero feedback — see C-4b). A
        // session that never touched guided mode still lands in freeform:
        // `guided` is null (fetchGuidedStateForSelect's fetch-and-tolerate).
        ...(guided !== null
          ? {
              guidedSession: guided.guided_session,
              guidedNextTurn: guided.next_turn,
              guidedTerminal: guided.terminal,
              guidedProposalReview: proposalReviewForTurn(guided.next_turn),
            }
          : {}),
      });

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
          compositionStateLoaded: false,
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
        // The fetch settled (failed) — the composition state is as known as
        // it will get without a retry. Marking loaded lets deferred
        // consumers (the #/{id}/yaml gate) resolve to "no content" instead
        // of waiting forever.
        set({
          error: "Failed to load session. Please refresh the page.",
          compositionStateLoaded: true,
        });
      }
    }
  },

  resetForTutorialSession(sessionId: string) {
    set({
      activeSessionId: sessionId,
      messages: [],
      compositionState: null,
      compositionStateLoaded: false,
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
  },

  unbindMissingSession(sessionId: string) {
    if (get().activeSessionId !== sessionId) {
      return;
    }
    set({
      activeSessionId: null,
      messages: [],
      compositionState: null,
      compositionStateLoaded: false,
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
    const progressPollGeneration =
      get().startComposerProgressPolling(activeSessionId);
    const inflightPollGeneration =
      get().startInflightMessagesPolling(activeSessionId);

    try {
      const stateId = get().compositionState?.id;
      const result = await api.sendMessage(activeSessionId, content, stateId, signal);
      if (get().activeSessionId !== activeSessionId) {
        return;
      }
      // Sync the chat panel against the durable DB state before applying the
      // POST's metadata. After this await, get().messages contains every
      // assistant row the compose loop persisted (and the canonical user row
      // — the optimistic local-* version is gone). The set() block below
      // then only needs to update derived state (compositionState, proposals,
      // isComposing) without re-appending the final assistant message that
      // the poll has already pulled in.
      await get().loadInflightMessages(activeSessionId);
      // Navigation can also happen during the post-completion message sync.
      // Keep the compose response scoped to the session that initiated it.
      if (get().activeSessionId !== activeSessionId) {
        return;
      }
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
      // freeform inline review widgets never render mid-session. Freeform is
      // fire-and-forget (`void`); only guided respondGuided awaits the refresh.
      void refreshInterpretationEventsForSession(activeSessionId);
    } catch (err) {
      let errorMessage: string;
      // Client-side abort (the useComposer COMPOSE_TIMEOUT_MS guard or any
      // user-supplied signal) rejects with the raw abort-reason string, or a
      // DOMException named 'AbortError' when aborted without a reason —
      // never a structured ApiError. The apiErr.detail fallback below would
      // otherwise mask it as a generic send failure.
      if (isComposeAbort(err)) {
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
      if (get().activeSessionId !== activeSessionId) {
        return;
      }
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
      if (isComposeAbort(err)) {
        // The turn ran (and was cancelled) server-side; pull its durable
        // partial results into view (see resyncAfterAbortedComposeTurn).
        await resyncAfterAbortedComposeTurn(
          activeSessionId,
          progressPollGeneration,
        );
      }
    } finally {
      get().stopInflightMessagesPolling(activeSessionId, inflightPollGeneration);
      get().stopComposerProgressPolling(activeSessionId, progressPollGeneration);
      // One-shot terminal pickup — only while this turn still owns the
      // poller; a newer turn's own polling handles it otherwise.
      if (progressPollGeneration === composerProgressPollGeneration) {
        await get().loadComposerProgress(activeSessionId);
      }
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
        get().compositionProposals.find((item) => item.id === proposalId)
          ?.pipeline_metadata?.draft_hash ?? null,
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
      // (see the invariant on refreshInterpretationEventsForSession). Freeform
      // is fire-and-forget (`void`); only guided respondGuided awaits it.
      void refreshInterpretationEventsForSession(activeSessionId);
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

  async loadComposerProgress(
    sessionId?: string,
    options?: { discardStaleTerminal?: boolean },
  ) {
    const targetSessionId = sessionId ?? get().activeSessionId;
    if (!targetSessionId) return;

    try {
      const progress = await api.fetchComposerProgress(targetSessionId);
      const current = get();
      if (current.activeSessionId !== targetSessionId) {
        return;
      }
      const isTerminal = TERMINAL_COMPOSER_PROGRESS_PHASES.has(progress.phase);
      if (options?.discardStaleTerminal && isTerminal && !composerProgressPollSeenNonTerminal) {
        // Stale carry-over from the PREVIOUS turn's terminal snapshot,
        // fetched before this compose's own "starting" event has landed in
        // the registry — drop it rather than flash a "done" state at the
        // start of a fresh compose. Callers outside the poll session (the
        // explicit final load in sendMessage/retryMessage/chatGuided's
        // finally, after stopComposerProgressPolling) don't pass
        // discardStaleTerminal — that one-shot load runs strictly after the
        // request settles, so the registry can only hold THIS turn's own
        // terminal event by then, never stale data.
        return;
      }
      if (progress.phase !== "idle" && !isTerminal) {
        composerProgressPollSeenNonTerminal = true;
      }
      set({ composerProgress: progress.phase === "idle" ? null : progress });
    } catch {
      // Composer progress is advisory. Keep the local heuristic fallback.
    }
  },

  startComposerProgressPolling(sessionId: string) {
    clearComposerProgressPollTimer();
    composerProgressPollGeneration += 1;
    composerProgressPollSessionId = sessionId;
    composerProgressPollSeenNonTerminal = false;
    set({ composerProgress: null });
    void get().loadComposerProgress(sessionId, { discardStaleTerminal: true });
    composerProgressPollTimer = setInterval(() => {
      if (composerProgressPollSessionId !== sessionId) return;
      void useSessionStore
        .getState()
        .loadComposerProgress(sessionId, { discardStaleTerminal: true });
    }, COMPOSER_PROGRESS_POLL_INTERVAL_MS);
    return composerProgressPollGeneration;
  },

  stopComposerProgressPolling(sessionId?: string, generation?: number) {
    if (generation !== undefined && generation !== composerProgressPollGeneration) {
      // A newer turn claimed the poller after this caller's start — the
      // teardown belongs to that turn now.
      return;
    }
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
    inflightMessagesPollGeneration += 1;
    inflightMessagesPollSessionId = sessionId;
    inflightMessagesPollTimer = setInterval(() => {
      if (inflightMessagesPollSessionId !== sessionId) return;
      void useSessionStore.getState().loadInflightMessages(sessionId);
    }, INFLIGHT_MESSAGES_POLL_INTERVAL_MS);
    return inflightMessagesPollGeneration;
  },

  stopInflightMessagesPolling(sessionId?: string, generation?: number) {
    if (generation !== undefined && generation !== inflightMessagesPollGeneration) {
      // A newer turn claimed the poller after this caller's start — the
      // teardown belongs to that turn now.
      return;
    }
    if (
      sessionId !== undefined &&
      inflightMessagesPollSessionId !== null &&
      inflightMessagesPollSessionId !== sessionId
    ) {
      return;
    }
    clearInflightMessagesPollTimer();
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
    const progressPollGeneration =
      get().startComposerProgressPolling(activeSessionId);
    const inflightPollGeneration =
      get().startInflightMessagesPolling(activeSessionId);

    try {
      // Use recompose (not sendMessage) — the user message is already
      // persisted from the original send. Calling sendMessage again
      // would insert a duplicate user message.
      const result = await api.recompose(activeSessionId, signal);
      if (get().activeSessionId !== activeSessionId) {
        return;
      }
      // Sync the chat panel against the DB state (see sendMessage for
      // rationale).
      await get().loadInflightMessages(activeSessionId);
      if (get().activeSessionId !== activeSessionId) {
        return;
      }
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
      // invariant on refreshInterpretationEventsForSession). Freeform is
      // fire-and-forget (`void`); only guided respondGuided awaits it.
      void refreshInterpretationEventsForSession(activeSessionId);
    } catch (err) {
      let errorMessage: string;
      if (isComposeAbort(err)) {
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

      if (get().activeSessionId !== activeSessionId) {
        return;
      }
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
      if (isComposeAbort(err)) {
        // The recompose turn ran (and was cancelled) server-side; pull its
        // durable partial results into view (see
        // resyncAfterAbortedComposeTurn).
        await resyncAfterAbortedComposeTurn(
          activeSessionId,
          progressPollGeneration,
        );
      }
    } finally {
      get().stopInflightMessagesPolling(activeSessionId, inflightPollGeneration);
      get().stopComposerProgressPolling(activeSessionId, progressPollGeneration);
      // One-shot terminal pickup — only while this turn still owns the
      // poller; a newer turn's own polling handles it otherwise.
      if (progressPollGeneration === composerProgressPollGeneration) {
        await get().loadComposerProgress(activeSessionId);
      }
    }
  },

  async forkFromMessage(messageId: string, newContent: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    clearComposerProgressPollTimer();
    clearInflightMessagesPollTimer();
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
        compositionStateLoaded: true,
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
    if (
      recoveryError.partial_state_save_failed === true ||
      typeof recoveredState.id !== "string" ||
      recoveredState.id.trim() === ""
    ) {
      set({
        error:
          "Recovered draft was not saved on the server. Discard recovery and retry the composer step.",
      });
      return { applied: false, needsConfirmation: false };
    }
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
    // Mirrors the active-session guard in loadComposerProgress:
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
        guidedProposalReview: proposalReviewForTurn(response.next_turn),
        compositionState: response.composition_state,
      });
    } catch (err) {
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      // Error path: set error string, leave existing guided state alone.
      // Mirrors selectSession lines 207-209: set error, don't clobber fields
      // that were already loaded. The caller can inspect error to decide whether
      // to surface a retry prompt. Surface the backend's typed detail when
      // present (mirroring respondGuided/chatGuided/convertToGuided) instead of
      // flattening every failure to the generic banner — a 400 that names a
      // recoverable mode-state boundary should reach the user (elspeth-e2c3dba6b5).
      const apiErr = err as ApiError;
      set({
        error:
          apiErr.detail ?? "Failed to load guided session. Please try again.",
      });
    }
  },

  async seedGuided(sessionId: string, profileKind: "live" | "tutorial") {
    const requestedSessionId = sessionId;
    const retry = acquireGuidedRetry("guided_start", sessionId, [profileKind]);
    let responseReceived = false;
    try {
      const response = await api.startGuidedSession(
        sessionId,
        profileKind,
        retry.operationId,
      );
      responseReceived = true;
      if (get().activeSessionId !== requestedSessionId) {
        clearGuidedRetry(retry);
        return;
      }
      await get().applyGuidedResponse(requestedSessionId, response);
      clearGuidedRetry(retry);
    } catch (err) {
      // Once POST returned, any failure is in local response application or
      // interpretation refresh. Keep the same id so retry exact-replays the
      // already committed start instead of creating a second operation.
      if (!responseReceived && !isAmbiguousGuidedRetryFailure(err)) {
        clearGuidedRetry(retry);
      }
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      const apiErr = err as ApiError;
      set({
        error:
          apiErr.detail ?? "Failed to start guided session. Please try again.",
      });
      throw err;
    }
  },

  async convertToGuided(sessionId: string) {
    // Capture the session identity before the await (stale-fetch guard,
    // mirroring startGuided). If the user switches sessions while the POST is in
    // flight, the response is dropped rather than overwriting the newly active
    // session's guided state.
    const requestedSessionId = sessionId;
    const retry = acquireGuidedRetry("guided_convert", sessionId, []);
    try {
      const response = await api.convertToGuided(sessionId, retry.operationId);
      clearGuidedRetry(retry);
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      // Atomically replace all 4 wire fields — server is authoritative (spec §7.3).
      set({
        guidedSession: response.guided_session,
        guidedNextTurn: response.next_turn,
        guidedTerminal: response.terminal,
        guidedProposalReview: proposalReviewForTurn(response.next_turn),
        compositionState: response.composition_state,
        error: null,
      });
    } catch (err) {
      if (!isAmbiguousGuidedRetryFailure(err)) {
        clearGuidedRetry(retry);
      }
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      // Surface the backend's typed detail when present, mirroring
      // respondGuided/reenterGuided. Convert 400s are rare (ownership only), but
      // an unbound catch would recreate the generic-banner defect this fix removes.
      const apiErr = err as ApiError;
      set({
        error:
          apiErr.detail ?? "Failed to switch to guided mode. Please try again.",
      });
    }
  },

  async respondGuided(body: GuidedRespondAction) {
    const { activeSessionId, guidedNextTurn, guidedTerminal } = get();
    // Offensive guard — caller must not invoke this without an active session.
    // Per CLAUDE.md: "Proactively detect invalid states and throw meaningful
    // exceptions." Using ?. to silently skip would mask a programmer error.
    if (activeSessionId === null) {
      throw new Error("respondGuided called without active session");
    }
    // Capture the session identity before the await (Codex #4 stale-fetch guard).
    // Mirrors the active-session guard in loadComposerProgress.
    const requestedSessionId = activeSessionId;
    const requestedTurnToken = guidedTerminal === null ? guidedNextTurn?.turn_token : null;
    if (guidedTerminal === null && requestedTurnToken === undefined) {
      throw new Error("respondGuided called without a current unanswered turn");
    }
    const retry = acquireGuidedRetry("guided_respond", requestedSessionId, [
      requestedTurnToken ?? "terminal",
      body,
    ]);
    const request: GuidedRespondRequest = {
      ...body,
      operation_id: retry.operationId,
      turn_token: requestedTurnToken ?? null,
    };
    const proposalBinding = body.proposal_id === null
      ? null
      : {
          proposal_id: body.proposal_id,
          draft_hash: body.draft_hash,
        };
    const proposalRetryAction = proposalRetryActionForBody(body);
    // Clear any stale self-heal notice at the start of the next attempt, per
    // its documented lifecycle (the resync notice describes the PREVIOUS
    // desync, not this one).
    set({
      guidedResponsePending: true,
      guidedSelfHealNotice: null,
      ...(proposalBinding === null
        ? {}
        : {
            guidedProposalReview: {
              status: "submitting",
              ...proposalBinding,
            },
          }),
    });
    let responseReceived = false;
    try {
      const response = await api.respondGuided(activeSessionId, request);
      responseReceived = true;
      // Stale-fetch guard (Codex #4): drop the response if the active session
      // changed while the request was in flight.
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      // Apply the response (atomic 4-field replace + B1/D12 interpretation
      // refresh + C-3 self-heal bookkeeping) via the shared helper — see
      // applyGuidedResponse below, also used by TutorialGuidedShell's
      // not-yet-active exit path.
      const applied = await get().applyGuidedResponse(requestedSessionId, response);
      if (!applied) {
        return;
      }
      clearGuidedRetriesForSession("guided_respond", requestedSessionId);
    } catch (err) {
      const isAmbiguousFailure =
        !responseReceived && isAmbiguousGuidedRetryFailure(err);
      if (!responseReceived && !isAmbiguousFailure) {
        clearGuidedRetry(retry);
      }
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      const apiErr = err as ApiError;

      // C-3 self-heal: "turn_not_emitted" means the client's view of the
      // current turn was stale (guided.py's respond handler couldn't find an
      // emitted TurnRecord for the current step). Refetch GET /guided
      // directly (NOT via startGuided — that action swallows its own
      // failures into `error`, which would stomp the notice we're about to
      // set; calling api.getGuided here keeps success/failure fully in this
      // block's control) so the current turn re-renders, and surface a calm
      // resync notice instead of a raw rejection — NOT the submitted body
      // resent: the rejected answer was never confirmed against a real
      // emitted turn, so blindly replaying it against whatever turn now
      // exists could apply a stale response to the wrong question. The user
      // re-submits manually against the refreshed turn. Capped at
      // MAX_TURN_NOT_EMITTED_SELF_HEALS per session so a refetch that
      // doesn't actually fix the staleness can't loop forever — a failed
      // resync, or an exhausted budget, falls through to the plain error
      // path below (whose `apiErr.detail` is already the backend's
      // plain-language "out of sync" copy, not the old raw protocol string).
      if (apiErr.error_type === "turn_not_emitted") {
        const priorAttempts =
          turnNotEmittedSelfHealCounts.get(requestedSessionId) ?? 0;
        if (priorAttempts < MAX_TURN_NOT_EMITTED_SELF_HEALS) {
          turnNotEmittedSelfHealCounts.set(requestedSessionId, priorAttempts + 1);
          try {
            const resynced = await api.getGuided(requestedSessionId);
            if (get().activeSessionId !== requestedSessionId) {
              return;
            }
            set({
              guidedSession: resynced.guided_session,
              guidedNextTurn: resynced.next_turn,
              guidedTerminal: resynced.terminal,
              guidedProposalReview: proposalReviewForTurn(resynced.next_turn),
              compositionState: resynced.composition_state,
              guidedResponsePending: false,
              error: null,
              errorDetails: null,
              guidedSelfHealNotice:
                "The wizard had fallen out of sync with the server. We've refreshed to the current step — please try again.",
            });
            return;
          } catch {
            // The resync fetch itself failed — fall through to the plain
            // error path below rather than pretending the self-heal
            // succeeded. Re-check the active session first: the resync await
            // invalidated the outer catch's entry guard, so without this a
            // session switch mid-resync would stomp the newly selected
            // session's UI with this (now-background) session's error.
            if (get().activeSessionId !== requestedSessionId) {
              return;
            }
          }
        } else {
          // Budget exhausted: stop self-healing silently and fall through to
          // the plain error state below (no infinite loop).
          turnNotEmittedSelfHealCounts.delete(requestedSessionId);
        }
      }

      // A proposal-authority conflict means the bound proposal changed before
      // settlement (or the server now requires a binding). Discard operation
      // custody and replace the actionable projection from GET; never replay
      // the rejected body against whatever proposal is current now. Other
      // 409s are actionable failures, not resync signals — notably a
      // wire_confirm_rejected response must retain its structured validation
      // details even when an authoritative GET would succeed.
      const isProposalAuthorityConflict =
        isHttpConflict(err) &&
        (body.proposal_id !== null ||
          apiErr.detail ===
            "the active guided proposal requires proposal_id and draft_hash" ||
          apiErr.detail ===
            "proposal_id and draft_hash do not identify the active guided proposal");
      if (isProposalAuthorityConflict) {
        clearGuidedRetry(retry);
        if (proposalBinding !== null) {
          set({
            guidedProposalReview: {
              status: "reloading",
              ...proposalBinding,
            },
          });
        }
        try {
          const resynced = await api.getGuided(requestedSessionId);
          if (get().activeSessionId !== requestedSessionId) {
            return;
          }
          await useInterpretationEventsStore
            .getState()
            .refreshAll(requestedSessionId);
          if (get().activeSessionId !== requestedSessionId) {
            return;
          }
          const authoritativeReview = proposalReviewForTurn(resynced.next_turn);
          const sameProposal =
            proposalBinding !== null &&
            authoritativeReview !== null &&
            authoritativeReview.proposal_id === proposalBinding.proposal_id &&
            authoritativeReview.draft_hash === proposalBinding.draft_hash;
          set({
            guidedSession: resynced.guided_session,
            guidedNextTurn: resynced.next_turn,
            guidedTerminal: resynced.terminal,
            guidedProposalReview:
              proposalBinding === null
                ? authoritativeReview
                : authoritativeReview !== null && !sameProposal
                  ? authoritativeReview
                  : { status: "stale", ...proposalBinding },
            compositionState: resynced.composition_state,
            error:
              apiErr.detail ??
              "The guided proposal changed. Review the refreshed step and try again.",
            errorDetails: null,
            guidedResponsePending: false,
            guidedSelfHealNotice: null,
          });
          return;
        } catch {
          if (get().activeSessionId !== requestedSessionId) {
            return;
          }
          // Preserve the original conflict as the actionable failure when the
          // authoritative reload is itself unavailable.
          if (proposalBinding !== null) {
            set({
              guidedProposalReview: {
                status: "error",
                ...proposalBinding,
                message:
                  "The proposal changed, but its authoritative replacement could not be loaded. Refresh the session before taking another action.",
                retryable: false,
                retry_action: null,
              },
            });
          }
          set({
            error:
              apiErr.detail ??
              "The guided proposal changed, but its authoritative replacement could not be loaded.",
            errorDetails: null,
            guidedResponsePending: false,
            guidedSelfHealNotice: null,
          });
          return;
        }
      }

      // Surface the backend's structured rejection when present — a wire-stage
      // confirm against an invalid pipeline returns 409 with a nested detail
      // ({code: "wire_confirm_rejected", detail, validation_errors}); showing
      // only a blanket "failed" would recreate the silent no-op confirm this
      // fix removes (elspeth-3b35abf148 variant 3). `validation_errors`
      // entries are backend ValidationEntry payloads ({component, message,
      // severity}) — read defensively so any shape still yields a line.
      const rejectionDetails = (apiErr.validation_errors ?? [])
        .map((entry) => {
          const raw = entry as unknown as Record<string, unknown>;
          const component =
            typeof raw.component === "string" && raw.component !== ""
              ? raw.component
              : null;
          const message = typeof raw.message === "string" ? raw.message : "";
          return component !== null && message !== ""
            ? `${component}: ${message}`
            : message;
        })
        .filter((line) => line !== "");
      const retainsRetryCustody = responseReceived || isAmbiguousFailure;
      const proposalErrorReview: GuidedProposalReviewState | null =
        proposalBinding === null
          ? null
          : retainsRetryCustody && proposalRetryAction !== null
            ? {
                status: "error",
                ...proposalBinding,
                message: responseReceived
                  ? "The server accepted the response, but local review state could not be refreshed. Retry the same action."
                  : apiErr.detail ??
                    "The proposal response was not received. Retry the same action.",
                retryable: true,
                retry_action: proposalRetryAction,
              }
            : {
                status: "error",
                ...proposalBinding,
                message:
                  apiErr.detail ??
                  "The proposal response failed. Refresh the session before taking another action.",
                retryable: false,
                retry_action: null,
              };
      set({
        error:
          apiErr.detail ?? "Failed to submit guided response. Please try again.",
        errorDetails: rejectionDetails.length > 0 ? rejectionDetails : null,
        guidedResponsePending: false,
        guidedSelfHealNotice: null,
        ...(proposalErrorReview === null
          ? {}
          : {
              guidedProposalReview: proposalErrorReview,
            }),
      });
    }
  },

  async applyGuidedResponse(sessionId: string, response: GuidedRespondResponse) {
    // Mirrors respondGuided's own stale-fetch guard: a response that arrives
    // for a session that is no longer active must not clobber whatever the
    // user has since switched to.
    if (get().activeSessionId !== sessionId) {
      return false;
    }
    // B1 (spec §5/D12): backend-surfaced pending interpretation cards must be
    // in interpretationEventsStore before the new turn is published. If this
    // refresh fails, the old token and retry custody stay aligned so the exact
    // action remains replayable.
    await refreshInterpretationEventsForSession(sessionId);
    if (get().activeSessionId !== sessionId) {
      return false;
    }
    // Publish the four authoritative fields atomically only after the
    // interpretation projection is ready.
    turnNotEmittedSelfHealCounts.delete(sessionId);
    set({
      guidedSession: response.guided_session,
      guidedNextTurn: response.next_turn,
      guidedTerminal: response.terminal,
      guidedProposalReview: proposalReviewForTurn(response.next_turn),
      compositionState: response.composition_state,
      guidedResponsePending: false,
      error: null,
      errorDetails: null,
      guidedSelfHealNotice: null,
    });
    return true;
  },

  async reenterGuided() {
    const { activeSessionId } = get();
    if (activeSessionId === null) {
      throw new Error("reenterGuided called without active session");
    }
    const requestedSessionId = activeSessionId;
    const retry = acquireGuidedRetry("guided_reenter", activeSessionId, []);
    try {
      const response = await api.reenterGuided(activeSessionId, retry.operationId);
      clearGuidedRetry(retry);
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      set({
        guidedSession: response.guided_session,
        guidedNextTurn: response.next_turn,
        guidedTerminal: response.terminal,
        guidedProposalReview: proposalReviewForTurn(response.next_turn),
        compositionState: response.composition_state,
        error: null,
      });
    } catch (err) {
      if (!isAmbiguousGuidedRetryFailure(err)) {
        clearGuidedRetry(retry);
      }
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      set({ error: "Failed to re-enter guided mode. Please try again." });
    }
  },

  async enterGuided() {
    // Unified "Switch to guided" entry point.  Sessions that were previously in
    // guided and exited via the operator's "Exit to freeform" button reach a
    // terminal of kind === "exited_to_freeform"; those go through reenterGuided,
    // because convert would return the terminal state and leave the
    // discriminator on freeform.
    //
    // Every other case routes through convertToGuided (POST /guided/convert),
    // which is idempotent and safe for all entry states:
    //   * empty / never-worked session   => lazy fresh wizard (like GET did)
    //   * worked freeform session         => fresh-wizard conversion — the one
    //                                        case GET /guided 400s on, and the
    //                                        whole point of this action
    //                                        (elspeth-e2c3dba6b5)
    //   * already-guided, non-terminal    => returned unchanged (idempotent)
    //   * completed terminal              => returned unchanged, so ChatPanel
    //                                        continues to render the completion
    //                                        summary. It is not re-entrable.
    const { activeSessionId, guidedSession } = get();
    if (activeSessionId === null) {
      throw new Error("enterGuided called without active session");
    }
    if (guidedSession?.terminal?.kind === "exited_to_freeform") {
      await get().reenterGuided();
      return;
    }
    await get().convertToGuided(activeSessionId);
  },

  async chatGuided(message: string, signal?: AbortSignal) {
    const { activeSessionId, guidedSession, guidedNextTurn } = get();
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
    if (guidedNextTurn === null) {
      throw new Error("chatGuided called without a current unanswered turn");
    }
    // Capture session + step identity before the await (stale-fetch guard
    // mirroring respondGuided / startGuided).  If the user switches
    // session or the wizard advances mid-flight, the response is dropped.
    const requestedSessionId = activeSessionId;
    const requestedTurnToken = guidedNextTurn.turn_token;
    const retry = acquireGuidedRetry("guided_chat", requestedSessionId, [
      requestedTurnToken,
      message,
    ]);

    // Slice 5: chat history is server-authoritative — no optimistic local
    // append.  The route handler appends both user + assistant turns to
    // `guidedSession.chat_history` and returns the updated session.  Only
    // `guidedChatPending` is local: it blocks rapid double-submits while
    // the round-trip is in flight.  Slice 4's optimistic-append pattern
    // produced visible drift if the server replied with a slightly
    // different ts_iso / seq than the client guessed.
    // Clear any stale self-heal notice at the start of the next attempt, per
    // its documented lifecycle — a successful advisory chat must not leave a
    // "we've refreshed — please try again" resync notice pinned above it.
    set({ guidedChatPending: true, guidedSelfHealNotice: null });
    // Mirrors sendMessage/retryMessage: the backend guided-chat route (any
    // step, not just step_2_sink — see post_guided_chat) now writes progress
    // snapshots the same way freeform compose does, so start polling here
    // too. Previously chatGuided never polled at all, which is why the
    // tutorial's step-2 substep indicator (composerProgress-derived) never
    // advanced in production (elspeth-a8eeebb3aa) — tests only ever passed
    // because they injected composerProgress directly via setState.
    const progressPollGeneration =
      get().startComposerProgressPolling(requestedSessionId);

    try {
      const response = await api.chatGuided(
        activeSessionId,
        {
          operation_id: retry.operationId,
          turn_token: requestedTurnToken,
          message,
        },
        signal,
      );
      clearGuidedRetry(retry);
      // Stale-fetch guard: drop the response if session changed mid-flight.
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      set({
        guidedSession: response.guided_session,
        guidedNextTurn: response.next_turn,
        guidedTerminal: response.terminal,
        guidedProposalReview: proposalReviewForTurn(response.next_turn),
        compositionState: response.composition_state,
        guidedChatPending: false,
      });
    } catch (err) {
      const ambiguous = isAmbiguousGuidedRetryFailure(err);
      if (!ambiguous) {
        clearGuidedRetry(retry);
      }
      if (get().activeSessionId !== requestedSessionId) {
        return;
      }
      const apiErr = err as ApiError;
      if (isHttpConflict(err)) {
        try {
          const resynced = await api.getGuided(requestedSessionId);
          if (get().activeSessionId !== requestedSessionId) {
            return;
          }
          set({
            guidedSession: resynced.guided_session,
            guidedNextTurn: resynced.next_turn,
            guidedTerminal: resynced.terminal,
            guidedProposalReview: proposalReviewForTurn(resynced.next_turn),
            compositionState: resynced.composition_state,
            error: apiErr.detail ?? "The guided turn changed. Review the refreshed step and try again.",
            guidedChatPending: false,
          });
          return;
        } catch {
          if (get().activeSessionId !== requestedSessionId) {
            return;
          }
          // Preserve the original conflict as the actionable failure when the
          // best-effort authoritative reload is itself unavailable.
        }
      }
      // Cancellation / client-timeout path (elspeth-fb4464cdf0): the guided
      // ChatInput's Stop button (or the COMPOSE_TIMEOUT_MS guard) aborted the
      // fetch. Reset the pending flag so the turn can be revised and re-sent,
      // and surface the same cancelled/timeout copy freeform uses — the
      // abort reason on the signal discriminates user-cancel from timeout.
      if (isComposeAbort(err)) {
        set({
          error: composeAbortMessage(signal),
          guidedChatPending: false,
        });
        // The turn ran (and was cancelled) server-side; pull its durable
        // partial results into view (see resyncAfterAbortedGuidedTurn).
        await resyncAfterAbortedGuidedTurn(
          requestedSessionId,
          progressPollGeneration,
        );
        return;
      }
      // HTTP-layer failure (network, 4xx/5xx).  Distinct from the
      // backend's synthetic-message path, which returns 200 with an
      // unavailable assistant message AND appends both turns to
      // chat_history — that path completes the optimistic write
      // server-side, so even a synthetic reply round-trips through this
      // success branch.  This catch fires when the request itself failed
      // (no response shape at all) OR the backend rejected with a 4xx/5xx.
      //
      // Surface the backend's `detail` when present: a 409 step-mismatch
      // tells the user to retry, a 400 names the bad step — far more
      // actionable than a blanket "failed", which forced the user to GUESS
      // the cause. Backend details are egress-safe by construction (Tier-3
      // row data is never placed in an HTTP detail — see guided.py); a bare
      // network failure with no structured body falls back to the generic line.
      set({
        error: apiErr.detail ?? "Failed to send chat message. Please try again.",
        guidedChatPending: false,
      });
    } finally {
      // Real finally (not duplicated into the success/catch branches above)
      // so polling stops on EVERY exit path, including the stale-session
      // early returns — mirrors sendMessage/retryMessage's teardown. The
      // extra loadComposerProgress picks up the backend's final
      // complete/failed/cancelled snapshot for the brief window before
      // guidedChatPending flips the pending strip away.
      get().stopComposerProgressPolling(requestedSessionId, progressPollGeneration);
      if (progressPollGeneration === composerProgressPollGeneration) {
        await get().loadComposerProgress(requestedSessionId);
      }
    }
  },

  async exitToFreeform() {
    // Sugar over respondGuided — sets control_signal and nulls all choice fields.
    // All state mutation is handled by respondGuided (via applyGuidedResponse).
    await get().respondGuided(EXIT_TO_FREEFORM_ACTION);
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
    const retry = acquireGuidedRetry("state_revert", activeSessionId, [stateId]);

    try {
      // R4-H3: Clear validation BEFORE updating compositionState
      // to prevent a frame where stale validation is visible with the new version
      getExecutionStore().clearValidation();

      const compositionState = await api.revertToVersion(
        activeSessionId,
        stateId,
        retry.operationId,
      );
      // Re-derive the guided surface from the reverted version. A revert can
      // cross the guided/freeform boundary — most visibly the recoverability
      // flow behind convertToGuided ("fresh wizard + consent",
      // elspeth-e2c3dba6b5): convert a worked freeform session to guided, then
      // revert to the prior freeform version to get the pipeline back. Patching
      // only compositionState would leave the stale cached guidedSession
      // rendering the guided wizard over restored freeform state (and the
      // reverse — reverting to a guided version would keep freeform). Probe
      // GET /guided (non-mutating; 400 => freeform-only) and set the wire
      // fields to what the reverted version actually is, mirroring
      // selectSession's discriminator.
      const guided = await fetchGuidedStateForSelect(activeSessionId, "throw");
      // Stale-guard: drop the result if the active session changed while the
      // revert + probe were in flight (mirrors startGuided/selectSession).
      if (get().activeSessionId !== activeSessionId) {
        clearGuidedRetry(retry);
        return;
      }
      // Clear selection — the reverted version may not contain the selected node
      set({
        compositionState: guided?.composition_state ?? compositionState,
        selectedNodeId: null,
        guidedSession: guided?.guided_session ?? null,
        guidedNextTurn: guided?.next_turn ?? null,
        guidedTerminal: guided?.terminal ?? null,
        guidedProposalReview: proposalReviewForTurn(guided?.next_turn ?? null),
      });
      clearGuidedRetry(retry);
    } catch (err) {
      if (!isAmbiguousGuidedRetryFailure(err)) {
        clearGuidedRetry(retry);
      }
      set({ error: "Failed to revert to version. Please try again." });
    }
  },

  applyResolvedInterpretation(newState: CompositionState | null) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    // Display sync: resolving an interpretation patches the pipeline server-side
    // (bakes the chosen prompt template / model / decision) and returns the new
    // composition. Reflect it so the rendered pipeline matches what will run.
    if (newState !== null) {
      set((s) => ({
        compositionState: newState,
        selectedNodeId:
          !s.selectedNodeId ||
          newState.nodes.some((n) => n.id === s.selectedNodeId)
            ? s.selectedNodeId
            : null,
      }));
    }

    // Gate clearing: re-validate explicitly. The run-gate (ExecuteButton) and
    // the validation system message are driven by validationResult, which goes
    // stale after a resolve — the auto-validate subscription only fires on a
    // composition-version bump, which a resolve does not guarantee (and
    // newState may be null). Without this the user resolves every review and
    // the Run button stays disabled with no signal of what changed. validate()
    // re-checks server-side pending interpretations, so once the last review is
    // resolved the gate opens.
    void getExecutionStore().validate(activeSessionId);
  },

  clearError() {
    set({ error: null, errorDetails: null, guidedSelfHealNotice: null });
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
    clearInflightMessagesPollTimer();
    clearAllGuidedRetries();
    // composeTimeoutReady resets to false via initialState; App.checkHealth
    // re-latches it on re-authentication. The module ceiling (composeTimeoutMs)
    // is a backend property that harmlessly persists — it is only read while
    // ready, which a fresh checkHealth re-establishes before any send.
    // Fresh array (not initialState.sessions) so loadSessions' reference
    // guard can't mistake a post-reset store for the pre-reset one it
    // captured before a still-in-flight fetch (logout/login ABA).
    set({ ...initialState, sessions: [] });
  },
}));
