// stores/subscriptions.ts
//
// Cross-store subscriptions extracted to break circular imports between
// sessionStore, executionStore, and auditReadinessStore. Call
// initStoreSubscriptions() once at app startup (e.g. in App.tsx).

import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";
import { useAuditReadinessStore } from "./auditReadinessStore";
import { useAuthStore } from "./authStore";
import type { ValidationResult } from "@/types/index";
import { hasCompositionContent } from "@/utils/compositionState";

let previousVersion: number | null = null;
let previousSessionIds: Set<string> = new Set();
let initialized = false;
let unsubscribe: (() => void) | null = null;

// Tracks the last-seen activeSessionId so the run-rehydration subscriber
// fires once per session activation, not on every sessionStore write.
let previousActiveSessionId: string | null = null;
let unsubscribeRunRehydration: (() => void) | null = null;

// Module-level state for the executionStore subscriber.
// Must be reset in _resetSubscriptionsForTesting().
let previousValidationFingerprint: string | null = null;
// Tracks whether the last surfaced validation outcome was a pending
// interpretation review. When the next outcome is clean-valid we use this to
// replace the "needs your okay" message with a "ready to run" nudge (and so
// clear the otherwise-stale pending message), without firing that nudge on
// ordinary mid-compose valid results.
let previousWasPendingReview = false;
let unsubscribeExecution: (() => void) | null = null;

const lastValidatedVersionBySession = new Map<string, number>();
let pendingValidateTarget: { sessionId: string; version: number } | null = null;
let validateInflight = false;
let unsubscribeAutoValidate: (() => void) | null = null;
let inflightValidateTarget: { sessionId: string; version: number } | null = null;
let unsubscribeAuth: (() => void) | null = null;

/**
 * Detects the backend's structured ``empty_pipeline`` validation outcome.
 *
 * Returned by ``web/execution/validation.py::validate_pipeline`` when the
 * composition has no source, no transforms, and no outputs. The frontend
 * uses this to suppress chat-injected error banners. The historical
 * feedback-to-LLM path also used this guard to avoid POSTing the failure to
 * ``/messages`` and prompting a confabulated placeholder ``set_pipeline``
 * fix; validation failures are now local-only, but the empty-state silence
 * remains intentional.
 */
function isEmptyPipelineResult(result: ValidationResult): boolean {
  return (
    !result.is_valid &&
    result.errors.length === 1 &&
    result.errors[0].error_code === "empty_pipeline"
  );
}

function isPendingInterpretationReviewResult(result: ValidationResult): boolean {
  const readiness = result.readiness;
  if (!readiness) return false;
  return (
    !result.is_valid &&
    readiness.authoring_valid &&
    !readiness.execution_ready &&
    readiness.completion_ready &&
    readiness.blockers.some((blocker) => blocker.code === "interpretation_review_pending")
  );
}

function validationFingerprint(result: ValidationResult | null): string | null {
  if (!result) return null;
  return JSON.stringify({
    is_valid: result.is_valid,
    readiness: result.readiness,
    errors: result.errors.map((err) => ({
      component_type: err.component_type ?? null,
      component_id: err.component_id ?? null,
      message: err.message,
      suggestion: err.suggestion ?? null,
    })),
    warnings: (result.warnings ?? []).map((warn) => ({
      component_type: warn.component_type ?? null,
      component_id: warn.component_id ?? null,
      message: warn.message,
      suggestion: warn.suggestion ?? null,
    })),
  });
}

/**
 * Returns a string that changes when the authenticated identity changes.
 * "anon" when no token is present; the user_id string otherwise.
 * Used by the auth subscription to detect user switches (logout / re-login
 * as a different user in the same tab) so per-user caches can be cleared.
 *
 * NOTE: UserProfile uses `user_id` (not `id`) — confirmed from types/index.ts.
 */
function authIdentityFingerprint(state: { token: string | null; user: { user_id: string } | null }): string {
  return state.token == null ? "anon" : `${state.user?.user_id ?? "unknown"}`;
}

/**
 * Clears all per-user module-level state without tearing down subscription
 * wiring. Called when the authenticated identity transitions (logout or
 * user switch). Does NOT touch validateInflight, initialized, or any
 * unsubscribe handle — those belong to the subscription wiring lifetime,
 * not the user session lifetime.
 *
 * inflightValidateTarget is deliberately preserved. The executionStore
 * subscriber uses `inflightValidateTarget.sessionId !== activeSessionId`
 * to drop a stale validationResult that resolves after the user has
 * already switched sessions or identities. Nulling it here would short-
 * circuit that guard and let a previous user's validation side-effect
 * system message fire on the new user's session. fireValidateLoop's own
 * try/finally clears the field once the in-flight validate() promise settles.
 */
function resetPerUserState(): void {
  previousVersion = null;
  previousValidationFingerprint = null;
  previousWasPendingReview = false;
  previousSessionIds = new Set();
  lastValidatedVersionBySession.clear();
  pendingValidateTarget = null;
  // Pre-run disclosure opt-outs are per user: the ack map survives
  // executionStore.reset() (session switches) by design, so the identity
  // transition is the one place it must be flushed.
  useExecutionStore.getState().clearRunDisclosureAcks();
}

/**
 * Wire up cross-store subscriptions. Must be called exactly once at
 * application startup, after all stores have been created.
 *
 * Current subscriptions:
 * - Auto-clear validation when compositionState.version changes.
 * - Auto-clear audit-readiness cache for any session that disappears from
 *   sessionStore.sessions (archive, 404 self-eviction, future removal paths).
 *   Uses a previous-id set tracked across firings to detect removals
 *   uniformly — no need to instrument each removal call site.
 * - Fire injectSystemMessage when useExecutionStore.validationResult
 *   transitions to a failing or warnings-only result. Phase 2C centralized
 *   the side-effect orchestration so keyboard and CommandPalette callers of
 *   validate() share the same local UI path.
 * - Auto-validate when compositionState.version increments, with a correctness
 *   loop that re-fires after in-flight validation settles if a newer version
 *   arrived in the meantime.
 * - Rehydrate a live run on session activation (elspeth-90db33baac): when
 *   activeSessionId changes, executionStore.rehydrateActiveRun re-attaches
 *   activeRunId + the progress WebSocket if the backend reports a pending or
 *   running run, so a page reload during execution keeps its Cancel control.
 *   Lives here rather than in sessionStore because sessionStore must not
 *   depend on execution wiring (same circular-import break as the rest of
 *   this module).
 */
export function initStoreSubscriptions(): void {
  if (initialized) return;
  initialized = true;

  // Seed previousSessionIds from the current store state. Otherwise the first
  // removal of any session that was already in sessionStore before
  // initStoreSubscriptions() was called would silently no-op (the empty
  // previousSessionIds set has no ids to detect as "removed"). Production
  // startup order makes this unreachable today (sessions starts empty and
  // init runs synchronously before loadSessions), but adding persist
  // middleware, SSR hydration, or a test that seeds sessions before init
  // would expose the gap. The seed costs one set construction.
  previousSessionIds = new Set(useSessionStore.getState().sessions.map((s) => s.id));

  unsubscribe = useSessionStore.subscribe((state) => {
    // Version-change clears validation.
    const currentVersion = state.compositionState?.version ?? null;
    if (previousVersion !== null && currentVersion !== previousVersion) {
      useExecutionStore.getState().clearValidation();
    }
    previousVersion = currentVersion;

    // Session removal clears audit-readiness cache.
    const currentIds = new Set(state.sessions.map((s) => s.id));
    for (const prevId of previousSessionIds) {
      if (!currentIds.has(prevId)) {
        useAuditReadinessStore.getState().clearSession(prevId);
        lastValidatedVersionBySession.delete(prevId);
      }
    }
    previousSessionIds = currentIds;
  });

  // Run/WebSocket rehydration on session activation. Seeded from current
  // state (like previousSessionIds above) so init itself does not fire a
  // rehydrate for a session that was already active before wiring.
  previousActiveSessionId = useSessionStore.getState().activeSessionId;
  unsubscribeRunRehydration = useSessionStore.subscribe((state) => {
    const sessionId = state.activeSessionId;
    if (sessionId === previousActiveSessionId) return;
    previousActiveSessionId = sessionId;
    if (!sessionId) return;
    // Fire-and-forget: rehydrateActiveRun guards internally against the
    // session changing again while its fetch is in flight.
    void useExecutionStore.getState().rehydrateActiveRun(sessionId);
  });

  const VALIDATION_MSG_ID = "system-validation-current";

  unsubscribeExecution = useExecutionStore.subscribe((state) => {
    const result = state.validationResult;
    if (!result) {
      previousValidationFingerprint = null;
      return;
    }

    if (
      inflightValidateTarget !== null &&
      inflightValidateTarget.sessionId !== useSessionStore.getState().activeSessionId
    ) {
      return;
    }

    const fingerprint = validationFingerprint(result);
    // Content guard: fire only when the validation outcome changes, not when
    // the same result is re-created as a fresh object during hydration or
    // auto-validation refreshes.
    if (fingerprint === previousValidationFingerprint) return;
    previousValidationFingerprint = fingerprint;

    // Empty-pipeline guard: when the backend reports the structured
    // ``empty_pipeline`` outcome, the user has not built anything yet
    // (e.g. immediately after exit_to_freeform). Keep that state silent
    // instead of injecting a local "fix these errors" message. The
    // fingerprint update above stays so a later non-empty failure with a
    // different fingerprint still surfaces.
    if (isEmptyPipelineResult(result)) return;

    const sessionStore = useSessionStore.getState();

    if (isPendingInterpretationReviewResult(result)) {
      previousWasPendingReview = true;
      // Human-centric, not a dump of the raw validation blockers: the review
      // cards above already describe each item in plain language, so this
      // message just orients the user toward them. The per-blocker detail
      // strings (validation.py) are machine-facing and would read as error
      // extracts here.
      const count = result.readiness.blockers.filter(
        (blocker) => blocker.code === "interpretation_review_pending",
      ).length;
      const message =
        count === 1
          ? "I made one choice while building this that I'd like you to okay. " +
            "Check the card above and pick **Use** or **Change it** — then your " +
            "pipeline's ready to run."
          : `I made ${count} choices while building this that I'd like you to okay. ` +
            "Check the cards above and pick **Use** or **Change it** for each — " +
            "once they're all approved, your pipeline's ready to run.";
      sessionStore.injectSystemMessage(message, VALIDATION_MSG_ID);
      return;
    }

    if (!result.is_valid && result.errors.length > 0) {
      previousWasPendingReview = false;
      const lines = ["**Validation failed** — fix the following errors before running:"];
      for (const err of result.errors) {
        lines.push(
          `- **[${err.component_type ?? "unknown"}] ${err.component_id ?? "unknown"}:** ${err.message}`,
        );
      }
      sessionStore.injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
    } else if (result.is_valid && result.warnings && result.warnings.length > 0) {
      previousWasPendingReview = false;
      const lines = ["**Validation passed with warnings:**"];
      for (const warn of result.warnings) {
        lines.push(
          `- **[${warn.component_type ?? "unknown"}] ${warn.component_id ?? "unknown"}:** ${warn.message}`,
        );
      }
      sessionStore.injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
    } else if (result.is_valid && previousWasPendingReview) {
      // The user just resolved the last pending interpretation review and the
      // pipeline is otherwise clean. Replace the now-stale "needs your okay"
      // message (same VALIDATION_MSG_ID) with a clear next step — the Run
      // button lives in the side rail, away from the chat where the user's
      // attention is, so name it explicitly. Gated on previousWasPendingReview
      // so ordinary mid-compose valid results stay quiet.
      previousWasPendingReview = false;
      sessionStore.injectSystemMessage(
        "All approved — your pipeline's ready. Select **Run pipeline** in the " +
          "side panel to start it.",
        VALIDATION_MSG_ID,
      );
    }
  });

  unsubscribeAutoValidate = useSessionStore.subscribe((state) => {
    const sessionId = state.activeSessionId;
    const version = state.compositionState?.version ?? null;
    if (!sessionId || version === null) return;
    if (!hasCompositionContent(state.compositionState)) return;
    if (lastValidatedVersionBySession.get(sessionId) === version) return;
    const exec = useExecutionStore.getState();
    if (exec.isExecuting || exec.progress?.status === "running") return;

    pendingValidateTarget = { sessionId, version };
    if (validateInflight) return;
    void fireValidateLoop();
  });

  // Auth identity subscription — clears per-user caches on logout or user
  // switch (same tab, different user). The subscription wiring itself is not
  // torn down; only the per-user state is cleared so the next user starts
  // fresh. `previousAuthFingerprint` is function-scoped so it is re-captured
  // fresh on every `initStoreSubscriptions()` call, which is what the test
  // seam in `_resetSubscriptionsForTesting` requires.
  let previousAuthFingerprint = authIdentityFingerprint(useAuthStore.getState());
  unsubscribeAuth = useAuthStore.subscribe((state) => {
    const fingerprint = authIdentityFingerprint(state);
    if (fingerprint === previousAuthFingerprint) return;
    previousAuthFingerprint = fingerprint;
    resetPerUserState();
  });
}

async function fireValidateLoop(): Promise<void> {
  validateInflight = true;
  try {
    while (pendingValidateTarget !== null) {
      const target = pendingValidateTarget;
      const execState = useExecutionStore.getState();
      if (execState.isExecuting || execState.progress?.status === "running") {
        pendingValidateTarget = null;
        break;
      }
      if (target.sessionId !== useSessionStore.getState().activeSessionId) {
        pendingValidateTarget = null;
        break;
      }
      if (lastValidatedVersionBySession.get(target.sessionId) === target.version) {
        pendingValidateTarget = null;
        break;
      }

      // FRAGILE: clear pending before awaiting validate() so any newer
      // compositionState.version that arrives during the await is re-queued
      // by the subscription and picked up by the next loop iteration.
      pendingValidateTarget = null;
      inflightValidateTarget = target;
      try {
        const validationApplied = await useExecutionStore
          .getState()
          .validate(target.sessionId, { expectedVersion: target.version });
        if (validationApplied !== false) {
          lastValidatedVersionBySession.set(target.sessionId, target.version);
        }
      } finally {
        inflightValidateTarget = null;
      }
    }
  } finally {
    validateInflight = false;
  }
}

/**
 * Cache-aware manual validate request. Mirrors the auto-validate
 * subscriber's enqueue logic so a manual trigger at an already-validated
 * version is a no-op. Use this from keyboard shortcuts and command-palette
 * actions instead of calling useExecutionStore.validate() directly.
 *
 * Skips validate when the active composition has no source, transforms,
 * or outputs. This mirrors the auto-validate subscription guard so that
 * keyboard / command-palette triggers cannot land the structured
 * ``empty_pipeline`` failure on a session immediately after
 * ``exit_to_freeform`` (where the composition_state version increments
 * but content is empty).
 */
export function requestValidate(sessionId: string, version: number): void {
  if (lastValidatedVersionBySession.get(sessionId) === version) return;
  if (!hasCompositionContent(useSessionStore.getState().compositionState)) {
    return;
  }
  const exec = useExecutionStore.getState();
  if (exec.isExecuting || exec.progress?.status === "running") return;
  pendingValidateTarget = { sessionId, version };
  if (validateInflight) return;
  void fireValidateLoop();
}

/**
 * Test-only helper. Resets the module-level state so each test starts from
 * a clean slate. Unsubscribes the active zustand subscriber so stale
 * callbacks do not accumulate across beforeEach resets. Not exported from
 * any index barrel.
 */
export function _resetSubscriptionsForTesting(): void {
  unsubscribe?.();
  unsubscribe = null;
  unsubscribeExecution?.();
  unsubscribeExecution = null;
  unsubscribeAutoValidate?.();
  unsubscribeAutoValidate = null;
  unsubscribeAuth?.();
  unsubscribeAuth = null;
  unsubscribeRunRehydration?.();
  unsubscribeRunRehydration = null;
  previousActiveSessionId = null;
  previousVersion = null;
  previousValidationFingerprint = null;
  previousWasPendingReview = false;
  previousSessionIds = new Set();
  lastValidatedVersionBySession.clear();
  pendingValidateTarget = null;
  validateInflight = false;
  inflightValidateTarget = null;
  initialized = false;
}
