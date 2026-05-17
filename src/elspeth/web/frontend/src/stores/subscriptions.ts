// stores/subscriptions.ts
//
// Cross-store subscriptions extracted to break circular imports between
// sessionStore, executionStore, and auditReadinessStore. Call
// initStoreSubscriptions() once at app startup (e.g. in App.tsx).

import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";
import { useAuditReadinessStore } from "./auditReadinessStore";
import type { ValidationResult } from "@/types/index";

let previousVersion: number | null = null;
let previousSessionIds: Set<string> = new Set();
let initialized = false;
let unsubscribe: (() => void) | null = null;

// Module-level state for the executionStore subscriber.
// Must be reset in _resetSubscriptionsForTesting().
let previousValidationFingerprint: string | null = null;
let unsubscribeExecution: (() => void) | null = null;

const lastValidatedVersionBySession = new Map<string, number>();
let pendingValidateTarget: { sessionId: string; version: number } | null = null;
let validateInflight = false;
let unsubscribeAutoValidate: (() => void) | null = null;
let inflightValidateTarget: { sessionId: string; version: number } | null = null;

function validationFingerprint(result: ValidationResult | null): string | null {
  if (!result) return null;
  return JSON.stringify({
    is_valid: result.is_valid,
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
 * Wire up cross-store subscriptions. Must be called exactly once at
 * application startup, after all stores have been created.
 *
 * Current subscriptions:
 * - Auto-clear validation when compositionState.version changes.
 * - Auto-clear audit-readiness cache for any session that disappears from
 *   sessionStore.sessions (archive, 404 self-eviction, future removal paths).
 *   Uses a previous-id set tracked across firings to detect removals
 *   uniformly — no need to instrument each removal call site.
 * - Fire injectSystemMessage + sendValidationFeedback when
 *   useExecutionStore.validationResult transitions to a failing or
 *   warnings-only result. Phase 2C centralized the side-effect orchestration
 *   so keyboard and CommandPalette callers of validate() share the same path.
 * - Auto-validate when compositionState.version increments, with a correctness
 *   loop that re-fires after in-flight validation settles if a newer version
 *   arrived in the meantime.
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
      }
    }
    previousSessionIds = currentIds;
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

    const sessionStore = useSessionStore.getState();

    if (!result.is_valid && result.errors.length > 0) {
      const lines = ["**Validation failed** — the following errors were sent to the agent:"];
      for (const err of result.errors) {
        lines.push(
          `- **[${err.component_type ?? "unknown"}] ${err.component_id ?? "unknown"}:** ${err.message}`,
        );
      }
      sessionStore.injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
      // sendValidationFeedback is fire-and-forget. Per CLAUDE.md audit-primacy,
      // the backend records the validation event in the audit Landscape; a
      // frontend telemetry breadcrumb would duplicate that record. The
      // user-visible system message is already injected above. Phase 8 is
      // the right owner if a frontend operational signal proves useful.
      void sessionStore.sendValidationFeedback(result);
    } else if (result.is_valid && result.warnings && result.warnings.length > 0) {
      const lines = ["**Validation passed with warnings:**"];
      for (const warn of result.warnings) {
        lines.push(
          `- **[${warn.component_type ?? "unknown"}] ${warn.component_id ?? "unknown"}:** ${warn.message}`,
        );
      }
      sessionStore.injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
    }
  });

  unsubscribeAutoValidate = useSessionStore.subscribe((state) => {
    const sessionId = state.activeSessionId;
    const version = state.compositionState?.version ?? null;
    if (!sessionId || version === null) return;
    if (lastValidatedVersionBySession.get(sessionId) === version) return;
    if (useExecutionStore.getState().isExecuting) return;

    pendingValidateTarget = { sessionId, version };
    if (validateInflight) return;
    void fireValidateLoop();
  });
}

async function fireValidateLoop(): Promise<void> {
  validateInflight = true;
  try {
    while (pendingValidateTarget !== null) {
      const target = pendingValidateTarget;
      if (useExecutionStore.getState().isExecuting) {
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
  previousVersion = null;
  previousValidationFingerprint = null;
  previousSessionIds = new Set();
  lastValidatedVersionBySession.clear();
  pendingValidateTarget = null;
  validateInflight = false;
  inflightValidateTarget = null;
  initialized = false;
}
