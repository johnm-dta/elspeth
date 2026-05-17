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
let previousValidationResult: ValidationResult | null = null;
let unsubscribeExecution: (() => void) | null = null;

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
 *   warnings-only result. Phase 2C subsumes the side-effect orchestration
 *   that previously lived in InspectorPanel.handleValidate; the
 *   InspectorPanel Validate button was removed in the same phase and the
 *   keyboard/CommandPalette callers of validate() are now the only triggers.
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
    // Reference-equality guard: fire only on result change, not on every
    // store update. Prevents duplicate injectSystemMessage / sendValidationFeedback
    // calls when validate() is invoked in quick succession.
    if (result === previousValidationResult) return;
    previousValidationResult = result;

    if (!result) return;

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
  previousVersion = null;
  previousValidationResult = null;
  previousSessionIds = new Set();
  initialized = false;
}
