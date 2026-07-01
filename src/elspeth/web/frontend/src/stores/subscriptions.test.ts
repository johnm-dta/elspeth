import { describe, it, expect, beforeEach, vi } from "vitest";
import { act, waitFor } from "@testing-library/react";
import { initStoreSubscriptions, _resetSubscriptionsForTesting, requestValidate } from "./subscriptions";
import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";
import { useAuditReadinessStore } from "./auditReadinessStore";
import { useAuthStore } from "./authStore";
import type { Session } from "../types/api";

const SESSION_A = "00000000-0000-0000-0000-000000000001";
const SESSION_B = "00000000-0000-0000-0000-000000000002";

function compositionWithSource(version: number) {
  return {
    version,
    sources: { source: { plugin: "text", options: { content: "hello" } } },
    nodes: [],
    edges: [],
    outputs: [],
  };
}

describe("subscriptions — auditReadiness clearSession on session removal", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    useSessionStore.setState({ sessions: [] });
    initStoreSubscriptions();
    vi.clearAllMocks();
  });

  it("calls clearSession when a session is removed from sessionStore.sessions", () => {
    const clearSpy = vi.spyOn(useAuditReadinessStore.getState(), "clearSession");
    // Seed: two sessions present.
    useSessionStore.setState({
      sessions: [
        { id: SESSION_A, title: "A", created_at: "", updated_at: "" } as Session,
        { id: SESSION_B, title: "B", created_at: "", updated_at: "" } as Session,
      ],
    });
    // Remove session A.
    useSessionStore.setState({
      sessions: [{ id: SESSION_B, title: "B", created_at: "", updated_at: "" } as Session],
    });
    expect(clearSpy).toHaveBeenCalledWith(SESSION_A);
    expect(clearSpy).not.toHaveBeenCalledWith(SESSION_B);
  });

  it("does not call clearSession on session addition", () => {
    const clearSpy = vi.spyOn(useAuditReadinessStore.getState(), "clearSession");
    useSessionStore.setState({ sessions: [{ id: SESSION_A, title: "A", created_at: "", updated_at: "" } as Session] });
    useSessionStore.setState({
      sessions: [
        { id: SESSION_A, title: "A", created_at: "", updated_at: "" } as Session,
        { id: SESSION_B, title: "B", created_at: "", updated_at: "" } as Session,
      ],
    });
    expect(clearSpy).not.toHaveBeenCalled();
  });

  it("calls clearSession exactly once per removed id, even if the array is rewritten", () => {
    const clearSpy = vi.spyOn(useAuditReadinessStore.getState(), "clearSession");
    useSessionStore.setState({
      sessions: [
        { id: SESSION_A, title: "A", created_at: "", updated_at: "" } as Session,
        { id: SESSION_B, title: "B", created_at: "", updated_at: "" } as Session,
      ],
    });
    useSessionStore.setState({ sessions: [] });
    expect(clearSpy).toHaveBeenCalledTimes(2);
    expect(clearSpy).toHaveBeenCalledWith(SESSION_A);
    expect(clearSpy).toHaveBeenCalledWith(SESSION_B);
  });

  it("clears audit readiness for sessions already present when init is called", () => {
    // Production startup is empty-first, but persist middleware or SSR
    // hydration would seed sessionStore before init. Verify the seed in
    // initStoreSubscriptions catches that case.
    _resetSubscriptionsForTesting();
    useSessionStore.setState({
      sessions: [{ id: SESSION_A, title: "A", created_at: "", updated_at: "" } as Session],
    });
    initStoreSubscriptions();
    const clearSpy = vi.spyOn(useAuditReadinessStore.getState(), "clearSession");

    // Now remove the session that was already present at init time.
    useSessionStore.setState({ sessions: [] });

    expect(clearSpy).toHaveBeenCalledWith(SESSION_A);
  });
});

describe("subscriptions — validation result side effects", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    vi.clearAllMocks();
  });

  it("injects local validation status but does not send raw validation errors to the LLM", () => {
    const injectSystemMessage = vi.fn();
    const sendMessage = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendMessage,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    // Act: set a failing validation result. The subscriber is synchronous —
    // no need for waitFor (mirrors the pattern in the existing subscriptions.test.ts).
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        errors: [
          {
            component_type: "source",
            component_id: "csv_source",
            message: "Required field 'path' contains expanded value sk-live-secret",
          },
        ],
        warnings: [],
      } as never,
    } as never);

    expect(injectSystemMessage).toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();

    const [message, stableId] = injectSystemMessage.mock.calls[0] as [string, string];
    expect(message).toContain("Validation failed");
    expect(message).toContain("csv_source");
    expect(message).not.toContain("sent to the agent");
    expect(stableId).toBe("system-validation-current");
  });

  it("does NOT inject system message or send a composer message for the structured empty_pipeline outcome", () => {
    // Regression: after exit_to_freeform the backend used to surface its
    // pydantic "ElspethSettings: source/sinks Field required" stack trace
    // here, which the subscription would (a) inject into chat and (b) POST
    // to /messages via the old automatic validation POST. Backend
    // now returns a structured ``empty_pipeline`` error_code; this guard
    // suppresses local chat noise for a user with no pipeline content.
    const injectSystemMessage = vi.fn();
    const sendMessage = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendMessage,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        errors: [
          {
            component_type: null,
            component_id: null,
            message: "Pipeline is empty. Add a source and at least one output to begin building.",
            suggestion: "Pick a source plugin and an output destination, then validate again.",
            error_code: "empty_pipeline",
          },
        ],
        warnings: [],
      } as never,
    } as never);

    expect(injectSystemMessage).not.toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("injects review-pending status but does NOT send a composer message for pending interpretation review", () => {
    const injectSystemMessage = vi.fn();
    const sendMessage = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendMessage,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        checks: [
          {
            name: "interpretation_review",
            passed: false,
            detail: "Interpretation review is pending for rate_node:cool.",
            affected_nodes: ["rate_node"],
            outcome_code: null,
          },
        ],
        errors: [
          {
            component_type: "transform",
            component_id: "rate_node",
            message: "Interpretation review is pending for 'cool'.",
            suggestion: "Resolve the pending interpretation review before running.",
            error_code: "interpretation_review_pending",
          },
        ],
        warnings: [],
        readiness: {
          authoring_valid: true,
          execution_ready: false,
          completion_ready: true,
          blockers: [
            {
              code: "interpretation_review_pending",
              component_id: "rate_node",
              component_type: "transform",
              detail: "rate_node:cool",
            },
          ],
        },
      } as never,
    } as never);

    expect(injectSystemMessage).toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
    const [message, stableId] = injectSystemMessage.mock.calls[0] as [string, string];
    // Human-centric copy that points at the review cards, NOT a dump of the
    // raw validation blockers. The machine-facing component id / detail
    // ("rate_node", "rate_node:cool") must NOT leak into the chat message.
    expect(message).toContain("okay");
    expect(message).toContain("ready to run");
    expect(message).not.toContain("rate_node");
    expect(stableId).toBe("system-validation-current");
  });

  it("replaces the pending message with a ready-to-run nudge once reviews resolve", () => {
    const injectSystemMessage = vi.fn();
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    // 1) Pending review surfaces the human "needs your okay" message.
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        checks: [],
        errors: [
          {
            component_type: "transform",
            component_id: "rate_node",
            message: "pending",
            error_code: "interpretation_review_pending",
          },
        ],
        warnings: [],
        readiness: {
          authoring_valid: true,
          execution_ready: false,
          completion_ready: true,
          blockers: [
            {
              code: "interpretation_review_pending",
              component_id: "rate_node",
              component_type: "transform",
              detail: "rate_node:cool",
            },
          ],
        },
      } as never,
    } as never);

    // 2) All reviews resolved → clean valid result → the stale pending message
    //    is replaced (same stable id) by a ready-to-run nudge that names Run.
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
        readiness: {
          authoring_valid: true,
          execution_ready: true,
          completion_ready: true,
          blockers: [],
        },
      } as never,
    } as never);

    const lastCall = injectSystemMessage.mock.calls.at(-1) as [string, string];
    expect(lastCall[0]).toContain("ready");
    expect(lastCall[0]).toContain("Run pipeline");
    expect(lastCall[1]).toBe("system-validation-current");
  });

  it("does NOT fire the ready-to-run nudge for an ordinary valid result", () => {
    const injectSystemMessage = vi.fn();
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    // A clean valid result with no preceding pending review must stay quiet —
    // the nudge is only for the just-resolved transition, not every compose.
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
        readiness: {
          authoring_valid: true,
          execution_ready: true,
          completion_ready: true,
          blockers: [],
        },
      } as never,
    } as never);

    expect(injectSystemMessage).not.toHaveBeenCalled();
  });

  it("injects local warning status without validation feedback egress", () => {
    const injectSystemMessage = vi.fn();
    const sendMessage = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendMessage,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        errors: [],
        warnings: [
          {
            component_type: "transform",
            component_id: "select_cols",
            message: "Identity passthrough detected",
          },
        ],
      } as never,
    } as never);

    expect(injectSystemMessage).toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();

    const [message] = injectSystemMessage.mock.calls[0] as [string, string];
    expect(message).toContain("Validation passed with warnings");
    expect(message).toContain("select_cols");
  });

  it("fires side effects exactly once when the same result reference is set twice (reference-equality guard)", () => {
    const injectSystemMessage = vi.fn();
    useSessionStore.setState({ activeSessionId: "sess-1", injectSystemMessage } as never);
    const result = {
      is_valid: false,
      errors: [{ component_type: "source", component_id: "s1", message: "boom" }],
      warnings: [],
    };

    // Start from null so the first setState transitions null → result.
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    // First setState: null → result; previousValidationResult becomes result;
    // side effects fire once.
    useExecutionStore.setState({ validationResult: result } as never);

    // Second setState: result === previousValidationResult; guard must prevent
    // a second fire.  The subscriber fires (Zustand 1-arg subscribe fires on
    // every setState), but the reference-equality check should short-circuit.
    useExecutionStore.setState({ validationResult: result } as never);

    // Exactly one call — not zero (first fire happened), not two (second blocked).
    expect(injectSystemMessage).toHaveBeenCalledTimes(1);
  });

  it("does not repeat side effects for a fresh object with the same validation outcome", () => {
    const injectSystemMessage = vi.fn();
    const sendMessage = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendMessage,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    const first = {
      is_valid: false,
      errors: [{ component_type: "source", component_id: "s1", message: "boom" }],
      warnings: [],
    };
    const second = {
      is_valid: false,
      errors: [{ component_type: "source", component_id: "s1", message: "boom" }],
      warnings: [],
    };

    useExecutionStore.setState({ validationResult: first } as never);
    useExecutionStore.setState({ validationResult: second } as never);

    expect(injectSystemMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage).not.toHaveBeenCalled();
  });
});

describe("auto-validate on composition-state version change", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
      sessions: [{ id: "sess-1", title: "x" } as never],
    } as never);
    useExecutionStore.setState({
      isExecuting: false,
      validationResult: null,
    } as never);
    initStoreSubscriptions();
  });

  it("fires validate when compositionState.version increments", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: compositionWithSource(1) as never,
    } as never);

    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-1", { expectedVersion: 1 }),
    );

    useSessionStore.setState({
      compositionState: compositionWithSource(2) as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
  });

  it("does not auto-validate a metadata-only guided exit state", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: {
        version: 1,
        sources: {},
        nodes: [],
        edges: [],
        outputs: [],
      } as never,
    } as never);

    await act(async () => {
      await Promise.resolve();
    });

    expect(validate).not.toHaveBeenCalled();
  });

  it("does not fire when version is unchanged (reference change only)", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: compositionWithSource(5) as never,
    } as never);
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-1", { expectedVersion: 5 }),
    );

    useSessionStore.setState({
      compositionState: compositionWithSource(5) as never,
    } as never);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).toHaveBeenCalledTimes(1);
  });

  it("does NOT fire while executing", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate, isExecuting: true } as never);

    useSessionStore.setState({
      compositionState: compositionWithSource(9) as never,
    } as never);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).not.toHaveBeenCalled();
  });

  it("does not fire when activeSessionId is null", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);
    useSessionStore.setState({ activeSessionId: null } as never);

    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).not.toHaveBeenCalled();
  });

  it("re-fires after the in-flight validate settles if a newer version arrived (correctness loop)", async () => {
    let resolveFirst: (() => void) | null = null;
    const firstCallPromise = new Promise<void>((r) => {
      resolveFirst = r;
    });
    const validate = vi
      .fn()
      .mockImplementationOnce(() => firstCallPromise)
      .mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(1));

    useSessionStore.setState({
      compositionState: compositionWithSource(2) as never,
    } as never);

    expect(validate).toHaveBeenCalledTimes(1);

    resolveFirst!();
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
    expect(validate).toHaveBeenLastCalledWith("sess-1", { expectedVersion: 2 });
  });

  it("resets per-session tracking when activeSessionId changes (cross-session isolation)", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-A",
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-A", { expectedVersion: 1 }),
    );

    useSessionStore.setState({
      activeSessionId: "sess-B",
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-B", { expectedVersion: 1 }),
    );
  });

  it("does not cache a version as validated when validation suppresses a stale result", async () => {
    const validate = vi
      .fn()
      .mockResolvedValueOnce(false)
      .mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(1));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    useSessionStore.setState({ activeSessionId: null } as never);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: compositionWithSource(1) as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
    expect(validate).toHaveBeenLastCalledWith("sess-1", { expectedVersion: 1 });
  });

  it("does not cache the failed version: a re-fired identical version retries instead of short-circuiting", async () => {
    // validate() returning false signals the catch path (backend 500 / network
    // error). The cache must stay empty so that re-firing the *same* version
    // triggers a second validate call.
    //
    // Discrimination:
    //   Fix:  cache empty after false → subscription re-runs on new object ref →
    //         lastValidatedVersionBySession.get('sess-1') === 1 is undefined===1
    //         → false → loop continues → validate called twice.
    //   Bug:  cache holds 1 → subscription re-runs → 1===1 → short-circuits →
    //         validate called only once. waitFor times out.
    const validate = vi
      .fn()
      .mockResolvedValueOnce(false)
      .mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(1));

    // Re-fire version 1 with a new object reference so the zustand subscription
    // callback re-runs. The cache check (line 151) is what discriminates:
    // under the fix, cache is empty and validate fires again; under the bug,
    // cache holds 1 and the subscription short-circuits.
    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
    expect(validate).toHaveBeenLastCalledWith("sess-1", { expectedVersion: 1 });
  });

  it("does not inject system message when the user switched sessions mid-validate (cross-session guard)", async () => {
    const injectSystemMessageSpy = vi.fn();
    const staleValidationResult = {
      is_valid: false,
      checks: [],
      errors: [
        {
          component_type: "source",
          component_id: "csv_source",
          message: "Missing path",
        } as never,
      ],
      warnings: [],
    };
    useSessionStore.setState({
      activeSessionId: "sess-A",
      sessions: [{ id: "sess-A" } as never, { id: "sess-B" } as never],
      injectSystemMessage: injectSystemMessageSpy,
    } as never);

    let resolveValidate: (() => void) | null = null;
    const validatePromise = new Promise<void>((r) => {
      resolveValidate = r;
    });
    const validate = vi.fn().mockImplementation(() => validatePromise);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-A", { expectedVersion: 1 }),
    );

    useSessionStore.setState({ activeSessionId: "sess-B" } as never);

    useExecutionStore.setState({
      validationResult: staleValidationResult as never,
    } as never);
    resolveValidate!();

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(injectSystemMessageSpy).not.toHaveBeenCalled();
  });

  it("does not let a suppressed stale result consume the validation fingerprint", async () => {
    const injectSystemMessageSpy = vi.fn();
    const sameContentResult = {
      is_valid: false,
      checks: [],
      errors: [
        {
          component_type: "source",
          component_id: "csv_source",
          message: "Missing path",
        } as never,
      ],
      warnings: [],
    };
    useSessionStore.setState({
      activeSessionId: "sess-A",
      sessions: [{ id: "sess-A" } as never, { id: "sess-B" } as never],
      injectSystemMessage: injectSystemMessageSpy,
    } as never);

    let resolveValidate: (() => void) | null = null;
    const validatePromise = new Promise<void>((r) => {
      resolveValidate = r;
    });
    const validate = vi.fn().mockImplementation(() => validatePromise);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-A", { expectedVersion: 1 }),
    );
    useSessionStore.setState({ activeSessionId: "sess-B" } as never);

    useExecutionStore.setState({ validationResult: sameContentResult as never } as never);
    resolveValidate!();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(injectSystemMessageSpy).not.toHaveBeenCalled();

    useExecutionStore.setState({
      validationResult: { ...sameContentResult } as never,
    } as never);
    await waitFor(() => expect(injectSystemMessageSpy).toHaveBeenCalledTimes(1));
  });
});

// ── Fix A: auto-validate guard checks progress?.status === "running" ─────────
//
// Discrimination: the old guard only checks isExecuting. If isExecuting is
// false but progress.status === "running" (the post-200 live-run window),
// the bug fires validate; the fix suppresses it.
describe("auto-validate guard respects progress.status === 'running' (Fix A)", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    useAuthStore.setState({ token: "tok", user: { user_id: "user-a" } as never });
    // compositionState starts null so no auto-validate fires during setup.
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
      sessions: [{ id: "sess-1", title: "x" } as never],
    } as never);
    useExecutionStore.setState({
      isExecuting: false,
      progress: null,
      validationResult: null,
    } as never);
    initStoreSubscriptions();
  });

  it("does NOT fire validate when progress.status is 'running' even though isExecuting is false", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    // Simulate: HTTP /execute returned 200 (isExecuting=false) but WS progress
    // is live (status="running").
    useExecutionStore.setState({
      isExecuting: false,
      progress: { status: "running", source_rows_processed: 0, tokens_succeeded: 0,
        tokens_failed: 0, tokens_quarantined: 0, tokens_routed_success: 0,
        tokens_routed_failure: 0, accounting: null, recent_errors: [] } as never,
    } as never);

    // Bump the composition version — this fires the auto-validate subscription.
    // Bug: only checks isExecuting (false) → validate fires.
    // Fix: also checks progress?.status === "running" → validate suppressed.
    useSessionStore.setState({
      compositionState: compositionWithSource(2) as never,
    } as never);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).not.toHaveBeenCalled();

    // Now clear the running progress — validate should fire.
    useExecutionStore.setState({ progress: null } as never);
    // Re-fire the same compositionState (new object ref) so the subscriber
    // callback runs again. The version is unchanged; what changed is the guard
    // condition (progress=null), which is checked from executionStore at fire
    // time. pendingValidateTarget was set to version 2 by the blocked
    // subscription fire above, so fireValidateLoop will run with target 2.
    useSessionStore.setState({
      compositionState: compositionWithSource(2) as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-1", { expectedVersion: 2 }));
  });
});

// ── Fix B: lastValidatedVersionBySession cleared on session removal ───────────
//
// Discrimination: without the fix, re-adding a session with the same id and
// the same version skips auto-validate (cache hit). With the fix, the cache
// was cleared at removal, so re-adding triggers validate.
describe("lastValidatedVersionBySession cleared when session is removed (Fix B)", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    useAuthStore.setState({ token: "tok", user: { user_id: "user-a" } as never });
    // compositionState starts null so no auto-validate fires during setup.
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
      sessions: [{ id: "sess-1", title: "x" } as never],
    } as never);
    useExecutionStore.setState({
      isExecuting: false,
      progress: null,
      validationResult: null,
    } as never);
    initStoreSubscriptions();
  });

  it("re-fires validate after a session is removed and re-added with the same version", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    // Seed the cache: trigger the initial auto-validate for sess-1 / version 1.
    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-1", { expectedVersion: 1 }),
    );
    expect(validate).toHaveBeenCalledTimes(1);

    // Remove sess-1 — Fix B: this clears the cache entry.
    // Bug: cache entry persists → re-adding with version 1 short-circuits.
    useSessionStore.setState({ sessions: [], activeSessionId: null } as never);

    // Re-add sess-1 with the same version 1.
    useSessionStore.setState({
      sessions: [{ id: "sess-1", title: "x" } as never],
      activeSessionId: "sess-1",
      compositionState: compositionWithSource(1) as never,
    } as never);

    // With the fix, cache was cleared at removal → validate fires again.
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
    expect(validate).toHaveBeenLastCalledWith("sess-1", { expectedVersion: 1 });
  });
});

// ── Fix C: per-user state resets on auth fingerprint change ──────────────────
//
// Discrimination: without the fix, the lastValidatedVersionBySession cache
// survives a user switch, so re-firing the same session/version with the new
// user short-circuits. With the fix, the cache is cleared on auth identity
// change, so validate fires for the new user.
//
// NOTE: UserProfile uses `user_id` (not `id`) — authIdentityFingerprint reads
// state.user?.user_id, confirmed from types/index.ts.
describe("per-user state resets on auth identity change (Fix C)", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    // Establish user-a as the initial identity BEFORE init, so
    // previousAuthFingerprint is captured as "user-a" at init time.
    useAuthStore.setState({ token: "token-a", user: { user_id: "user-a" } as never });
    // compositionState starts null so no auto-validate fires during setup.
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
      sessions: [{ id: "sess-1", title: "x" } as never],
    } as never);
    useExecutionStore.setState({
      isExecuting: false,
      progress: null,
      validationResult: null,
    } as never);
    initStoreSubscriptions();
  });

  it("fires validate after user switches even when session/version are unchanged", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    // Seed the cache: trigger the initial auto-validate for user-a / sess-1 / version 1.
    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-1", { expectedVersion: 1 }),
    );
    expect(validate).toHaveBeenCalledTimes(1);

    // Switch to user-b — resetPerUserState() clears the cache.
    // Bug: cache survives → re-firing version 1 short-circuits.
    // Fix: cache cleared → validate fires again for user-b.
    useAuthStore.setState({ token: "token-b", user: { user_id: "user-b" } as never });

    // Re-fire compositionState (new object ref) to trigger the auto-validate
    // subscription. The session and version are unchanged.
    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
    expect(validate).toHaveBeenLastCalledWith("sess-1", { expectedVersion: 1 });
  });
});

// ── requestValidate: cache-aware manual validate entry point ──────────────────
//
// Discrimination tests: requestValidate must respect the same guard conditions
// as the auto-validate subscriber — cache hit, isExecuting, progress.status.
// These tests also verify that requestValidate does NOT export or mutate the
// module-level state directly; it is a controlled entry point only.
describe("requestValidate — cache-aware manual validate entry point", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    useAuthStore.setState({ token: "tok", user: { user_id: "user-a" } as never });
    useSessionStore.setState({
      activeSessionId: "sess-1",
      // Default to a non-empty state so the hasCompositionContent guard
      // inside requestValidate does not short-circuit every test below.
      // The empty-state case has its own dedicated test further down.
      compositionState: compositionWithSource(0) as never,
      sessions: [{ id: "sess-1", title: "x" } as never],
    } as never);
    useExecutionStore.setState({
      isExecuting: false,
      progress: null,
      validationResult: null,
    } as never);
    initStoreSubscriptions();
  });

  it("skips validate when active compositionState has no content (post-exit_to_freeform metadata-only state)", async () => {
    // After exit_to_freeform the backend returns a state with version=N
    // but source=null nodes=[] outputs=[]. Without the guard, requestValidate
    // would queue validate and land the structured ``empty_pipeline``
    // failure — which the executionStore subscription used to broadcast
    // via injectSystemMessage and the old automatic validation POST
    // to /messages as role=user.
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: { version: 1, sources: {}, nodes: [], edges: [], outputs: [] } as never,
    } as never);

    // Manually invoke as if from CommandPalette / Ctrl+Shift+V / CompletionSummary.
    requestValidate("sess-1", 1);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).not.toHaveBeenCalled();
  });

  it("is a no-op at an already-validated version (cache hit)", async () => {
    // Seed the cache by triggering an auto-validate for version 1.
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: compositionWithSource(1) as never,
    } as never);
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-1", { expectedVersion: 1 }),
    );
    expect(validate).toHaveBeenCalledTimes(1);

    // Manual trigger at the same (already-validated) version — must be a no-op.
    requestValidate("sess-1", 1);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    // Still exactly one call — requestValidate short-circuited on cache hit.
    expect(validate).toHaveBeenCalledTimes(1);
  });

  it("enqueues validate when version is unvalidated", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    // No auto-validate has fired; version 2 is unvalidated.
    requestValidate("sess-1", 2);

    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith("sess-1", { expectedVersion: 2 }),
    );
    expect(validate).toHaveBeenCalledTimes(1);
  });

  it("skips validate when isExecuting is true", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate, isExecuting: true } as never);

    requestValidate("sess-1", 3);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).not.toHaveBeenCalled();
  });

  it("skips validate when progress.status is 'running'", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({
      validate,
      isExecuting: false,
      progress: {
        status: "running",
        source_rows_processed: 0,
        tokens_succeeded: 0,
        tokens_failed: 0,
        tokens_quarantined: 0,
        tokens_routed_success: 0,
        tokens_routed_failure: 0,
        accounting: null,
        recent_errors: [],
      } as never,
    } as never);

    requestValidate("sess-1", 4);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).not.toHaveBeenCalled();
  });
});
