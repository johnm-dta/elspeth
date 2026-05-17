import { describe, it, expect, beforeEach, vi } from "vitest";
import { initStoreSubscriptions, _resetSubscriptionsForTesting } from "./subscriptions";
import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";
import { useAuditReadinessStore } from "./auditReadinessStore";
import type { Session } from "../types/api";

const SESSION_A = "00000000-0000-0000-0000-000000000001";
const SESSION_B = "00000000-0000-0000-0000-000000000002";

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

  it("calls injectSystemMessage and sendValidationFeedback when validation fails", () => {
    const injectSystemMessage = vi.fn();
    const sendValidationFeedback = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendValidationFeedback,
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
            message: "Required field 'path' is missing",
          },
        ],
        warnings: [],
      } as never,
    } as never);

    expect(injectSystemMessage).toHaveBeenCalled();
    expect(sendValidationFeedback).toHaveBeenCalled();

    const [message, stableId] = injectSystemMessage.mock.calls[0] as [string, string];
    expect(message).toContain("Validation failed");
    expect(message).toContain("csv_source");
    expect(stableId).toBe("system-validation-current");
  });

  it("calls injectSystemMessage but NOT sendValidationFeedback when validation passes with warnings", () => {
    const injectSystemMessage = vi.fn();
    const sendValidationFeedback = vi.fn();
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendValidationFeedback,
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
    expect(sendValidationFeedback).not.toHaveBeenCalled();

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
});
