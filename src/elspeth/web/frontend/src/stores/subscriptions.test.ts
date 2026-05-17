import { describe, it, expect, beforeEach, vi } from "vitest";
import { act, waitFor } from "@testing-library/react";
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

  it("does not repeat side effects for a fresh object with the same validation outcome", () => {
    const injectSystemMessage = vi.fn();
    const sendValidationFeedback = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendValidationFeedback,
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
    expect(sendValidationFeedback).toHaveBeenCalledTimes(1);
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
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-1"));

    useSessionStore.setState({
      compositionState: { version: 2, source: null, nodes: [], outputs: [] } as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
  });

  it("does not fire when version is unchanged (reference change only)", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: { version: 5, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(1));

    useSessionStore.setState({
      compositionState: { version: 5, source: null, nodes: [], outputs: [] } as never,
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
      compositionState: { version: 9, source: null, nodes: [], outputs: [] } as never,
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
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
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
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(1));

    useSessionStore.setState({
      compositionState: { version: 2, source: null, nodes: [], outputs: [] } as never,
    } as never);

    expect(validate).toHaveBeenCalledTimes(1);

    resolveFirst!();
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
    expect(validate).toHaveBeenLastCalledWith("sess-1");
  });

  it("resets per-session tracking when activeSessionId changes (cross-session isolation)", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-A",
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-A"));

    useSessionStore.setState({
      activeSessionId: "sess-B",
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-B"));
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
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-A"));

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
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-A"));
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
