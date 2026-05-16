import { describe, it, expect, beforeEach, vi } from "vitest";
import { initStoreSubscriptions, _resetSubscriptionsForTesting } from "./subscriptions";
import { useSessionStore } from "./sessionStore";
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
});
