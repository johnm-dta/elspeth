import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetStore } from "@/test/store-helpers";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useSessionLifecycle } from "./useSession";

/**
 * useSessionLifecycle reset-effect tests.
 *
 * The effect must call reset() unconditionally on every activeSessionId change
 * (including transitions to null), and call loadRuns() only when there is a
 * non-null activeSessionId.
 *
 * Each test is deliberately discriminating: it will FAIL under the original
 * bug (where reset() was inside the `if (activeSessionId)` guard) and PASS
 * under the fix (where reset() is unconditional).
 *
 * Setup discipline (per advisor):
 *   1. resetStore() runs first to restore the real reset() so it doesn't
 *      crash on leftover module-level state (validationRequestSeq, wsConnection).
 *   2. vi.fn() mocks are injected via setState() AFTER resetStore() so they
 *      survive subsequent partial-merge setState() calls (spyOn orphans).
 *   3. loadSessions is mocked to prevent the loadSessions effect from hitting
 *      the real API on mount.
 *   4. loadRuns reference is captured once and never re-injected, because
 *      changing it would re-trigger the effect (it's a dep) and skew counts.
 */

describe("useSessionLifecycle — reset-effect guard", () => {
  let resetMock: ReturnType<typeof vi.fn>;
  let loadRunsMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // Step 1: restore stores to initial state (calls the real reset()).
    resetStore(useExecutionStore);
    resetStore(useSessionStore);

    // Step 2: inject mocks AFTER resetStore so they aren't consumed by it.
    resetMock = vi.fn();
    loadRunsMock = vi.fn().mockResolvedValue(undefined);

    useExecutionStore.setState({ reset: resetMock, loadRuns: loadRunsMock } as never);

    // Prevent the loadSessions effect from hitting the real API.
    useSessionStore.setState({ loadSessions: vi.fn().mockResolvedValue(undefined) } as never);
  });

  // ── Test 1: initial mount with null activeSessionId ─────────────────────
  //
  // On mount the effect fires once. Under the fix reset() is unconditional,
  // so it fires even though activeSessionId is null.
  //
  // Under the bug: reset() is inside `if (activeSessionId)` — it does NOT fire
  // when activeSessionId is null. Expected calls = 0. Assertion fails.
  // Under the fix: reset() fires unconditionally. Expected calls = 1. Passes.

  it("calls reset() once on initial mount when activeSessionId is null", () => {
    useSessionStore.setState({ activeSessionId: null } as never);

    renderHook(() => useSessionLifecycle());

    expect(resetMock).toHaveBeenCalledTimes(1);
    expect(loadRunsMock).not.toHaveBeenCalled();
  });

  // ── Test 2: transition from "sess-1" to null ────────────────────────────
  //
  // Mount with sess-1 (reset fires once), then transition to null (reset fires
  // again). Under the fix total = 2.
  //
  // Under the bug: the null transition is guarded — reset() is NOT called.
  // Total = 1. Assertion `toHaveBeenCalledTimes(2)` fails.
  // Under the fix: total = 2. Passes.

  it("calls reset() again when activeSessionId transitions to null", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    renderHook(() => useSessionLifecycle());

    // Initial mount: reset called once, loadRuns called with "sess-1".
    expect(resetMock).toHaveBeenCalledTimes(1);
    expect(loadRunsMock).toHaveBeenCalledWith("sess-1");

    act(() => {
      useSessionStore.setState({ activeSessionId: null } as never);
    });

    // Null transition: reset must fire again.
    expect(resetMock).toHaveBeenCalledTimes(2);
    // loadRuns must NOT be called with null (the guard still protects it).
    expect(loadRunsMock).toHaveBeenCalledTimes(1);
    expect(loadRunsMock).not.toHaveBeenCalledWith(null);
  });

  // ── Test 3: session-to-session transition ───────────────────────────────
  //
  // Mount with sess-1, then switch to sess-2. reset() fires on each change;
  // loadRuns() fires for each non-null session.
  //
  // Both behaviours are the same under bug and fix for non-null transitions,
  // but this test confirms the fix didn't break the existing happy path.

  it("calls reset() and loadRuns(id) on a session-to-session transition", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    renderHook(() => useSessionLifecycle());

    expect(resetMock).toHaveBeenCalledTimes(1);
    expect(loadRunsMock).toHaveBeenCalledWith("sess-1");

    act(() => {
      useSessionStore.setState({ activeSessionId: "sess-2" } as never);
    });

    expect(resetMock).toHaveBeenCalledTimes(2);
    expect(loadRunsMock).toHaveBeenCalledWith("sess-2");
    expect(loadRunsMock).toHaveBeenCalledTimes(2);
  });

  // ── Test 4: loadRuns is never called with null ──────────────────────────
  //
  // Transitions to null must never invoke loadRuns(null). This confirms the
  // inner guard that protects loadRuns is intact under the fix.

  it("does not call loadRuns when activeSessionId is null (guard preserved)", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    renderHook(() => useSessionLifecycle());

    act(() => {
      useSessionStore.setState({ activeSessionId: null } as never);
    });

    // loadRuns must have been called exactly once (for the initial "sess-1" mount),
    // never with null.
    expect(loadRunsMock).toHaveBeenCalledTimes(1);
    expect(loadRunsMock).toHaveBeenCalledWith("sess-1");
    expect(loadRunsMock).not.toHaveBeenCalledWith(null);
    expect(loadRunsMock).not.toHaveBeenCalledWith(undefined);
  });
});
