import { describe, expect, it, afterEach, vi } from "vitest";
import { renderHook } from "@testing-library/react";

import {
  DEFAULT_COMPOSE_TIMEOUT_MS,
  COMPOSE_CLIENT_GRACE_MS,
  getComposeTimeoutMs,
  applyServerComposerTimeout,
  resetComposeTimeout,
  COMPOSE_TIMEOUT_ABORT_REASON,
  COMPOSE_USER_CANCEL_ABORT_REASON,
} from "@/config/composer";
import { useComposer } from "@/hooks/useComposer";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";

describe("compose timeout ceiling", () => {
  afterEach(() => {
    // The ceiling is module-level state; the readiness gate lives in the
    // store. Restore both so tests stay independent.
    resetComposeTimeout();
    resetStore(useSessionStore);
    vi.useRealTimers();
  });

  it("defaults to outliving the checked-in backend wall clock (270s + grace)", () => {
    // ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS (deploy/elspeth-web.env) — the
    // backend's own wall-clock budget for a compose turn. Healthy freeform
    // multi-tool turns legitimately run right up to it (11-22 tool calls
    // observed in the 2026-07-10 battery), so the client must not abort
    // first: a client abort in the 90-270s band killed healthy turns 4/4
    // and left sessions wedged behind zombie server turns
    // (elspeth-e08063c3a5). The client cap exists only for a truly dead
    // connection; the grace margin covers response transit after the
    // backend deadline fires.
    const defaultBackendWallClockMs = 270_000;

    expect(DEFAULT_COMPOSE_TIMEOUT_MS).toBe(295_000);
    expect(getComposeTimeoutMs()).toBe(DEFAULT_COMPOSE_TIMEOUT_MS);
    expect(getComposeTimeoutMs()).toBeGreaterThan(defaultBackendWallClockMs);
  });

  it("derives the ceiling from the configured backend wall clock plus grace", () => {
    // composer_timeout_seconds is deployment-configurable with NO fixed
    // maximum (only transport-ceiling headroom) — e.g. 300s is a supported
    // configuration. A hard-coded 295s cap would abort those healthy
    // requests before the backend's structured 422; the ceiling must
    // track the configured wall clock.
    applyServerComposerTimeout(300);
    expect(getComposeTimeoutMs()).toBe(300_000 + COMPOSE_CLIENT_GRACE_MS);
    expect(getComposeTimeoutMs()).toBeGreaterThan(300_000);
  });

  it("ignores nonsense server values and keeps the current ceiling", () => {
    const before = getComposeTimeoutMs();
    applyServerComposerTimeout(0);
    applyServerComposerTimeout(-5);
    applyServerComposerTimeout(Number.NaN);
    applyServerComposerTimeout(Number.POSITIVE_INFINITY);
    expect(getComposeTimeoutMs()).toBe(before);
  });

  it("reads the ceiling at call time so a boot-applied server value governs the abort", async () => {
    vi.useFakeTimers();
    // Tiny wall clock so the test doesn't advance 295s of fake time:
    // 1s backend wall + grace = 26s ceiling.
    applyServerComposerTimeout(1);
    let captured: AbortSignal | undefined;
    useSessionStore.setState({
      // Post-boot: the readiness gate is open, so the send proceeds and the
      // hook schedules the abort at the derived ceiling.
      composeTimeoutReady: true,
      sendMessage: vi.fn(async (_content: string, signal?: AbortSignal) => {
        captured = signal;
        await new Promise<void>((resolve) => {
          signal?.addEventListener("abort", () => resolve());
        });
      }),
    });

    const { result } = renderHook(() => useComposer());
    const sendPromise = result.current.sendMessage("hello");

    await vi.advanceTimersByTimeAsync(1_000 + COMPOSE_CLIENT_GRACE_MS - 1);
    expect(captured?.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(2);
    expect(captured?.aborted).toBe(true);
    expect(captured?.reason).toBe(COMPOSE_TIMEOUT_ABORT_REASON);
    await sendPromise;
  });

  it("uses distinct abort reasons for timeout and user cancel paths", () => {
    expect(COMPOSE_TIMEOUT_ABORT_REASON).not.toBe(COMPOSE_USER_CANCEL_ABORT_REASON);
  });

  it("does not start a send while the readiness gate is closed (bootstrap window)", async () => {
    // The gate is the client-outlives-server invariant during boot: before
    // GET /api/system/status supplies the wall clock, no request may start —
    // it would schedule an abort from the stale default ceiling.
    const storeSend = vi.fn();
    useSessionStore.setState({ composeTimeoutReady: false, sendMessage: storeSend });

    const { result } = renderHook(() => useComposer());
    await result.current.sendMessage("hello");

    expect(storeSend).not.toHaveBeenCalled();
  });

  it("does not start a retry while the readiness gate is closed", async () => {
    const storeRetry = vi.fn();
    useSessionStore.setState({ composeTimeoutReady: false, retryMessage: storeRetry });

    const { result } = renderHook(() => useComposer());
    await result.current.retryMessage("msg-1");

    expect(storeRetry).not.toHaveBeenCalled();
  });

  it("starts send and retry once the readiness gate opens", async () => {
    applyServerComposerTimeout(300);
    const storeSend = vi.fn().mockResolvedValue(undefined);
    const storeRetry = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      composeTimeoutReady: true,
      sendMessage: storeSend,
      retryMessage: storeRetry,
    });

    const { result } = renderHook(() => useComposer());
    await result.current.sendMessage("hello");
    await result.current.retryMessage("msg-1");

    expect(storeSend).toHaveBeenCalledWith("hello", expect.any(AbortSignal));
    expect(storeRetry).toHaveBeenCalledWith("msg-1", expect.any(AbortSignal));
  });
});
