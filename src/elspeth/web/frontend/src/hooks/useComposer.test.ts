import { describe, expect, it, afterEach, vi } from "vitest";
import { renderHook } from "@testing-library/react";

import {
  DEFAULT_COMPOSE_TIMEOUT_MS,
  COMPOSE_CLIENT_GRACE_MS,
  getComposeTimeoutMs,
  applyServerComposerTimeout,
  COMPOSE_TIMEOUT_ABORT_REASON,
  COMPOSE_USER_CANCEL_ABORT_REASON,
} from "@/config/composer";
import { useComposer } from "@/hooks/useComposer";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";

describe("compose timeout ceiling", () => {
  afterEach(() => {
    // The ceiling is module-level state: restore the default-equivalent
    // derivation (backend 270s + grace = 295s) so tests stay independent.
    applyServerComposerTimeout(DEFAULT_COMPOSE_TIMEOUT_MS / 1000 - COMPOSE_CLIENT_GRACE_MS / 1000);
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
});
