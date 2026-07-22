import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_COMPOSE_TIMEOUT_MS,
  COMPOSE_CLIENT_GRACE_MS,
  COMPOSE_TIMEOUT_ABORT_REASON,
  getComposeTimeoutMs,
  applyServerComposerTimeout,
  resetComposeTimeoutForTests,
  runComposeWithTimeout,
} from "@/config/composer";

// The compose abort ceiling is module-level state; resetComposeTimeoutForTests
// restores it to the boot default so tests stay independent.
describe("compose timeout ceiling derivation", () => {
  beforeEach(() => {
    resetComposeTimeoutForTests();
  });
  afterEach(() => {
    resetComposeTimeoutForTests();
    vi.useRealTimers();
  });

  it("boots at the checked-in default (270s + grace)", () => {
    expect(DEFAULT_COMPOSE_TIMEOUT_MS).toBe(295_000);
    expect(getComposeTimeoutMs()).toBe(DEFAULT_COMPOSE_TIMEOUT_MS);
  });

  it("adopts the backend wall clock plus grace and reports applied", () => {
    // composer_timeout_seconds is deployment-configurable with NO fixed
    // maximum — 300s is a supported configuration the stale 295s default
    // would abort early. App.checkHealth uses the boolean return to latch the
    // store readiness gate only when a known-good ceiling was derived.
    const applied = applyServerComposerTimeout(300);
    expect(applied).toBe(true);
    expect(getComposeTimeoutMs()).toBe(300_000 + COMPOSE_CLIENT_GRACE_MS);
  });

  it("ignores non-finite / non-positive values and reports not-applied", () => {
    for (const bad of [0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(applyServerComposerTimeout(bad)).toBe(false);
    }
    // A garbage value leaves the ceiling untouched (a safe floor) AND does not
    // let readiness latch — the store gate stays closed.
    expect(getComposeTimeoutMs()).toBe(DEFAULT_COMPOSE_TIMEOUT_MS);
  });

  it("a garbage value AFTER a good ceiling leaves the applied ceiling intact", () => {
    // Deliberate asymmetry App.checkHealth's else-guard relies on: a transient
    // partial/garbage health response must never shrink a ceiling already
    // derived from a real backend wall clock back toward the stale default.
    expect(applyServerComposerTimeout(300)).toBe(true);
    const goodCeiling = getComposeTimeoutMs();
    for (const bad of [0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(applyServerComposerTimeout(bad)).toBe(false);
    }
    expect(getComposeTimeoutMs()).toBe(goodCeiling);
  });
});

describe("runComposeWithTimeout — the shared freeform/guided send primitive", () => {
  beforeEach(() => {
    resetComposeTimeoutForTests();
  });
  afterEach(() => {
    resetComposeTimeoutForTests();
    vi.useRealTimers();
  });

  it("does not start the runner when readiness is false (bootstrap window)", async () => {
    const ref = { current: null as AbortController | null };
    const runner = vi.fn(async () => {});

    await runComposeWithTimeout(ref, false, runner);

    // No API request, no controller, no timer scheduled from an unknown
    // ceiling. This backs the disabled Send affordance for programmatic
    // callers (SideRailValidationBanner).
    expect(runner).not.toHaveBeenCalled();
    expect(ref.current).toBeNull();
  });

  it("aborts a healthy request only AFTER the applied wall clock + grace, never at the stale default", async () => {
    vi.useFakeTimers();
    applyServerComposerTimeout(300); // ceiling = 300s + 25s grace = 325s

    const ref = { current: null as AbortController | null };
    let captured: AbortSignal | undefined;
    const runner = vi.fn(async (signal: AbortSignal) => {
      captured = signal;
      await new Promise<void>((resolve) => {
        signal.addEventListener("abort", () => resolve());
      });
    });

    const p = runComposeWithTimeout(ref, true, runner);
    expect(runner).toHaveBeenCalledTimes(1);
    expect(ref.current).not.toBeNull();

    // Past the STALE 295s default — the original bug aborted a healthy 300s
    // turn here. The fix must keep the request alive.
    await vi.advanceTimersByTimeAsync(295_000 + 1);
    expect(captured?.aborted).toBe(false);

    // Past the applied 325s ceiling — now the last-resort guard fires, after
    // the backend has had its chance to return the structured 422.
    await vi.advanceTimersByTimeAsync(325_000 - (295_000 + 1) + 1);
    expect(captured?.aborted).toBe(true);
    expect(captured?.reason).toBe(COMPOSE_TIMEOUT_ABORT_REASON);

    await p;
    // Controller ref is released once the run settles.
    expect(ref.current).toBeNull();
  });
});
