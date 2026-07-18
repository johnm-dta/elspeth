import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  acquireGuidedRetry,
  clearAllGuidedRetries,
  clearGuidedRetry,
  GUIDED_RETRY_STORAGE_KEY,
  isAmbiguousGuidedRetryFailure,
} from "./guidedOperationRetry";

const SESSION_A = "00000000-0000-4000-8000-000000000201";
const SESSION_B = "00000000-0000-4000-8000-000000000202";

function storedEnvelope(): { descriptors: unknown[] } {
  return JSON.parse(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "null") as {
    descriptors: unknown[];
  };
}

describe("guided operation retry custody", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    window.sessionStorage.clear();
    clearAllGuidedRetries();
  });

  it("expires an ambiguous descriptor after 24 hours", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-18T00:00:00Z"));
    const first = acquireGuidedRetry("guided_reenter", SESSION_A, []);

    vi.setSystemTime(new Date("2026-07-19T00:00:00.001Z"));
    const expired = acquireGuidedRetry("guided_reenter", SESSION_A, []);

    expect(expired.operationId).not.toBe(first.operationId);
    expect(storedEnvelope().descriptors).toHaveLength(1);
  });

  it("caps descriptor count and serialized storage bytes", () => {
    for (let index = 0; index < 40; index += 1) {
      const suffix = (300 + index).toString().padStart(12, "0");
      acquireGuidedRetry("guided_reenter", `00000000-0000-4000-8000-${suffix}`, []);
    }

    const encoded = window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "";
    expect(storedEnvelope().descriptors.length).toBeLessThanOrEqual(16);
    expect(new TextEncoder().encode(encoded).byteLength).toBeLessThanOrEqual(8192);
  });

  it("drops malformed storage instead of reusing an unvalidated operation id", () => {
    window.sessionStorage.setItem(
      GUIDED_RETRY_STORAGE_KEY,
      JSON.stringify({
        schema: "guided-operation-retries.v1",
        descriptors: [{ kind: "guided_reenter", sessionId: SESSION_A, operationId: "not-a-uuid" }],
      }),
    );

    const handle = acquireGuidedRetry("guided_reenter", SESSION_A, []);

    expect(handle.operationId).toMatch(/^[0-9a-f-]{36}$/);
    expect(storedEnvelope().descriptors).toHaveLength(1);
  });

  it("stores only a fingerprint, never the raw request identity", () => {
    acquireGuidedRetry("state_revert", SESSION_A, ["raw-state-id-must-not-survive"]);

    const encoded = window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "";
    expect(encoded).not.toContain("raw-state-id-must-not-survive");
    expect(encoded).toContain(SESSION_A);
  });

  it("retains guided conversion custody without storing request content", () => {
    const first = acquireGuidedRetry("guided_convert", SESSION_A, []);
    const retry = acquireGuidedRetry("guided_convert", SESSION_A, []);

    expect(retry.operationId).toBe(first.operationId);
    expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY)).toContain('"kind":"guided_convert"');
  });

  it("a different action evicts the prior same-kind session descriptor", () => {
    const first = acquireGuidedRetry("state_revert", SESSION_A, ["state-a"]);
    acquireGuidedRetry("state_revert", SESSION_A, ["state-b"]);
    const stateAAgain = acquireGuidedRetry("state_revert", SESSION_A, ["state-a"]);

    expect(stateAAgain.operationId).not.toBe(first.operationId);
    expect(storedEnvelope().descriptors).toHaveLength(1);
  });

  it("a different operation kind evicts prior custody for the same session", () => {
    const revert = acquireGuidedRetry("state_revert", SESSION_A, ["state-a"]);
    acquireGuidedRetry("guided_reenter", SESSION_A, []);
    const revertAgain = acquireGuidedRetry("state_revert", SESSION_A, ["state-a"]);

    expect(revertAgain.operationId).not.toBe(revert.operationId);
    expect(storedEnvelope().descriptors).toHaveLength(1);
  });

  it("retains retry custody in bounded memory when sessionStorage writes fail", () => {
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });

    const first = acquireGuidedRetry("guided_reenter", SESSION_A, []);
    const retry = acquireGuidedRetry("guided_reenter", SESSION_A, []);

    expect(retry.operationId).toBe(first.operationId);
    setItem.mockRestore();
  });

  it("keeps the newer fallback authoritative when quota leaves a stale envelope readable", () => {
    const stale = acquireGuidedRetry("state_revert", SESSION_A, ["state-a"]);
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });

    const replacement = acquireGuidedRetry("state_revert", SESSION_A, ["state-b"]);
    const retry = acquireGuidedRetry("state_revert", SESSION_A, ["state-b"]);

    expect(replacement.operationId).not.toBe(stale.operationId);
    expect(retry.operationId).toBe(replacement.operationId);
    setItem.mockRestore();
  });

  it("does not resurrect completed custody when removing stale storage fails", () => {
    const completed = acquireGuidedRetry("guided_reenter", SESSION_A, []);
    const removeItem = vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });

    clearGuidedRetry(completed);
    expect(removeItem).toHaveBeenCalledWith(GUIDED_RETRY_STORAGE_KEY);
    expect(storedEnvelope().descriptors).toHaveLength(0);
    const nextAction = acquireGuidedRetry("guided_reenter", SESSION_A, []);

    expect(nextAction.operationId).not.toBe(completed.operationId);
    removeItem.mockRestore();
  });

  it("hydrates fallback custody before a later sessionStorage read failure", async () => {
    const stored = acquireGuidedRetry("guided_reenter", SESSION_A, []);
    vi.resetModules();
    const reloaded = await import("./guidedOperationRetry");
    const hydrated = reloaded.acquireGuidedRetry("guided_reenter", SESSION_A, []);
    expect(hydrated.operationId).toBe(stored.operationId);
    const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });

    const retry = reloaded.acquireGuidedRetry("guided_reenter", SESSION_A, []);

    expect(retry.operationId).toBe(stored.operationId);
    getItem.mockRestore();
  });

  it("persists an empty generation when removal fails so reload cannot resurrect custody", async () => {
    const completed = acquireGuidedRetry("guided_reenter", SESSION_A, []);
    const removeItem = vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });

    clearGuidedRetry(completed);
    removeItem.mockRestore();
    vi.resetModules();
    const reloaded = await import("./guidedOperationRetry");
    const nextAction = reloaded.acquireGuidedRetry("guided_reenter", SESSION_A, []);

    expect(nextAction.operationId).not.toBe(completed.operationId);
  });

  it("still removes stale custody when the empty-generation write fails", async () => {
    const completed = acquireGuidedRetry("guided_reenter", SESSION_A, []);
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });
    const removeItem = vi.spyOn(Storage.prototype, "removeItem");

    clearGuidedRetry(completed);
    expect(removeItem).toHaveBeenCalledWith(GUIDED_RETRY_STORAGE_KEY);
    setItem.mockRestore();
    removeItem.mockRestore();
    vi.resetModules();
    const reloaded = await import("./guidedOperationRetry");
    const nextAction = reloaded.acquireGuidedRetry("guided_reenter", SESSION_A, []);

    expect(nextAction.operationId).not.toBe(completed.operationId);
  });

  it("rejects non-canonical or oversized session ids", () => {
    expect(() => acquireGuidedRetry("guided_reenter", "sess-1", [])).toThrow(
      "canonical UUID",
    );
    expect(() =>
      acquireGuidedRetry("guided_reenter", `${SESSION_B}${"x".repeat(256)}`, []),
    ).toThrow("canonical UUID");
  });

  it("retains only network, abort, timeout, and 5xx failures", () => {
    expect(isAmbiguousGuidedRetryFailure(new TypeError("Failed to fetch"))).toBe(true);
    expect(isAmbiguousGuidedRetryFailure(new DOMException("aborted", "AbortError"))).toBe(true);
    expect(isAmbiguousGuidedRetryFailure({ name: "TimeoutError" })).toBe(true);
    expect(isAmbiguousGuidedRetryFailure({ status: 503 })).toBe(true);
    expect(isAmbiguousGuidedRetryFailure({ status: 409 })).toBe(false);
    expect(isAmbiguousGuidedRetryFailure(new Error("application error"))).toBe(false);
  });
});
