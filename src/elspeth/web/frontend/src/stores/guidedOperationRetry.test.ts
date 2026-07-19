import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  acquireGuidedRetry,
  clearAllGuidedRetries,
  clearGuidedRetry,
  clearGuidedRetriesForSession,
  findGuidedRetry,
  GUIDED_RETRY_STORAGE_KEY,
  isAmbiguousGuidedRetryFailure,
} from "./guidedOperationRetry";
import type {
  GuidedRetryAcquisition,
  GuidedRetryHandle,
  GuidedRetryKind,
} from "./guidedOperationRetry";

const SESSION_A = "00000000-0000-4000-8000-000000000201";
const SESSION_B = "00000000-0000-4000-8000-000000000202";

function storedEnvelope(): { descriptors: unknown[] } {
  return JSON.parse(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "null") as {
    descriptors: unknown[];
  };
}

function expectAcquired(acquisition: GuidedRetryAcquisition): GuidedRetryHandle {
  expect(acquisition.status).toBe("acquired");
  if (acquisition.status === "conflict") {
    throw new Error(`unexpected ${acquisition.existing.kind} retry conflict`);
  }
  return acquisition.handle;
}

function acquireHandle(
  kind: GuidedRetryKind,
  sessionId: string,
  requestIdentity: readonly unknown[],
): GuidedRetryHandle {
  return expectAcquired(acquireGuidedRetry(kind, sessionId, requestIdentity));
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
    const first = acquireHandle("guided_reenter", SESSION_A, []);

    vi.setSystemTime(new Date("2026-07-19T00:00:00.001Z"));
    const expired = acquireHandle("guided_reenter", SESSION_A, []);

    expect(expired.operationId).not.toBe(first.operationId);
    expect(storedEnvelope().descriptors).toHaveLength(1);
  });

  it("caps descriptor count and serialized storage bytes", () => {
    for (let index = 0; index < 40; index += 1) {
      const suffix = (300 + index).toString().padStart(12, "0");
      acquireHandle("guided_reenter", `00000000-0000-4000-8000-${suffix}`, []);
    }

    const encoded = window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "";
    expect(storedEnvelope().descriptors.length).toBeLessThanOrEqual(16);
    expect(new TextEncoder().encode(encoded).byteLength).toBeLessThanOrEqual(8192);
  });

  it("drops malformed storage instead of reusing an unvalidated operation id", () => {
    window.sessionStorage.setItem(
      GUIDED_RETRY_STORAGE_KEY,
      JSON.stringify({
        schema: "guided-operation-retries.v2",
        descriptors: [{ kind: "guided_reenter", sessionId: SESSION_A, operationId: "not-a-uuid" }],
      }),
    );

    const handle = acquireHandle("guided_reenter", SESSION_A, []);

    expect(handle.operationId).toMatch(/^[0-9a-f-]{36}$/);
    expect(storedEnvelope().descriptors).toHaveLength(1);
  });

  it("stores only a fingerprint, never the raw request identity", () => {
    acquireHandle("state_revert", SESSION_A, ["raw-state-id-must-not-survive"]);

    const encoded = window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "";
    expect(encoded).not.toContain("raw-state-id-must-not-survive");
    expect(encoded).toContain(SESSION_A);
  });

  it("retains guided conversion custody without storing request content", () => {
    const first = acquireHandle("guided_convert", SESSION_A, []);
    const retry = acquireHandle("guided_convert", SESSION_A, []);

    expect(retry.operationId).toBe(first.operationId);
    expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY)).toContain('"kind":"guided_convert"');
  });

  it("retains guided plan custody under the v2-only envelope", () => {
    const first = acquireHandle("guided_plan", SESSION_A, ["Build the exact graph"]);
    const retry = acquireHandle("guided_plan", SESSION_A, ["Build the exact graph"]);

    expect(retry.operationId).toBe(first.operationId);
    expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY)).toContain('"schema":"guided-operation-retries.v2"');
    expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY)).toContain('"kind":"guided_plan"');
  });

  it("finds existing guided-start custody after the prompt is no longer in memory", () => {
    const created = acquireHandle("guided_start", SESSION_A, ["prompt that must not persist"]);

    const recovered = findGuidedRetry("guided_start", SESSION_A);

    expect(recovered).toEqual(created);
    expect(recovered).not.toBe(created);
    expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY)).not.toContain(
      "prompt that must not persist",
    );
  });

  it("does not read the retired v1 key or envelope", () => {
    window.sessionStorage.setItem(
      "elspeth_guided_operation_retries_v1",
      JSON.stringify({
        schema: "guided-operation-retries.v1",
        descriptors: [{
          kind: "guided_reenter",
          sessionId: SESSION_A,
          requestFingerprint: "a".repeat(64),
          operationId: "00000000-0000-4000-8000-000000000399",
          createdAt: Date.now(),
        }],
      }),
    );

    const acquired = acquireHandle("guided_reenter", SESSION_A, []);

    expect(acquired.operationId).not.toBe("00000000-0000-4000-8000-000000000399");
    expect(GUIDED_RETRY_STORAGE_KEY).toBe("elspeth_guided_operation_retries_v2");
  });

  it("reuses one session fork operation without storing edited content", () => {
    const identity = ["00000000-0000-4000-8000-000000000301", "private edited request"];
    const first = acquireHandle("session_fork", SESSION_A, identity);
    const retry = acquireHandle("session_fork", SESSION_A, identity);

    expect(retry.operationId).toBe(first.operationId);
    const encoded = window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "";
    expect(encoded).toContain('"kind":"session_fork"');
    expect(encoded).not.toContain(identity[1]);
  });

  it("rehydrates a lost-response session fork operation after module reload", async () => {
    const identity = ["00000000-0000-4000-8000-000000000301", "private edited request"];
    const first = acquireHandle("session_fork", SESSION_A, identity);

    vi.resetModules();
    const reloaded = await import("./guidedOperationRetry");
    const retry = expectAcquired(reloaded.acquireGuidedRetry("session_fork", SESSION_A, identity));

    expect(retry.operationId).toBe(first.operationId);
  });

  it("reuses one guided chat operation for an ambiguous retry without storing the message", () => {
    const identity = ["a".repeat(64), "use the uploaded customer list"];
    const first = acquireHandle("guided_chat", SESSION_A, identity);
    const retry = acquireHandle("guided_chat", SESSION_A, identity);

    expect(retry.operationId).toBe(first.operationId);
    const encoded = window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "";
    expect(encoded).toContain('"kind":"guided_chat"');
    expect(encoded).not.toContain(identity[1]);
  });

  it("canonicalizes object key order before deriving retry identity", () => {
    const first = acquireHandle("guided_respond", SESSION_A, [
      "a".repeat(64),
      { chosen: ["csv"], edited_values: null },
    ]);
    const retry = acquireHandle("guided_respond", SESSION_A, [
      "a".repeat(64),
      { edited_values: null, chosen: ["csv"] },
    ]);

    expect(retry.operationId).toBe(first.operationId);
  });

  it("derives canonical identity without locale-sensitive key ordering", () => {
    const localeCompare = vi
      .spyOn(String.prototype, "localeCompare")
      .mockImplementation(() => {
        throw new Error("locale ordering must not participate");
      });

    const first = acquireHandle("guided_respond", SESSION_A, [
      { z: true, a: false },
    ]);
    const retry = acquireHandle("guided_respond", SESSION_A, [
      { a: false, z: true },
    ]);

    expect(retry.operationId).toBe(first.operationId);
    localeCompare.mockRestore();
  });

  it("rejects non-plain objects that would collapse to an empty JSON record", () => {
    expect(() => acquireHandle("guided_respond", SESSION_A, [new Date(0)])).toThrow(
      "plain records",
    );
  });

  it("clears every descriptor of one kind for one authoritative session", () => {
    const createdAt = Date.now();
    const descriptor = (kind: string, sessionId: string, suffix: string, fingerprint: string) => ({
      kind,
      sessionId,
      requestFingerprint: fingerprint.repeat(64),
      operationId: `00000000-0000-4000-8000-${suffix.padStart(12, "0")}`,
      createdAt,
    });
    window.sessionStorage.setItem(
      GUIDED_RETRY_STORAGE_KEY,
      JSON.stringify({
        schema: "guided-operation-retries.v2",
        descriptors: [
          descriptor("guided_respond", SESSION_A, "301", "a"),
          descriptor("guided_respond", SESSION_A, "302", "b"),
          descriptor("guided_reenter", SESSION_A, "303", "c"),
          descriptor("guided_respond", SESSION_B, "304", "d"),
        ],
      }),
    );

    clearGuidedRetriesForSession("guided_respond", SESSION_A);

    expect(storedEnvelope().descriptors).toEqual([
      descriptor("guided_reenter", SESSION_A, "303", "c"),
      descriptor("guided_respond", SESSION_B, "304", "d"),
    ]);
  });

  it("rejects a different action while preserving prior same-kind session custody", () => {
    const first = acquireHandle("state_revert", SESSION_A, ["state-a"]);
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const removeItem = vi.spyOn(Storage.prototype, "removeItem");
    const conflict = acquireGuidedRetry("state_revert", SESSION_A, ["state-b"]);

    expect(conflict).toEqual({ status: "conflict", existing: first });
    expect(setItem).not.toHaveBeenCalled();
    expect(removeItem).not.toHaveBeenCalled();
    const stateAAgain = acquireHandle("state_revert", SESSION_A, ["state-a"]);

    expect(stateAAgain.operationId).toBe(first.operationId);
    expect(storedEnvelope().descriptors).toHaveLength(1);
  });

  it("keeps independent operation kinds in custody for the same session", () => {
    const revert = acquireHandle("state_revert", SESSION_A, ["state-a"]);
    acquireHandle("guided_reenter", SESSION_A, []);
    const revertAgain = acquireHandle("state_revert", SESSION_A, ["state-a"]);

    expect(revertAgain.operationId).toBe(revert.operationId);
    expect(storedEnvelope().descriptors).toHaveLength(2);
  });

  it("retains retry custody in bounded memory when sessionStorage writes fail", () => {
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });

    const first = acquireHandle("guided_reenter", SESSION_A, []);
    const retry = acquireHandle("guided_reenter", SESSION_A, []);

    expect(retry.operationId).toBe(first.operationId);
    setItem.mockRestore();
  });

  it("keeps the newer fallback authoritative when quota leaves a stale envelope readable", () => {
    const stale = acquireHandle("state_revert", SESSION_A, ["state-a"]);
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });
    const removeItem = vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    clearGuidedRetry(stale);

    const replacement = acquireHandle("state_revert", SESSION_A, ["state-b"]);
    const retry = acquireHandle("state_revert", SESSION_A, ["state-b"]);

    expect(replacement.operationId).not.toBe(stale.operationId);
    expect(retry.operationId).toBe(replacement.operationId);
    setItem.mockRestore();
    removeItem.mockRestore();
  });

  it("does not resurrect completed custody when removing stale storage fails", () => {
    const completed = acquireHandle("guided_reenter", SESSION_A, []);
    const removeItem = vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });

    clearGuidedRetry(completed);
    expect(removeItem).toHaveBeenCalledWith(GUIDED_RETRY_STORAGE_KEY);
    expect(storedEnvelope().descriptors).toHaveLength(0);
    const nextAction = acquireHandle("guided_reenter", SESSION_A, []);

    expect(nextAction.operationId).not.toBe(completed.operationId);
    removeItem.mockRestore();
  });

  it("hydrates fallback custody before a later sessionStorage read failure", async () => {
    const stored = acquireHandle("guided_reenter", SESSION_A, []);
    vi.resetModules();
    const reloaded = await import("./guidedOperationRetry");
    const hydrated = expectAcquired(reloaded.acquireGuidedRetry("guided_reenter", SESSION_A, []));
    expect(hydrated.operationId).toBe(stored.operationId);
    const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });

    const retry = expectAcquired(reloaded.acquireGuidedRetry("guided_reenter", SESSION_A, []));

    expect(retry.operationId).toBe(stored.operationId);
    getItem.mockRestore();
  });

  it("persists an empty generation when removal fails so reload cannot resurrect custody", async () => {
    const completed = acquireHandle("guided_reenter", SESSION_A, []);
    const removeItem = vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });

    clearGuidedRetry(completed);
    removeItem.mockRestore();
    vi.resetModules();
    const reloaded = await import("./guidedOperationRetry");
    const nextAction = expectAcquired(reloaded.acquireGuidedRetry("guided_reenter", SESSION_A, []));

    expect(nextAction.operationId).not.toBe(completed.operationId);
  });

  it("still removes stale custody when the empty-generation write fails", async () => {
    const completed = acquireHandle("guided_reenter", SESSION_A, []);
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
    const nextAction = expectAcquired(reloaded.acquireGuidedRetry("guided_reenter", SESSION_A, []));

    expect(nextAction.operationId).not.toBe(completed.operationId);
  });

  it("rejects non-canonical or oversized session ids", () => {
    expect(() => acquireHandle("guided_reenter", "sess-1", [])).toThrow(
      "canonical UUID",
    );
    expect(() =>
      acquireHandle("guided_reenter", `${SESSION_B}${"x".repeat(256)}`, []),
    ).toThrow("canonical UUID");
  });

  it("retains only network, abort, timeout, and 5xx failures", () => {
    expect(isAmbiguousGuidedRetryFailure(new TypeError("Failed to fetch"))).toBe(true);
    expect(isAmbiguousGuidedRetryFailure(new DOMException("aborted", "AbortError"))).toBe(true);
    expect(isAmbiguousGuidedRetryFailure({ name: "TimeoutError" })).toBe(true);
    expect(isAmbiguousGuidedRetryFailure({ status: 503 })).toBe(true);
    expect(isAmbiguousGuidedRetryFailure({ status: 502, error_type: "proxy_error" })).toBe(true);
    expect(
      isAmbiguousGuidedRetryFailure({
        status: 500,
        error_type: "guided_operation_terminal_failure",
      }),
    ).toBe(false);
    expect(isAmbiguousGuidedRetryFailure({ status: 409 })).toBe(false);
    expect(isAmbiguousGuidedRetryFailure(new Error("application error"))).toBe(false);
  });
});
