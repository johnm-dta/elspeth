import { describe, it, expect, beforeEach, vi } from "vitest";
import { useAuditReadinessStore, getInitialState } from "./auditReadinessStore";
import * as api from "../api/auditReadiness";
import type { AuditReadinessSnapshot, AuditReadinessExplain } from "../types/api";

vi.mock("../api/auditReadiness");

const SESSION_ID = "00000000-0000-0000-0000-000000000001";
const READY_READINESS = {
  authoring_valid: true,
  execution_ready: true,
  completion_ready: true,
  blockers: [],
};

function snapshot(version: number): AuditReadinessSnapshot {
  return {
    session_id: SESSION_ID,
    composition_version: version,
    checked_at: new Date().toISOString(),
    rows: [
      { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
      { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
      { id: "provenance", label: "Provenance", status: "ok", summary: "Complete lineage", detail: null, component_ids: [] },
      { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
      { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
      { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets referenced", detail: null, component_ids: [] },
    ],
    validation_result: {
      is_valid: true,
      checks: [],
      errors: [],
      warnings: [],
      readiness: READY_READINESS,
      semantic_contracts: [],
    },
  };
}

describe("useAuditReadinessStore", () => {
  beforeEach(() => {
    useAuditReadinessStore.setState(getInitialState());
    vi.clearAllMocks();
  });

  it("loadSnapshot fetches and stores by sessionId", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(snapshot(1));

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);

    const state = useAuditReadinessStore.getState();
    expect(state.snapshotsBySession[SESSION_ID]?.composition_version).toBe(1);
    expect(state.isLoadingBySession[SESSION_ID]).toBe(false);
    expect(state.errorBySession[SESSION_ID]).toBeNull();
  });

  it("loadSnapshot is a no-op when the cached snapshot's version matches", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(snapshot(2));

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);

    expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(1);
  });

  it("loadSnapshot force option bypasses a matching-version cached snapshot", async () => {
    vi.mocked(api.fetchAuditReadiness)
      .mockResolvedValueOnce(snapshot(2))
      .mockResolvedValueOnce(snapshot(2));

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2, { force: true });

    expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(2);
  });

  it("loadSnapshot refetches when the version advances", async () => {
    vi.mocked(api.fetchAuditReadiness)
      .mockResolvedValueOnce(snapshot(1))
      .mockResolvedValueOnce(snapshot(2));

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);

    expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(2);
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]
        ?.composition_version,
    ).toBe(2);
  });

  it("loadSnapshot stores error on 404", async () => {
    vi.mocked(api.fetchAuditReadiness).mockRejectedValueOnce({
      status: 404,
      detail: "No composition state for this session",
    });

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);

    const state = useAuditReadinessStore.getState();
    expect(state.errorBySession[SESSION_ID]).toContain("No composition state");
    expect(state.snapshotsBySession[SESSION_ID]).toBeUndefined();
  });

  it("loadExplain fetches narrative and caches by version", async () => {
    const expl: AuditReadinessExplain = {
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "When you run this pipeline, ELSPETH will record …",
    };
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce(expl);

    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);
    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);

    expect(api.fetchAuditReadinessExplain).toHaveBeenCalledTimes(1);
    expect(
      useAuditReadinessStore.getState().explainsBySession[SESSION_ID]?.narrative,
    ).toContain("ELSPETH will record");
  });

  it("loadExplain refetches when the version advances", async () => {
    vi.mocked(api.fetchAuditReadinessExplain)
      .mockResolvedValueOnce({ session_id: SESSION_ID, composition_version: 1, narrative: "v1 text" })
      .mockResolvedValueOnce({ session_id: SESSION_ID, composition_version: 2, narrative: "v2 text" });

    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);
    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 2);

    expect(api.fetchAuditReadinessExplain).toHaveBeenCalledTimes(2);
    expect(
      useAuditReadinessStore.getState().explainsBySession[SESSION_ID]?.narrative,
    ).toBe("v2 text");
  });

  it("clearSession removes both snapshot and explain", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(snapshot(1));
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "text",
    });
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);

    useAuditReadinessStore.getState().clearSession(SESSION_ID);

    const state = useAuditReadinessStore.getState();
    expect(state.snapshotsBySession[SESSION_ID]).toBeUndefined();
    expect(state.explainsBySession[SESSION_ID]).toBeUndefined();
  });

  it("clearSession during in-flight fetch aborts the request and resets per-session loading", async () => {
    let reject!: (err: unknown) => void;
    vi.mocked(api.fetchAuditReadiness).mockReturnValueOnce(
      new Promise<AuditReadinessSnapshot>((_, rej) => {
        reject = rej;
      }),
    );

    // Start the fetch but do not await it — leaves the AbortController in-flight.
    const inFlight = useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    const ctrl = useAuditReadinessStore.getState().abortControllers[SESSION_ID];
    expect(ctrl).toBeDefined();

    // clearSession must abort the controller, remove all per-session entries.
    useAuditReadinessStore.getState().clearSession(SESSION_ID);
    expect(ctrl?.signal.aborted).toBe(true);

    // Simulate the native fetch surfacing AbortError after abort.
    reject(Object.assign(new Error("AbortError"), { name: "AbortError" }));
    await inFlight;

    const state = useAuditReadinessStore.getState();
    expect(state.snapshotsBySession[SESSION_ID]).toBeUndefined();
    expect(state.abortControllers[SESSION_ID]).toBeUndefined();
    expect(state.isLoadingBySession[SESSION_ID]).toBeUndefined();
    expect(state.errorBySession[SESSION_ID]).toBeUndefined();
  });

  it("tracked snapshot abort removes the controller and clears loading", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce((_sid, signal) => (
      new Promise<AuditReadinessSnapshot>((_, reject) => {
        signal?.addEventListener("abort", () => {
          reject(Object.assign(new Error("AbortError"), { name: "AbortError" }));
        });
      })
    ));

    const inFlight = useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    const ctrl = useAuditReadinessStore.getState().abortControllers[SESSION_ID];
    expect(ctrl).toBeDefined();

    ctrl?.abort();
    await inFlight;

    const state = useAuditReadinessStore.getState();
    expect(state.abortControllers[SESSION_ID]).toBeUndefined();
    expect(state.isLoadingBySession[SESSION_ID]).toBe(false);
  });

  it("tracked explain abort removes the controller and clears loading", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockImplementationOnce((_sid, signal) => (
      new Promise<AuditReadinessExplain>((_, reject) => {
        signal?.addEventListener("abort", () => {
          reject(Object.assign(new Error("AbortError"), { name: "AbortError" }));
        });
      })
    ));

    const inFlight = useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);
    const ctrl = useAuditReadinessStore.getState().explainAbortControllers[SESSION_ID];
    expect(ctrl).toBeDefined();

    ctrl?.abort();
    await inFlight;

    const state = useAuditReadinessStore.getState();
    expect(state.explainAbortControllers[SESSION_ID]).toBeUndefined();
    expect(state.isLoadingExplainBySession[SESSION_ID]).toBe(false);
  });

  // --- Monotonic write-guard contract ---
  // This test exercises the version monotonicity guard (loadSnapshot discards
  // a response whose composition_version is lower than what's already cached).
  // It is SEQUENTIAL: v2 completes before v1 starts. It does NOT exercise the
  // AbortController cancellation path — see the abort-cancellation test below.
  it("monotonic write guard — sequential ordering: fast-v2 + slow-v1 interleaved — v2 payload wins", async () => {
    // Deferred resolver pattern: v2 starts first and resolves first; v1
    // starts after v2 but resolves last (simulating a slow in-flight v1 that
    // was superseded by a version bump).
    let resolveV1!: (s: AuditReadinessSnapshot) => void;
    const slowV1 = new Promise<AuditReadinessSnapshot>((res) => { resolveV1 = res; });

    vi.mocked(api.fetchAuditReadiness)
      .mockReturnValueOnce(Promise.resolve(snapshot(2)))   // fast v2
      .mockReturnValueOnce(slowV1);                        // slow v1

    // Kick off v2 fetch first — await it fully before starting v1.
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]?.composition_version,
    ).toBe(2);

    // Start v1 fetch (simulates a race from a component that received an
    // older compositionVersion).
    const v1Promise = useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    resolveV1(snapshot(1)); // now the slow v1 resolve arrives
    await v1Promise;

    // Monotonic guard must have discarded v1 — v2 still wins.
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]?.composition_version,
    ).toBe(2);
    // Invariant: abortControllers holds only in-flight controllers. The
    // resolved-and-discarded v1 controller must not linger after the guard
    // arm fires.
    expect(
      useAuditReadinessStore.getState().abortControllers[SESSION_ID],
    ).toBeUndefined();
  });

  // --- AbortController cancellation contract ---
  // This test exercises the abort path: when a second loadSnapshot call starts
  // while the first is still in-flight, the store must abort the first fetch's
  // AbortController and clear isLoadingBySession after the second completes.
  // It is CONCURRENT: both fetches are in-flight simultaneously. It covers a
  // different contract from the monotonic-guard test above — both are required.
  it("abort-cancellation: second in-flight fetch aborts the first; isLoadingBySession resets cleanly", async () => {
    let resolveFirst!: (s: AuditReadinessSnapshot) => void;
    let rejectFirst!: (err: unknown) => void;
    const firstFetch = new Promise<AuditReadinessSnapshot>((res, rej) => {
      resolveFirst = res;
      rejectFirst = rej;
    });

    vi.mocked(api.fetchAuditReadiness)
      .mockReturnValueOnce(firstFetch)                          // slow first fetch
      .mockReturnValueOnce(Promise.resolve(snapshot(2)));       // fast second fetch

    // Start BOTH fetches without awaiting the first — they are concurrently in-flight.
    // Capture the first controller (from store state) before the second call overwrites it.
    // (loadSnapshot sets abortControllers[sessionId] synchronously inside its set() call.)
    const firstPromise = useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    const storedCtrl = useAuditReadinessStore.getState().abortControllers[SESSION_ID];

    // Now start the second fetch — this aborts the first controller.
    const secondPromise = useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);

    // The first fetch's AbortController must have been aborted.
    expect(storedCtrl?.signal.aborted).toBe(true);

    // Reject the first fetch with an AbortError (simulating native fetch abort).
    rejectFirst(Object.assign(new Error("AbortError"), { name: "AbortError" }));
    await firstPromise;

    // Complete the second fetch.
    await secondPromise;

    // isLoadingBySession[SESSION_ID] must be false — the abort arm must have cleared it.
    expect(useAuditReadinessStore.getState().isLoadingBySession[SESSION_ID]).toBe(false);
    // Second fetch's result must be stored.
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]?.composition_version,
    ).toBe(2);

    // Suppress unused-variable warning from the unused resolve binding.
    void resolveFirst;
  });
});
