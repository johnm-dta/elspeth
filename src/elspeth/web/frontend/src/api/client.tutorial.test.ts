import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cancelTutorialRun,
  deleteTutorialOrphans,
  getRunAuditSummary,
  renameSession,
  runTutorialPipeline,
  sendTutorialAbandonBeacon,
} from "./client";

describe("api/client tutorial helpers", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("POSTs tutorial run requests to /api/tutorial/run keyed by session alone", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          run_id: "run-1",
          output: {
            rows: [{ url: "https://www.australia.gov.au", score: 6 }],
            source_data_hash: "a7f3e2",
            discarded_row_count: 0,
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );

    const result = await runTutorialPipeline({ session_id: "session-1" });

    expect(result.run_id).toBe("run-1");
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/tutorial/run");
    expect(init?.method).toBe("POST");
    // The deleted tutorial cache keyed on `prompt`; the request body carries
    // only the session id now (TutorialRunRequest contract).
    expect(JSON.parse(init?.body as string)).toEqual({
      session_id: "session-1",
    });
  });

  it("surfaces tutorial run backend failures", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "unknown session" }), {
        status: 404,
        statusText: "Not Found",
        headers: { "content-type": "application/json" },
      }),
    );

    await expect(
      runTutorialPipeline({ session_id: "missing" }),
    ).rejects.toMatchObject({ status: 404 });
  });

  it("POSTs the best-effort server cancel with keepalive", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ cancelled: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const result = await cancelTutorialRun("session-1");

    expect(result.cancelled).toBe(true);
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/tutorial/cancel");
    expect(init?.method).toBe("POST");
    // keepalive lets the cancel survive the user closing the tab right after
    // clicking Cancel.
    expect(init?.keepalive).toBe(true);
    expect(JSON.parse(init?.body as string)).toEqual({
      session_id: "session-1",
    });
  });

  it("sends the abandon beacon as an authenticated keepalive POST", () => {
    localStorage.setItem("auth_token", "beacon-token");
    try {
      fetchSpy.mockResolvedValueOnce(new Response(null, { status: 204 }));

      sendTutorialAbandonBeacon();

      const [url, init] = fetchSpy.mock.calls[0];
      expect(url).toBe("/api/tutorial/abandon");
      expect(init?.method).toBe("POST");
      // The endpoint requires the bearer header, which navigator.sendBeacon
      // cannot carry — so the beacon is a keepalive fetch that survives
      // pagehide.
      expect(init?.keepalive).toBe(true);
      expect((init?.headers as Record<string, string>).Authorization).toBe(
        "Bearer beacon-token",
      );
    } finally {
      localStorage.removeItem("auth_token");
    }
  });

  it("swallows abandon-beacon failures (best-effort telemetry)", () => {
    fetchSpy.mockRejectedValueOnce(new TypeError("network down"));
    expect(() => sendTutorialAbandonBeacon()).not.toThrow();
  });

  it("GETs the run audit-story projection by session and run id", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          run_id: "run-1",
          session_id: "session-1",
          llm_call_count: 5,
          source_data_hash: "cafe",
          started_at: "2026-05-19T12:00:00Z",
          plugin_versions: { web_scrape: "1.0.0" },
          seeded_from_cache: true,
          cache_key: "abc123",
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );

    const summary = await getRunAuditSummary("session-1", "run-1");

    expect(summary.llm_call_count).toBe(5);
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/sessions/session-1/runs/run-1/audit-story");
    expect(init?.method).toBeUndefined();
  });

  it("PATCHes session titles through renameSession", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: "session-1",
          title: "hello-world (cool government pages)",
          created_at: "2026-05-19T12:00:00Z",
          updated_at: "2026-05-19T12:00:00Z",
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );

    const session = await renameSession(
      "session-1",
      "hello-world (cool government pages)",
    );

    expect(session.title).toBe("hello-world (cool government pages)");
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/sessions/session-1");
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(init?.body as string)).toEqual({
      title: "hello-world (cool government pages)",
    });
  });

  it("DELETEs tutorial orphans and returns the cleanup count", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ deleted_count: 2 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const result = await deleteTutorialOrphans();

    expect(result.deleted_count).toBe(2);
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/tutorial/orphans");
    expect(init?.method).toBe("DELETE");
  });
});
