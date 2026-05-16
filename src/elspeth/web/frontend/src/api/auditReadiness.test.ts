import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  fetchAuditReadiness,
  fetchAuditReadinessExplain,
} from "./auditReadiness";

const SESSION_ID = "00000000-0000-0000-0000-000000000001";

describe("auditReadiness API client", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  it("fetchAuditReadiness GETs the right URL with auth header", async () => {
    const body = {
      session_id: SESSION_ID,
      composition_version: 3,
      rows: [
        { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
        { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
        { id: "provenance", label: "Provenance", status: "warning", summary: "Identity passthrough detected", detail: "Identity passthrough — provenance gap on transform 'select_columns'.", component_ids: ["select_columns"] },
        { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
        { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms in this pipeline", detail: null, component_ids: [] },
        { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secret references in this pipeline", detail: null, component_ids: [] },
      ],
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } }),
    );

    const snapshot = await fetchAuditReadiness(SESSION_ID);
    expect(snapshot.composition_version).toBe(3);
    expect(snapshot.rows).toHaveLength(6);
    expect(snapshot.rows[2].status).toBe("warning");

    const mock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const [url, init] = mock.mock.calls[0];
    expect(url).toBe(`/api/sessions/${SESSION_ID}/audit-readiness`);
    expect(init?.method).toBe("GET");
  });

  it("fetchAuditReadiness throws on non-2xx", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response("server error", { status: 500 }),
    );
    await expect(fetchAuditReadiness(SESSION_ID)).rejects.toThrow();
  });

  it("fetchAuditReadiness propagates 404 (session missing or no state) as ApiError", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify({ detail: "No composition state for this session" }),
        { status: 404, headers: { "content-type": "application/json" } },
      ),
    );
    await expect(fetchAuditReadiness(SESSION_ID)).rejects.toMatchObject({ status: 404 });
  });

  it("fetchAuditReadinessExplain GETs the explain URL and returns narrative", async () => {
    const body = {
      session_id: SESSION_ID,
      composition_version: 3,
      narrative: "When you run this pipeline, ELSPETH will record:\n\n• Source data — 5 URLs ...",
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } }),
    );

    const explain = await fetchAuditReadinessExplain(SESSION_ID);
    expect(explain.narrative).toContain("ELSPETH will record");
    expect(explain.composition_version).toBe(3);

    const mock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const [url, init] = mock.mock.calls[0];
    expect(url).toBe(`/api/sessions/${SESSION_ID}/audit-readiness/explain`);
    expect(init?.method).toBe("GET");
  });

  it("fetchAuditReadinessExplain throws on non-2xx", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response("nope", { status: 500 }),
    );
    await expect(fetchAuditReadinessExplain(SESSION_ID)).rejects.toThrow();
  });

  it("fetchAuditReadiness propagates network TypeError", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new TypeError("Failed to fetch"),
    );
    await expect(fetchAuditReadiness(SESSION_ID)).rejects.toThrow(TypeError);
  });

  it("fetchAuditReadiness propagates 401 as ApiError with status 401", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Unauthorized" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      }),
    );
    await expect(fetchAuditReadiness(SESSION_ID)).rejects.toMatchObject({ status: 401 });
  });

  it("fetchAuditReadiness throws when the response body has wrong shape", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify({ unexpected: "shape" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    await expect(fetchAuditReadiness(SESSION_ID)).rejects.toMatchObject({
      detail: expect.stringMatching(/Unexpected response shape/),
    });
  });
});
