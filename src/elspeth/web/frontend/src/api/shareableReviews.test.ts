import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  fetchShareableLink,
  fetchSharedInspect,
  markReadyForReview,
} from "./shareableReviews";
import { useAuthStore } from "../stores/authStore";

const SESSION_ID = "00000000-0000-0000-0000-000000000001";
const READY_READINESS = {
  authoring_valid: true,
  execution_ready: true,
  completion_ready: true,
  blockers: [],
};

const _validReadinessSnapshot = {
  session_id: SESSION_ID,
  composition_version: 3,
  checked_at: new Date().toISOString(),
  rows: [
    { id: "validation", label: "Validation", status: "ok", summary: "ok", detail: null, component_ids: [] },
    { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "ok", detail: null, component_ids: [] },
    { id: "provenance", label: "Provenance", status: "ok", summary: "ok", detail: null, component_ids: [] },
    { id: "retention", label: "Retention", status: "ok", summary: "ok", detail: null, component_ids: [] },
    { id: "llm_interpretations", label: "LLM interpretations", status: "ok", summary: "ok", detail: null, component_ids: [] },
    { id: "secrets", label: "Secrets", status: "ok", summary: "ok", detail: null, component_ids: [] },
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

describe("shareableReviews API client", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
    localStorage.clear();
    useAuthStore.setState({
      token: null,
      user: null,
      loginError: null,
      isLoading: false,
    } as never);
  });

  // ── markReadyForReview ──────────────────────────────────────────────

  it("markReadyForReview POSTs the right URL with auth header", async () => {
    localStorage.setItem("auth_token", "test-token");
    try {
      const body = {
        token: "ZmFrZS10b2tlbg",
        share_url: "/#/shared/ZmFrZS10b2tlbg",
        expires_at: "2026-06-19T00:00:00+00:00",
        payload_digest: "sha256:" + "ab".repeat(32),
      };
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      const response = await markReadyForReview(SESSION_ID);
      expect(response.token).toBe("ZmFrZS10b2tlbg");
      expect(response.payload_digest.startsWith("sha256:")).toBe(true);

      const mock = globalThis.fetch as ReturnType<typeof vi.fn>;
      const [url, init] = mock.mock.calls[0];
      expect(url).toBe(`/api/sessions/${SESSION_ID}/mark-ready-for-review`);
      expect(init?.method).toBe("POST");
      const headers = init?.headers as Record<string, string> | undefined;
      expect(headers?.Authorization).toBe("Bearer test-token");
    } finally {
      localStorage.removeItem("auth_token");
    }
  });

  it("markReadyForReview rejects malformed responses (missing token)", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          share_url: "/#/shared/abc",
          expires_at: "2026-06-19T00:00:00+00:00",
          payload_digest: "sha256:" + "ab".repeat(32),
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    await expect(markReadyForReview(SESSION_ID)).rejects.toMatchObject({
      detail: expect.stringContaining("mark-ready-for-review"),
    });
  });

  // ── fetchShareableLink ──────────────────────────────────────────────

  it("fetchShareableLink GETs the right URL and parses the response", async () => {
    const body = {
      token: "tk-1",
      share_url: "/#/shared/tk-1",
      expires_at: "2026-06-19T00:00:00+00:00",
      state_id: "11111111-1111-1111-1111-111111111111",
      payload_digest: "sha256:" + "cd".repeat(32),
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const response = await fetchShareableLink(SESSION_ID);
    expect(response.state_id).toBe("11111111-1111-1111-1111-111111111111");
    expect(response.payload_digest.startsWith("sha256:")).toBe(true);

    const mock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const [url, init] = mock.mock.calls[0];
    expect(url).toBe(`/api/sessions/${SESSION_ID}/shareable-link`);
    expect(init?.method).toBe("GET");
  });

  it("fetchShareableLink rejects malformed responses (missing state_id)", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          token: "tk",
          share_url: "/#/shared/tk",
          expires_at: "2026-06-19T00:00:00+00:00",
          payload_digest: "sha256:" + "cd".repeat(32),
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    await expect(fetchShareableLink(SESSION_ID)).rejects.toMatchObject({
      detail: expect.stringContaining("shareable-link"),
    });
  });

  // ── fetchSharedInspect ──────────────────────────────────────────────

  it("fetchSharedInspect GETs /shared/{token} URL-encoded", async () => {
    const token = "abc/def+ghi=";
    const body = {
      session_id: SESSION_ID,
      state_id: "22222222-2222-2222-2222-222222222222",
      pipeline_metadata: { name: "Demo", description: "" },
      composition_snapshot: {
        version: 1,
        metadata: { name: "Demo", description: "" },
        sources: {},
        nodes: [],
        edges: [],
        outputs: [],
      },
      yaml: "version: 1\n",
      audit_readiness: _validReadinessSnapshot,
      created_by_user_id: "alice",
      created_at: "2026-05-19T00:00:00+00:00",
      expires_at: "2026-06-19T00:00:00+00:00",
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const response = await fetchSharedInspect(token);
    expect(response.session_id).toBe(SESSION_ID);
    expect(response.audit_readiness.rows).toHaveLength(6);

    const mock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const [url] = mock.mock.calls[0];
    // URL-encoded so '/' and '+' don't break the path.
    expect(url).toBe(`/api/sessions/shared/${encodeURIComponent(token)}`);
  });

  it("fetchSharedInspect rejects malformed audit_readiness (not a record)", async () => {
    const body = {
      session_id: SESSION_ID,
      state_id: "22222222-2222-2222-2222-222222222222",
      pipeline_metadata: { name: "Demo", description: "" },
      composition_snapshot: {
        version: 1,
        metadata: { name: "Demo", description: "" },
        sources: {},
        nodes: [],
        edges: [],
        outputs: [],
      },
      yaml: "version: 1\n",
      audit_readiness: "not-an-object", // wrong type
      created_by_user_id: "alice",
      created_at: "2026-05-19T00:00:00+00:00",
      expires_at: "2026-06-19T00:00:00+00:00",
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  // ── FIX-K: tighter Tier-3 → Tier-1 validation at the API boundary ─────

  function _buildValidSharedInspectBody(
    overrides: Partial<Record<string, unknown>> = {},
  ): Record<string, unknown> {
    return {
      session_id: SESSION_ID,
      state_id: "22222222-2222-2222-2222-222222222222",
      pipeline_metadata: { name: "Demo", description: "" },
      composition_snapshot: {
        version: 1,
        metadata: { name: "Demo", description: "" },
        sources: {},
        nodes: [],
        edges: [],
        outputs: [],
      },
      yaml: "version: 1\n",
      audit_readiness: _validReadinessSnapshot,
      created_by_user_id: "alice",
      created_at: "2026-05-19T00:00:00+00:00",
      expires_at: "2026-06-19T00:00:00+00:00",
      ...overrides,
    };
  }

  function _mockJsonResponse(body: unknown, status = 200): void {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
      }),
    );
  }

  it("fetchSharedInspect accepts a fully-shaped happy-path body (smoke)", async () => {
    _mockJsonResponse(_buildValidSharedInspectBody());
    const response = await fetchSharedInspect("happy-token");
    expect(response.audit_readiness.rows).toHaveLength(6);
    expect(response.pipeline_metadata.name).toBe("Demo");
    expect(response.composition_snapshot.nodes).toEqual([]);
  });

  it("fetchSharedInspect rejects audit_readiness with malformed row (missing required row fields)", async () => {
    const bad_readiness = {
      ..._validReadinessSnapshot,
      rows: [{ id: "validation" }], // missing label/status/summary/detail/component_ids
    };
    _mockJsonResponse(_buildValidSharedInspectBody({ audit_readiness: bad_readiness }));
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  it("fetchSharedInspect rejects audit_readiness with non-object row (null)", async () => {
    const bad_readiness = { ..._validReadinessSnapshot, rows: [null] };
    _mockJsonResponse(_buildValidSharedInspectBody({ audit_readiness: bad_readiness }));
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  it("fetchSharedInspect rejects audit_readiness with non-object row (string)", async () => {
    const bad_readiness = { ..._validReadinessSnapshot, rows: ["string-row"] };
    _mockJsonResponse(_buildValidSharedInspectBody({ audit_readiness: bad_readiness }));
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  it("fetchSharedInspect rejects pipeline_metadata with missing name", async () => {
    _mockJsonResponse(
      _buildValidSharedInspectBody({ pipeline_metadata: { description: "" } }),
    );
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  it("fetchSharedInspect rejects pipeline_metadata with wrong type for name", async () => {
    _mockJsonResponse(
      _buildValidSharedInspectBody({ pipeline_metadata: { name: 123, description: "" } }),
    );
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  it("fetchSharedInspect rejects composition_snapshot with missing nodes", async () => {
    _mockJsonResponse(
      _buildValidSharedInspectBody({
        composition_snapshot: {
          version: 1,
          metadata: { name: "Demo", description: "" },
          sources: {},
          edges: [],
          outputs: [],
        },
      }),
    );
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  it("fetchSharedInspect rejects composition_snapshot with wrong type for nodes", async () => {
    _mockJsonResponse(
      _buildValidSharedInspectBody({
        composition_snapshot: {
          version: 1,
          metadata: { name: "Demo", description: "" },
          sources: {},
          nodes: "not-array",
          edges: [],
          outputs: [],
        },
      }),
    );
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  it("fetchSharedInspect rejects composition_snapshot with missing metadata", async () => {
    _mockJsonResponse(
      _buildValidSharedInspectBody({
        composition_snapshot: {
          version: 1,
          sources: {},
          nodes: [],
          edges: [],
          outputs: [],
        },
      }),
    );
    await expect(fetchSharedInspect("any-token")).rejects.toMatchObject({
      detail: expect.stringContaining("shared-inspect"),
    });
  });

  it("fetchSharedInspect treats 401 as a capability-token error without clearing auth", async () => {
    localStorage.setItem("auth_token", "reviewer-session-token");
    useAuthStore.setState({
      token: "reviewer-session-token",
      user: {
        user_id: "reviewer",
        username: "reviewer",
        display_name: null,
        email: null,
        groups: [],
      },
      loginError: null,
      isLoading: false,
    } as never);
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Invalid or expired share token" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      }),
    );
    await expect(fetchSharedInspect("bad-token")).rejects.toMatchObject({
      status: 401,
    });
    expect(useAuthStore.getState().token).toBe("reviewer-session-token");
    expect(useAuthStore.getState().user?.username).toBe("reviewer");
    expect(localStorage.getItem("auth_token")).toBe("reviewer-session-token");
  });

  it("fetchSharedInspect propagates 404 from the parser (blob expired)", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          detail: "Shared snapshot is no longer available; ask the sender for a fresh link",
        }),
        { status: 404, headers: { "content-type": "application/json" } },
      ),
    );
    await expect(fetchSharedInspect("good-but-stale-token")).rejects.toMatchObject({
      status: 404,
    });
  });
});
