import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { SharedInspectView } from "./SharedInspectView";
import * as api from "@/api/shareableReviews";

const _validReadiness = {
  session_id: "00000000-0000-0000-0000-000000000001",
  composition_version: 3,
  checked_at: "2026-05-19T00:00:00+00:00",
  rows: [
    { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
    { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All tier 1/2", detail: null, component_ids: [] },
    { id: "provenance", label: "Provenance", status: "warning", summary: "Identity passthrough", detail: "passthrough", component_ids: ["t1"] },
    { id: "retention", label: "Retention", status: "not_applicable", summary: "90 days", detail: null, component_ids: [] },
    { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM", detail: null, component_ids: [] },
    { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets", detail: null, component_ids: [] },
  ],
  validation_result: {
    is_valid: true,
    checks: [],
    errors: [],
    warnings: [],
    semantic_contracts: [],
  },
};

const _validResponse = {
  session_id: "00000000-0000-0000-0000-000000000001",
  state_id: "11111111-1111-1111-1111-111111111111",
  pipeline_metadata: { name: "Demo Pipeline", description: "A test" },
  composition_snapshot: { version: 1, nodes: [], edges: [], outputs: [] },
  yaml: "version: 1\nname: Demo\n",
  audit_readiness: _validReadiness,
  created_by_user_id: "alice",
  created_at: "2026-05-19T00:00:00+00:00",
  expires_at: "2026-06-19T00:00:00+00:00",
} as never;

describe("SharedInspectView", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the loading state while the request is in flight", () => {
    vi.spyOn(api, "fetchSharedInspect").mockImplementation(
      () => new Promise(() => {}), // never resolves
    );
    render(<SharedInspectView token="abc" />);
    expect(screen.getByTestId("shared-inspect-loading")).toBeInTheDocument();
  });

  it("renders the pipeline metadata + YAML + audit-readiness panel on success", async () => {
    vi.spyOn(api, "fetchSharedInspect").mockResolvedValueOnce(_validResponse);
    render(<SharedInspectView token="abc" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-loaded")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("shared-inspect-pipeline-name")).toHaveTextContent("Demo Pipeline");
    expect(screen.getByTestId("shared-inspect-pipeline-description")).toHaveTextContent("A test");
    expect(screen.getByTestId("shared-inspect-yaml")).toHaveTextContent("version: 1");
    // Six readiness rows visible.
    expect(screen.getByTestId("shared-inspect-readiness-row-validation")).toBeInTheDocument();
    expect(screen.getByTestId("shared-inspect-readiness-row-plugin_trust")).toBeInTheDocument();
    expect(screen.getByTestId("shared-inspect-readiness-row-provenance")).toBeInTheDocument();
    expect(screen.getByTestId("shared-inspect-readiness-row-retention")).toBeInTheDocument();
    expect(screen.getByTestId("shared-inspect-readiness-row-llm_interpretations")).toBeInTheDocument();
    expect(screen.getByTestId("shared-inspect-readiness-row-secrets")).toBeInTheDocument();
  });

  it("renders the 401 error path for tampered/expired tokens", async () => {
    vi.spyOn(api, "fetchSharedInspect").mockRejectedValueOnce({
      status: 401,
      detail: "Invalid or expired share token",
    });
    render(<SharedInspectView token="bad-token" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-error")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("shared-inspect-error")).toHaveTextContent(/no longer valid/i);
    expect(screen.getByTestId("shared-inspect-return-link")).toBeInTheDocument();
  });

  it("renders the 404 error path for expired blobs", async () => {
    vi.spyOn(api, "fetchSharedInspect").mockRejectedValueOnce({
      status: 404,
      detail: "Shared snapshot is no longer available",
    });
    render(<SharedInspectView token="stale-token" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-error")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("shared-inspect-error")).toHaveTextContent(
      /no longer available|fresh link/i,
    );
  });

  it("renders a generic error path for network failures", async () => {
    vi.spyOn(api, "fetchSharedInspect").mockRejectedValueOnce(new Error("network down"));
    render(<SharedInspectView token="any" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-error")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("shared-inspect-error")).toHaveTextContent("network down");
  });

  it("re-fetches when the token prop changes", async () => {
    const fetchSpy = vi
      .spyOn(api, "fetchSharedInspect")
      .mockResolvedValue(_validResponse);
    const { rerender } = render(<SharedInspectView token="t1" />);
    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(1));
    rerender(<SharedInspectView token="t2" />);
    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(2));
    expect(fetchSpy.mock.calls[0][0]).toBe("t1");
    expect(fetchSpy.mock.calls[1][0]).toBe("t2");
  });
});
