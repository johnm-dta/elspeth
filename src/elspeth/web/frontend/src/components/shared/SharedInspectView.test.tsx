import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";
import { SharedInspectView } from "./SharedInspectView";
import { ReadOnlyProvider, useReadOnly } from "@/contexts/ReadOnlyContext";
import * as api from "@/api/shareableReviews";
import type {
  AuditReadinessSnapshot,
  SharedInspectResponse,
} from "@/types/api";

expect.extend(toHaveNoViolations);

const _validReadiness: AuditReadinessSnapshot = {
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

const _validResponse: SharedInspectResponse = {
  session_id: "00000000-0000-0000-0000-000000000001",
  state_id: "11111111-1111-1111-1111-111111111111",
  pipeline_metadata: { name: "Demo Pipeline", description: "A test" },
  composition_snapshot: { version: 1, nodes: [], edges: [], outputs: [] },
  yaml: "version: 1\nname: Demo\n",
  audit_readiness: _validReadiness,
  created_by_user_id: "alice",
  created_at: "2026-05-19T00:00:00+00:00",
  expires_at: "2026-06-19T00:00:00+00:00",
};

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

  // FIX-C: structural assertions for the refactored composition that
  // uses SharedAuditReadinessPanel / YamlDisplay / GraphMiniView
  // primitives wrapped in ReadOnlyProvider.

  it("renders the SharedAuditReadinessPanel primitive (not an inline table)", async () => {
    vi.spyOn(api, "fetchSharedInspect").mockResolvedValueOnce(_validResponse);
    render(<SharedInspectView token="abc" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-loaded")).toBeInTheDocument(),
    );
    expect(
      screen.getByTestId("shared-audit-readiness-panel"),
    ).toBeInTheDocument();
    // The pre-FIX-C inline 3-column table is gone — verify there's no
    // <th>Status</th> column header anymore (we don't render a table).
    expect(
      screen.queryByRole("columnheader", { name: /^Status$/i }),
    ).not.toBeInTheDocument();
  });

  it("renders the YamlDisplay primitive in place of the inline <pre>", async () => {
    vi.spyOn(api, "fetchSharedInspect").mockResolvedValueOnce(_validResponse);
    render(<SharedInspectView token="abc" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-loaded")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("yaml-display")).toBeInTheDocument();
    // The Copy / Download chrome from YamlDisplay must be present.
    expect(
      screen.getByRole("button", { name: "Copy YAML to clipboard" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Download YAML file" }),
    ).toBeInTheDocument();
  });

  it("renders the GraphMiniView primitive with the frozen composition", async () => {
    vi.spyOn(api, "fetchSharedInspect").mockResolvedValueOnce(_validResponse);
    render(<SharedInspectView token="abc" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-loaded")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("shared-inspect-graph")).toBeInTheDocument();
    // The fixture's composition_snapshot has no source / nodes / outputs
    // — the mini view renders its empty-state badge in that case (using
    // the OVERRIDE, not the store, which is what we're verifying: the
    // composer's session store is not populated in this test, so an
    // un-overridden GraphMiniView would also be empty — to disambiguate,
    // use a fixture with non-empty composition in the next test).
    expect(screen.getByTestId("graph-mini-empty")).toBeInTheDocument();
  });

  it("renders the populated GraphMiniView when the composition_snapshot has nodes", async () => {
    const populated: SharedInspectResponse = {
      ..._validResponse,
      composition_snapshot: {
        version: 1,
        source: { plugin: "csv", options: {} },
        nodes: [
          {
            id: "tx-1",
            node_type: "transform",
            plugin: "field_mapper",
            options: {},
          },
        ],
        edges: [],
        outputs: [{ name: "out", plugin: "stdout", options: {} }],
      },
    };
    vi.spyOn(api, "fetchSharedInspect").mockResolvedValueOnce(populated);
    render(<SharedInspectView token="abc" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-loaded")).toBeInTheDocument(),
    );
    // Populated override → the clickable wrapper renders (not empty).
    expect(
      screen.getByRole("button", { name: /pipeline graph/i }),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("graph-mini-empty")).not.toBeInTheDocument();
  });

  it("propagates a true read-only signal through ReadOnlyProvider", async () => {
    function ReadOnlyProbe(): JSX.Element {
      const readOnly = useReadOnly();
      return <span data-testid="read-only-probe">{String(readOnly)}</span>;
    }
    // Inject the probe by wrapping the test render in a custom shell —
    // since we can't pass children to SharedInspectView, we verify the
    // contract by mounting a probe inside the same provider tree that
    // SharedInspectView establishes. The probe component itself is
    // identical to one inside a ReadOnlyProvider wrapper.
    //
    // The contract here is that SharedInspectView wraps its loaded
    // render in <ReadOnlyProvider value={true}> — this is exercised by
    // every existing rendered-tree test (the absence of clickable
    // audit-readiness buttons is the indirect proof). Add a direct
    // assertion by patching the loaded render via composition: render
    // the view, then walk the DOM for the data-testid the panel emits
    // and confirm no <button> within it.
    const provenanceErrorResponse: SharedInspectResponse = {
      ..._validResponse,
      audit_readiness: {
        ..._validResponse.audit_readiness,
        rows: _validResponse.audit_readiness.rows.map((r) =>
          r.id === "provenance" ? { ...r, status: "error" as const } : r,
        ),
      },
    };
    vi.spyOn(api, "fetchSharedInspect").mockResolvedValueOnce(
      provenanceErrorResponse,
    );
    render(
      <>
        <ReadOnlyProvider value={false}>
          <ReadOnlyProbe />
        </ReadOnlyProvider>
        <SharedInspectView token="abc" />
      </>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-loaded")).toBeInTheDocument(),
    );
    // The probe outside SharedInspectView reads false (its explicit
    // wrapper). Inside SharedInspectView, ReadOnlyProvider value={true}
    // is active — verify by checking that the provenance row (now
    // status=error, which would be a clickable button in the composer
    // path) renders as a static group, not a button.
    expect(screen.getByTestId("read-only-probe")).toHaveTextContent("false");
    expect(
      screen.queryByRole("button", { name: /provenance/i }),
    ).not.toBeInTheDocument();
  });

  // Plan line 470: jest-axe accessibility assertion on the shared
  // inspect view. The loaded state is the user-visible terminal happy
  // path — every reviewer who follows a shared link lands here.
  it("has no accessibility violations in the loaded state", async () => {
    vi.spyOn(api, "fetchSharedInspect").mockResolvedValueOnce(_validResponse);
    const { container } = render(<SharedInspectView token="abc" />);
    await waitFor(() =>
      expect(screen.getByTestId("shared-inspect-loaded")).toBeInTheDocument(),
    );
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
