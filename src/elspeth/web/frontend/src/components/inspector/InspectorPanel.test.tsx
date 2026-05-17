import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InspectorPanel } from "./InspectorPanel";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { fetchAuditReadiness } from "@/api/auditReadiness";
import { makeComposition } from "@/test/composerFixtures";
import type { CompositionState, CompositionStateVersion } from "@/types/index";

// Mock API client and websocket to prevent real calls
vi.mock("@/api/client", () => ({
  fetchSessions: vi.fn(),
  createSession: vi.fn(),
  fetchMessages: vi.fn(),
  fetchCompositionState: vi.fn(),
  sendMessage: vi.fn(),
  revertToVersion: vi.fn(),
  fetchStateVersions: vi.fn().mockResolvedValue([]),
  archiveSession: vi.fn(),
  validatePipeline: vi.fn(),
  executePipeline: vi.fn(),
  fetchRuns: vi.fn().mockResolvedValue([]),
  fetchYaml: vi.fn().mockResolvedValue({ yaml: "source:\n  plugin: text\n" }),
  cancelExecution: vi.fn(),
  listSources: vi.fn().mockResolvedValue([]),
  listTransforms: vi.fn().mockResolvedValue([]),
  listSinks: vi.fn().mockResolvedValue([]),
  getPluginSchema: vi.fn(),
}));

vi.mock("@/api/websocket", () => ({
  connectToRun: vi.fn().mockReturnValue({ close: vi.fn() }),
}));

vi.mock("@/api/auditReadiness", () => ({
  fetchAuditReadiness: vi.fn(),
  fetchAuditReadinessExplain: vi.fn(),
}));

function makeState(
  overrides: Partial<CompositionState> = {},
): CompositionState {
  return {
    id: "state-1",
    version: 1,
    source: null,
    nodes: [
      {
        id: "t1",
        node_type: "transform" as const,
        plugin: "uppercase",
        input: "source_out",
        on_success: "main",
        on_error: null,
        options: {},
      },
    ],
    edges: [],
    outputs: [],
    metadata: { name: "test", description: "" },
    ...overrides,
  };
}

describe("ValidationDot in InspectorPanel", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: null,
      stateVersions: [],
      isLoadingVersions: false,
    });
    useExecutionStore.setState({
      validationResult: null,
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
  });

  it("shows amber dot when not validated", () => {
    useSessionStore.setState({
      compositionState: makeState(),
    });
    render(<InspectorPanel />);
    const dot = screen.getByLabelText("Not validated");
    expect(dot).toBeInTheDocument();
  });

  it("shows green dot when validation passed", () => {
    useSessionStore.setState({
      compositionState: makeState(),
    });
    useExecutionStore.setState({
      validationResult: { is_valid: true, summary: "All checks passed", checks: [], errors: [], warnings: [] },
    });
    render(<InspectorPanel />);
    const dot = screen.getByLabelText("Validation passed");
    expect(dot).toBeInTheDocument();
  });

  it("shows red dot when validation failed", () => {
    useSessionStore.setState({
      compositionState: makeState(),
    });
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        summary: "Validation failed",
        checks: [],
        errors: [
          {
            component_id: "source",
            component_type: "source",
            message: "Missing source",
            suggestion: null,
          },
        ],
        warnings: [],
      },
    });
    render(<InspectorPanel />);
    const dot = screen.getByLabelText("Validation failed");
    expect(dot).toBeInTheDocument();
  });

  it("hides dot when no pipeline", () => {
    useSessionStore.setState({
      compositionState: null,
    });
    render(<InspectorPanel />);
    expect(screen.queryByLabelText("Not validated")).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Validation passed"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Validation failed"),
    ).not.toBeInTheDocument();
  });

  it("hides dot when pipeline has no nodes", () => {
    useSessionStore.setState({
      compositionState: makeState({ nodes: [] }),
    });
    render(<InspectorPanel />);
    expect(screen.queryByLabelText("Not validated")).not.toBeInTheDocument();
  });
});

describe("InspectorPanel three-state validation indicator", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: {
        id: "state-1",
        version: 1,
        source: { plugin: "csv", options: {} },
        nodes: [],
        edges: [],
        outputs: [{ name: "out", plugin: "json", options: {} }],
        metadata: { name: null, description: null },
      },
      stateVersions: [],
      isLoadingVersions: false,
    });
    useExecutionStore.setState({
      validationResult: null,
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
  });

  it("shows hollow circle when not validated", () => {
    useExecutionStore.setState({ validationResult: null });
    render(<InspectorPanel />);
    expect(screen.getByLabelText("Not validated")).toBeInTheDocument();
  });

  it("shows checkmark for valid pipeline (no warnings)", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        summary: "All checks passed",
        checks: [],
        errors: [],
        warnings: [],
      },
    });
    render(<InspectorPanel />);
    expect(screen.getByLabelText("Validation passed")).toBeInTheDocument();
  });

  it("shows warning indicator for valid-with-warnings", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        summary: "Passed with warnings",
        checks: [],
        errors: [],
        warnings: [
          {
            component_id: "source",
            component_type: "source",
            message: "No explicit schema",
            suggestion: "Add schema",
          },
        ],
      },
    });
    render(<InspectorPanel />);
    expect(screen.getByLabelText("Validation passed with warnings")).toBeInTheDocument();
  });

  it("shows error indicator for invalid pipeline", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        summary: "Validation failed",
        checks: [],
        errors: [
          {
            component_id: "llm",
            component_type: "transform",
            message: "Missing model",
            suggestion: null,
          },
        ],
        warnings: [],
      },
    });
    render(<InspectorPanel />);
    expect(screen.getByLabelText("Validation failed")).toBeInTheDocument();
  });
});

describe("Version selector and catalog", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: null,
      stateVersions: [],
      isLoadingVersions: false,
    });
    useExecutionStore.setState({
      validationResult: null,
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
  });

  it("renders version selector with current version", () => {
    useSessionStore.setState({
      compositionState: makeState({ version: 3 }),
    });
    render(<InspectorPanel />);
    // The VersionSelector trigger button shows "v{N} ▾"
    expect(screen.getByText(/v3/)).toBeInTheDocument();
  });

  it("version dropdown opens on click", async () => {
    const versions: CompositionStateVersion[] = [
      { id: "state-2", version: 2, created_at: "2026-03-31T00:00:00Z", node_count: 5 },
      { id: "state-1", version: 1, created_at: "2026-03-30T00:00:00Z", node_count: 3 },
    ];
    useSessionStore.setState({
      compositionState: makeState({ version: 2 }),
      stateVersions: versions,
    });
    render(<InspectorPanel />);
    const user = userEvent.setup();
    // Click the version trigger button
    const trigger = screen.getByRole("button", { name: /Version 2/ });
    await user.click(trigger);
    // Dropdown listbox should be visible
    expect(screen.getByRole("listbox")).toBeInTheDocument();
  });

  it("keeps version options free of nested actions and wires trigger aria-controls", async () => {
    const { fetchStateVersions } = await import("@/api/client");
    const versions: CompositionStateVersion[] = [
      { id: "state-2", version: 2, created_at: "2026-03-31T00:00:00Z", node_count: 5 },
      { id: "state-1", version: 1, created_at: "2026-03-30T00:00:00Z", node_count: 3 },
    ];
    (fetchStateVersions as ReturnType<typeof vi.fn>).mockResolvedValue(versions);
    useSessionStore.setState({
      compositionState: makeState({ version: 2 }),
      stateVersions: versions,
    });
    render(<InspectorPanel />);
    const user = userEvent.setup();

    const trigger = screen.getByRole("button", { name: /Version 2/ });
    await user.click(trigger);

    const listbox = screen.getByRole("listbox", { name: "Version history" });
    await within(listbox).findByRole("option", { name: /Version 1/ });
    expect(listbox.id).not.toBe("");
    expect(trigger).toHaveAttribute("aria-controls", listbox.id);
    expect(within(listbox).queryByRole("button", { name: /Revert/ })).not.toBeInTheDocument();
  });

  it("reverts the selected non-current version with an action outside the listbox", async () => {
    const { fetchStateVersions } = await import("@/api/client");
    const versions: CompositionStateVersion[] = [
      { id: "state-2", version: 2, created_at: "2026-03-31T00:00:00Z", node_count: 5 },
      { id: "state-1", version: 1, created_at: "2026-03-30T00:00:00Z", node_count: 3 },
    ];
    (fetchStateVersions as ReturnType<typeof vi.fn>).mockResolvedValue(versions);
    useSessionStore.setState({
      compositionState: makeState({ version: 2 }),
      stateVersions: versions,
    });
    render(<InspectorPanel />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Version 2/ }));
    const listbox = screen.getByRole("listbox", { name: "Version history" });
    await user.click(await within(listbox).findByRole("option", { name: /Version 1/ }));
    await user.click(screen.getByRole("button", { name: "Revert selected version 1" }));

    expect(screen.getByRole("alertdialog", { name: "Revert pipeline" })).toBeInTheDocument();
  });

  it("opens the revert confirmation when Enter is pressed on a focused non-current option", async () => {
    const { fetchStateVersions } = await import("@/api/client");
    const versions: CompositionStateVersion[] = [
      { id: "state-2", version: 2, created_at: "2026-03-31T00:00:00Z", node_count: 5 },
      { id: "state-1", version: 1, created_at: "2026-03-30T00:00:00Z", node_count: 3 },
    ];
    (fetchStateVersions as ReturnType<typeof vi.fn>).mockResolvedValue(versions);
    useSessionStore.setState({
      compositionState: makeState({ version: 2 }),
      stateVersions: versions,
    });
    render(<InspectorPanel />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Version 2/ }));
    const listbox = screen.getByRole("listbox", { name: "Version history" });
    await within(listbox).findByRole("option", { name: /Version 1/ });
    await user.keyboard("{ArrowDown}{Enter}");

    expect(screen.getByRole("alertdialog", { name: "Revert pipeline" })).toBeInTheDocument();
  });

  it("catalog button toggles drawer", async () => {
    useSessionStore.setState({
      compositionState: makeState(),
    });
    render(<InspectorPanel />);
    const user = userEvent.setup();

    // Drawer should not be open initially
    expect(screen.queryByText("Plugin Catalog")).not.toBeInTheDocument();

    // Click Catalog button to open
    const catalogBtn = screen.getByRole("button", { name: /Catalog/i });
    await user.click(catalogBtn);
    expect(screen.getByText("Plugin Catalog")).toBeInTheDocument();

    // Click again to close
    await user.click(catalogBtn);
    expect(screen.queryByText("Plugin Catalog")).not.toBeInTheDocument();
  });

  it("opens the catalog drawer when the global open-catalog event fires", () => {
    useSessionStore.setState({
      compositionState: makeState(),
    });
    render(<InspectorPanel />);

    expect(screen.queryByText("Plugin Catalog")).not.toBeInTheDocument();

    fireEvent(window, new CustomEvent("open-catalog"));

    expect(screen.getByText("Plugin Catalog")).toBeInTheDocument();
  });
});

describe("InspectorPanel validation dot colour", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
      stateVersions: [],
      isLoadingVersions: false,
    });
    useExecutionStore.setState({
      validationResult: null,
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
  });

  it("renders unchecked state in muted text colour, not warning orange", () => {
    render(<InspectorPanel />);

    const dot = screen.getByLabelText("Not validated");
    expect(dot.getAttribute("style")).toContain("var(--color-text-muted)");
    expect(dot.getAttribute("style")).not.toContain("var(--color-warning)");
  });

  it("hides the decorative validation symbol from assistive technology", () => {
    render(<InspectorPanel />);

    const dot = screen.getByLabelText("Not validated");
    const symbol = dot.querySelector("[aria-hidden='true']");

    expect(symbol).toHaveTextContent("\u25CB");
  });

  it("renders warning state in warning colour", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        summary: "Passed with warnings",
        checks: [],
        errors: [],
        warnings: [{ component_id: "src", component_type: "source", message: "x", suggestion: null }],
      },
    });

    render(<InspectorPanel />);

    const dot = screen.getByLabelText("Validation passed with warnings");
    expect(dot.getAttribute("style")).toContain("var(--color-warning)");
  });
});

describe("InspectorPanel execution feedback", () => {
  beforeEach(async () => {
    const { executePipeline, fetchRuns } = await import("@/api/client");
    (executePipeline as ReturnType<typeof vi.fn>).mockReset();
    (fetchRuns as ReturnType<typeof vi.fn>).mockReset();
    (fetchRuns as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: {
        id: "state-1",
        version: 1,
        source: { plugin: "text", options: {} },
        nodes: [],
        edges: [],
        outputs: [{ name: "out", plugin: "json", options: {} }],
        metadata: { name: null, description: null },
      },
      stateVersions: [],
      isLoadingVersions: false,
    });
    useExecutionStore.setState({
      runs: [],
      activeRunId: null,
      validationResult: {
        is_valid: true,
        summary: "All checks passed",
        checks: [],
        errors: [],
        warnings: [],
      },
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
  });

  it("does not host Execute after the button moves to the side rail", async () => {
    render(<InspectorPanel />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("tab", { name: "YAML" }));
    expect(screen.getByRole("tab", { name: "YAML" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      screen.queryByRole("button", { name: "Execute pipeline" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Runs" })).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "YAML" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });
});

describe("InspectorPanel Runs tab removal", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
      stateVersions: [],
      isLoadingVersions: false,
    });
    useExecutionStore.setState({
      runs: [],
      activeRunId: null,
      validationResult: {
        is_valid: true,
        summary: "All checks passed",
        checks: [],
        errors: [],
        warnings: [],
      },
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
  });

  it("keeps Graph and YAML while removing Spec and Runs from the tab strip", () => {
    render(<InspectorPanel />);

    const tablist = screen.getByRole("tablist", { name: /Inspector tabs/ });
    expect(within(tablist).getByRole("tab", { name: "Graph" })).toBeInTheDocument();
    expect(within(tablist).getByRole("tab", { name: "YAML" })).toBeInTheDocument();
    expect(within(tablist).queryByRole("tab", { name: "Spec" })).not.toBeInTheDocument();
    expect(within(tablist).queryByRole("tab", { name: "Runs" })).not.toBeInTheDocument();
  });

  it("defaults the inspector to the Graph tab", () => {
    render(<InspectorPanel />);

    expect(screen.getByRole("tab", { name: "Graph" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("keeps the current tab after Execute is removed from the inspector", async () => {
    render(<InspectorPanel />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("tab", { name: "YAML" }));

    expect(
      screen.queryByRole("button", { name: "Execute pipeline" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Runs" })).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "YAML" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("selects the node and moves validation component clicks to Graph", async () => {
    const user = userEvent.setup();
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        summary: "Validation failed",
        checks: [],
        errors: [
          {
            component_id: "t1",
            component_type: "transform",
            message: "Bad transform",
            suggestion: null,
          },
        ],
        warnings: [],
      },
    });

    render(<InspectorPanel />);
    await user.click(screen.getByRole("tab", { name: "YAML" }));
    await user.click(screen.getByRole("button", { name: /transform:t1/ }));

    expect(useSessionStore.getState().selectedNodeId).toBe("t1");
    expect(screen.getByRole("tab", { name: "Graph" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });
});

describe("InspectorPanel aria-live scope", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s1",
      compositionState: makeState(),
      stateVersions: [],
      isLoadingVersions: false,
    });
    useExecutionStore.setState({
      validationResult: null,
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
  });

  it("does not place aria-live on the tab panel container", () => {
    render(<InspectorPanel />);

    const tabPanel = screen.getByRole("tabpanel");
    expect(tabPanel.getAttribute("aria-live")).toBeNull();
  });
});

describe("AuditReadinessPanel mount in InspectorPanel", () => {
  beforeEach(() => {
    // Use the canonical fixture; pass an overrides map to clear nodes when
    // the test wants an empty pipeline. SourceSpec is { plugin, options }
    // (NOT kind/config — that shape never existed; the previous draft of
    // this plan inherited the drift from a stale memo).
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: makeComposition(1, { nodes: [] }),
    } as never);
    useExecutionStore.setState({
      validationResult: null,
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
    vi.mocked(fetchAuditReadiness).mockResolvedValue({
      session_id: "s-1",
      composition_version: 1,
      checked_at: new Date().toISOString(),
      rows: [
        { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
        { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
        { id: "provenance", label: "Provenance", status: "ok", summary: "Complete lineage", detail: null, component_ids: [] },
        { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
        { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
        { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets", detail: null, component_ids: [] },
      ],
      validation_result: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
        semantic_contracts: [],
      },
    });
  });

  it("renders the audit readiness panel inside the inspector", async () => {
    render(<InspectorPanel />);
    await waitFor(() => {
      expect(screen.getByLabelText("Audit readiness")).toBeInTheDocument();
    });
  });

  it("renders the audit readiness panel above the tab strip", async () => {
    render(<InspectorPanel />);
    const panel = await screen.findByLabelText("Audit readiness");
    const tablist = screen.getByRole("tablist", { name: /Inspector tabs/ });
    // compareDocumentPosition returns 4 (DOCUMENT_POSITION_FOLLOWING) when
    // the argument follows the receiver — the panel must come first.
    expect(panel.compareDocumentPosition(tablist)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  // W6 — added 2026-05-17

  it("panel remains present when every tab is activated in turn", async () => {
    // The panel mounts above the tab strip (not inside any tab panel), so it
    // must survive tab switching. We iterate dynamically over whatever tabs
    // are rendered, so this test is resilient to future tab-list changes.
    const user = userEvent.setup();
    render(<InspectorPanel />);
    // Wait for initial mount and panel to appear.
    await screen.findByLabelText("Audit readiness");
    const tabs = screen.getAllByRole("tab");
    for (const tab of tabs) {
      await user.click(tab);
      // Panel must still be present after each tab switch.
      expect(screen.getByLabelText("Audit readiness")).toBeInTheDocument();
    }
  });

  it("renders without crashing when activeSessionId and compositionState are null", () => {
    // Protects against the null-compositionState path omitted from the initial
    // test set. The panel must not throw on a store that has not yet populated.
    useSessionStore.setState({
      activeSessionId: null,
      compositionState: null,
    } as never);
    // Should not throw.
    render(<InspectorPanel />);
    // The panel is either absent or renders an empty/placeholder state — either
    // is correct. The assertion is that no runtime error is thrown and the
    // tablist still renders (InspectorPanel skeleton remains usable).
    expect(screen.getByRole("tablist", { name: /Inspector tabs/ })).toBeInTheDocument();
  });
});

describe("Validate button removal (Phase 2C)", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: makeComposition(1, { nodes: [] }),
    } as never);
    useExecutionStore.setState({
      validationResult: null,
      isValidating: false,
      isExecuting: false,
      progress: null,
      error: null,
    });
  });

  it("does not render a button labelled 'Validate' (subsumed by audit-readiness panel)", () => {
    render(<InspectorPanel />);
    expect(
      screen.queryByRole("button", { name: /^Validate$/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /^Validate pipeline$/ }),
    ).not.toBeInTheDocument();
  });

  it("renders Catalog while Execute is owned by the side rail", () => {
    render(<InspectorPanel />);
    expect(
      screen.queryByRole("button", { name: /^Execute pipeline$/ }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Catalog/ })).toBeInTheDocument();
  });

  it("the tab strip still renders and arrow navigation still works", () => {
    render(<InspectorPanel />);
    const tablist = screen.getByRole("tablist", { name: /Inspector tabs/ });
    expect(tablist).toBeInTheDocument();
    const tabs = within(tablist).getAllByRole("tab");
    expect(tabs.length).toBeGreaterThan(1);
    tabs[0].focus();
    fireEvent.keyDown(tabs[0], { key: "ArrowRight" });
    // The next tab should be focused after arrow navigation; the focus
    // assertion verifies the keyboard navigation hasn't regressed when the
    // header layout changed.
    expect(document.activeElement).toBe(tabs[1]);
  });
});
