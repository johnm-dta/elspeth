import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InspectorPanel } from "./InspectorPanel";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
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

  it("switches to Runs after Execute starts a run", async () => {
    const { executePipeline } = await import("@/api/client");
    (executePipeline as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "run-1",
    });

    render(<InspectorPanel />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("tab", { name: "YAML" }));
    expect(screen.getByRole("tab", { name: "YAML" })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    await user.click(screen.getByRole("button", { name: "Execute pipeline" }));

    await waitFor(() =>
      expect(executePipeline).toHaveBeenCalledWith("session-1"),
    );
    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "Runs" })).toHaveAttribute(
        "aria-selected",
        "true",
      ),
    );
  });
});
