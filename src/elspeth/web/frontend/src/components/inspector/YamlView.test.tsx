import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { YamlView } from "./YamlView";
import { useSessionStore } from "@/stores/sessionStore";
import type { CompositionProposal } from "@/types/api";

vi.mock("@/api/client", () => ({
  fetchYaml: vi.fn(),
}));

function makeState(version = 1) {
  return {
    id: "state-1",
    version,
    sources: { source: { plugin: "text", options: { content: "hello" } } },
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: "test", description: "" },
  };
}

function makeMetadataOnlyState(version = 1) {
  return {
    ...makeState(version),
    sources: {},
  };
}

function makeProposal(
  overrides: Partial<CompositionProposal> = {},
): CompositionProposal {
  return {
    id: "proposal-1",
    session_id: "session-1",
    tool_call_id: "call-1",
    tool_name: "set_pipeline",
    status: "pending",
    summary: "Replace the pipeline.",
    rationale: "Requested by the current composer turn.",
    affects: ["yaml"],
    arguments_redacted_json: {},
    base_state_id: null,
    committed_state_id: null,
    audit_event_id: "event-1",
    created_at: "2026-05-14T00:00:00Z",
    updated_at: "2026-05-14T00:00:00Z",
    ...overrides,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
}

describe("YamlView", () => {
  beforeEach(async () => {
    useSessionStore.setState({
      activeSessionId: null,
      compositionState: null,
      compositionProposals: [],
      // Reset the export sidecar binding (setState merges); the binding tests
      // below would otherwise leak their stashed value across cases.
      exportedYamlBlobBinding: null,
    });

    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockReset();
  });

  it("renders the empty state when there is no composition state", () => {
    render(<YamlView />);

    expect(
      screen.getByText("YAML will appear here once your pipeline has components."),
    ).toBeInTheDocument();
  });

  it("does not fetch YAML for a metadata-only guided exit state", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockResolvedValue({
      yaml: "source:\n  plugin: text\n",
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeMetadataOnlyState(),
    });

    render(<YamlView />);

    expect(
      screen.getByText("YAML will appear here once your pipeline has components."),
    ).toBeInTheDocument();
    expect(fetchYaml).not.toHaveBeenCalled();
  });

  it("shows a validation-blocked alert when YAML export returns 409", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockRejectedValue({
      status: 409,
      detail: "Current composition state is invalid. Fix validation errors before exporting YAML.",
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
    });

    render(<YamlView />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "YAML export is blocked by validation errors.",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Current composition state is invalid. Fix validation errors before exporting YAML.",
    );
    expect(
      screen.queryByText("YAML will appear here once your pipeline has components."),
    ).not.toBeInTheDocument();
  });

  it("clears stale YAML controls while refetching after a composition version change", async () => {
    const { fetchYaml } = await import("@/api/client");
    const secondFetch = deferred<{ yaml: string }>();
    vi.mocked(fetchYaml)
      .mockResolvedValueOnce({
        yaml: "source:\n  plugin: old_text\n",
      })
      .mockReturnValueOnce(secondFetch.promise);

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(1),
    });

    render(<YamlView />);

    expect(await screen.findByText("old_text")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Copy YAML to clipboard" }),
    ).toBeEnabled();

    act(() => {
      useSessionStore.setState({
        compositionState: makeState(2),
      });
    });

    expect(screen.getByRole("status")).toHaveTextContent("Loading YAML...");
    expect(screen.queryByText("old_text")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Copy YAML to clipboard" }),
    ).not.toBeInTheDocument();

    await act(async () => {
      secondFetch.resolve({
        yaml: "source:\n  plugin: new_text\n",
      });
      await secondFetch.promise;
    });

    expect(await screen.findByText("new_text")).toBeInTheDocument();
  });

  it("exposes copied state as a data attribute for forced-colors CSS", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockResolvedValue({
      yaml: "source:\n  plugin: text\n",
    });
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
    });

    render(<YamlView />);
    const user = userEvent.setup();

    const copyButton = await screen.findByRole("button", {
      name: "Copy YAML to clipboard",
    });
    expect(copyButton).toHaveAttribute("data-copied", "false");

    await user.click(copyButton);

    await waitFor(() => {
      expect(copyButton).toHaveAttribute("data-copied", "true");
    });
  });

  it("hides visual line numbers from assistive technology", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockResolvedValue({
      yaml: "source:\n  plugin: text\n",
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
    });

    const { container } = render(<YamlView />);

    await screen.findByRole("button", { name: "Copy YAML to clipboard" });
    const lineNumbers = container.querySelectorAll(".yaml-view-line-number");

    expect(lineNumbers.length).toBeGreaterThan(0);
    lineNumbers.forEach((lineNumber) => {
      expect(lineNumber).toHaveAttribute("aria-hidden", "true");
    });
  });

  it("renders pending YAML change summary when proposals affect yaml", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockResolvedValue({
      yaml: "source:\n  plugin: text\n",
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
      compositionProposals: [makeProposal()],
    });

    render(<YamlView />);

    expect(await screen.findByText(/Pending YAML change/)).toBeInTheDocument();
    expect(screen.getByText(/Replace the pipeline/)).toBeInTheDocument();
  });

  it("keeps pending YAML proposal actions visible when current YAML export is invalid", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockRejectedValue({
      status: 409,
      detail: "Current composition state is invalid.",
    });
    const acceptProposal = vi.fn();
    const rejectProposal = vi.fn();

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
      compositionProposals: [makeProposal()],
      acceptProposal,
      rejectProposal,
    });

    render(<YamlView />);
    const user = userEvent.setup();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "YAML export is blocked by validation errors.",
    );
    expect(screen.getByText(/Pending YAML change/)).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", {
        name: "Accept YAML proposal: Replace the pipeline.",
      }),
    );

    expect(acceptProposal).toHaveBeenCalledWith("proposal-1");
  });

  // ── source_blob_ids sidecar capture (for the import round-trip) ─────────────

  it("stashes the source_blob_ids sidecar paired to the session + exact YAML on export fetch", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockResolvedValue({
      yaml: "source:\n  plugin: text\n",
      source_blob_ids: { source: "22222222-2222-2222-2222-222222222222" },
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
    });

    render(<YamlView />);
    await screen.findByRole("button", { name: "Copy YAML to clipboard" });

    expect(useSessionStore.getState().exportedYamlBlobBinding).toEqual({
      sessionId: "session-1",
      yaml: "source:\n  plugin: text\n",
      sourceBlobIds: { source: "22222222-2222-2222-2222-222222222222" },
    });
  });

  it("clears a stale sidecar binding when the export fetch returns no source_blob_ids", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockResolvedValue({ yaml: "source:\n  plugin: text\n" });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
      exportedYamlBlobBinding: {
        sessionId: "session-1",
        yaml: "stale",
        sourceBlobIds: { source: "old" },
      },
    } as never);

    render(<YamlView />);
    await screen.findByRole("button", { name: "Copy YAML to clipboard" });

    expect(useSessionStore.getState().exportedYamlBlobBinding).toBeNull();
  });
});
