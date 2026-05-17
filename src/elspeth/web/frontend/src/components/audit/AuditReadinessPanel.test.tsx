import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AuditReadinessPanel } from "./AuditReadinessPanel";
import { useSessionStore } from "../../stores/sessionStore";
import { useAuditReadinessStore, getInitialState } from "../../stores/auditReadinessStore";
import * as api from "../../api/auditReadiness";
import type { AuditReadinessSnapshot } from "../../types/api";
import { makeComposition, makeAbortablePromise } from "@/test/composerFixtures";

vi.mock("../../api/auditReadiness");

const SESSION_ID = "00000000-0000-0000-0000-000000000001";

function allGreenSnapshot(version: number): AuditReadinessSnapshot {
  return {
    session_id: SESSION_ID,
    composition_version: version,
    rows: [
      { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
      { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
      { id: "provenance", label: "Provenance", status: "ok", summary: "Complete lineage", detail: null, component_ids: [] },
      { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
      { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
      { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets", detail: null, component_ids: [] },
    ],
  };
}

function snapshotWithProvenanceWarning(version: number): AuditReadinessSnapshot {
  const base = allGreenSnapshot(version);
  return {
    ...base,
    rows: base.rows.map((r) =>
      r.id === "provenance"
        ? {
            ...r,
            status: "warning",
            summary: "Identity passthrough detected",
            detail: "Identity passthrough — provenance gap on 'select_columns'.",
            component_ids: ["select_columns"],
          }
        : r,
    ),
  };
}

describe("AuditReadinessPanel", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      compositionState: makeComposition(1),
    });
    // Use the canonical reset factory so every per-session dict resets,
    // including abortControllers/explainAbortControllers — matches the
    // pattern in auditReadinessStore.test.ts:27 and survives store-shape
    // extensions automatically.
    useAuditReadinessStore.setState(getInitialState());
    vi.clearAllMocks();
  });

  it("auto-fetches on mount using compositionState.version", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(allGreenSnapshot(1), { signal }),
    );
    render(<AuditReadinessPanel />);
    await waitFor(() => {
      expect(api.fetchAuditReadiness).toHaveBeenCalledWith(
        SESSION_ID,
        expect.any(AbortSignal),
      );
    });
  });

  it("collapses to a single 'Audit ready' summary when all rows are ok/not_applicable", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(allGreenSnapshot(1), { signal }),
    );
    render(<AuditReadinessPanel />);
    expect(await screen.findByText(/Audit ready/i)).toBeInTheDocument();
    // Six row labels are not all rendered up-front in collapsed mode.
    expect(screen.queryByText("Plugin trust")).not.toBeInTheDocument();
    // The collapsed summary is a disclosure button: screen-reader users
    // must learn from aria-expanded that it can be opened. aria-controls
    // is intentionally NOT set — the rows list is not in the DOM while
    // collapsed, so pointing at a missing id would be incorrect.
    const summary = screen.getByRole("button", { name: /Audit ready/i });
    expect(summary).toHaveAttribute("aria-expanded", "false");
  });

  it("expands to all rows when the summary is clicked", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(allGreenSnapshot(1), { signal }),
    );
    const user = userEvent.setup();
    render(<AuditReadinessPanel />);
    const summary = await screen.findByRole("button", { name: /Audit ready/i });
    await user.click(summary);
    expect(screen.getByText("Validation")).toBeInTheDocument();
    expect(screen.getByText("Plugin trust")).toBeInTheDocument();
    expect(screen.getByText("Provenance")).toBeInTheDocument();
    expect(screen.getByText("Retention")).toBeInTheDocument();
  });

  it("shows all rows by default when any row has warning or error status", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(snapshotWithProvenanceWarning(1), { signal }),
    );
    render(<AuditReadinessPanel />);
    await waitFor(() => {
      expect(screen.getByText("Validation")).toBeInTheDocument();
    });
    expect(screen.getByText("Provenance")).toBeInTheDocument();
    expect(screen.getByText("Identity passthrough detected")).toBeInTheDocument();
  });

  it("renders all six rows when every row is warning or error (no collapse path)", async () => {
    const everyRowActionable: AuditReadinessSnapshot = {
      session_id: SESSION_ID,
      composition_version: 1,
      rows: [
        { id: "validation", label: "Validation", status: "error", summary: "Two errors", detail: "Missing source plugin.", component_ids: ["source"] },
        { id: "plugin_trust", label: "Plugin trust", status: "warning", summary: "One Tier 3 plugin", detail: null, component_ids: ["web_scrape"] },
        { id: "provenance", label: "Provenance", status: "warning", summary: "Identity passthrough", detail: null, component_ids: ["select_columns"] },
        { id: "retention", label: "Retention", status: "warning", summary: "Retention shorter than expected", detail: null, component_ids: [] },
        { id: "llm_interpretations", label: "LLM interpretations", status: "warning", summary: "Untracked LLM output", detail: null, component_ids: ["classify"] },
        { id: "secrets", label: "Secrets", status: "error", summary: "Missing required secret", detail: "secret 'OPENAI_API_KEY' is not declared.", component_ids: [] },
      ],
    };
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(everyRowActionable, { signal }),
    );
    render(<AuditReadinessPanel />);
    await waitFor(() => {
      expect(screen.getByText("Validation")).toBeInTheDocument();
    });
    // All six row labels must appear — no all-green collapse can apply.
    expect(screen.getByText("Plugin trust")).toBeInTheDocument();
    expect(screen.getByText("Provenance")).toBeInTheDocument();
    expect(screen.getByText("Retention")).toBeInTheDocument();
    expect(screen.getByText("LLM interpretations")).toBeInTheDocument();
    expect(screen.getByText("Secrets")).toBeInTheDocument();
    // Each summary string should be present (sanity check on row-rendering loop).
    expect(screen.getByText("Two errors")).toBeInTheDocument();
    expect(screen.getByText(/Missing required secret/)).toBeInTheDocument();
  });

  it("refetches when compositionState.version advances", async () => {
    vi.mocked(api.fetchAuditReadiness)
      .mockImplementationOnce(
        (_sid, signal) => makeAbortablePromise(allGreenSnapshot(1), { signal }),
      )
      .mockImplementationOnce(
        (_sid, signal) => makeAbortablePromise(allGreenSnapshot(2), { signal }),
      );
    const { rerender } = render(<AuditReadinessPanel />);
    await waitFor(() => expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(1));

    useSessionStore.setState({ compositionState: makeComposition(2) });
    rerender(<AuditReadinessPanel />);

    await waitFor(() => expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(2));
  });

  it("renders a loading state on first fetch", async () => {
    let resolve!: (s: AuditReadinessSnapshot) => void;
    vi.mocked(api.fetchAuditReadiness).mockReturnValueOnce(
      new Promise<AuditReadinessSnapshot>((r) => {
        resolve = r;
      }),
    );
    render(<AuditReadinessPanel />);
    expect(screen.getByText(/Checking audit readiness/i)).toBeInTheDocument();
    resolve(allGreenSnapshot(1));
    await waitFor(() => {
      expect(screen.queryByText(/Checking audit readiness/i)).not.toBeInTheDocument();
    });
  });

  it("renders an error message on fetch failure", async () => {
    vi.mocked(api.fetchAuditReadiness).mockRejectedValueOnce({
      status: 500,
      detail: "Internal server error",
    });
    render(<AuditReadinessPanel />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/Internal server error/);
  });

  it("mounts the Explain dialog when Explain → is clicked (narrative content asserted in 14c)", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(allGreenSnapshot(1), { signal }),
    );
    // The placeholder ExplainDialog still calls loadExplain on mount, so mock
    // the response to keep the store happy. The narrative content is NOT
    // asserted here — that belongs to 14c's ExplainDialog.test.tsx.
    vi.mocked(api.fetchAuditReadinessExplain).mockImplementationOnce(
      (_sid, signal) =>
        makeAbortablePromise(
          {
            session_id: SESSION_ID,
            composition_version: 1,
            narrative: "stub for placeholder mount; content assertion lives in 14c",
          },
          { signal },
        ),
    );
    const user = userEvent.setup();
    render(<AuditReadinessPanel />);
    const summary = await screen.findByRole("button", { name: /Audit ready/i });
    await user.click(summary); // expand
    const explainBtn = screen.getByRole("button", { name: /Explain/i });
    await user.click(explainBtn);
    // 14b asserts mount only — the placeholder's data-testid was added by the
    // W2 a11y fix (role="dialog" removed; proper dialog semantics are 14c's
    // responsibility). When 14c replaces the placeholder, this assertion must
    // change to findByRole("dialog") to validate the real modal contract.
    expect(
      await screen.findByTestId("explaindialog-placeholder"),
    ).toBeInTheDocument();
  });

  it("renders nothing when there is no active session", () => {
    useSessionStore.setState({ activeSessionId: null, compositionState: null });
    const { container } = render(<AuditReadinessPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the composition is empty (no source, no nodes, no outputs)", () => {
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      compositionState: {
        ...makeComposition(1),
        source: null,
        nodes: [],
        outputs: [],
      },
    });
    const { container } = render(<AuditReadinessPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it("aborts the in-flight fetch on unmount without clearing cached data", async () => {
    // Seed a prior cached snapshot for SESSION_ID so we can verify the cleanup
    // does NOT call clearSession (which would wipe this entry).
    const seeded = allGreenSnapshot(1);
    useAuditReadinessStore.setState({
      snapshotsBySession: { [SESSION_ID]: seeded },
    });
    // Force the auto-fetch to fire by activating composition version 2 (≠ 1).
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      compositionState: makeComposition(2),
    });

    // Signal-aware mock with a delay long enough that the cleanup runs first.
    // makeAbortablePromise rejects with AbortError when the signal aborts, so
    // the store's AbortError catch arm fires under test (production parity).
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) =>
        makeAbortablePromise(allGreenSnapshot(2), { signal, delay: 10_000 }),
    );

    const { unmount } = render(<AuditReadinessPanel />);

    // Wait for loadSnapshot to start and store the AbortController.
    let capturedCtrl: AbortController | undefined;
    await waitFor(() => {
      capturedCtrl =
        useAuditReadinessStore.getState().abortControllers[SESSION_ID];
      expect(capturedCtrl).toBeDefined();
    });

    // The cleanup must abort this controller on unmount.
    unmount();
    expect(capturedCtrl!.signal.aborted).toBe(true);

    // Let the AbortError microtask run through the store's catch arm.
    await waitFor(() => {
      expect(
        useAuditReadinessStore.getState().isLoadingBySession[SESSION_ID],
      ).toBe(false);
    });

    // The seeded snapshot must still be cached — cleanup must NOT have called
    // clearSession (which would have removed snapshotsBySession[SESSION_ID]),
    // and the AbortError catch arm preserves cached snapshot/error.
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID],
    ).toEqual(seeded);

    // Note: the prior form of this test asserted a "late-resolve ratchet"
    // — that an aborted fetch which later resolves still writes via the
    // monotonic guard. That race is production-unreachable: real fetch
    // rejects with AbortError when its signal aborts, so the success arm
    // never fires for an aborted call. Signal-aware test mocks match that
    // production semantic, which is why the sub-assertion is not preserved
    // here. The store's monotonic-guard property itself (later snapshot for
    // a higher version overwrites an earlier one) is covered by the
    // version-advances tests in auditReadinessStore.test.ts.
  });

  it("preserves the user's expand preference across component unmount/remount (Phase 3B remount safety)", async () => {
    // Seed a cached all-green snapshot directly — no fetch needed; the
    // component reads from the store, and version parity means loadSnapshot
    // is a no-op. This isolates the test to userExpanded behaviour only.
    useAuditReadinessStore.setState({
      snapshotsBySession: { [SESSION_ID]: allGreenSnapshot(1) },
    });

    const user = userEvent.setup();

    // Mount the panel; with an all-green snapshot it collapses to "Audit ready".
    const { unmount } = render(<AuditReadinessPanel />);
    expect(await screen.findByRole("button", { name: /Audit ready/i })).toBeInTheDocument();

    // User clicks to expand — sets userExpanded=true via the toggle.
    await user.click(screen.getByRole("button", { name: /Audit ready/i }));
    // Confirm expansion: the full row list is visible.
    expect(screen.getByText("Validation")).toBeInTheDocument();

    // Simulate the Phase 3B remount: unmount the current tree, then render a fresh instance.
    unmount();
    render(<AuditReadinessPanel />);

    // The panel must still be expanded — userExpanded survived the remount via the store.
    // With component-local useState this test FAILS: the new instance starts with
    // useState(false), anyActionable is false (all-green), and showExpanded = false.
    expect(screen.getByText("Validation")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Audit ready/i })).not.toBeInTheDocument();
  });

  it("auto-collapses when a subsequent refetch returns all-green (no sticky expansion)", async () => {
    // Arrange: render with an actionable snapshot at version 1; the panel
    // auto-expands because anyActionable is true.
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(snapshotWithProvenanceWarning(1), { signal }),
    );
    const { rerender } = render(<AuditReadinessPanel />);
    await waitFor(() =>
      expect(screen.getByText(/Provenance/)).toBeInTheDocument(),
    );

    // Act: bump version; mock returns an all-green snapshot for v2.
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(allGreenSnapshot(2), { signal }),
    );
    useSessionStore.setState({ compositionState: makeComposition(2) });
    rerender(<AuditReadinessPanel />);

    // Assert: panel falls back to the collapsed "Audit ready" summary —
    // anyActionable is now false and the user never explicitly expanded, so
    // showExpanded = anyActionable || userExpanded = false. This is the
    // sticky-expansion regression that elspeth-82ef9d5bd0 fixed: the prior
    // useState+useEffect form set `expanded = true` permanently on the first
    // warning snapshot, leaving the panel stuck open even after all-green.
    await waitFor(() =>
      expect(screen.queryByText(/Provenance/)).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: /Audit ready/i })).toBeInTheDocument();
  });
});
