import { describe, it, expect, beforeEach, vi } from "vitest";
import { act } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AuditReadinessPanel } from "./AuditReadinessPanel";
import { useSessionStore } from "../../stores/sessionStore";
import { useAuditReadinessStore, getInitialState } from "../../stores/auditReadinessStore";
import { useExecutionStore } from "../../stores/executionStore";
import { useInlineSourceStore } from "@/stores/inlineSourceStore";
import { resetStore } from "@/test/store-helpers";
import * as api from "../../api/auditReadiness";
import type { AuditReadinessSnapshot, ValidationResult } from "../../types/api";
import { makeComposition, makeAbortablePromise } from "@/test/composerFixtures";

vi.mock("../../api/auditReadiness");

const SESSION_ID = "00000000-0000-0000-0000-000000000001";
const OTHER_SESSION_ID = "00000000-0000-0000-0000-000000000002";

function allGreenSnapshot(version: number): AuditReadinessSnapshot {
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

function snapshotWithValidationErrorAndProvenanceWarning(version: number): AuditReadinessSnapshot {
  const base = snapshotWithProvenanceWarning(version);
  return {
    ...base,
    rows: base.rows.map((r) =>
      r.id === "validation"
        ? {
            ...r,
            status: "error",
            summary: "Missing source plugin",
            detail: "A source is required before execution.",
            component_ids: ["source"],
          }
        : r,
    ),
    validation_result: {
      is_valid: false,
      checks: [
        {
          name: "settings_load",
          passed: false,
          detail: "A source is required before execution.",
          affected_nodes: [],
          outcome_code: null,
        },
      ],
      errors: [
        {
          component_id: "source",
          component_type: "source",
          message: "A source is required before execution.",
          suggestion: null,
        },
      ],
      warnings: [],
      semantic_contracts: [],
    },
  };
}

function snapshotWithRawValidationResult(version: number, validationResult: ValidationResult): AuditReadinessSnapshot {
  const base = allGreenSnapshot(version);
  return {
    ...base,
    rows: base.rows.map((r) =>
      r.id === "validation"
        ? {
            ...r,
            status: validationResult.is_valid ? "ok" : "error",
            summary: validationResult.is_valid ? "All checks pass" : `${validationResult.errors.length} errors — see details`,
            detail: validationResult.errors.map((err) => err.message).join("\n"),
            component_ids: validationResult.errors
              .map((err) => err.component_id)
              .filter((id): id is string => id !== null),
          }
        : r,
    ),
    validation_result: validationResult,
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
    useExecutionStore.setState({ validationResult: null } as never);
    resetStore(useInlineSourceStore);
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

  it("includes status, label, and summary in actionable row button accessible names", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) =>
        makeAbortablePromise(snapshotWithValidationErrorAndProvenanceWarning(1), { signal }),
    );
    render(<AuditReadinessPanel />);

    expect(
      await screen.findByRole("button", {
        name: /error.*validation.*missing source plugin/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: /warning.*provenance.*identity passthrough detected/i,
      }),
    ).toBeInTheDocument();
  });

  it("announces loading and row updates through a polite non-atomic live region", async () => {
    let resolve!: (s: AuditReadinessSnapshot) => void;
    vi.mocked(api.fetchAuditReadiness).mockReturnValueOnce(
      new Promise<AuditReadinessSnapshot>((r) => {
        resolve = r;
      }),
    );

    render(<AuditReadinessPanel />);
    const loading = screen.getByText(/Checking audit readiness/i);
    const loadingLiveRegion = loading.closest("[aria-live='polite']");
    expect(loadingLiveRegion).toHaveAttribute("aria-atomic", "false");

    resolve(snapshotWithProvenanceWarning(1));
    const rows = await screen.findByRole("list");
    expect(rows).toHaveAttribute("aria-live", "polite");
    expect(rows).toHaveAttribute("aria-atomic", "false");
  });

  it("shows loading instead of stale rows during stale-cache refetch", async () => {
    useAuditReadinessStore.setState({
      snapshotsBySession: { [SESSION_ID]: snapshotWithProvenanceWarning(1) },
    });
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      compositionState: makeComposition(2),
    });

    let resolve!: (s: AuditReadinessSnapshot) => void;
    vi.mocked(api.fetchAuditReadiness).mockReturnValueOnce(
      new Promise<AuditReadinessSnapshot>((r) => {
        resolve = r;
      }),
    );

    render(<AuditReadinessPanel />);

    expect(screen.getByText(/Checking audit readiness/i)).toBeInTheDocument();
    expect(screen.queryByText("Identity passthrough detected")).not.toBeInTheDocument();

    resolve(snapshotWithProvenanceWarning(2));
    expect(await screen.findByText("Identity passthrough detected")).toBeInTheDocument();
    const panel = screen.getByRole("region", { name: /audit readiness/i });
    expect(panel).not.toHaveAttribute("aria-busy");
  });

  it("does not render a cached snapshot when its version differs from the current composition", async () => {
    useAuditReadinessStore.setState({
      snapshotsBySession: { [SESSION_ID]: allGreenSnapshot(1) },
    });
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      compositionState: makeComposition(2),
    });

    vi.mocked(api.fetchAuditReadiness).mockReturnValueOnce(
      new Promise<AuditReadinessSnapshot>(() => {}),
    );

    render(<AuditReadinessPanel />);

    expect(screen.getByText(/Checking audit readiness/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Audit ready/i })).not.toBeInTheDocument();
  });

  it("projects an OK validation row into the execution validation state", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(allGreenSnapshot(1), { signal }),
    );

    render(<AuditReadinessPanel />);

    await screen.findByRole("button", { name: /Audit ready/i });
    expect(useExecutionStore.getState().validationResult).toMatchObject({
      is_valid: true,
      errors: [],
    });
  });

  it("projects the raw snapshot validation result without collapsing component attribution", async () => {
    const rawValidation: ValidationResult = {
      is_valid: false,
      checks: [
        {
          name: "settings_load",
          passed: false,
          detail: "settings failed",
          affected_nodes: [],
          outcome_code: null,
        },
      ],
      errors: [
        {
          component_id: "first",
          component_type: "transform",
          message: "First transform is invalid.",
          suggestion: "Fix first.",
        },
        {
          component_id: "second",
          component_type: "transform",
          message: "Second transform is invalid.",
          suggestion: "Fix second.",
        },
      ],
      warnings: [],
      semantic_contracts: [],
    };
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) =>
        makeAbortablePromise(snapshotWithRawValidationResult(1, rawValidation), { signal }),
    );

    render(<AuditReadinessPanel />);

    await waitFor(() => {
      expect(useExecutionStore.getState().validationResult).toEqual(rawValidation);
    });
  });

  it("renders a freshness indicator with the checked time and composition version", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(snapshotWithProvenanceWarning(1), { signal }),
    );

    render(<AuditReadinessPanel />);

    expect(
      await screen.findByLabelText(/audit readiness checked just now.*v1/i),
    ).toBeInTheDocument();
  });

  it("force-refreshes the current composition version when Refresh is clicked", async () => {
    useAuditReadinessStore.setState({
      snapshotsBySession: { [SESSION_ID]: snapshotWithProvenanceWarning(1) },
    });
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      compositionState: makeComposition(1),
    });
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(snapshotWithProvenanceWarning(1), { signal }),
    );

    const user = userEvent.setup();
    render(<AuditReadinessPanel />);

    expect(api.fetchAuditReadiness).not.toHaveBeenCalled();
    await user.click(
      screen.getByRole("button", { name: /refresh audit check now/i }),
    );

    await waitFor(() => expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(1));
  });

  it("projects a forced-refresh validation row into the execution validation state", async () => {
    useAuditReadinessStore.setState({
      snapshotsBySession: { [SESSION_ID]: allGreenSnapshot(1) },
    });
    useExecutionStore.getState().setValidationResult({
      is_valid: true,
      checks: [],
      errors: [],
      warnings: [],
      semantic_contracts: [],
    });
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) =>
        makeAbortablePromise(snapshotWithValidationErrorAndProvenanceWarning(1), {
          signal,
        }),
    );

    const user = userEvent.setup();
    render(<AuditReadinessPanel />);

    await user.click(await screen.findByRole("button", { name: /Audit ready/i }));
    await user.click(
      screen.getByRole("button", { name: /refresh audit check now/i }),
    );

    await waitFor(() => {
      expect(useExecutionStore.getState().validationResult).toMatchObject({
        is_valid: false,
        errors: [
          {
            component_id: "source",
            message: "A source is required before execution.",
          },
        ],
      });
    });
  });

  it("does not project a forced refresh after the active session changes", async () => {
    useAuditReadinessStore.setState({
      snapshotsBySession: {
        [SESSION_ID]: allGreenSnapshot(1),
        [OTHER_SESSION_ID]: { ...allGreenSnapshot(1), session_id: OTHER_SESSION_ID },
      },
    });
    const previousValidation = {
      is_valid: true,
      checks: [],
      errors: [],
      warnings: [],
      semantic_contracts: [],
    };
    useExecutionStore.getState().setValidationResult(previousValidation);
    let resolveRefresh!: (snapshot: AuditReadinessSnapshot) => void;
    vi.mocked(api.fetchAuditReadiness).mockReturnValueOnce(
      new Promise<AuditReadinessSnapshot>((resolve) => {
        resolveRefresh = resolve;
      }),
    );

    const user = userEvent.setup();
    const { rerender } = render(<AuditReadinessPanel />);

    await user.click(await screen.findByRole("button", { name: /Audit ready/i }));
    await user.click(
      screen.getByRole("button", { name: /refresh audit check now/i }),
    );
    act(() => {
      useSessionStore.setState({
        activeSessionId: OTHER_SESSION_ID,
        compositionState: makeComposition(1),
      });
    });
    rerender(<AuditReadinessPanel />);

    resolveRefresh(snapshotWithValidationErrorAndProvenanceWarning(1));
    await waitFor(() => expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]).toMatchObject({
        rows: expect.arrayContaining([
          expect.objectContaining({ id: "validation", status: "error" }),
        ]),
      }),
    );
    expect(useExecutionStore.getState().validationResult).toEqual(previousValidation);
  });

  it("renders all six rows when every row is warning or error (no collapse path)", async () => {
    const everyRowActionable: AuditReadinessSnapshot = {
      session_id: SESSION_ID,
      composition_version: 1,
      checked_at: new Date().toISOString(),
      rows: [
        { id: "validation", label: "Validation", status: "error", summary: "Two errors", detail: "Missing source plugin.", component_ids: ["source"] },
        { id: "plugin_trust", label: "Plugin trust", status: "warning", summary: "One Tier 3 plugin", detail: null, component_ids: ["web_scrape"] },
        { id: "provenance", label: "Provenance", status: "warning", summary: "Identity passthrough", detail: null, component_ids: ["select_columns"] },
        { id: "retention", label: "Retention", status: "warning", summary: "Retention shorter than expected", detail: null, component_ids: [] },
        { id: "llm_interpretations", label: "LLM interpretations", status: "warning", summary: "Untracked LLM output", detail: null, component_ids: ["classify"] },
        { id: "secrets", label: "Secrets", status: "error", summary: "Missing required secret", detail: "secret 'OPENAI_API_KEY' is not declared.", component_ids: [] },
      ],
      validation_result: {
        is_valid: false,
        checks: [],
        errors: [
          {
            component_id: "source",
            component_type: "source",
            message: "Missing source plugin.",
            suggestion: null,
          },
        ],
        warnings: [],
        semantic_contracts: [],
      },
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

  it("mounts the Explain dialog when Explain → is clicked", async () => {
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) => makeAbortablePromise(allGreenSnapshot(1), { signal }),
    );
    vi.mocked(api.fetchAuditReadinessExplain).mockImplementationOnce(
      (_sid, signal) =>
        makeAbortablePromise(
          {
            session_id: SESSION_ID,
            composition_version: 1,
            narrative: "Narrative content rendered by ExplainDialog.",
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
    // The real ExplainDialog (Phase 2C) renders role="dialog" labelled by the
    // heading "What this pipeline will record". Content assertions for the
    // dialog itself live in ExplainDialog.test.tsx.
    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveTextContent(/What this pipeline will record/i);
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

  it("renders inline-content-hashed provenance row when the source is inline_blob-backed (Phase 5a.7)", async () => {
    // Seed an inline-source summary so the panel's projection branch fires.
    useInlineSourceStore.getState().setSummary(SESSION_ID, {
      blobId: "blob-uuid",
      filename: "chat.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://finance.gov.au",
      rowCount: 1,
      contentHash: "abc123def456789",
      provenance: "verbatim",
    });
    // Snapshot must be actionable so the panel auto-expands and the
    // provenance row is in the DOM (collapsed mode hides the row list).
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) =>
        makeAbortablePromise(snapshotWithProvenanceWarning(1), { signal }),
    );

    render(<AuditReadinessPanel />);

    // The override replaces the backend-supplied summary text on the
    // provenance row with "Inline content hashed (SHA-256: <prefix>…)".
    expect(
      await screen.findByText(/inline content hashed/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/abc123def456/)).toBeInTheDocument();
    // And the backend summary string for that row is NOT shown — we
    // replaced it, not appended to it.
    expect(
      screen.queryByText(/identity passthrough detected/i),
    ).not.toBeInTheDocument();
  });

  it("renders the default backend-supplied provenance summary when no inline source is bound (Phase 5a.7)", async () => {
    // No inline source seeded — store returns null for getSummary(SESSION_ID).
    vi.mocked(api.fetchAuditReadiness).mockImplementationOnce(
      (_sid, signal) =>
        makeAbortablePromise(snapshotWithProvenanceWarning(1), { signal }),
    );

    render(<AuditReadinessPanel />);

    // Default backend-supplied summary IS rendered; the inline override
    // is NOT applied.
    expect(
      await screen.findByText(/identity passthrough detected/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/inline content hashed/i),
    ).not.toBeInTheDocument();
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
