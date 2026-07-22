import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { NarrativeResults } from "./NarrativeResults";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useExecutionStore } from "@/stores/executionStore";
import { resetStore } from "@/test/store-helpers";
import type { InterpretationEvent } from "@/types/interpretation";
import type {
  Run,
  RunOutputArtifact,
  RunOutputArtifactPreview,
  RunOutputsResponse,
} from "@/types/index";
import {
  downloadRunOutputContent,
  fetchRunOutputPreview,
  fetchRunOutputs,
} from "@/api/client";

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>("@/api/client");
  return {
    ...actual,
    fetchRunOutputs: vi.fn(),
    fetchRunOutputPreview: vi.fn(),
    downloadRunOutputContent: vi.fn(),
  };
});

function fileArtifact(overrides: Partial<RunOutputArtifact> = {}): RunOutputArtifact {
  return {
    artifact_id: "art-1",
    sink_node_id: "results",
    producer_kind: "sink_effect",
    produced_by_state_id: null,
    sink_effect_id: "effect-1",
    artifact_type: "file",
    path_or_uri: "file:///data/outputs/results.jsonl",
    content_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    size_bytes: 1024,
    publication_performed: true,
    publication_evidence_kind: "returned",
    created_at: "2026-05-19T10:04:00Z",
    exists_now: true,
    downloadable: true,
    storage_kind: "sink_file",
    ...overrides,
  };
}

function outputsResponse(artifacts: RunOutputArtifact[]): RunOutputsResponse {
  return {
    run_id: "run-1",
    landscape_run_id: "run-1",
    artifacts,
  };
}

function jsonlPreview(rows: Array<Record<string, unknown>>, overrides: Partial<RunOutputArtifactPreview> = {}): RunOutputArtifactPreview {
  return {
    artifact_id: "art-1",
    content_type: "jsonl",
    preview_text: rows.map((r) => JSON.stringify(r)).join("\n"),
    truncated: false,
    total_size_bytes: 256,
    row_count_preview: rows.length,
    ...overrides,
  };
}

// ── Fixtures ────────────────────────────────────────────────────────────────

function makeRun(overrides: Partial<Run> = {}): Run {
  return {
    id: "run-1",
    session_id: "sess-1",
    status: "completed",
    accounting: null,
    error: null,
    started_at: "2026-05-19T10:00:00Z",
    finished_at: "2026-05-19T10:05:00Z",
    composition_version: 1,
    ...overrides,
  };
}

function makeResolvedEvent(overrides: Partial<InterpretationEvent> = {}): InterpretationEvent {
  return {
    id: "evt-1",
    session_id: "sess-1",
    composition_state_id: "state-1",
    affected_node_id: "node-1",
    tool_call_id: "tool-1",
    user_term: "cool",
    kind: "vague_term",
    llm_draft: "engaging",
    accepted_value: "engaging",
    choice: "accepted_as_drafted",
    created_at: "2026-05-19T10:02:00Z",
    resolved_at: "2026-05-19T10:02:00Z",
    actor: "user:owner:u-1",
    interpretation_source: "user_approved",
    model_identifier: "anthropic/claude-opus-4-7",
    model_version: "20260518",
    provider: "anthropic",
    composer_skill_hash: "deadbeef",
    arguments_hash: null,
    hash_domain_version: null,
    runtime_model_identifier_at_resolve: null,
    runtime_model_version_at_resolve: null,
    resolved_prompt_template_hash: null,
    ...overrides,
  };
}

describe("NarrativeResults", () => {
  beforeEach(() => {
    useSessionStore.setState({ activeSessionId: null } as never);
    resetStore(useInterpretationEventsStore);
    // Reset execution-store run state so a prior test's runs/activeRunId
    // don't bleed into the next test's narrative-overlay derivation.
    useExecutionStore.setState({ runs: [], activeRunId: null } as never);
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Some tests below pin Date.now() with fake timers; ensure they
    // don't bleed past the test that set them.
    vi.useRealTimers();
  });

  it("renders the no-summary placeholder when summaryOverride is undefined (live mode without an aggregated summary)", () => {
    render(<NarrativeResults />);
    expect(screen.getByTestId("narrative-results")).toBeInTheDocument();
    expect(screen.getByTestId("narrative-results-no-summary")).toBeInTheDocument();
    expect(screen.queryByTestId("narrative-results-summary")).toBeNull();
  });

  it("renders the supplied summary when summaryOverride is provided", () => {
    render(<NarrativeResults summaryOverride="The pipeline achieved an F1 of 0.87." />);
    expect(screen.getByTestId("narrative-results-summary")).toHaveTextContent(
      "The pipeline achieved an F1 of 0.87.",
    );
    expect(screen.queryByTestId("narrative-results-no-summary")).toBeNull();
  });

  it("renders the no-summary placeholder when summaryOverride is explicitly null", () => {
    render(<NarrativeResults summaryOverride={null} />);
    expect(screen.getByTestId("narrative-results-no-summary")).toBeInTheDocument();
  });

  it("renders the no-summary placeholder when summaryOverride is the empty string", () => {
    render(<NarrativeResults summaryOverride="" />);
    expect(screen.getByTestId("narrative-results-no-summary")).toBeInTheDocument();
  });

  it("does not render the interpretation overlay when no session is active", () => {
    render(<NarrativeResults summaryOverride="anything" />);
    expect(screen.queryByTestId("narrative-results-interpretations")).toBeNull();
  });

  it("does not render the interpretation overlay when the session has not opted out", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useInterpretationEventsStore.setState({
      optedOutBySession: { "sess-1": false },
    } as never);
    render(<NarrativeResults summaryOverride="anything" />);
    expect(screen.queryByTestId("narrative-results-interpretations")).toBeNull();
  });

  it("renders the interpretation overlay when the active session has opted out of LLM surfacing", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useInterpretationEventsStore.setState({
      optedOutBySession: { "sess-1": true },
    } as never);
    render(<NarrativeResults summaryOverride="anything" />);
    expect(screen.getByTestId("narrative-results-interpretations")).toBeInTheDocument();
    expect(screen.getByTestId("narrative-results-interpretations")).toHaveTextContent(
      /opt-out/i,
    );
  });

  // ── Plan-mandated tests 5/6/7 (load-bearing run-window filter) ─────────────
  // 19b-phase-6b-frontend.md §Task 6 lines 311-337.

  it("plan test 5: events whose resolved_at falls inside the run window render in the overlay", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
    } as never);
    useInterpretationEventsStore.setState({
      resolvedBySession: {
        "sess-1": [
          makeResolvedEvent({
            id: "evt-in-window",
            user_term: "cool",
            accepted_value: "engaging",
            resolved_at: "2026-05-19T10:02:00Z",
          }),
        ],
      },
    } as never);

    render(<NarrativeResults summaryOverride="anything" />);

    const overlay = screen.getByTestId("narrative-results-interpretation-overlay");
    expect(overlay).toBeInTheDocument();
    const row = screen.getByTestId("narrative-overlay-event-evt-in-window");
    expect(row).toHaveTextContent("cool");
    expect(row).toHaveTextContent("engaging");
  });

  it("plan test 6: prior-run events (resolved_at < runStart) do NOT render — guards against session-aggregate over-count", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
    } as never);
    useInterpretationEventsStore.setState({
      resolvedBySession: {
        "sess-1": [
          // Resolved at 09:55Z — 5 minutes BEFORE run-1's 10:00Z start.
          // This is the explicit failure mode the plan guards against:
          // without the wall-clock filter the overlay would surface a
          // resolution that belongs to a prior run in the same session.
          makeResolvedEvent({
            id: "evt-prior-run",
            user_term: "fast",
            accepted_value: "responsive",
            resolved_at: "2026-05-19T09:55:00Z",
            created_at: "2026-05-19T09:55:00Z",
          }),
        ],
      },
    } as never);

    render(<NarrativeResults summaryOverride="anything" />);

    // The overlay container is rendered only when the filtered list is
    // non-empty — so the absence of the testid is the load-bearing assertion.
    expect(
      screen.queryByTestId("narrative-results-interpretation-overlay"),
    ).toBeNull();
    expect(screen.queryByTestId("narrative-overlay-event-evt-prior-run")).toBeNull();
  });

  it("plan test 7: in-flight runs (finished_at == null) use Date.now() as the upper bound", () => {
    // Pin the clock so the in-flight Date.now() branch is deterministic.
    // The fixed "now" is 2026-05-19T10:00:00Z.
    const fixedNow = new Date("2026-05-19T10:00:00Z");
    vi.useFakeTimers();
    vi.setSystemTime(fixedNow);

    const runStartedIso = new Date(fixedNow.getTime() - 60_000).toISOString(); // -60s
    const inWindowResolvedIso = new Date(fixedNow.getTime() - 30_000).toISOString(); // -30s
    const beforeStartResolvedIso = new Date(fixedNow.getTime() - 90_000).toISOString(); // -90s

    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [
        makeRun({
          id: "run-inflight",
          status: "running",
          started_at: runStartedIso,
          finished_at: null,
        }),
      ],
      activeRunId: "run-inflight",
    } as never);
    useInterpretationEventsStore.setState({
      resolvedBySession: {
        "sess-1": [
          makeResolvedEvent({
            id: "evt-in-flight-window",
            user_term: "smart",
            accepted_value: "clever",
            resolved_at: inWindowResolvedIso,
            created_at: inWindowResolvedIso,
          }),
          makeResolvedEvent({
            id: "evt-before-start",
            user_term: "stale",
            accepted_value: "old",
            resolved_at: beforeStartResolvedIso,
            created_at: beforeStartResolvedIso,
          }),
        ],
      },
    } as never);

    render(<NarrativeResults summaryOverride="anything" />);

    // In-window event (resolved 30s ago, run started 60s ago, "now" is now)
    // renders.
    expect(
      screen.getByTestId("narrative-overlay-event-evt-in-flight-window"),
    ).toBeInTheDocument();
    // Event resolved 90s ago (BEFORE the run's 60-s-ago start) does NOT render.
    expect(
      screen.queryByTestId("narrative-overlay-event-evt-before-start"),
    ).toBeNull();
  });

  // ── FIX-D: AC1 Markdown / AC3 Download / AC4 live-summary extraction ───────
  // Plan 19b lines 309 (Markdown rendering), 342 ("Download full output"
  // link), 349 (find-last-output-row-with-summary live extraction).

  it("AC1: renders the summary string as Markdown when supplied as summaryOverride", () => {
    render(<NarrativeResults summaryOverride="The pipeline produced **bold** results and reached _F1=0.87_." />);
    // MarkdownRenderer surfaces inline emphasis as <strong>/<em>. We assert
    // structural Markdown rendering (not raw text) — matching the
    // MarkdownRenderer.test.tsx convention of asserting on element tagName.
    const bold = screen.getByText("bold");
    expect(bold.tagName).toBe("STRONG");
    const em = screen.getByText("F1=0.87");
    expect(em.tagName).toBe("EM");
  });

  it("AC1: renders Markdown headings and code in the summary", () => {
    render(<NarrativeResults summaryOverride={"## Verdict\n\nUse `set_source` to retry."} />);
    const heading = screen.getByRole("heading", { level: 2 });
    expect(heading).toHaveTextContent("Verdict");
    const code = screen.getByText("set_source");
    expect(code.tagName).toBe("CODE");
  });

  it("AC3: renders a 'Download full output' affordance with the load-bearing test id when a terminal run has a downloadable file artifact", async () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
    } as never);
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      outputsResponse([fileArtifact()]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonlPreview([{ score: 0.87 }]),
    );

    render(<NarrativeResults />);

    const link = await screen.findByTestId("narrative-results-download-link");
    expect(link).toBeInTheDocument();
    // The backend `/content` endpoint requires `Authorization: Bearer`
    // (api/client.ts:877-883) — a plain <a href> would 401 on top-level
    // navigation. The shipped pattern (RunOutputsPanel:126-138) uses a
    // button that triggers an authenticated fetch + object-URL download.
    expect(link.tagName).toBe("BUTTON");
  });

  it("AC3: clicking 'Download full output' invokes downloadRunOutputContent against the chosen artifact", async () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
    } as never);
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      outputsResponse([fileArtifact({ artifact_id: "art-chosen" })]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonlPreview([{ score: 0.87 }]),
    );
    const blob = new Blob(["bytes"], { type: "application/octet-stream" });
    (downloadRunOutputContent as ReturnType<typeof vi.fn>).mockResolvedValue({
      data: blob,
      filename: "results.jsonl",
    });
    const createSpy = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
    const revokeSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    render(<NarrativeResults />);

    const link = await screen.findByTestId("narrative-results-download-link");
    fireEvent.click(link);

    await waitFor(() =>
      expect(downloadRunOutputContent).toHaveBeenCalledWith("run-1", "art-chosen"),
    );
    expect(createSpy).toHaveBeenCalledWith(blob);

    createSpy.mockRestore();
    revokeSpy.mockRestore();
  });

  it("AC3: hides the download affordance when no terminal run is active (no activeRunId)", () => {
    // No activeRunId — store remains at default (null).
    render(<NarrativeResults />);
    expect(screen.queryByTestId("narrative-results-download-link")).toBeNull();
  });

  it("AC3: hides the download affordance when the run manifest has no file artifacts", async () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
    } as never);
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      outputsResponse([]),
    );

    render(<NarrativeResults />);

    // Wait for the manifest fetch to settle, then assert absence.
    await waitFor(() => expect(fetchRunOutputs).toHaveBeenCalledWith("run-1"));
    expect(screen.queryByTestId("narrative-results-download-link")).toBeNull();
  });

  it("AC4: live mode extracts the summary from the last output row whose `summary` field is non-empty (JSONL preview)", async () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
    } as never);
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      outputsResponse([fileArtifact()]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonlPreview([
        { score: 0.65 },
        { summary: "Earlier partial summary." },
        { score: 0.87, summary: "Final verdict: classifier converged at F1=0.87." },
      ]),
    );

    render(<NarrativeResults />);

    // Plan 19b:349 — "find the last output row that has a `summary` field;
    // if multiple, concatenate with blank lines." With two `summary`-bearing
    // rows the rendered surface concatenates them in document order.
    const surface = await screen.findByTestId("narrative-results-summary");
    expect(surface).toHaveTextContent("Earlier partial summary.");
    expect(surface).toHaveTextContent(
      "Final verdict: classifier converged at F1=0.87.",
    );
    expect(screen.queryByTestId("narrative-results-no-summary")).toBeNull();
  });

  it("AC4: live mode renders the no-summary placeholder when no output row carries a `summary` field", async () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
    } as never);
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      outputsResponse([fileArtifact()]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonlPreview([{ score: 0.65 }, { score: 0.87 }]),
    );

    render(<NarrativeResults />);

    await waitFor(() =>
      expect(fetchRunOutputPreview).toHaveBeenCalledWith("run-1", "art-1"),
    );
    // The narrative renderer settled and saw no `summary` field — render
    // the documented no-summary placeholder rather than fabricating text.
    expect(
      await screen.findByTestId("narrative-results-no-summary"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("narrative-results-summary")).toBeNull();
  });

  it("AC4: summaryOverride takes precedence over live extraction (read-only inspect-view contract)", async () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
    } as never);
    // If live extraction were not gated on summaryOverride === undefined,
    // this preview's "live summary" would clobber the override. The
    // SharedInspectView contract is: when a frozen blob supplies the
    // summary, the live execution-store path must not race against it.
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      outputsResponse([fileArtifact()]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonlPreview([{ summary: "live-extracted summary" }]),
    );

    render(<NarrativeResults summaryOverride="frozen-blob summary" />);

    expect(
      screen.getByTestId("narrative-results-summary"),
    ).toHaveTextContent("frozen-blob summary");
    // Live extraction must not be initiated when the override is supplied.
    expect(fetchRunOutputs).not.toHaveBeenCalled();
  });
});
