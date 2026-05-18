import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { NarrativeResults } from "./NarrativeResults";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useExecutionStore } from "@/stores/executionStore";
import { resetStore } from "@/test/store-helpers";
import type { InterpretationEvent } from "@/types/interpretation";
import type { Run } from "@/types/index";

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
});
