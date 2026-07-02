import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import {
  ExecuteButton,
  INTERPRETATION_PENDING_RUN_BLOCK_TITLE,
  buildRunEgressSummary,
} from "./ExecuteButton";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import type { CompositionState } from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";

function makeInterpretationEvent(
  overrides: Partial<InterpretationEvent> = {},
): InterpretationEvent {
  return {
    id: "evt-1",
    session_id: "sess-1",
    composition_state_id: "state-1",
    affected_node_id: "llm_classify",
    tool_call_id: "tc-1",
    user_term: "cool",
    kind: "vague_term",
    llm_draft: "engaging",
    accepted_value: null,
    choice: "pending",
    created_at: "2026-05-18T10:00:00Z",
    resolved_at: null,
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

function makeComposition(
  overrides: Partial<CompositionState> = {},
): CompositionState {
  return {
    id: "state-1",
    version: 1,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
    ...overrides,
  };
}

describe("ExecuteButton", () => {
  beforeEach(() => {
    useExecutionStore.setState({
      validationResult: null,
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
      runDisclosureAckBySession: {},
    } as never);
    useSessionStore.setState({
      activeSessionId: null,
      compositionState: null,
    } as never);
    resetStore(useInterpretationEventsStore);
  });

  it("renders nothing when there is no active session", () => {
    const { container } = render(<ExecuteButton />);
    expect(container.firstChild).toBeNull();
  });

  it("renders a Run pipeline button when validation has passed", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExecuteButton />);

    expect(
      screen.getByRole("button", { name: /run pipeline/i }),
    ).toBeInTheDocument();
  });

  it("disables the Run pipeline button when validation is failing", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        checks: [],
        errors: [
          {
            component_type: "source",
            component_id: "csv_source",
            message: "x",
          } as never,
        ],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExecuteButton />);

    expect(screen.getByRole("button", { name: /run pipeline/i })).toBeDisabled();
  });

  it("stays a co-equal plain .btn (never btn-primary) even when runnable (elspeth-0d37694c8c)", () => {
    // CompletionBar's contract (its docstring, per plan 19b §"Scope
    // boundaries"): Save-for-review / Run / Export YAML are co-equal verbs
    // with no primary emphasis. A conditional btn-primary previously singled
    // Run out as the lone filled accent button whenever the composition was
    // valid — the common case.
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExecuteButton />);

    const button = screen.getByRole("button", { name: /run pipeline/i });
    expect(button).not.toBeDisabled();
    expect(button).toHaveClass("btn");
    expect(button).not.toHaveClass("btn-primary");
  });

  // ── Pre-run egress disclosure (elspeth-c18ad229cc) ────────────────────────
  //
  // The tutorial's Run step disclosures before firing; the production Run —
  // the surface touching live credentials — must too. A ConfirmDialog gates
  // execute(), summarising what the run will reach (derived from the live
  // composition), with a per-session "don't ask again" opt-out.

  it("opens the egress disclosure instead of executing directly, then executes on confirm", () => {
    const execute = vi.fn();
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
      execute,
    } as never);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: makeComposition({
        sources: {
          source: { plugin: "csv", options: {}, on_success: "classify_in" },
        },
        nodes: [
          {
            id: "classify",
            node_type: "transform",
            plugin: "llm",
            input: "classify_in",
            on_success: "results",
            on_error: null,
            options: { model: "openrouter/anthropic/claude-sonnet-4.6" },
          },
        ],
        outputs: [{ name: "results", plugin: "csv", options: {} }],
      }),
    } as never);

    render(<ExecuteButton />);
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));

    // Confirm gates execute(): nothing fired yet.
    expect(execute).not.toHaveBeenCalled();
    const dialog = screen.getByRole("alertdialog", { name: /run pipeline\?/i });
    // The summary derives from the actual composition.
    expect(dialog).toHaveTextContent("source (csv)");
    expect(dialog).toHaveTextContent(
      "classify (model openrouter/anthropic/claude-sonnet-4.6)",
    );
    expect(dialog).toHaveTextContent("results (csv)");

    fireEvent.click(
      within(dialog).getByRole("button", { name: /^run pipeline$/i }),
    );
    expect(execute).toHaveBeenCalledWith("sess-1");
  });

  it("does not execute when the disclosure is cancelled", () => {
    const execute = vi.fn();
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
      execute,
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExecuteButton />);
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));
    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));

    expect(execute).not.toHaveBeenCalled();
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("skips the disclosure once 'don't ask again' has been confirmed for the session", () => {
    const execute = vi.fn();
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
      execute,
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExecuteButton />);

    // First run: tick the opt-out, confirm.
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));
    const dialog = screen.getByRole("alertdialog", { name: /run pipeline\?/i });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /don't ask again/i }),
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: /^run pipeline$/i }),
    );
    expect(execute).toHaveBeenCalledTimes(1);
    expect(
      useExecutionStore.getState().runDisclosureAckBySession["sess-1"],
    ).toBe(true);

    // Second run: no dialog, executes directly.
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(execute).toHaveBeenCalledTimes(2);
  });

  it("executes directly when the session already opted out of the disclosure", () => {
    const execute = vi.fn();
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
      execute,
      runDisclosureAckBySession: { "sess-1": true },
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExecuteButton />);
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));

    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(execute).toHaveBeenCalledWith("sess-1");
  });

  // ── Phase 5b.18b.7 — interpretation-review run gating ──────────────────────
  //
  // Three tests pin the spec contract (18b lines 702-722, test §6-§8):
  //
  //  §6 — pending event AND NOT opted out → aria-disabled + focusable +
  //       no-op activation + title + aria-describedby. Native `disabled`
  //       must NOT be set: it removes the button from the tab order and
  //       makes the blocked reason unreachable for exactly the keyboard/
  //       screen-reader users it exists for (WCAG 4.1.2, elspeth-94c32de486).
  //  §7 — last pending event resolved → button re-enables.
  //  §8 — opted-out session → button enabled regardless of any residual
  //       store entries (opt-out clears pendingBySession in production, but
  //       this test guards the predicate logic directly so a future
  //       store-reorder doesn't silently regress the contract).

  it("blocks Run pipeline as focusable aria-disabled when a pending interpretation event exists and the session is NOT opted out (Phase 5b.18b.7 §6)", () => {
    const execute = vi.fn();
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
      execute,
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useInterpretationEventsStore.setState({
      pendingBySession: { "sess-1": { "evt-1": makeInterpretationEvent() } },
      optedOutBySession: {},
    });

    render(<ExecuteButton />);

    const button = screen.getByRole("button", { name: /run pipeline/i });
    // NOT native-disabled: the button must stay in the tab order so the
    // aria-describedby reason is reachable (elspeth-94c32de486).
    expect(button).not.toBeDisabled();
    expect(button).toHaveAttribute("aria-disabled", "true");
    button.focus();
    expect(button).toHaveFocus();
    expect(button).toHaveAttribute(
      "title",
      INTERPRETATION_PENDING_RUN_BLOCK_TITLE,
    );
    // aria-describedby points at a visually-hidden span that carries the same
    // text — the WCAG-canonical announcement path.
    const describedById = button.getAttribute("aria-describedby");
    expect(describedById).toBeTruthy();
    const desc = document.getElementById(describedById!);
    expect(desc).not.toBeNull();
    expect(desc?.textContent).toBe(INTERPRETATION_PENDING_RUN_BLOCK_TITLE);
    // Activation is a no-op while blocked.
    fireEvent.click(button);
    expect(execute).not.toHaveBeenCalled();
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("re-enables Run pipeline after the last pending interpretation event is resolved (Phase 5b.18b.7 §7)", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    // Start with a pending event so the button is initially disabled.
    useInterpretationEventsStore.setState({
      pendingBySession: { "sess-1": { "evt-1": makeInterpretationEvent() } },
      optedOutBySession: {},
    });

    const { rerender } = render(<ExecuteButton />);
    expect(
      screen.getByRole("button", { name: /run pipeline/i }),
    ).toHaveAttribute("aria-disabled", "true");

    // Clear the pending map (simulating resolveEvent's store mutation).
    useInterpretationEventsStore.setState({
      pendingBySession: { "sess-1": {} },
    });
    rerender(<ExecuteButton />);

    const button = screen.getByRole("button", { name: /run pipeline/i });
    expect(button).not.toBeDisabled();
    expect(button).not.toHaveAttribute("title");
    expect(button).not.toHaveAttribute("aria-describedby");
  });

  it("keeps Run pipeline enabled when the session is opted out, even with residual store entries (Phase 5b.18b.7 §8)", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
      } as never,
      isExecuting: false,
      progress: null,
    } as never);
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    // Opt-out true AND a residual pending entry. The production opt-out
    // flow clears pendingBySession; this test pins the predicate's NOT-
    // opted-out conjunction so a regression that flipped the conjunction
    // (or omitted the opt-out check) would be caught here.
    useInterpretationEventsStore.setState({
      pendingBySession: { "sess-1": { "evt-1": makeInterpretationEvent() } },
      optedOutBySession: { "sess-1": true },
    });

    render(<ExecuteButton />);

    const button = screen.getByRole("button", { name: /run pipeline/i });
    expect(button).not.toBeDisabled();
    expect(button).not.toHaveAttribute("title");
  });
});

describe("buildRunEgressSummary", () => {
  it("returns no lines for a null composition", () => {
    expect(buildRunEgressSummary(null)).toEqual([]);
  });

  it("derives sources, LLM/model nodes, network fetches, and sinks from the composition", () => {
    const lines = buildRunEgressSummary(
      makeComposition({
        sources: {
          source: { plugin: "csv", options: {}, on_success: "fetch_in" },
        },
        nodes: [
          {
            id: "fetch_page",
            node_type: "transform",
            plugin: "web_scrape",
            input: "fetch_in",
            on_success: "classify_in",
            on_error: null,
            options: {},
          },
          {
            id: "classify",
            node_type: "transform",
            plugin: "llm",
            input: "classify_in",
            on_success: "results",
            on_error: null,
            options: { model: "openrouter/anthropic/claude-sonnet-4.6" },
          },
        ],
        outputs: [
          { name: "results", plugin: "csv", options: {} },
          { name: "errors", plugin: "json", options: {} },
        ],
      }),
    );

    expect(lines).toEqual([
      "Reads source data: source (csv).",
      "Sends rows to the configured LLM: classify (model openrouter/anthropic/claude-sonnet-4.6).",
      "Fetches over the network: fetch_page (web_scrape).",
      "Writes output: results (csv), errors (json).",
    ]);
  });

  it("omits categories the composition does not contain", () => {
    const lines = buildRunEgressSummary(
      makeComposition({
        outputs: [{ name: "results", plugin: "csv", options: {} }],
      }),
    );
    expect(lines).toEqual(["Writes output: results (csv)."]);
  });
});
