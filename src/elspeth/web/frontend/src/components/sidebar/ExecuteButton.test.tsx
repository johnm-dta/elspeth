import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import {
  ExecuteButton,
  INTERPRETATION_PENDING_RUN_BLOCK_TITLE,
} from "./ExecuteButton";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
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

describe("ExecuteButton", () => {
  beforeEach(() => {
    useExecutionStore.setState({
      validationResult: null,
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    useSessionStore.setState({ activeSessionId: null } as never);
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

  it("invokes execute with the active session id when clicked", () => {
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

    expect(execute).toHaveBeenCalledWith("sess-1");
  });

  // ── Phase 5b.18b.7 — interpretation-review run gating ──────────────────────
  //
  // Three tests pin the spec contract (18b lines 702-722, test §6-§8):
  //
  //  §6 — pending event AND NOT opted out → disabled + title + aria-describedby.
  //  §7 — last pending event resolved → button re-enables.
  //  §8 — opted-out session → button enabled regardless of any residual
  //       store entries (opt-out clears pendingBySession in production, but
  //       this test guards the predicate logic directly so a future
  //       store-reorder doesn't silently regress the contract).

  it("disables Run pipeline when a pending interpretation event exists and the session is NOT opted out (Phase 5b.18b.7 §6)", () => {
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
    useInterpretationEventsStore.setState({
      pendingBySession: { "sess-1": { "evt-1": makeInterpretationEvent() } },
      optedOutBySession: {},
    });

    render(<ExecuteButton />);

    const button = screen.getByRole("button", { name: /run pipeline/i });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-disabled", "true");
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
    ).toBeDisabled();

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
