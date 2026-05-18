import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { CompletionBar } from "./CompletionBar";
import { useShareableReviewStore } from "@/stores/shareableReviewStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";

function _validValidation() {
  return {
    is_valid: true,
    checks: [],
    errors: [],
  } as never;
}

function _invalidValidation() {
  return {
    is_valid: false,
    checks: [],
    errors: [
      {
        component_id: "node1",
        component_type: "transform",
        message: "boom",
        suggestion: null,
      },
    ],
  } as never;
}

describe("CompletionBar", () => {
  beforeEach(() => {
    useSessionStore.setState({ activeSessionId: null } as never);
    useExecutionStore.setState({
      validationResult: null,
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    useShareableReviewStore.getState().reset();
    resetStore(useInterpretationEventsStore);
  });

  it("renders nothing without an active session", () => {
    const { container } = render(<CompletionBar />);
    expect(container.firstChild).toBeNull();
  });

  it("renders three co-equal buttons when a session is active and validation passes", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      validationResult: _validValidation(),
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    render(<CompletionBar />);
    expect(screen.getByTestId("completion-bar")).toBeInTheDocument();
    expect(screen.getByTestId("completion-bar-save-for-review")).toBeInTheDocument();
    expect(screen.getByTestId("completion-bar-run-pipeline")).toBeInTheDocument();
    expect(screen.getByTestId("completion-bar-export-yaml")).toBeInTheDocument();
  });

  it("disables Save for review when validation has not run", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    // validationResult: null — no validation has been run.
    render(<CompletionBar />);
    const button = screen.getByTestId("completion-bar-save-for-review") as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(button.getAttribute("title")).toMatch(/fix validation/i);
  });

  it("disables Save for review when validation is invalid", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      validationResult: _invalidValidation(),
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    render(<CompletionBar />);
    const button = screen.getByTestId("completion-bar-save-for-review") as HTMLButtonElement;
    expect(button.disabled).toBe(true);
  });

  it("enables Save for review when validation is valid", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      validationResult: _validValidation(),
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    render(<CompletionBar />);
    const button = screen.getByTestId("completion-bar-save-for-review") as HTMLButtonElement;
    expect(button.disabled).toBe(false);
  });

  it("clicking Save for review invokes openAndMark with the active session id", () => {
    useSessionStore.setState({ activeSessionId: "sess-XYZ" } as never);
    useExecutionStore.setState({
      validationResult: _validValidation(),
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    const openAndMarkSpy = vi.fn();
    useShareableReviewStore.setState({ openAndMark: openAndMarkSpy } as never);

    render(<CompletionBar />);
    fireEvent.click(screen.getByTestId("completion-bar-save-for-review"));
    expect(openAndMarkSpy).toHaveBeenCalledWith("sess-XYZ");
  });

  it("disables Save for review while a mark request is in flight", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      validationResult: _validValidation(),
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    useShareableReviewStore.setState({ inFlight: true } as never);
    render(<CompletionBar />);
    const button = screen.getByTestId("completion-bar-save-for-review") as HTMLButtonElement;
    expect(button.disabled).toBe(true);
  });
});
