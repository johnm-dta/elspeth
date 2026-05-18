import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { CompletionBar } from "./CompletionBar";
import { ExportYamlModal } from "@/components/sidebar/ExportYamlModal";
import { useShareableReviewStore } from "@/stores/shareableReviewStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";

// The Export-YAML dialog body renders YamlView, which pulls in session state
// and YAML rendering machinery irrelevant to the assertions in this file.
// Stub it so the dialog mounts deterministically and AC 7 can assert only the
// dialog itself (plan 19b:232).
vi.mock("@/components/inspector/YamlView", () => ({
  YamlView: () => (
    <button type="button" data-testid="yaml-view-stub">
      stub
    </button>
  ),
}));

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

  // Plan 19b:229 — AC 4: Export-YAML button is ALWAYS enabled, even when no
  // validation has run and when validation is invalid. The button is the only
  // completion-bar verb that the design doc (09) marks "Available always —
  // even with warning status", so we pin both extremes: null validation (no
  // run yet) and explicit invalid validation.
  it("Export-YAML button is always enabled regardless of validation state (plan 19b:229, AC 4)", () => {
    // Case 1: no validation run yet (validationResult === null).
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    // beforeEach already sets validationResult: null.
    const { unmount } = render(<CompletionBar />);
    const exportContainer = screen.getByTestId("completion-bar-export-yaml");
    const exportButtonNull = exportContainer.querySelector("button") as HTMLButtonElement;
    expect(exportButtonNull).not.toBeNull();
    expect(exportButtonNull.disabled).toBe(false);
    expect(exportButtonNull.getAttribute("aria-disabled")).toBeNull();
    unmount();

    // Case 2: explicit invalid validation — Save-for-review and Run-pipeline
    // both become disabled, but Export-YAML must remain enabled.
    useExecutionStore.setState({
      validationResult: _invalidValidation(),
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    render(<CompletionBar />);
    const exportContainerInvalid = screen.getByTestId("completion-bar-export-yaml");
    const exportButtonInvalid = exportContainerInvalid.querySelector("button") as HTMLButtonElement;
    expect(exportButtonInvalid).not.toBeNull();
    expect(exportButtonInvalid.disabled).toBe(false);
    expect(exportButtonInvalid.getAttribute("aria-disabled")).toBeNull();
  });

  // Plan 19b:231 — AC 6: Clicking Run-pipeline calls the existing Execute
  // action from executionStore. The ExecuteButton primitive carries the
  // signature `execute(activeSessionId)` (see ExecuteButton.tsx line 74).
  it("clicking Run-pipeline calls executionStore.execute with the active session id (plan 19b:231, AC 6)", () => {
    const executeSpy = vi.fn();
    useSessionStore.setState({ activeSessionId: "sess-RUN" } as never);
    useExecutionStore.setState({
      validationResult: _validValidation(),
      isExecuting: false,
      progress: null,
      execute: executeSpy,
    } as never);

    render(<CompletionBar />);
    const runContainer = screen.getByTestId("completion-bar-run-pipeline");
    const runButton = runContainer.querySelector("button") as HTMLButtonElement;
    expect(runButton).not.toBeNull();
    expect(runButton.disabled).toBe(false);

    fireEvent.click(runButton);

    expect(executeSpy).toHaveBeenCalledTimes(1);
    expect(executeSpy).toHaveBeenCalledWith("sess-RUN");
  });

  // Plan 19b:232 — AC 7: Clicking Export-YAML opens the existing YamlView
  // modal. ExportYamlButton dispatches a window CustomEvent
  // (OPEN_YAML_MODAL_EVENT) that ExportYamlModal listens for; rendering both
  // components together lets us assert the dialog becomes visible end-to-end.
  it("clicking Export-YAML opens the ExportYamlModal dialog (plan 19b:232, AC 7)", async () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useExecutionStore.setState({
      validationResult: _validValidation(),
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);

    render(
      <>
        <CompletionBar />
        <ExportYamlModal />
      </>,
    );

    // Dialog should not be mounted before the click.
    expect(screen.queryByRole("dialog")).toBeNull();

    const exportContainer = screen.getByTestId("completion-bar-export-yaml");
    const exportButton = exportContainer.querySelector("button") as HTMLButtonElement;
    fireEvent.click(exportButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog", { name: /export yaml/i })).toBeInTheDocument();
    });
  });
});
