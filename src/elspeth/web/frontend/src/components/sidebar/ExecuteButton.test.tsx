import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ExecuteButton } from "./ExecuteButton";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";

describe("ExecuteButton", () => {
  beforeEach(() => {
    useExecutionStore.setState({
      validationResult: null,
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    useSessionStore.setState({ activeSessionId: null } as never);
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
});
