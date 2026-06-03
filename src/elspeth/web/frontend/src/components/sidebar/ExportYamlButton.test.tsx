import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ExportYamlButton } from "./ExportYamlButton";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";

describe("ExportYamlButton", () => {
  beforeEach(() => {
    useSessionStore.setState({ activeSessionId: null } as never);
  });

  it("renders nothing when there is no active session", () => {
    const { container } = render(<ExportYamlButton />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the export button when a session is active", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExportYamlButton />);

    expect(
      screen.getByRole("button", { name: /export yaml/i }),
    ).toBeInTheDocument();
  });

  it("dispatches OPEN_YAML_MODAL_EVENT on click", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);

    render(<ExportYamlButton />);
    fireEvent.click(screen.getByRole("button", { name: /export yaml/i }));

    expect(handler).toHaveBeenCalled();
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
  });
});
