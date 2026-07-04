import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ImportYamlButton } from "./ImportYamlButton";
import { useSessionStore } from "@/stores/sessionStore";

vi.mock("@/api/client", () => ({
  importCompositionYaml: vi.fn(),
}));

describe("ImportYamlButton", () => {
  afterEach(() => {
    useSessionStore.setState({
      activeSessionId: null,
      compositionState: null,
      guidedSession: null,
    } as never);
  });

  it("renders nothing when there is no active session", () => {
    const { container } = render(<ImportYamlButton />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the import button when a session is active", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ImportYamlButton />);

    expect(
      screen.getByRole("button", { name: /import yaml/i }),
    ).toBeInTheDocument();
  });

  it("opens the import modal on click", () => {
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
    } as never);

    render(<ImportYamlButton />);
    expect(screen.queryByRole("dialog")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /import yaml/i }));

    expect(
      screen.getByRole("dialog", { name: /import yaml/i }),
    ).toBeInTheDocument();
  });

  it("closes the modal when the modal's Cancel action fires", () => {
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
    } as never);

    render(<ImportYamlButton />);
    fireEvent.click(screen.getByRole("button", { name: /import yaml/i }));
    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
