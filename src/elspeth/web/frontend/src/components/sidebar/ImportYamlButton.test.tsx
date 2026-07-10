import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ImportYamlButton } from "./ImportYamlButton";
import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_IMPORT_YAML_MODAL_EVENT } from "@/lib/composer-events";

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

  it("dispatches the app-level open event instead of mounting a side-rail dialog", () => {
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
    } as never);
    const handler = vi.fn();
    window.addEventListener(OPEN_IMPORT_YAML_MODAL_EVENT, handler);

    try {
      render(<ImportYamlButton />);

      fireEvent.click(screen.getByRole("button", { name: /import yaml/i }));

      expect(handler).toHaveBeenCalledOnce();
      expect(screen.queryByRole("dialog", { name: /import yaml/i })).toBeNull();
    } finally {
      window.removeEventListener(OPEN_IMPORT_YAML_MODAL_EVENT, handler);
    }
  });
});
