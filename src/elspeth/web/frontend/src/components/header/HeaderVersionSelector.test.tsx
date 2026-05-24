import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { HeaderVersionSelector } from "./HeaderVersionSelector";
import { useSessionStore } from "@/stores/sessionStore";

describe("HeaderVersionSelector", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: {
        version: 3,
        sources: {},
        nodes: [],
        edges: [],
        outputs: [],
      } as never,
      stateVersions: [],
      isLoadingVersions: false,
      loadStateVersions: vi.fn(),
      revertToVersion: vi.fn(),
    } as never);
  });

  it("renders nothing when no active session", () => {
    useSessionStore.setState({ activeSessionId: null } as never);
    const { container } = render(<HeaderVersionSelector />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the current composition version label", () => {
    render(<HeaderVersionSelector />);
    expect(screen.getByText(/v3|version 3/i)).toBeInTheDocument();
  });

  it("uses the design-spec label 'Composition history' on the dropdown trigger", () => {
    render(<HeaderVersionSelector />);
    expect(
      screen.getByRole("button", { name: /composition history/i }),
    ).toBeInTheDocument();
  });

  it("calls loadStateVersions when the dropdown opens", () => {
    const loadStateVersions = vi.fn();
    useSessionStore.setState({ loadStateVersions } as never);
    render(<HeaderVersionSelector />);

    fireEvent.click(
      screen.getByRole("button", { name: /composition history/i }),
    );

    expect(loadStateVersions).toHaveBeenCalled();
  });

  it("confirms and calls revertToVersion when the user picks an older version", () => {
    const revertToVersion = vi.fn();
    useSessionStore.setState({
      stateVersions: [
        {
          id: "st-1",
          version: 1,
          created_at: "2026-05-15T10:00:00Z",
          node_count: 1,
        } as never,
        {
          id: "st-2",
          version: 2,
          created_at: "2026-05-15T10:10:00Z",
          node_count: 2,
        } as never,
        {
          id: "st-3",
          version: 3,
          created_at: "2026-05-15T10:20:00Z",
          node_count: 3,
        } as never,
      ],
      revertToVersion,
    } as never);
    render(<HeaderVersionSelector />);

    fireEvent.click(
      screen.getByRole("button", { name: /composition history/i }),
    );
    fireEvent.click(screen.getByRole("option", { name: /^version 2$/i }));
    fireEvent.click(screen.getByRole("button", { name: /revert to version 2/i }));
    fireEvent.click(screen.getByRole("button", { name: /^revert$/i }));

    expect(revertToVersion).toHaveBeenCalledWith("st-2");
  });
});
