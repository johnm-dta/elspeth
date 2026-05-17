import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { GraphModal } from "./GraphModal";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";

vi.mock("@/components/inspector/GraphView", () => ({
  GraphView: () => <div data-testid="graph-view-stub" />,
}));

describe("GraphModal", () => {
  it("renders nothing when not opened", () => {
    const { container } = render(<GraphModal />);
    expect(container.querySelector("[role='dialog']")).toBeNull();
  });

  it("opens on OPEN_GRAPH_MODAL_EVENT", () => {
    render(<GraphModal />);

    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));

    expect(
      screen.getByRole("dialog", { name: /pipeline graph/i }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("graph-view-stub")).toBeInTheDocument();
  });

  it("closes on Escape", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));

    fireEvent.keyDown(document, { key: "Escape" });

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes when the backdrop is clicked", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));

    fireEvent.click(screen.getByTestId("graph-modal-backdrop"));

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes when the close button is clicked", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));

    fireEvent.click(screen.getByRole("button", { name: /close graph/i }));

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
