import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GraphModal } from "./GraphModal";
// (Close control uses the standard × glyph — pinned below alongside the
// aria-label so the lowercase-'x' regression cannot recur; elspeth-83eb51334f
// wave-3 carry-over, same defect class as elspeth-bff8043d33's modal close.)
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";

// The stub must include a focusable element so the focus trap has two nodes
// (Close button + stub button) and the wrap-around logic is exercised. A
// stub with zero focusable children would produce a single-element cycle that
// passes trivially even without useFocusTrap wired.
vi.mock("@/components/inspector/GraphView", () => ({
  GraphView: () => (
    <button type="button" data-testid="graph-view-stub">
      stub
    </button>
  ),
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

  it("uses the standard × close glyph (not a lowercase 'x')", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));

    const closeBtn = screen.getByRole("button", { name: /close graph/i });
    expect(closeBtn.textContent?.trim()).toBe("×");
  });

  // ── Focus trap ──────────────────────────────────────────────────────────────
  // useFocusTrap is wired with initialFocusSelector ".graph-modal-close" so
  // the close button receives focus on open.  Tab cycles forward to the stub
  // button and Shift+Tab wraps back — verifying the trap does not leak focus
  // outside the modal.

  it("moves focus to the close button when the modal opens", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));

    expect(screen.getByRole("button", { name: /close graph/i })).toHaveFocus();
  });

  it("traps Tab forward through focusable elements without escaping the modal", async () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));

    const closeBtn = screen.getByRole("button", { name: /close graph/i });
    const stubBtn = screen.getByTestId("graph-view-stub");

    closeBtn.focus();
    await userEvent.tab();
    expect(stubBtn).toHaveFocus();

    // Wraps back to first — focus stays inside the modal
    await userEvent.tab();
    expect(closeBtn).toHaveFocus();
  });

  it("traps Shift+Tab backward through focusable elements without escaping the modal", async () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));

    const closeBtn = screen.getByRole("button", { name: /close graph/i });
    const stubBtn = screen.getByTestId("graph-view-stub");

    closeBtn.focus();
    await userEvent.tab({ shift: true });
    // Wraps from first to last
    expect(stubBtn).toHaveFocus();

    await userEvent.tab({ shift: true });
    expect(closeBtn).toHaveFocus();
  });
});
