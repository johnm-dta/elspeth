import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ExportYamlModal } from "./ExportYamlModal";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

// The stub must include a focusable element so the focus trap has two nodes
// (Close button + stub button) and the wrap-around logic is exercised. A
// stub with zero focusable children would produce a single-element cycle that
// passes trivially even without useFocusTrap wired.
vi.mock("@/components/inspector/YamlView", () => ({
  YamlView: () => (
    <button type="button" data-testid="yaml-view-stub">
      stub
    </button>
  ),
}));

describe("ExportYamlModal", () => {
  it("renders nothing until opened", () => {
    const { container } = render(<ExportYamlModal />);
    expect(container.querySelector("[role='dialog']")).toBeNull();
  });

  it("opens on OPEN_YAML_MODAL_EVENT", () => {
    render(<ExportYamlModal />);

    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));

    expect(
      screen.getByRole("dialog", { name: /export yaml/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("dialog", { name: /review yaml/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("yaml-view-stub")).toBeInTheDocument();
  });

  it("closes on Escape", () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));

    fireEvent.keyDown(document, { key: "Escape" });

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes when the backdrop is clicked", () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));

    fireEvent.click(screen.getByTestId("yaml-modal-backdrop"));

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes when the close button is clicked", () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));

    fireEvent.click(screen.getByRole("button", { name: /close export yaml/i }));

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  // ── Focus trap ──────────────────────────────────────────────────────────────
  // useFocusTrap is wired with initialFocusSelector ".yaml-modal-close" so
  // the close button receives focus on open.  Tab cycles forward to the stub
  // button and Shift+Tab wraps back — verifying the trap does not leak focus
  // outside the modal.

  it("moves focus to the close button when the modal opens", () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));

    expect(
      screen.getByRole("button", { name: /close export yaml/i }),
    ).toHaveFocus();
  });

  it("traps Tab forward through focusable elements without escaping the modal", async () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));

    const closeBtn = screen.getByRole("button", { name: /close export yaml/i });
    const stubBtn = screen.getByTestId("yaml-view-stub");

    closeBtn.focus();
    await userEvent.tab();
    expect(stubBtn).toHaveFocus();

    // Wraps back to first — focus stays inside the modal
    await userEvent.tab();
    expect(closeBtn).toHaveFocus();
  });

  it("traps Shift+Tab backward through focusable elements without escaping the modal", async () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));

    const closeBtn = screen.getByRole("button", { name: /close export yaml/i });
    const stubBtn = screen.getByTestId("yaml-view-stub");

    closeBtn.focus();
    await userEvent.tab({ shift: true });
    // Wraps from first to last
    expect(stubBtn).toHaveFocus();

    await userEvent.tab({ shift: true });
    expect(closeBtn).toHaveFocus();
  });
});
