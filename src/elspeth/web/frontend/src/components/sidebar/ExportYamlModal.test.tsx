import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ExportYamlModal } from "./ExportYamlModal";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

vi.mock("@/components/inspector/YamlView", () => ({
  YamlView: () => <div data-testid="yaml-view-stub" />,
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
});
