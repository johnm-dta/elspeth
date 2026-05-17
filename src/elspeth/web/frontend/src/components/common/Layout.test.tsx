import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { Layout } from "./Layout";

vi.mock("./DefaultModeChangedBanner", () => ({
  DefaultModeChangedBanner: () => (
    <div data-testid="default-mode-changed-banner" />
  ),
}));

describe("Layout", () => {
  it("uses the fixed 320px composer side rail width", () => {
    const { container } = render(
      <Layout
        chat={<div>Chat</div>}
        siderail={<div>Side rail</div>}
      />,
    );

    const layout = container.querySelector(".app-layout") as HTMLElement;
    expect(layout.style.gridTemplateColumns).toBe("minmax(0, 1fr) 320px");
  });

  it("keeps the default-mode banner inside the chat column", () => {
    render(
      <Layout
        chat={<div>Chat</div>}
        siderail={<div>Side rail</div>}
      />,
    );

    const chatColumn = screen.getByTestId("layout-chat");
    expect(
      within(chatColumn).getByTestId("default-mode-changed-banner"),
    ).toBeInTheDocument();
  });

  it("does not render the retired overlay, toggle, or resize controls", () => {
    render(
      <Layout
        chat={<div>Chat</div>}
        siderail={<div>Side rail</div>}
      />,
    );

    expect(screen.queryByRole("separator")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /side rail/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /close side rail/i }),
    ).not.toBeInTheDocument();
  });
});
