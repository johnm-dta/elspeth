import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { Layout } from "./Layout";

vi.mock("./DefaultModeChangedBanner", () => ({
  DefaultModeChangedBanner: () => (
    <div data-testid="default-mode-changed-banner" />
  ),
}));

// cwd-relative paths per the established test idiom (GraphView.test.tsx reads
// its stylesheet the same way; vitest runs from the frontend root).
const sharedCss = readFileSync("src/styles/shared.css", "utf8");
const tokensCss = readFileSync("src/styles/tokens.css", "utf8");

describe("Layout", () => {
  it("leaves column sizing to the stylesheet so the responsive collapse can restructure the grid", () => {
    // The rail width was previously a hardcoded inline style
    // (grid-template-columns: minmax(0, 1fr) 320px) which out-specified every
    // stylesheet rule — no media query could collapse the shell below 960px
    // (elspeth-49dd290c7a). Sizing now lives in shared.css via the
    // --siderail-width token; the component must not reintroduce it inline.
    const { container } = render(
      <Layout
        chat={<div>Chat</div>}
        siderail={<div>Side rail</div>}
      />,
    );

    const layout = container.querySelector(".app-layout") as HTMLElement;
    expect(layout).not.toBeNull();
    expect(layout.style.gridTemplateColumns).toBe("");
  });

  it("sizes the rail from the --siderail-width token in the stylesheet", () => {
    expect(tokensCss).toMatch(/--siderail-width:\s*320px;/);
    expect(sharedCss).toMatch(
      /\.app-layout\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s*var\(--siderail-width\);/,
    );
  });

  it("collapses to a single column with a scrollable stacked rail below 960px", () => {
    // WCAG 1.4.10 Reflow: below the 960px breakpoint the shell must reflow
    // (single column, rail stacked and scrollable) rather than clip. Mirrors
    // the guided (900px) / tutorial (760px) collapse discipline.
    const mediaBlockMatch = /@media \(max-width: 960px\)\s*\{([\s\S]*?)\n\}/.exec(
      sharedCss,
    );
    expect(mediaBlockMatch, "shared.css must define the 960px collapse").not.toBeNull();
    const mediaBlock = mediaBlockMatch![1];
    expect(mediaBlock).toMatch(
      /\.app-layout\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\);/,
    );
    expect(mediaBlock).toContain('"chat"');
    expect(mediaBlock).toContain('"siderail"');
    expect(mediaBlock).toMatch(/\.app-layout\s*\{[^}]*min-width:\s*0;/);
    expect(mediaBlock).toMatch(/\.layout-siderail\s*\{[^}]*overflow-y:\s*auto;/);
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
