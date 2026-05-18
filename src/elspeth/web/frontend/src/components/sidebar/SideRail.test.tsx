import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { SideRail } from "./SideRail";

describe("SideRail", () => {
  it("renders the audit-readiness slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-audit-readiness"),
    ).toBeInTheDocument();
  });

  it("renders the graph mini slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-graph-mini"),
    ).toBeInTheDocument();
  });

  it("renders the validation banner slot region above the graph mini slot", () => {
    render(
      <SideRail
        validationBannerSlot={<div>Validation banner</div>}
        graphMiniSlot={<div>Graph mini</div>}
      />,
    );

    const validationSlot = screen.getByTestId(
      "siderail-slot-validation-banner",
    );
    const graphSlot = screen.getByTestId("siderail-slot-graph-mini");
    expect(within(validationSlot).getByText("Validation banner")).toBeInTheDocument();
    expect(validationSlot.compareDocumentPosition(graphSlot)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  it("renders the catalog slot region", () => {
    render(<SideRail />);
    expect(screen.getByTestId("siderail-slot-catalog")).toBeInTheDocument();
  });

  it("renders the completion-bar slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-completion-bar"),
    ).toBeInTheDocument();
  });

  it("renders content passed via the completionBar slot prop", () => {
    render(
      <SideRail
        completionBarSlot={<button type="button">Run pipeline</button>}
      />,
    );
    const slot = screen.getByTestId("siderail-slot-completion-bar");
    expect(
      within(slot).getByRole("button", { name: /run pipeline/i }),
    ).toBeInTheDocument();
  });

  it("does not render the retired transitional inspector mount", () => {
    const { container } = render(<SideRail />);
    expect(container.querySelector("[class*='transitional']")).toBeNull();
  });

  it("no longer exposes the retired exportYaml / executeButton slot regions", () => {
    // Phase 6B Tasks 9-10 retired the standalone ExportYaml + Execute
    // button slots in favour of the consolidated CompletionBar slot.
    // Per the No Legacy Code Policy (CLAUDE.md), the placeholder slots
    // must not linger as dead `null`-pass-through DOM nodes. This test
    // pins their absence so a future maintainer who restores the props
    // is forced to confront whether the slot model has actually
    // regressed.
    render(<SideRail />);
    expect(screen.queryByTestId("siderail-slot-export-yaml")).toBeNull();
    expect(screen.queryByTestId("siderail-slot-execute-button")).toBeNull();
  });
});
