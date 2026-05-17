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

  it("renders the export-yaml slot region", () => {
    render(<SideRail />);
    expect(screen.getByTestId("siderail-slot-export-yaml")).toBeInTheDocument();
  });

  it("renders the execute-button slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-execute-button"),
    ).toBeInTheDocument();
  });

  it("renders the completion-bar slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-completion-bar"),
    ).toBeInTheDocument();
  });

  it("renders content passed via the executeButton slot prop", () => {
    render(<SideRail executeButtonSlot={<button>Run</button>} />);
    expect(screen.getByRole("button", { name: /run/i })).toBeInTheDocument();
  });

  it("places executeButtonSlot content in the execute-button slot", () => {
    render(
      <SideRail
        executeButtonSlot={<button type="button">Run pipeline</button>}
      />,
    );
    const slot = screen.getByTestId("siderail-slot-execute-button");
    expect(
      within(slot).getByRole("button", { name: /run pipeline/i }),
    ).toBeInTheDocument();
  });

  it("does not render the retired transitional inspector mount", () => {
    const { container } = render(<SideRail />);
    expect(container.querySelector("[class*='transitional']")).toBeNull();
  });
});
