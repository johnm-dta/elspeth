import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
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

  it("renders children passed through (the transitional inspector mount)", () => {
    render(
      <SideRail>
        <div data-testid="children-marker" />
      </SideRail>,
    );
    expect(screen.getByTestId("children-marker")).toBeInTheDocument();
  });
});
