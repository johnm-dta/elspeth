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

  it("positions the completion-bar slot below audit-readiness and above the catalog slot (plan 19b:602)", () => {
    // Plan 19b:602 mandates: "<CompletionBar /> is present in the expected
    // position (below audit-readiness, above Catalog)." This test pins the
    // DOM order so a regression that re-shuffles the slots is caught here
    // rather than at staging.
    const { container } = render(
      <SideRail
        auditReadinessSlot={<div>Audit readiness</div>}
        validationBannerSlot={<div>Validation banner</div>}
        graphMiniSlot={<div>Graph mini</div>}
        catalogSlot={<div>Catalog</div>}
        completionBarSlot={<div>Completion bar</div>}
      />,
    );

    const slots = Array.from(
      container.querySelectorAll<HTMLElement>('[data-testid^="siderail-slot-"]'),
    );
    const indexOf = (testid: string): number =>
      slots.findIndex((node) => node.dataset.testid === testid);

    const auditIdx = indexOf("siderail-slot-audit-readiness");
    const completionIdx = indexOf("siderail-slot-completion-bar");
    const catalogIdx = indexOf("siderail-slot-catalog");

    expect(auditIdx).toBeGreaterThanOrEqual(0);
    expect(completionIdx).toBeGreaterThanOrEqual(0);
    expect(catalogIdx).toBeGreaterThanOrEqual(0);
    expect(auditIdx).toBeLessThan(completionIdx);
    expect(completionIdx).toBeLessThan(catalogIdx);
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
