import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AuditCharacteristicIcon } from "./AuditCharacteristicIcon";

describe("AuditCharacteristicIcon", () => {
  it("renders the label and glyph for a known flag", () => {
    render(<AuditCharacteristicIcon flag="io_read" />);
    expect(screen.getByText(/reads i\/?o/i)).toBeInTheDocument();
  });

  it("uses a positive-tone class for io_read", () => {
    const { container } = render(<AuditCharacteristicIcon flag="io_read" />);
    expect(container.firstChild).toHaveClass("audit-icon-positive");
  });

  it("uses an attention-tone class for external_call", () => {
    const { container } = render(<AuditCharacteristicIcon flag="external_call" />);
    expect(container.firstChild).toHaveClass("audit-icon-attention");
  });

  it("renders the tooltip on the title attribute", () => {
    render(<AuditCharacteristicIcon flag="quarantine" />);
    const el = screen.getByText(/quarantines/i);
    // Tooltip via title for keyboard / screen-reader access without
    // pulling in a tooltip library.
    expect(el.closest("[title]")?.getAttribute("title")).toMatch(/sink/i);
  });

  it("renders unknown flags as a fallback chip with the raw flag string", () => {
    render(<AuditCharacteristicIcon flag="future_flag_2027" />);
    expect(screen.getByText("future_flag_2027")).toBeInTheDocument();
  });

  it("applies an 'audit-icon-unknown' class for unknown flags", () => {
    const { container } = render(
      <AuditCharacteristicIcon flag="future_flag_2027" />,
    );
    expect(container.firstChild).toHaveClass("audit-icon-unknown");
  });
});
