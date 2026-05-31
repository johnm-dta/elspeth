import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AuditReadinessRow, type RowPresentation } from "./AuditReadinessRow";
import { ReadOnlyProvider } from "../../contexts/ReadOnlyContext";
import type { ReadinessStatus } from "../../types/api";

function makeRow(status: ReadinessStatus): RowPresentation {
  return {
    id: "provenance",
    status,
    heading: "Provenance",
    summaryText: `Provenance ${status}`,
    glyph: status === "error" ? "✗" : status === "warning" ? "⚠" : "✓",
    ariaStatusLabel:
      status === "error"
        ? "Error"
        : status === "warning"
          ? "Warning"
          : status === "ok"
            ? "OK"
            : "Not applicable",
  };
}

describe("AuditReadinessRow", () => {
  it("renders the static variant for an ok row regardless of onSelect", () => {
    const onSelect = vi.fn();
    render(
      <ul>
        <AuditReadinessRow row={makeRow("ok")} onSelect={onSelect} />
      </ul>,
    );
    expect(
      screen.queryByRole("button", { name: /provenance/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Provenance ok")).toBeInTheDocument();
  });

  it("renders an actionable button when status is warning AND onSelect is supplied", async () => {
    const onSelect = vi.fn();
    render(
      <ul>
        <AuditReadinessRow row={makeRow("warning")} onSelect={onSelect} />
      </ul>,
    );
    const btn = screen.getByRole("button");
    expect(btn).toHaveTextContent("Provenance");
    const user = userEvent.setup();
    await user.click(btn);
    expect(onSelect).toHaveBeenCalledWith("provenance");
  });

  it("renders an actionable button for error status", () => {
    render(
      <ul>
        <AuditReadinessRow row={makeRow("error")} onSelect={() => {}} />
      </ul>,
    );
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("renders the static variant when onSelect is omitted (no actionability without a handler)", () => {
    render(
      <ul>
        <AuditReadinessRow row={makeRow("warning")} />
      </ul>,
    );
    expect(
      screen.queryByRole("button", { name: /provenance/i }),
    ).not.toBeInTheDocument();
  });

  it("forces the static variant inside ReadOnlyProvider even for actionable status + onSelect", () => {
    const onSelect = vi.fn();
    render(
      <ReadOnlyProvider value={true}>
        <ul>
          <AuditReadinessRow row={makeRow("error")} onSelect={onSelect} />
        </ul>
      </ReadOnlyProvider>,
    );
    expect(
      screen.queryByRole("button", { name: /provenance/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Provenance error")).toBeInTheDocument();
  });

  it("applies extraClassName and testId when supplied", () => {
    render(
      <ul>
        <AuditReadinessRow
          row={{
            ...makeRow("not_applicable"),
            extraClassName: "audit-readiness-row--llm-interpretations",
            testId: "audit-readiness-row-llm-interpretations",
          }}
        />
      </ul>,
    );
    const li = screen.getByTestId("audit-readiness-row-llm-interpretations");
    expect(li.className).toMatch(/audit-readiness-row--llm-interpretations/);
    expect(li.className).toMatch(/audit-readiness-row--not_applicable/);
  });
});
