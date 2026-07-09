import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AuditReadinessRow, type RowPresentation } from "./AuditReadinessRow";
import { ReadOnlyProvider } from "../../contexts/ReadOnlyContext";
import type { ReadinessRowId, ReadinessStatus } from "../../types/api";

function makeRow(
  status: ReadinessStatus,
  id: ReadinessRowId = "provenance",
): RowPresentation {
  return {
    id,
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

  // ── Gate legibility (elspeth-088bf83922 T-2, option (a)) ───────────────────
  //
  // Every row carries a "Blocks Run" / "Advisory" text badge next to its
  // heading, classified by isRunGatingReadinessRow (ExecuteButton.tsx) —
  // the same file that owns canExecute. Only `validation` and
  // `llm_interpretations` are gating; the other four ids are always
  // advisory. The badge is visible text (not aria-hidden) and also
  // exposed programmatically via `data-gate` on the row's <li>, so both
  // sighted and assistive-tech users, and tests/tooling, get the same
  // classification.

  it.each([
    ["validation", "blocks", "Blocks Run"],
    ["llm_interpretations", "blocks", "Blocks Run"],
    ["plugin_trust", "advisory", "Advisory"],
    ["provenance", "advisory", "Advisory"],
    ["retention", "advisory", "Advisory"],
    ["secrets", "advisory", "Advisory"],
  ] as const)(
    "labels the %s row data-gate=%s with visible text %j",
    (id, expectedGate, expectedLabel) => {
      render(
        <ul>
          <AuditReadinessRow row={makeRow("ok", id)} />
        </ul>,
      );
      const li = screen.getByRole("listitem");
      expect(li).toHaveAttribute("data-gate", expectedGate);
      expect(screen.getByText(expectedLabel)).toBeInTheDocument();
    },
  );

  it("keeps the heading exact-text queryable alongside the gate badge (no string concatenation)", () => {
    // Regression guard: the badge must NOT be appended into row.heading's
    // own text node, or every existing `getByText("Validation")`-style
    // assertion across the panel test suites would break.
    render(
      <ul>
        <AuditReadinessRow row={makeRow("ok", "validation")} />
      </ul>,
    );
    expect(screen.getByText("Provenance")).toBeInTheDocument();
    expect(screen.getByText("Blocks Run")).toBeInTheDocument();
  });

  it("renders the gate badge as visible text (not aria-hidden) for both actionable and static variants", () => {
    const onSelect = vi.fn();
    const { rerender } = render(
      <ul>
        <AuditReadinessRow row={makeRow("error", "secrets")} onSelect={onSelect} />
      </ul>,
    );
    const badge = screen.getByText("Advisory");
    expect(badge).not.toHaveAttribute("aria-hidden");

    rerender(
      <ul>
        <AuditReadinessRow row={makeRow("ok", "secrets")} />
      </ul>,
    );
    expect(screen.getByText("Advisory")).not.toHaveAttribute("aria-hidden");
  });
});
