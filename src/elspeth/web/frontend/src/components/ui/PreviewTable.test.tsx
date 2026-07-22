import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PreviewTable } from "./PreviewTable";

const structuredPreviewCss = readFileSync(
  "src/components/ui/structured-preview.css",
  "utf8",
).replace(/\/\*[\s\S]*?\*\//g, "");

function ruleBody(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = new RegExp(`${escaped}\\s*\\{(?<body>[\\s\\S]*?)\\n\\}`).exec(
    structuredPreviewCss,
  );
  if (!match?.groups?.body) {
    throw new Error(`Could not find ${selector} rule in structured-preview.css`);
  }
  return match.groups.body;
}

describe("PreviewTable", () => {
  it("renders headers as th scope=col and data rows as td", () => {
    const { container } = render(
      <PreviewTable
        table={{
          headers: ["id", "status"],
          rows: [
            ["1", "ok"],
            ["2", "error"],
          ],
        }}
      />,
    );

    const headerCells = screen.getAllByRole("columnheader");
    expect(headerCells.map((el) => el.textContent)).toEqual(["id", "status"]);
    headerCells.forEach((cell) => {
      expect(cell.tagName).toBe("TH");
      expect(cell.getAttribute("scope")).toBe("col");
    });

    // Body cells are plain <td> — no bold/background fake-header styling.
    const bodyCells = container.querySelectorAll("tbody td");
    expect(bodyCells).toHaveLength(4);
    expect(screen.getByText("error")).toBeInTheDocument();
  });

  it("pads rows shorter than the header count with empty cells", () => {
    const { container } = render(
      <PreviewTable
        table={{
          headers: ["a", "b", "c"],
          rows: [["only-a"]],
        }}
      />,
    );

    const bodyRow = container.querySelector("tbody tr");
    expect(bodyRow?.querySelectorAll("td")).toHaveLength(3);
  });

  it("renders a single-column table (e.g. jsonl rows) with a real header cell", () => {
    render(
      <PreviewTable
        table={{
          headers: ["value"],
          rows: [['{"a":1,"b":2}'], ['{"a":3,"b":4}']],
        }}
      />,
    );

    const header = screen.getByRole("columnheader", { name: "value" });
    expect(header.tagName).toBe("TH");
    expect(header.getAttribute("scope")).toBe("col");
    expect(screen.getByText('{"a":1,"b":2}')).toBeInTheDocument();
  });

  it("keeps wide preview cells inspectable through horizontal scroll instead of forced wrapping", () => {
    const wrapperRule = ruleBody(".structured-preview-table-wrap");
    const tableRule = ruleBody(".structured-preview-table");
    const cellRule = ruleBody(".structured-preview-table th,\n.structured-preview-table td");

    expect(wrapperRule).toContain("overflow: auto;");
    expect(tableRule).toContain("width: max-content;");
    expect(tableRule).toContain("min-width: 100%;");
    expect(cellRule).toContain("white-space: pre;");
    expect(cellRule).not.toContain("overflow-wrap: anywhere;");
  });
});
