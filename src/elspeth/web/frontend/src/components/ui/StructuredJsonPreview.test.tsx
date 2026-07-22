import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { StructuredJsonPreview } from "./StructuredJsonPreview";

// The JSON view now routes through the shared CodeBlock (elspeth-865bc4fcfc),
// which highlights via Prism line-<div>s rather than a plain text node — so
// `.textContent` no longer contains literal "\n" between lines. Assertions
// below check substrings within a line instead of newline-joined snippets.
function jsonViewPre(container: HTMLElement): HTMLElement | null {
  return container.querySelector('[data-codeblock-format]');
}

describe("StructuredJsonPreview", () => {
  it("renders the JSON view through CodeBlock, highlighted and with a copy button", () => {
    const { container } = render(
      <StructuredJsonPreview text='{"id":1,"status":"ok"}' />,
    );

    const pre = jsonViewPre(container);
    expect(pre).not.toBeNull();
    expect(pre!.getAttribute("data-codeblock-format")).toBe("json");
    expect(pre!.querySelectorAll("span").length).toBeGreaterThan(0);
    expect(
      screen.getByRole("button", { name: /copy value/i }),
    ).toBeInTheDocument();
  });

  it("preserves unsafe integer literals in the JSON text view", () => {
    const { container } = render(
      <StructuredJsonPreview text='{"id":9007199254740993}' />,
    );

    expect(jsonViewPre(container)?.textContent).toContain("9007199254740993");
    expect(jsonViewPre(container)?.textContent).not.toContain(
      "9007199254740992",
    );
    expect(
      screen.queryByRole("button", { name: "Table view" }),
    ).not.toBeInTheDocument();
  });

  it.each([
    ['{"id":9007199254740993.0}', "9007199254740993.0"],
    ['{"id":9007199254740993e0}', "9007199254740993e0"],
    ['{"id":-9007199254740993e0}', "-9007199254740993e0"],
    ['{"delta":-0}', "-0"],
  ])("preserves lossy number literal forms in %s", (text, expectedLiteral) => {
    const { container } = render(<StructuredJsonPreview text={text} />);

    const pre = jsonViewPre(container);
    expect(pre?.textContent).toContain(expectedLiteral);
    expect(pre?.textContent).not.toContain("9007199254740992");
    // Still routed through the JSON-highlighted CodeBlock path, not the
    // "could not parse" plain fallback — the guard renders the original
    // text verbatim without abandoning highlighting.
    expect(pre?.getAttribute("data-codeblock-format")).toBe("json");
    expect(
      screen.queryByRole("button", { name: "Table view" }),
    ).not.toBeInTheDocument();
  });

  it("does not flag an ordinary Python-emitted float (e.g. 100.0) as lossy", () => {
    // Regression guard for the reviewed, load-bearing decimal-normalisation
    // behaviour: over-flagging values like 100.0 would wrongly suppress
    // Table view for perfectly representable JSON.
    render(<StructuredJsonPreview text='{"amount":100.0}' />);

    expect(
      screen.getByRole("button", { name: "Table view" }),
    ).toBeInTheDocument();
  });

  it("keeps table view for JSON that can be represented losslessly", async () => {
    const user = userEvent.setup();
    render(<StructuredJsonPreview text='[{"id":1,"status":"ok"}]' />);

    await user.click(screen.getByRole("button", { name: "Table view" }));

    expect(screen.getByText("id")).toBeInTheDocument();
    expect(screen.getByText("status")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
  });

  it("renders the JSON table view through the shared PreviewTable (real th scope=col)", async () => {
    // Parity guard for elspeth-611a05668e: the JSON table view and
    // RunOutputsPanel's csv/jsonl preview now share one PreviewTable
    // component, so header cells must be real th[scope=col] here too —
    // not just for the tabular consumer.
    const user = userEvent.setup();
    const { container } = render(
      <StructuredJsonPreview text='[{"id":1,"status":"ok"}]' />,
    );

    await user.click(screen.getByRole("button", { name: "Table view" }));

    const table = container.querySelector(".structured-preview-table");
    expect(table).not.toBeNull();
    const headerCells = screen.getAllByRole("columnheader");
    headerCells.forEach((cell) => {
      expect(cell.tagName).toBe("TH");
      expect(cell.getAttribute("scope")).toBe("col");
    });
    expect(table?.querySelectorAll("tbody td").length).toBeGreaterThan(0);
  });

  it("preserves first-seen header order across rows with differing keys", async () => {
    const user = userEvent.setup();
    render(
      <StructuredJsonPreview text='[{"b":1,"a":2},{"c":3,"a":4},{"b":5,"d":6}]' />,
    );

    await user.click(screen.getByRole("button", { name: "Table view" }));

    const headerTexts = screen
      .getAllByRole("columnheader")
      .map((el) => el.textContent);
    expect(headerTexts).toEqual(["b", "a", "c", "d"]);
  });

  it("shows no table markup until Table view is actually selected", async () => {
    // Behavioural proxy for the lazy table-model split (elspeth-37dc3472de):
    // this only proves the TableModel isn't *rendered* before toggling, not
    // that the build itself is deferred — the memo's laziness isn't
    // independently observable from the DOM without instrumenting the
    // module internals, which is out of scope here.
    const user = userEvent.setup();
    const { container } = render(
      <StructuredJsonPreview text='[{"id":1,"status":"ok"}]' />,
    );

    expect(container.querySelector(".structured-preview-table")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Table view" }));

    expect(
      container.querySelector(".structured-preview-table"),
    ).not.toBeNull();
  });
});
