import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { StructuredJsonPreview } from "./StructuredJsonPreview";

describe("StructuredJsonPreview", () => {
  it("preserves unsafe integer literals in the JSON text view", () => {
    const { container } = render(
      <StructuredJsonPreview text='{"id":9007199254740993}' />,
    );

    expect(
      container.querySelector(".structured-preview-pre")?.textContent,
    ).toContain("9007199254740993");
    expect(
      container.querySelector(".structured-preview-pre")?.textContent,
    ).not.toContain("9007199254740992");
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

    expect(
      container.querySelector(".structured-preview-pre")?.textContent,
    ).toContain(expectedLiteral);
    expect(
      container.querySelector(".structured-preview-pre")?.textContent,
    ).not.toContain("9007199254740992");
    expect(
      screen.queryByRole("button", { name: "Table view" }),
    ).not.toBeInTheDocument();
  });

  it("keeps table view for JSON that can be represented losslessly", async () => {
    const user = userEvent.setup();
    render(<StructuredJsonPreview text='[{"id":1,"status":"ok"}]' />);

    await user.click(screen.getByRole("button", { name: "Table view" }));

    expect(screen.getByText("id")).toBeInTheDocument();
    expect(screen.getByText("status")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
  });
});
