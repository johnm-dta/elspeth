import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MarkdownRenderer } from "./MarkdownRenderer";

describe("MarkdownRenderer", () => {
  it("renders plain text as a paragraph", () => {
    render(<MarkdownRenderer content="Hello world" />);
    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("renders headings", () => {
    render(<MarkdownRenderer content="## Section Title" />);
    const heading = screen.getByRole("heading", { level: 2 });
    expect(heading).toHaveTextContent("Section Title");
  });

  it("renders inline code", () => {
    render(<MarkdownRenderer content="Use `set_source` to configure input." />);
    const code = screen.getByText("set_source");
    expect(code.tagName).toBe("CODE");
  });

  it("renders code blocks with language class", () => {
    const content = "```yaml\nsource:\n  plugin: csv\n```";
    const { container } = render(<MarkdownRenderer content={content} />);
    const pre = container.querySelector("pre");
    expect(pre).toBeInTheDocument();
    const code = pre?.querySelector("code");
    expect(code).toBeInTheDocument();
    expect(code?.textContent).toContain("source:");
  });

  it("renders tables from GFM markdown", () => {
    const content = "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |";
    render(<MarkdownRenderer content={content} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("Col A")).toBeInTheDocument();
  });

  it("renders a mermaid container for mermaid code blocks", () => {
    const content = "```mermaid\ngraph TD\n  A --> B\n```";
    const { container } = render(<MarkdownRenderer content={content} />);
    const mermaidDiv = container.querySelector(".mermaid-container");
    expect(mermaidDiv).toBeInTheDocument();
  });

  it("does not render mermaid blocks as regular code", () => {
    const content = "```mermaid\ngraph TD\n  A --> B\n```";
    const { container } = render(<MarkdownRenderer content={content} />);
    const codeBlocks = container.querySelectorAll("pre > code");
    for (const block of codeBlocks) {
      expect(block.textContent).not.toContain("graph TD");
    }
  });
});

describe("MarkdownRenderer fenced code blocks", () => {
  it("renders YAML fenced blocks with Prism token spans", () => {
    const md = "```yaml\nsource:\n  type: csv_file\n```";
    render(<MarkdownRenderer content={md} />);
    // Prism produces token spans inside the pre element
    const pre = document.querySelector("pre.code-block");
    expect(pre).not.toBeNull();
    expect(pre!.querySelector("span")).not.toBeNull();
  });

  it("renders a copy button that copies the code to the clipboard", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    const md = "```yaml\nfoo: bar\n```";
    render(<MarkdownRenderer content={md} />);

    const copy = screen.getByRole("button", { name: /copy/i });
    await user.click(copy);

    expect(writeText).toHaveBeenCalledWith("foo: bar");
  });
});
