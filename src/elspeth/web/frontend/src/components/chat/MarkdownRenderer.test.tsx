import { readFileSync } from "node:fs";

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MarkdownRenderer, mermaidThemeFromTokens } from "./MarkdownRenderer";

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

  it("renders external links with safe new-tab attributes", () => {
    render(<MarkdownRenderer content="[Docs](https://example.com/docs)" />);

    const link = screen.getByRole("link", { name: "Docs" });
    expect(link).toHaveAttribute("href", "https://example.com/docs");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
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

describe("MarkdownRenderer Mermaid theme", () => {
  it("derives Mermaid theme variables from CSS tokens", () => {
    document.documentElement.style.setProperty(
      "--color-surface-elevated",
      "rgb(1, 2, 3)",
    );
    document.documentElement.style.setProperty("--color-text", "rgb(4, 5, 6)");
    document.documentElement.style.setProperty(
      "--color-border-strong",
      "rgba(7, 8, 9, 0.4)",
    );
    document.documentElement.style.setProperty(
      "--color-text-muted",
      "rgb(10, 11, 12)",
    );
    document.documentElement.style.setProperty(
      "--color-surface-raised",
      "rgb(13, 14, 15)",
    );
    document.documentElement.style.setProperty("--color-bg", "rgb(16, 17, 18)");

    const config = mermaidThemeFromTokens("dark");

    expect(config.theme).toBe("dark");
    expect(config.themeVariables).toMatchObject({
      primaryColor: "rgb(1, 2, 3)",
      primaryTextColor: "rgb(4, 5, 6)",
      primaryBorderColor: "rgba(7, 8, 9, 0.4)",
      lineColor: "rgb(10, 11, 12)",
      secondaryColor: "rgb(13, 14, 15)",
      tertiaryColor: "rgb(16, 17, 18)",
    });
  });

  it("does not hardcode Mermaid theme hex literals in the renderer", () => {
    const source = readFileSync("src/components/chat/MarkdownRenderer.tsx", "utf8");

    expect(source).not.toMatch(/primaryColor:\s*"#[0-9a-fA-F]+"/);
    expect(source).not.toMatch(/primaryTextColor:\s*"#[0-9a-fA-F]+"/);
    expect(source).not.toMatch(/primaryBorderColor:\s*"#[0-9a-fA-F]+"/);
  });
});

describe("MarkdownRenderer fenced code blocks", () => {
  it("renders YAML fenced blocks with Prism token spans", () => {
    const md = "```yaml\nsource:\n  type: csv_file\n```";
    const { container } = render(<MarkdownRenderer content={md} />);
    // Prism produces token spans inside the pre element
    const pre = container.querySelector("pre.code-block");
    expect(pre).not.toBeNull();
    expect(pre!.querySelector("span")).not.toBeNull();
  });

  it("renders a copy button that copies the code to the clipboard", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
      writable: true,
    });

    const md = "```yaml\nfoo: bar\n```";
    render(<MarkdownRenderer content={md} />);

    const copy = screen.getByRole("button", { name: /copy/i });
    await user.click(copy);

    expect(writeText).toHaveBeenCalledWith("foo: bar");
  });

  it("does not crash and does not show 'Copied' when clipboard write rejects", async () => {
    // Clipboard API may throw in non-secure contexts (HTTP) or when
    // permission is denied.  The component swallows the rejection — verify
    // that swallow behaviour: no thrown error, no false 'Copied' affordance.
    const user = userEvent.setup();
    const writeText = vi.fn().mockRejectedValue(new Error("permission denied"));
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
      writable: true,
    });

    const md = "```yaml\nfoo: bar\n```";
    render(<MarkdownRenderer content={md} />);

    const copy = screen.getByRole("button", { name: /copy code/i });
    await user.click(copy);

    expect(writeText).toHaveBeenCalledWith("foo: bar");
    // Button label must NOT flip to "Copied" on failure — that would lie to
    // the user about the operation succeeding.
    expect(screen.queryByRole("button", { name: /^copied$/i })).toBeNull();
    // Original 'Copy code' affordance still present.
    expect(screen.getByRole("button", { name: /copy code/i })).toBeInTheDocument();
  });
});
