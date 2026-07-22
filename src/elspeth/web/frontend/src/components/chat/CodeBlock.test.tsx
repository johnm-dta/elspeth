import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { CodeBlock, hasLossyNumberLiteral } from "./CodeBlock";

describe("CodeBlock — JSON pretty-print", () => {
  it("parses and 2-space pretty-prints valid JSON, highlighting it", () => {
    const { container } = render(
      <CodeBlock code='{"name":"Ada","amount":42}' prettyJson />,
    );

    const pre = container.querySelector("pre");
    expect(pre).not.toBeNull();
    // Highlighted JSON, not the plain fallback.
    expect(pre!.getAttribute("data-codeblock-format")).toBe("json");
    // 2-space pretty-printing: keys are indented, and Prism splits the object
    // into one line <div> per source line (4 lines for a two-key object).
    expect(pre!.textContent).toContain('"name"');
    expect(pre!.textContent).toContain('  "name"');
    expect(pre!.querySelectorAll(":scope > div").length).toBeGreaterThanOrEqual(3);
    // Prism emits token <span>s when highlighting.
    expect(pre!.querySelectorAll("span").length).toBeGreaterThan(0);
  });

  it("falls back to plain monospace on invalid JSON (never throws, never fabricates)", () => {
    const raw = "name,amount\nAda,42";
    const { container } = render(<CodeBlock code={raw} prettyJson />);

    const pre = container.querySelector("pre");
    expect(pre).not.toBeNull();
    expect(pre!.getAttribute("data-codeblock-format")).toBe("plain");
    expect(pre!.className).toContain("code-block--plain");
    // The raw value is rendered verbatim — no invented structure.
    expect(pre!.textContent).toBe(raw);
  });

  it("highlights with the given language when not pretty-printing JSON", () => {
    const { container } = render(<CodeBlock code="const x = 1;" language="javascript" />);

    const pre = container.querySelector("pre");
    expect(pre!.getAttribute("data-codeblock-format")).toBe("highlighted");
  });

  it("exposes a copy affordance by default and an accessible label", () => {
    render(<CodeBlock code='{"a":1}' prettyJson ariaLabel="Source data" />);
    expect(screen.getByRole("button", { name: /copy value/i })).toBeTruthy();
    expect(screen.getByLabelText("Source data")).toBeTruthy();
  });

  it("omits the copy affordance when showCopy is false", () => {
    render(<CodeBlock code='{"a":1}' prettyJson showCopy={false} />);
    expect(screen.queryByRole("button", { name: /copy/i })).toBeNull();
  });
});

describe("CodeBlock — lossy-literal guard (elspeth-865bc4fcfc)", () => {
  it.each([
    // 20-digit integer id — past Number.MAX_SAFE_INTEGER, would round on
    // JSON.parse -> JSON.stringify.
    ['{"id":12345678901234567890}', "12345678901234567890"],
    // Unsafe integer written with a trailing ".0" — JSON.stringify(value)
    // would drop the fractional zero AND round the integer part.
    ['{"id":9007199254740993.0}', "9007199254740993.0"],
    ['{"delta":-0}', "-0"],
  ])(
    "renders %s verbatim (never re-serialised) while still JSON-highlighting",
    (raw) => {
      const { container } = render(<CodeBlock code={raw} prettyJson />);

      const pre = container.querySelector("pre");
      expect(pre).not.toBeNull();
      // Still the JSON-highlighted path, not the plain "could not parse"
      // fallback — the guard skips re-stringifying, it doesn't abandon
      // highlighting.
      expect(pre!.getAttribute("data-codeblock-format")).toBe("json");
      expect(pre!.querySelectorAll("span").length).toBeGreaterThan(0);
      // Verbatim: the exact source text, not a JSON.parse -> stringify
      // round trip (which would have renamed/rounded the literal).
      expect(pre!.textContent).toBe(raw);
    },
  );

  it("does not flag an ordinary Python-emitted float (e.g. 100.0) as lossy", () => {
    // Load-bearing regression guard: over-flagging trailing-zero decimals
    // would wrongly skip pretty-printing for common, perfectly
    // representable JSON.
    expect(hasLossyNumberLiteral('{"amount":100.0}')).toBe(false);

    const { container } = render(
      <CodeBlock code='{"amount":100.0}' prettyJson />,
    );
    const pre = container.querySelector("pre");
    // Normal path: parsed and re-stringified (2-space), same as any other
    // losslessly-representable JSON.
    expect(pre!.textContent).toContain('"amount"');
  });

  it("still falls back to plain monospace on invalid JSON even with a lossy-looking literal", () => {
    const raw = '{"id":9007199254740993, invalid}';
    const { container } = render(<CodeBlock code={raw} prettyJson />);

    const pre = container.querySelector("pre");
    expect(pre!.getAttribute("data-codeblock-format")).toBe("plain");
    expect(pre!.textContent).toBe(raw);
  });
});
