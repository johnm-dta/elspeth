import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WordMark } from "./WordMark";

describe("WordMark", () => {
  it("renders the literal ELSPETH text in a span by default", () => {
    render(<WordMark />);
    const el = screen.getByText("ELSPETH");
    expect(el.tagName).toBe("SPAN");
  });

  it("renders the chosen element via the as prop", () => {
    render(<WordMark as="h1" />);
    const el = screen.getByText("ELSPETH");
    expect(el.tagName).toBe("H1");
  });

  it("applies the canonical wordmark letter-spacing token", () => {
    render(<WordMark />);
    expect(screen.getByText("ELSPETH")).toHaveStyle({
      letterSpacing: "var(--tracking-wordmark)",
    });
  });
});
