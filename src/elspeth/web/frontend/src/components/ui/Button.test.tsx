import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Button } from "./Button";

describe("Button", () => {
  it("renders the base .btn class, label, and type=button", () => {
    render(<Button>Run</Button>);
    const btn = screen.getByRole("button", { name: "Run" });
    expect(btn).toHaveClass("btn");
    expect(btn).toHaveAttribute("type", "button");
  });
  it("applies the variant modifier (secondary = no modifier)", () => {
    render(<Button variant="primary">Go</Button>);
    expect(screen.getByRole("button", { name: "Go" })).toHaveClass("btn", "btn-primary");
  });
  it("uses the standalone compact base class", () => {
    render(<Button compact>x</Button>);
    const b = screen.getByRole("button", { name: "x" });
    expect(b).toHaveClass("btn-compact");
    expect(b).not.toHaveClass("btn");
  });
  it("forwards className and DOM props", () => {
    render(<Button className="x" disabled aria-label="lbl" />);
    const b = screen.getByRole("button", { name: "lbl" });
    expect(b).toHaveClass("x");
    expect(b).toBeDisabled();
  });
});
