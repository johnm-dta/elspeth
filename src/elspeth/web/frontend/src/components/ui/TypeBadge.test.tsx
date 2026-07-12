import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TypeBadge } from "./TypeBadge";

describe("TypeBadge", () => {
  it("renders the base + type modifier class", () => {
    render(<TypeBadge type="gate" />);
    const badge = screen.getByText("gate");
    expect(badge).toHaveClass("type-badge", "type-badge-gate");
  });
  it("defaults the label to the type name", () => {
    render(<TypeBadge type="transform" />);
    expect(screen.getByText("transform")).toBeInTheDocument();
  });
  it("defaults type to source", () => {
    render(<TypeBadge />);
    const badge = screen.getByText("source");
    expect(badge).toHaveClass("type-badge", "type-badge-source");
  });
  it("lets children override the label", () => {
    render(<TypeBadge type="sink">Output</TypeBadge>);
    const badge = screen.getByText("Output");
    expect(badge).toHaveClass("type-badge-sink");
  });
  it("forwards className and DOM props", () => {
    render(<TypeBadge type="source" className="x" data-testid="tb" />);
    const badge = screen.getByTestId("tb");
    expect(badge).toHaveClass("type-badge", "type-badge-source", "x");
  });
  it("renders a queue badge with the base + queue modifier class and accessible label", () => {
    render(<TypeBadge type="queue" />);
    const badge = screen.getByText("queue");
    expect(badge).toHaveClass("type-badge", "type-badge-queue");
  });
  it("lets children override the queue label while keeping the queue class (compact contexts)", () => {
    render(<TypeBadge type="queue">Inbound queue</TypeBadge>);
    const badge = screen.getByText("Inbound queue");
    expect(badge).toHaveClass("type-badge-queue");
  });
  it("forwards className and DOM props for a queue badge", () => {
    render(<TypeBadge type="queue" className="x" data-testid="qb" />);
    const badge = screen.getByTestId("qb");
    expect(badge).toHaveClass("type-badge", "type-badge-queue", "x");
  });
});
