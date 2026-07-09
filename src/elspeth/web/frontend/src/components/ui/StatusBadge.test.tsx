import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusBadge } from "./StatusBadge";

describe("StatusBadge", () => {
  it("maps completed_with_failures to the completed colour and shows the ⚠ glyph", () => {
    render(<StatusBadge status="completed_with_failures" data-testid="sb" />);
    const badge = screen.getByTestId("sb");
    expect(badge).toHaveClass("status-badge", "status-badge-completed");
    expect(badge).not.toHaveClass("status-badge-completed_with_failures");
    expect(badge).toHaveTextContent("⚠");
  });
  it("renders empty with its own colour and the ∅ glyph", () => {
    render(<StatusBadge status="empty" data-testid="sb" />);
    const badge = screen.getByTestId("sb");
    expect(badge).toHaveClass("status-badge", "status-badge-empty");
    expect(badge).toHaveTextContent("∅");
  });
  it("maps cancelling to the cancelled colour", () => {
    render(<StatusBadge status="cancelling" data-testid="sb" />);
    const badge = screen.getByTestId("sb");
    expect(badge).toHaveClass("status-badge", "status-badge-cancelled");
  });
  it("renders running with no glyph", () => {
    render(<StatusBadge status="running" data-testid="sb" />);
    const badge = screen.getByTestId("sb");
    expect(badge).toHaveClass("status-badge", "status-badge-running");
    expect(badge).toHaveTextContent("running");
    expect(badge).not.toHaveTextContent("⚠");
    expect(badge).not.toHaveTextContent("∅");
  });
  it("defaults to pending and lets children override the label", () => {
    render(<StatusBadge data-testid="sb">Queued</StatusBadge>);
    const badge = screen.getByTestId("sb");
    expect(badge).toHaveClass("status-badge", "status-badge-pending");
    expect(badge).toHaveTextContent("Queued");
  });
  it("forwards className", () => {
    render(<StatusBadge status="failed" className="x" data-testid="sb" />);
    const badge = screen.getByTestId("sb");
    expect(badge).toHaveClass("status-badge", "status-badge-failed", "x");
  });
});
