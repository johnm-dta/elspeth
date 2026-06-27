import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AlertBanner } from "./AlertBanner";

describe("AlertBanner", () => {
  it("defaults to the error tone with role=alert", () => {
    render(<AlertBanner>Backend unavailable</AlertBanner>);
    const el = screen.getByRole("alert");
    expect(el).toHaveClass("alert-banner");
    expect(el).not.toHaveClass("alert-banner--warning");
    expect(el).toHaveTextContent("Backend unavailable");
  });

  it("uses role=status and a tone class for non-error tones", () => {
    render(<AlertBanner tone="warning">Heads up</AlertBanner>);
    const el = screen.getByRole("status");
    expect(el).toHaveClass("alert-banner", "alert-banner--warning");
  });

  it("allows the role to be overridden", () => {
    render(
      <AlertBanner role="status" data-testid="banner">
        Quiet error
      </AlertBanner>,
    );
    expect(screen.getByTestId("banner")).toHaveAttribute("role", "status");
  });

  it("renders a right-aligned action when provided", () => {
    render(
      <AlertBanner action={<button>Retry</button>}>Request failed</AlertBanner>,
    );
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
