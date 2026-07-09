import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AppHeader } from "./AppHeader";

describe("AppHeader", () => {
  it("renders the ELSPETH brand", () => {
    render(<AppHeader onOpenSettings={() => {}} onSignOut={() => {}} />);
    expect(screen.getByText(/ELSPETH/i)).toBeInTheDocument();
  });

  it("renders the session switcher", () => {
    // "No session" is the no-active-session trigger label — the switcher no
    // longer mints a competing "Untitled" default (elspeth-ef8c18a6cb).
    render(<AppHeader onOpenSettings={() => {}} onSignOut={() => {}} />);
    expect(
      screen.getByRole("button", { name: /session switcher: no session/i }),
    ).toBeInTheDocument();
  });

  it("renders the user menu", () => {
    render(<AppHeader onOpenSettings={() => {}} onSignOut={() => {}} />);
    expect(screen.getByRole("button", { name: /account/i })).toBeInTheDocument();
  });
});
