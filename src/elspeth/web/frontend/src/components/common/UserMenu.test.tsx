import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UserMenu } from "./UserMenu";

describe("UserMenu", () => {
  it("is closed by default — menu items not in the document", () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    expect(
      screen.queryByRole("menuitem", { name: /settings/i }),
    ).not.toBeInTheDocument();
  });

  it("shows Settings + Sign out items when opened", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    expect(
      screen.getByRole("menuitem", { name: /settings/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("menuitem", { name: /sign out/i }),
    ).toBeInTheDocument();
  });

  it("calls onOpenSettings when Settings is clicked, then closes", async () => {
    const openSettings = vi.fn();
    render(<UserMenu onOpenSettings={openSettings} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /settings/i }));
    expect(openSettings).toHaveBeenCalled();
    expect(
      screen.queryByRole("menuitem", { name: /settings/i }),
    ).not.toBeInTheDocument();
  });

  it("calls onSignOut when Sign out is clicked", async () => {
    const signOut = vi.fn();
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={signOut} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /sign out/i }));
    expect(signOut).toHaveBeenCalled();
  });

  it("closes when clicking outside the menu", async () => {
    render(
      <div>
        <UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />
        <button type="button">outside</button>
      </div>,
    );
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    expect(
      screen.getByRole("menuitem", { name: /settings/i }),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /outside/i }));
    expect(
      screen.queryByRole("menuitem", { name: /settings/i }),
    ).not.toBeInTheDocument();
  });

  it("Escape closes the menu and returns focus to the trigger", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    const trigger = screen.getByRole("button", { name: /account/i });
    await userEvent.click(trigger);
    expect(
      screen.getByRole("menuitem", { name: /settings/i }),
    ).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(
      screen.queryByRole("menuitem", { name: /settings/i }),
    ).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("Tab navigates between menu items (project convention: Tab not arrows)", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.tab();
    expect(
      screen.getByRole("menuitem", { name: /settings/i }),
    ).toHaveFocus();
    await userEvent.tab();
    expect(
      screen.getByRole("menuitem", { name: /sign out/i }),
    ).toHaveFocus();
  });
});
