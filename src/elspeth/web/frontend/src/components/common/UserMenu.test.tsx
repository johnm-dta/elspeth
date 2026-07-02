import { beforeEach, describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UserMenu } from "./UserMenu";

// Role contract: this component is a disclosure/popover, NOT a WAI-ARIA
// `menu` widget. Tests query items by their implicit `button` role rather
// than `menuitem` — the menu role was dropped because we don't implement
// the arrow-key/Home/End/type-ahead keyboard contract that the menu
// pattern demands. See UserMenu.tsx module comment.
describe("UserMenu", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
    document.documentElement.style.colorScheme = "";
  });

  it("is closed by default — action buttons not in the document", () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    expect(
      screen.queryByRole("button", { name: /composer preferences/i }),
    ).not.toBeInTheDocument();
  });

  it("shows theme, Composer preferences, and Sign out items when opened", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    expect(
      screen.getByRole("button", { name: /switch to light theme/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /composer preferences/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /sign out/i }),
    ).toBeInTheDocument();
  });

  it("calls onOpenSettings when Composer preferences is clicked, then closes", async () => {
    const openSettings = vi.fn();
    render(<UserMenu onOpenSettings={openSettings} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.click(
      screen.getByRole("button", { name: /composer preferences/i }),
    );
    expect(openSettings).toHaveBeenCalled();
    expect(
      screen.queryByRole("button", { name: /composer preferences/i }),
    ).not.toBeInTheDocument();
  });

  it("toggles the theme from the account menu", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));

    await userEvent.click(
      screen.getByRole("button", { name: /switch to light theme/i }),
    );

    expect(localStorage.getItem("elspeth_theme")).toBe("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(
      screen.queryByRole("button", { name: /switch to dark theme/i }),
    ).not.toBeInTheDocument();
  });

  it("calls onSignOut when Sign out is clicked", async () => {
    const signOut = vi.fn();
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={signOut} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.click(screen.getByRole("button", { name: /sign out/i }));
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
      screen.getByRole("button", { name: /composer preferences/i }),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /outside/i }));
    expect(
      screen.queryByRole("button", { name: /composer preferences/i }),
    ).not.toBeInTheDocument();
  });

  it("Escape closes the menu and returns focus to the trigger", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    const trigger = screen.getByRole("button", { name: /account/i });
    await userEvent.click(trigger);
    expect(
      screen.getByRole("button", { name: /composer preferences/i }),
    ).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(
      screen.queryByRole("button", { name: /composer preferences/i }),
    ).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("Tab navigates between action buttons (project convention: Tab not arrows)", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.tab();
    expect(
      screen.getByRole("button", { name: /switch to light theme/i }),
    ).toHaveFocus();
    await userEvent.tab();
    expect(
      screen.getByRole("button", { name: /composer preferences/i }),
    ).toHaveFocus();
    await userEvent.tab();
    expect(
      screen.getByRole("link", { name: /help & documentation/i }),
    ).toHaveFocus();
    await userEvent.tab();
    expect(
      screen.getByRole("button", { name: /sign out/i }),
    ).toHaveFocus();
  });

  // elspeth-8225736807: one honest help entry — a link to the repository
  // docs directory (the deployment serves no docs site of its own).
  it("offers a 'Help & documentation' link to the project docs", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    const help = screen.getByRole("link", { name: /help & documentation/i });
    expect(help).toHaveAttribute(
      "href",
      "https://github.com/johnm-dta/elspeth/tree/main/docs",
    );
    // New tab, no opener leakage.
    expect(help).toHaveAttribute("target", "_blank");
    expect(help).toHaveAttribute("rel", "noreferrer");
  });

  // elspeth-83eb51334f: focus leaving the menu subtree closes it — a
  // keyboard user must not be able to Tab away while the popup stays open.
  it("closes when focus moves outside the menu subtree", async () => {
    render(
      <div>
        <UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />
        <button type="button">outside</button>
      </div>,
    );
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    const signOut = screen.getByRole("button", { name: /sign out/i });
    signOut.focus();
    const outside = screen.getByRole("button", { name: /^outside$/i });
    fireEvent.blur(signOut, { relatedTarget: outside });
    expect(
      screen.queryByRole("button", { name: /sign out/i }),
    ).not.toBeInTheDocument();
  });

  it("stays open when focus moves between items inside the menu", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    const theme = screen.getByRole("button", { name: /switch to/i });
    const signOut = screen.getByRole("button", { name: /sign out/i });
    theme.focus();
    fireEvent.blur(theme, { relatedTarget: signOut });
    expect(
      screen.getByRole("button", { name: /sign out/i }),
    ).toBeInTheDocument();
  });

  it("trigger advertises aria-haspopup=true (disclosure, not menu)", () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    // Regression pin for the role-contract fix: the trigger MUST NOT
    // assert aria-haspopup="menu" because the component doesn't honour
    // the WAI-ARIA menu keyboard contract (arrow keys, Home/End,
    // type-ahead). "true" is the no-promise-of-specific-popup-role
    // value the disclosure pattern uses.
    const trigger = screen.getByRole("button", { name: /account/i });
    expect(trigger).toHaveAttribute("aria-haspopup", "true");
  });
});
