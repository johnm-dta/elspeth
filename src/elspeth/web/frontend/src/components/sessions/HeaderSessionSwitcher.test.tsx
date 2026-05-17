import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HeaderSessionSwitcher } from "./HeaderSessionSwitcher";
import { useSessionStore } from "@/stores/sessionStore";

describe("HeaderSessionSwitcher", () => {
  beforeEach(() => {
    useSessionStore.setState({
      sessions: [
        { id: "s1", title: "First", updated_at: "2026-05-15T00:00:00Z" } as never,
        { id: "s2", title: "Second", updated_at: "2026-05-14T00:00:00Z" } as never,
      ],
      activeSessionId: "s1",
    } as never);
  });

  it("shows the active session title as the trigger label", () => {
    render(<HeaderSessionSwitcher />);
    expect(
      screen.getByRole("button", { name: /first/i }),
    ).toBeInTheDocument();
  });

  it("opens a menu of all sessions when clicked", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    expect(screen.getByRole("menuitem", { name: /^first$/i })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /^second$/i })).toBeInTheDocument();
  });

  it("calls selectSession when a menu item is clicked", async () => {
    const selectSession = vi.fn();
    useSessionStore.setState({ selectSession } as never);
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /^second$/i }));
    expect(selectSession).toHaveBeenCalledWith("s2");
  });

  it("offers a 'New session' verb at the top of the dropdown", async () => {
    const createSession = vi.fn();
    useSessionStore.setState({ createSession } as never);
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /new session/i }));
    expect(createSession).toHaveBeenCalled();
  });

  it("renames a session from the header menu", async () => {
    const renameSession = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ renameSession } as never);
    const user = userEvent.setup();

    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /first/i }));
    await user.click(screen.getByRole("menuitem", { name: /^rename first$/i }));
    const input = screen.getByRole("textbox", { name: /^rename session$/i });
    await user.clear(input);
    await user.type(input, "Updated title");
    await user.click(screen.getByRole("button", { name: /save session name/i }));

    expect(renameSession).toHaveBeenCalledWith("s1", "Updated title");
  });

  it("archives a session from the header menu after confirmation", async () => {
    const archiveSession = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ archiveSession } as never);
    const user = userEvent.setup();

    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /first/i }));
    await user.click(screen.getByRole("menuitem", { name: /^archive second$/i }));
    await user.click(screen.getByRole("button", { name: /^archive$/i }));

    expect(archiveSession).toHaveBeenCalledWith("s2");
  });

  it("includes session titles in rename and archive action labels", async () => {
    render(<HeaderSessionSwitcher />);

    await userEvent.click(screen.getByRole("button", { name: /first/i }));

    expect(
      screen.getByRole("menuitem", { name: /^rename first$/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("menuitem", { name: /^archive first$/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("menuitem", { name: /^rename second$/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("menuitem", { name: /^archive second$/i }),
    ).toBeInTheDocument();
  });

  it("closes on Escape", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    expect(
      screen.getByRole("menuitem", { name: /^second$/i }),
    ).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(
      screen.queryByRole("menuitem", { name: /^second$/i }),
    ).not.toBeInTheDocument();
  });

  it("renders 'untitled' fallback when no session is active", () => {
    useSessionStore.setState({ activeSessionId: null } as never);
    render(<HeaderSessionSwitcher />);
    expect(screen.getByRole("button")).toHaveTextContent(/untitled/i);
  });

  it("shows 'New session' even when the sessions list is empty (just-archived edge)", async () => {
    useSessionStore.setState({
      sessions: [],
      activeSessionId: "sess-orphaned",
    } as never);
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("menuitem", { name: /new session/i })).toBeInTheDocument();
  });

  it("ArrowDown moves focus through menu items", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    const items = screen.getAllByRole("menuitem");
    expect(items[0]).toHaveFocus();
    await userEvent.keyboard("{ArrowDown}");
    expect(items[1]).toHaveFocus();
    await userEvent.keyboard("{ArrowDown}");
    expect(items[2]).toHaveFocus();
  });

  it("ArrowUp from the first item wraps to the last", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    const items = screen.getAllByRole("menuitem");
    expect(items[0]).toHaveFocus();
    await userEvent.keyboard("{ArrowUp}");
    expect(items[items.length - 1]).toHaveFocus();
  });

  it("Tab from inside the menu closes it and returns focus to the trigger", async () => {
    render(<HeaderSessionSwitcher />);
    const trigger = screen.getByRole("button", { name: /first/i });
    await userEvent.click(trigger);
    expect(screen.getByRole("menu")).toBeInTheDocument();
    await userEvent.tab();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });
});
