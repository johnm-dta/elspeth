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

  // Under the bug: the menu-level Enter keydown handler preventDefaulted and
  // called activateMenuIndex(focusIndex), which (focusIndex defaults to 0)
  // fired createSession() instead of saveRename(). The Save button's native
  // submit was suppressed. Under the fix: events originating inside the
  // rename form are skipped, so the form's onSubmit fires renameSession.
  it("does not swallow Enter on the rename Save button", async () => {
    const renameSession = vi.fn().mockResolvedValue(undefined);
    const createSession = vi.fn();
    useSessionStore.setState({ renameSession, createSession } as never);
    const user = userEvent.setup();

    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /first/i }));
    await user.click(screen.getByRole("menuitem", { name: /^rename first$/i }));
    const input = screen.getByRole("textbox", { name: /^rename session$/i });
    await user.clear(input);
    await user.type(input, "Renamed via keyboard");
    // Tab into the Save button, then press Enter on it. Under the bug the
    // ul-level handler would intercept Enter and run createSession; under the
    // fix the form's submit handler runs renameSession.
    await user.tab();
    const saveBtn = screen.getByRole("button", { name: /save session name/i });
    expect(saveBtn).toHaveFocus();
    await user.keyboard("{Enter}");

    expect(renameSession).toHaveBeenCalledWith("s1", "Renamed via keyboard");
    expect(createSession).not.toHaveBeenCalled();
  });

  // Symmetric guard: Enter on the Cancel button must run cancelRename's
  // onClick handler, not the menu activator. Under the bug, focusIndex=0
  // would fire createSession. (We focus Cancel directly via .focus() because
  // the ul's Tab handler closes the menu when Tab bubbles from a child
  // without stopPropagation — natural keyboard navigation past Save would
  // exit the menu before reaching Cancel.)
  it("does not swallow Enter on the rename Cancel button", async () => {
    const renameSession = vi.fn();
    const createSession = vi.fn();
    useSessionStore.setState({ renameSession, createSession } as never);
    const user = userEvent.setup();

    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /first/i }));
    await user.click(screen.getByRole("menuitem", { name: /^rename first$/i }));
    // The rename input is open; the rename form is in the DOM.
    expect(
      screen.getByRole("textbox", { name: /^rename session$/i }),
    ).toBeInTheDocument();
    const cancelBtn = screen.getByRole("button", { name: /cancel session rename/i });
    cancelBtn.focus();
    expect(cancelBtn).toHaveFocus();
    await user.keyboard("{Enter}");

    // Cancel closes the rename form: the textbox is gone.
    expect(
      screen.queryByRole("textbox", { name: /^rename session$/i }),
    ).not.toBeInTheDocument();
    expect(createSession).not.toHaveBeenCalled();
    expect(renameSession).not.toHaveBeenCalled();
  });

  // ── Phase 8.4 filter + archive polish ──────────────────────────────────────

  it("renders a session filter input", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    expect(
      screen.getByRole("textbox", { name: /find a session/i }),
    ).toBeInTheDocument();
  });

  it("filters sessions by title (case-insensitive substring)", async () => {
    useSessionStore.setState({
      sessions: [
        { id: "s1", title: "Tender review", updated_at: "2026-05-15T00:00:00Z" } as never,
        { id: "s2", title: "Weather monitor", updated_at: "2026-05-14T00:00:00Z" } as never,
        { id: "s3", title: "Document QA pipeline", updated_at: "2026-05-13T00:00:00Z" } as never,
      ],
      activeSessionId: "s1",
    } as never);

    const user = userEvent.setup();
    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /tender review/i }));
    await user.type(screen.getByRole("textbox", { name: /find a session/i }), "weather");

    expect(screen.getByRole("menuitem", { name: /^weather monitor$/i })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: /^tender review$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: /^document qa pipeline$/i })).not.toBeInTheDocument();
  });

  it("hides archived sessions by default and shows them when toggled", async () => {
    useSessionStore.setState({
      sessions: [
        { id: "s1", title: "Active session", updated_at: "2026-05-15T00:00:00Z" } as never,
        { id: "s2", title: "Old archived session", updated_at: "2026-05-14T00:00:00Z", archived: true } as never,
      ],
      activeSessionId: "s1",
    } as never);

    const user = userEvent.setup();
    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /active session/i }));

    // Archived session is hidden by default.
    expect(screen.getByRole("menuitem", { name: /^active session$/i })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: /^old archived session$/i })).not.toBeInTheDocument();

    // Toggle shows the archived session.
    await user.click(screen.getByRole("checkbox", { name: /show archived/i }));
    expect(screen.getByRole("menuitem", { name: /^old archived session$/i })).toBeInTheDocument();
  });

  // Q9: backend archive failure must surface an error region AND preserve the
  // row in the active list. Without these tests an optimistic-UI implementation
  // that removes the row before awaiting the network response would ship
  // undetected — the row would disappear even though the backend state is
  // unchanged.
  //
  // The error-region assertion is split into two cases (Error vs non-Error
  // rejection) because the regex used to share one test passed against both
  // the preserved-message branch ("Could not archive session: backend
  // unavailable") AND the generic fallback ("Could not archive session.
  // Please try again."). A regression dropping ``err.message`` would not have
  // been caught.
  it("preserves err.message in the inline alert when archive rejects with an Error", async () => {
    const archiveSession = vi.fn().mockRejectedValue(new Error("backend unavailable"));
    useSessionStore.setState({
      sessions: [
        { id: "s1", title: "Tender review", updated_at: "2026-05-15T00:00:00Z" } as never,
      ],
      activeSessionId: "s1",
      archiveSession,
    } as never);

    const user = userEvent.setup();
    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /tender review/i }));
    await user.click(screen.getByRole("menuitem", { name: /^archive tender review$/i }));
    await user.click(screen.getByRole("button", { name: /^archive$/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toBeInTheDocument();
    // The backend message must be verbatim in the alert — the test fails if
    // confirmArchive falls back to the generic message when it had an
    // Error.message available.
    expect(alert.textContent).toContain("backend unavailable");

    // Row is still present (no optimistic removal on failure).
    await user.click(screen.getByRole("button", { name: /tender review/i }));
    expect(screen.getByRole("menuitem", { name: /^tender review$/i })).toBeInTheDocument();
  });

  it("uses the generic fallback when archive rejects with a non-Error value", async () => {
    // Stringly-typed rejection from an upstream layer — the inline alert
    // must NOT leak the raw value into the UI (it has no useful operator
    // semantics) and must NOT match the preserved-message phrasing.
    const archiveSession = vi.fn().mockRejectedValue("not-an-error-instance");
    useSessionStore.setState({
      sessions: [
        { id: "s1", title: "Tender review", updated_at: "2026-05-15T00:00:00Z" } as never,
      ],
      activeSessionId: "s1",
      archiveSession,
    } as never);

    const user = userEvent.setup();
    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /tender review/i }));
    await user.click(screen.getByRole("menuitem", { name: /^archive tender review$/i }));
    await user.click(screen.getByRole("button", { name: /^archive$/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/please try again/i);
    expect(alert.textContent).not.toContain("not-an-error-instance");
  });

  // S1: reopening the menu clears stale inline alerts so the user doesn't
  // see a previous failure persisting next to a fresh interaction.  Pins
  // HeaderSessionSwitcher.tsx onClick reset of archiveError + renameError.
  it("clears a stale archive error when the menu is reopened", async () => {
    const archiveSession = vi.fn().mockRejectedValue(new Error("backend unavailable"));
    useSessionStore.setState({
      sessions: [
        { id: "s1", title: "Tender review", updated_at: "2026-05-15T00:00:00Z" } as never,
      ],
      activeSessionId: "s1",
      archiveSession,
    } as never);

    const user = userEvent.setup();
    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /tender review/i }));
    await user.click(screen.getByRole("menuitem", { name: /^archive tender review$/i }));
    await user.click(screen.getByRole("button", { name: /^archive$/i }));

    // Confirm the alert is present after the failure.
    expect(await screen.findByRole("alert")).toBeInTheDocument();

    // First trigger click closes the (already-closed-after-confirm) menu
    // state, second click reopens — that's the path the reset hook fires on.
    await user.click(screen.getByRole("button", { name: /tender review/i }));
    await user.click(screen.getByRole("button", { name: /tender review/i }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  // I3: rename failure must surface an inline alert co-located with the
  // rename form rather than relying on the global composer error region.
  // Mirror of the archive path.
  it("preserves err.message in the rename-form alert when rename rejects with an Error", async () => {
    const renameSession = vi
      .fn()
      .mockRejectedValue(new Error("title contains forbidden character"));
    useSessionStore.setState({ renameSession } as never);
    const user = userEvent.setup();

    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /first/i }));
    await user.click(screen.getByRole("menuitem", { name: /^rename first$/i }));
    const input = screen.getByRole("textbox", { name: /^rename session$/i });
    await user.clear(input);
    await user.type(input, "Updated title");
    await user.click(screen.getByRole("button", { name: /save session name/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("title contains forbidden character");
  });

  it("uses the generic fallback when rename rejects with a non-Error value", async () => {
    const renameSession = vi.fn().mockRejectedValue("not-an-error-instance");
    useSessionStore.setState({ renameSession } as never);
    const user = userEvent.setup();

    render(<HeaderSessionSwitcher />);
    await user.click(screen.getByRole("button", { name: /first/i }));
    await user.click(screen.getByRole("menuitem", { name: /^rename first$/i }));
    const input = screen.getByRole("textbox", { name: /^rename session$/i });
    await user.clear(input);
    await user.type(input, "Updated title");
    await user.click(screen.getByRole("button", { name: /save session name/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/please try again/i);
    expect(alert.textContent).not.toContain("not-an-error-instance");
  });
});
