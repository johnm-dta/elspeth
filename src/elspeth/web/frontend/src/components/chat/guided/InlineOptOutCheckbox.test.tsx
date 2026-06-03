import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InlineOptOutCheckbox } from "./InlineOptOutCheckbox";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";

vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
  fetchSessions: vi.fn(),
  createSession: vi.fn(),
}));

describe("InlineOptOutCheckbox", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      bannerDismissedAt: null,
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });
    useSessionStore.setState({ activeSessionId: null });
    vi.clearAllMocks();
  });

  it("is unchecked when default is guided", () => {
    render(<InlineOptOutCheckbox />);
    expect(screen.getByRole("checkbox")).not.toBeChecked();
  });

  it("is checked when default is freeform", () => {
    usePreferencesStore.setState({ defaultMode: "freeform" });
    render(<InlineOptOutCheckbox />);
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  it("writes 'freeform' when ticked from guided default (forwards activeSessionId watermark)", async () => {
    const setDefault = vi
      .spyOn(usePreferencesStore.getState(), "setDefaultMode")
      .mockResolvedValueOnce(undefined);
    render(<InlineOptOutCheckbox />);
    await userEvent.click(screen.getByRole("checkbox"));
    // Second arg is the banner-timing watermark (null here = no session
    // active in test setup).
    expect(setDefault).toHaveBeenCalledWith("freeform", null);
  });

  it("writes 'guided' when re-ticked from freeform default", async () => {
    usePreferencesStore.setState({ defaultMode: "freeform" });
    const setDefault = vi
      .spyOn(usePreferencesStore.getState(), "setDefaultMode")
      .mockResolvedValueOnce(undefined);
    render(<InlineOptOutCheckbox />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(setDefault).toHaveBeenCalledWith("guided", null);
  });

  it("forwards the active session id when one is set", async () => {
    useSessionStore.setState({ activeSessionId: "sess-9" });
    const setDefault = vi
      .spyOn(usePreferencesStore.getState(), "setDefaultMode")
      .mockResolvedValueOnce(undefined);
    render(<InlineOptOutCheckbox />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(setDefault).toHaveBeenCalledWith("freeform", "sess-9");
  });

  it("disables the checkbox during write", () => {
    usePreferencesStore.setState({ writing: true });
    render(<InlineOptOutCheckbox />);
    expect(screen.getByRole("checkbox")).toBeDisabled();
  });

  it("returns null before preferences load", () => {
    usePreferencesStore.setState({ loaded: false, defaultMode: null });
    const { container } = render(<InlineOptOutCheckbox />);
    expect(container.firstChild).toBeNull();
  });

  it("surfaces writeError via role=alert when a PATCH fails (Panel a11y F2)", () => {
    usePreferencesStore.setState({
      writeError: "Couldn't save your preference: 503",
    });
    render(<InlineOptOutCheckbox />);
    expect(screen.getByRole("alert")).toHaveTextContent(/503/);
  });
});
