import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DefaultModeChangedBanner } from "./DefaultModeChangedBanner";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";
import { updateUserComposerPreferences } from "@/api/client";

vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
  fetchSessions: vi.fn(),
  createSession: vi.fn(),
}));

describe("DefaultModeChangedBanner", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    useSessionStore.setState({ activeSessionId: null });
    vi.clearAllMocks();
  });

  it("does not render before preferences load", () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: false,
      bannerDismissedAt: null,
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });
    render(<DefaultModeChangedBanner />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("does not render when banner has been dismissed", () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: "2026-05-15T00:00:00Z",
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });
    render(<DefaultModeChangedBanner />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("does not render when user is on guided", () => {
    usePreferencesStore.setState({
      defaultMode: "guided",
      loaded: true,
      bannerDismissedAt: null,
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });
    render(<DefaultModeChangedBanner />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders for opted-out users who haven't dismissed (and no watermark)", () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: null,
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });
    render(<DefaultModeChangedBanner />);
    // Spec 05 wording: user-choice acknowledgment, not system migration.
    expect(screen.getByRole("status")).toHaveTextContent(
      /future sessions will start in/i,
    );
    expect(screen.getByRole("status")).toHaveTextContent(/freeform mode/i);
    // No system-migration narration.
    expect(screen.getByRole("status")).not.toHaveTextContent(
      /we changed the default/i,
    );
  });

  it("SUPPRESSES in the session of opt-out (banner cluster timing fix)", () => {
    // The user opted out in session 'sess-current', and they're still
    // in that session. The banner is the "first session after" surface,
    // not the "in this session" surface.
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: null,
      writing: false,
      writeError: null,
      optedOutAtSessionId: "sess-current",
    });
    useSessionStore.setState({ activeSessionId: "sess-current" });
    render(<DefaultModeChangedBanner />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("SHOWS in the NEXT session after opt-out (timing watermark crossover)", () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: null,
      writing: false,
      writeError: null,
      optedOutAtSessionId: "sess-original",
    });
    useSessionStore.setState({ activeSessionId: "sess-new" });
    render(<DefaultModeChangedBanner />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("dismisses on click", async () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: null,
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });
    const dismiss = vi
      .spyOn(usePreferencesStore.getState(), "dismissDefaultChangedBanner")
      .mockResolvedValueOnce(undefined);
    render(<DefaultModeChangedBanner />);
    await userEvent.click(
      screen.getByRole("button", { name: /got it|dismiss/i }),
    );
    expect(dismiss).toHaveBeenCalled();
  });

  it("surfaces dismiss failures in the visible banner", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    vi.mocked(updateUserComposerPreferences).mockRejectedValueOnce(
      new Error("503 Service Unavailable"),
    );
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: null,
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });

    render(<DefaultModeChangedBanner />);
    await userEvent.click(
      screen.getByRole("button", { name: /got it|dismiss/i }),
    );

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /couldn't dismiss the banner: 503 service unavailable/i,
      );
    });
    consoleError.mockRestore();
  });

  it("disables the dismiss button while a write is in flight (P0.8 offensive guard)", () => {
    // P0.8: the store's dismissDefaultChangedBanner throws if invoked
    // while writing=true. The UI must prevent that programmatic
    // collision by disabling the trigger; otherwise a concurrent
    // setDefaultMode PATCH would race against the dismiss action.
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: null,
      writing: true,
      writeError: null,
      optedOutAtSessionId: null,
    });
    render(<DefaultModeChangedBanner />);
    expect(
      screen.getByRole("button", { name: /got it|dismiss/i }),
    ).toBeDisabled();
  });

  it("on dismiss, moves focus to the chat input (WCAG 2.4.3 — no stranded focus)", async () => {
    // Stage a chat input fixture so the focus-target query succeeds.
    const input = document.createElement("textarea");
    input.setAttribute("data-chat-input", "");
    document.body.appendChild(input);
    try {
      usePreferencesStore.setState({
        defaultMode: "freeform",
        loaded: true,
        bannerDismissedAt: null,
        writing: false,
        writeError: null,
        optedOutAtSessionId: null,
      });
      vi.spyOn(
        usePreferencesStore.getState(),
        "dismissDefaultChangedBanner",
      ).mockResolvedValueOnce(undefined);
      render(<DefaultModeChangedBanner />);
      await userEvent.click(
        screen.getByRole("button", { name: /got it|dismiss/i }),
      );
      // The chat input must receive focus after the dismiss resolves.
      await waitFor(() => {
        expect(document.activeElement).toBe(input);
      });
    } finally {
      input.remove();
    }
  });
});
