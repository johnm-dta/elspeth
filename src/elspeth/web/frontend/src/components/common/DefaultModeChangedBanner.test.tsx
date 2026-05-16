import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DefaultModeChangedBanner } from "./DefaultModeChangedBanner";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";

vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));

describe("DefaultModeChangedBanner", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    vi.clearAllMocks();
  });

  it("does not render before preferences load", () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: false,
      bannerDismissedAt: null,
      writing: false,
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
    });
    render(<DefaultModeChangedBanner />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders for opted-out users who haven't dismissed", () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: null,
      writing: false,
    });
    render(<DefaultModeChangedBanner />);
    expect(screen.getByRole("status")).toHaveTextContent(/freeform/i);
  });

  it("dismisses on click", async () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      loaded: true,
      bannerDismissedAt: null,
      writing: false,
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
});
