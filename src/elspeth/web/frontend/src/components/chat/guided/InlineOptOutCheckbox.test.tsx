import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InlineOptOutCheckbox } from "./InlineOptOutCheckbox";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";

vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));

describe("InlineOptOutCheckbox", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      bannerDismissedAt: null,
      writing: false,
    });
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

  it("writes 'freeform' when ticked from guided default", async () => {
    const setDefault = vi
      .spyOn(usePreferencesStore.getState(), "setDefaultMode")
      .mockResolvedValueOnce(undefined);
    render(<InlineOptOutCheckbox />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(setDefault).toHaveBeenCalledWith("freeform");
  });

  it("writes 'guided' when re-ticked from freeform default", async () => {
    usePreferencesStore.setState({ defaultMode: "freeform" });
    const setDefault = vi
      .spyOn(usePreferencesStore.getState(), "setDefaultMode")
      .mockResolvedValueOnce(undefined);
    render(<InlineOptOutCheckbox />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(setDefault).toHaveBeenCalledWith("guided");
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
});
