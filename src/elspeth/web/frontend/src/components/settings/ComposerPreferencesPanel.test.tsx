import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ComposerPreferencesForm } from "./ComposerPreferencesPanel";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";

// API mock — the real preferencesStore module imports these helpers and
// would receive undefined without the entries.
vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));

describe("ComposerPreferencesForm", () => {
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

  it("renders the current default-mode selection (freeform)", () => {
    usePreferencesStore.setState({ defaultMode: "freeform" });
    render(<ComposerPreferencesForm />);
    expect(screen.getByLabelText(/freeform/i)).toBeChecked();
    expect(screen.getByLabelText(/guided/i)).not.toBeChecked();
  });

  it("writes the new default-mode on selection", async () => {
    const setDefault = vi
      .spyOn(usePreferencesStore.getState(), "setDefaultMode")
      .mockResolvedValueOnce(undefined);

    render(<ComposerPreferencesForm />);
    await userEvent.click(screen.getByLabelText(/freeform/i));

    expect(setDefault).toHaveBeenCalledWith("freeform");
  });

  it("disables inputs while writing", () => {
    usePreferencesStore.setState({ writing: true });
    render(<ComposerPreferencesForm />);
    expect(screen.getByLabelText(/guided/i)).toBeDisabled();
    expect(screen.getByLabelText(/freeform/i)).toBeDisabled();
  });

  it("returns null before preferences load (defaultMode null)", () => {
    usePreferencesStore.setState({ loaded: false, defaultMode: null });
    const { container } = render(<ComposerPreferencesForm />);
    // Component must gate on loaded === true; defaultMode is null pre-bootstrap.
    expect(container.firstChild).toBeNull();
  });
});
