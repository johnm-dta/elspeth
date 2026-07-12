import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { usePreferencesStore } from "@/stores/preferencesStore";
import { FreeformIntroduction } from "./FreeformIntroduction";

describe("FreeformIntroduction", () => {
  beforeEach(() => {
    usePreferencesStore.setState({
      loaded: true,
      freeformIntroDismissedAt: null,
      writing: false,
      dismissFreeformIntro: vi.fn().mockResolvedValue(undefined),
    });
  });

  it("briefly explains how to build an auditable pipeline", () => {
    render(<FreeformIntroduction />);

    expect(
      screen.getByRole("heading", { name: "Build a pipeline" }),
    ).toBeVisible();
    expect(
      screen.getByText(/what ELSPETH should read.*how the data should change/is),
    ).toBeVisible();
  });

  it("does not render before preferences load or after account dismissal", () => {
    usePreferencesStore.setState({ loaded: false });
    const { rerender } = render(<FreeformIntroduction />);
    expect(screen.queryByRole("heading")).not.toBeInTheDocument();

    usePreferencesStore.setState({
      loaded: true,
      freeformIntroDismissedAt: "2026-07-12T03:04:05Z",
    });
    rerender(<FreeformIntroduction />);
    expect(screen.queryByRole("heading")).not.toBeInTheDocument();
  });

  it("dismisses through the account preference action", async () => {
    const user = userEvent.setup();
    const dismiss = vi.fn().mockResolvedValue(undefined);
    usePreferencesStore.setState({ dismissFreeformIntro: dismiss });
    render(<FreeformIntroduction />);

    await user.click(
      screen.getByRole("button", { name: "Don’t show this again" }),
    );

    expect(dismiss).toHaveBeenCalledOnce();
  });
});
