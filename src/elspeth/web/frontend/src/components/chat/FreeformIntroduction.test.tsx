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

  it("explains the building blocks and wiring of an auditable pipeline", () => {
    render(<FreeformIntroduction />);

    expect(
      screen.getByRole("heading", { name: "How pipelines work", level: 2 }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", {
        name: "The three building blocks",
        level: 3,
      }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Wiring the flow", level: 3 }),
    ).toBeVisible();

    for (const term of [
      "Sources",
      "Transforms",
      "Sinks",
      "Gate",
      "Fork",
      "Coalesce",
      "Aggregate",
      "Queue",
      "Expand",
    ]) {
      expect(screen.getByText(term, { selector: "dt" })).toBeVisible();
    }

    expect(
      screen.getByText(
        /think of each record as a case moving through a controlled workplace/,
      ),
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

  it("disables the action and reports progress while a preference write is active", () => {
    usePreferencesStore.setState({ writing: true });
    render(<FreeformIntroduction />);

    expect(screen.getByRole("button", { name: "Hiding…" })).toBeDisabled();
  });
});
