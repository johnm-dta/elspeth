import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
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
      screen.getByText(
        "A pipeline is a controlled route for information. You choose what " +
          "enters, what happens to it, and where the result goes. ELSPETH " +
          "records each step so you can review how every output was produced.",
      ),
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

    const definitions = [
      [
        "Sources",
        "bring records into the pipeline from files, databases, APIs, or " +
          "text. ELSPETH tracks each incoming record through the run.",
      ],
      [
        "Transforms",
        "examine or change records. They can clean fields, enrich content, " +
          "apply an LLM, or prepare data for the next step.",
      ],
      [
        "Sinks",
        "receive records at the end of a route. They can write results to " +
          "files, data stores, or other configured destinations; records " +
          "requiring attention can follow a separate route.",
      ],
      [
        "Gate",
        "is a sorting desk. It sends each case along the appropriate route " +
          "according to a stated condition.",
      ],
      [
        "Fork",
        "sends controlled copies of one case to several specialist teams. " +
          "ELSPETH tracks each parallel path independently.",
      ],
      [
        "Coalesce",
        "waits for the required specialist responses, then combines their " +
          "findings into one case that can continue.",
      ],
      [
        "Aggregate",
        "brings a group of cases together for batch work, such as producing " +
          "totals, statistics, or a report.",
      ],
      [
        "Queue",
        "is a shared inbox. It accepts cases from several upstream teams and " +
          "feeds one next step while keeping every case separate.",
      ],
      [
        "Expand",
        "opens a bundled case into several independently tracked cases.",
      ],
    ];

    for (const [term, description] of definitions) {
      const termElement = screen.getByText(term, { selector: "dt" });
      const definitionRow = termElement.parentElement;

      expect(termElement).toBeVisible();
      expect(definitionRow).not.toBeNull();
      expect(
        within(definitionRow!).getByText(description, { selector: "dd" }),
      ).toBeVisible();
    }

    expect(
      screen.getByText(
        "Wiring is the set of connections between these components. A simple " +
          "pipeline runs from source to transforms to sink. For a more " +
          "involved flow, think of each record as a case moving through a " +
          "controlled workplace:",
      ),
    ).toBeVisible();
    expect(
      screen.getByText(
        "Describe the outcome you need in ordinary language. ELSPETH will " +
          "propose the components and their wiring; review the graph and " +
          "details before you run it.",
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
