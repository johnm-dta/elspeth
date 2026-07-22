import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ModeSwitchButton } from "./ModeSwitchButton";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";

describe("ModeSwitchButton", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
  });

  it("target=guided, no work: a single click switches immediately (no confirm)", () => {
    const enterGuided = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ enterGuided });

    render(<ModeSwitchButton target="guided" hasWork={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Switch to guided" }));

    expect(enterGuided).toHaveBeenCalledTimes(1);
    expect(
      screen.queryByRole("button", { name: /^confirm/i }),
    ).toBeNull();
  });

  it("target=guided, with work: click reveals a confirm; confirm switches", () => {
    const enterGuided = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ enterGuided });

    render(<ModeSwitchButton target="guided" hasWork />);
    fireEvent.click(screen.getByRole("button", { name: "Switch to guided" }));

    // First click only arms the confirmation — it must NOT switch yet.
    expect(enterGuided).not.toHaveBeenCalled();
    fireEvent.click(
      screen.getByRole("button", { name: "Confirm switch to guided" }),
    );
    expect(enterGuided).toHaveBeenCalledTimes(1);
  });

  it("target=guided, with work: the confirm is honest that guided starts fresh and the pipeline is recoverable", () => {
    // "Fresh wizard + consent" (elspeth-e2c3dba6b5): converting a worked
    // freeform session reseeds a fresh wizard and sets the current pipeline
    // aside. The two-step confirm must disclose this rather than implying a
    // lossless in-place switch — the recoverability (version history) is what
    // makes the discard consented rather than surprising.
    const enterGuided = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ enterGuided });

    render(<ModeSwitchButton target="guided" hasWork />);
    fireEvent.click(screen.getByRole("button", { name: "Switch to guided" }));

    expect(screen.getByText(/fresh/i)).toBeInTheDocument();
    expect(screen.getByText(/version history/i)).toBeInTheDocument();
  });

  it("target=guided, with work: renders an explicit contextual confirmation panel", () => {
    const enterGuided = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ enterGuided });

    const { container } = render(<ModeSwitchButton target="guided" hasWork />);
    fireEvent.click(screen.getByRole("button", { name: "Switch to guided" }));

    expect(container.querySelector(".mode-switch-confirm-card")).not.toBeNull();
    expect(screen.getByText("Switch to guided mode?")).toBeInTheDocument();
    expect(
      screen.getByRole("group", { name: "Confirm switch to guided" }),
    ).toHaveAttribute("aria-describedby");
    expect(
      screen.getByRole("button", { name: "Confirm switch to guided" }),
    ).toHaveAccessibleDescription(/version history/i);
  });

  it("target=freeform, with work: the confirm does NOT carry the fresh-wizard note", () => {
    // The disclosure is guided-direction only; exiting to freeform is a
    // genuinely lossless in-place switch and must keep its terse confirm.
    const exitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform });

    render(<ModeSwitchButton target="freeform" hasWork />);
    fireEvent.click(screen.getByRole("button", { name: "Exit to freeform" }));

    expect(screen.queryByText(/version history/i)).toBeNull();
  });

  it("target=freeform, with work: names the current guided context before confirming", () => {
    const exitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform });

    render(<ModeSwitchButton target="freeform" hasWork />);
    fireEvent.click(screen.getByRole("button", { name: "Exit to freeform" }));

    expect(screen.getByText("Exit to freeform mode?")).toBeInTheDocument();
    expect(screen.getByText(/continue in the freeform composer/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Confirm exit to freeform" }),
    ).toHaveAccessibleDescription(/continue in the freeform composer/i);
  });

  it("with work: Cancel dismisses the confirm without switching", () => {
    const enterGuided = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ enterGuided });

    render(<ModeSwitchButton target="guided" hasWork />);
    fireEvent.click(screen.getByRole("button", { name: "Switch to guided" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(enterGuided).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Switch to guided" }),
    ).toBeInTheDocument();
  });

  it("target=freeform: labelled 'Exit to freeform' and calls exitToFreeform", () => {
    const exitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform });

    render(<ModeSwitchButton target="freeform" hasWork={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Exit to freeform" }));

    expect(exitToFreeform).toHaveBeenCalledTimes(1);
  });

  it("target=freeform, with work: confirm label is 'Confirm exit to freeform'", () => {
    const exitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform });

    render(<ModeSwitchButton target="freeform" hasWork />);
    fireEvent.click(screen.getByRole("button", { name: "Exit to freeform" }));

    expect(exitToFreeform).not.toHaveBeenCalled();
    fireEvent.click(
      screen.getByRole("button", { name: "Confirm exit to freeform" }),
    );
    expect(exitToFreeform).toHaveBeenCalledTimes(1);
  });

  // ── C-4b: permanently-terminal guided sessions ─────────────────────────────

  it("disabledReason set: renders a disabled button with the explanation, never calls enterGuided", () => {
    const enterGuided = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ enterGuided });

    render(
      <ModeSwitchButton
        target="guided"
        hasWork={false}
        disabledReason="Guided ended for this session — start a new session to use guided."
      />,
    );

    const button = screen.getByRole("button", { name: "Switch to guided" });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(enterGuided).not.toHaveBeenCalled();

    expect(
      screen.getByText(
        "Guided ended for this session — start a new session to use guided.",
      ),
    ).toBeInTheDocument();
    // The explanation is programmatically associated with the button, not
    // just visually nearby — a screen reader must not silently skip it.
    expect(button).toHaveAttribute("aria-describedby");
    const describedById = button.getAttribute("aria-describedby");
    expect(document.getElementById(describedById!)).toHaveTextContent(
      "Guided ended for this session",
    );
  });

  it("disabledReason unset: the normal switch/confirm flow renders instead", () => {
    render(<ModeSwitchButton target="guided" hasWork={false} />);

    expect(
      screen.getByRole("button", { name: "Switch to guided" }),
    ).not.toBeDisabled();
  });
});
