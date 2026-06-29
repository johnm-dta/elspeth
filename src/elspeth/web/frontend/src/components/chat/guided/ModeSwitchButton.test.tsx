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
});
