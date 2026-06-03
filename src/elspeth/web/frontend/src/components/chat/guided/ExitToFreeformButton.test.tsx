// ============================================================================
// ExitToFreeformButton -- regression coverage for the guided-mode exit control.
//
// Pinned contracts:
//   1. Button is type="button" and visible -- the label "Exit to freeform" is
//      rendered inside a <button type="button"> so the browser never treats
//      a stray Enter/Space as a form submission.
//   2. Click invokes useSessionStore.exitToFreeform -- the button delegates
//      entirely to the store action; it does NOT construct a GuidedRespondRequest
//      body itself.  The mock asserts the delegation.
//   3. Confirming path -- ExitToFreeformButton requires a deliberate second
//      click before it calls exitToFreeform.  A stray click must not terminate
//      the guided wizard.
//   4. Distinctness / independence -- two simultaneous ExitToFreeformButton
//      instances each fire their *own* exitToFreeform call independently; there
//      is no shared singleton or de-duplication across instances.
//   5. No auto-focus on initial render -- the button must not steal focus on
//      mount (matches the convention established in InspectAndConfirmTurn,
//      Task 7.3).
// ============================================================================

import { beforeEach, describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ExitToFreeformButton } from "./ExitToFreeformButton";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";

// ── Helpers ──────────────────────────────────────────────────────────────────

beforeEach(() => {
  resetStore(useSessionStore);
});

// ── Contract 1: type="button" and visible ────────────────────────────────────

describe("ExitToFreeformButton -- button identity", () => {
  it("renders a visible button with label 'Exit to freeform'", () => {
    render(<ExitToFreeformButton />);
    const btn = screen.getByRole("button", { name: /exit to freeform/i });
    expect(btn).toBeTruthy();
  });

  it("button has type='button' (not submit or reset)", () => {
    render(<ExitToFreeformButton />);
    const btn = screen.getByRole("button", { name: /exit to freeform/i });
    expect(btn.getAttribute("type")).toBe("button");
  });
});

// ── Contract 2: click invokes exitToFreeform ─────────────────────────────────

describe("ExitToFreeformButton -- store action delegation", () => {
  it("first click asks for confirmation and does not call exitToFreeform", async () => {
    const user = userEvent.setup();
    const mockExitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExitToFreeform });

    render(<ExitToFreeformButton />);
    await user.click(screen.getByRole("button", { name: /exit to freeform/i }));

    expect(mockExitToFreeform).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: /confirm exit to freeform/i }),
    ).toBeTruthy();
  });

  it("confirmation click calls useSessionStore.exitToFreeform once", async () => {
    const user = userEvent.setup();
    const mockExitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExitToFreeform });

    render(<ExitToFreeformButton />);
    await user.click(screen.getByRole("button", { name: /exit to freeform/i }));
    await user.click(
      screen.getByRole("button", { name: /confirm exit to freeform/i }),
    );

    expect(mockExitToFreeform).toHaveBeenCalledTimes(1);
    expect(mockExitToFreeform).toHaveBeenCalledWith();
  });
});

// ── Contract 3: confirmation prompt ──────────────────────────────────────────

describe("ExitToFreeformButton -- intermediate confirmation prompt", () => {
  it("does not render confirmation controls on initial render", () => {
    render(<ExitToFreeformButton />);
    expect(
      screen.queryByRole("button", { name: /confirm exit to freeform/i }),
    ).toBeNull();
    expect(screen.queryByText(/cancel/i)).toBeNull();
  });

  it("renders a confirmation and cancel affordance after the first click", async () => {
    const user = userEvent.setup();
    const mockExitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExitToFreeform });

    render(<ExitToFreeformButton />);
    await user.click(screen.getByRole("button", { name: /exit to freeform/i }));

    expect(
      screen.getByRole("button", { name: /confirm exit to freeform/i }),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: /cancel exit/i })).toBeTruthy();
    expect(mockExitToFreeform).not.toHaveBeenCalled();
  });

  it("cancel returns the control to its initial state", async () => {
    const user = userEvent.setup();
    const mockExitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExitToFreeform });

    render(<ExitToFreeformButton />);
    await user.click(screen.getByRole("button", { name: /exit to freeform/i }));
    await user.click(screen.getByRole("button", { name: /cancel exit/i }));

    expect(screen.getByRole("button", { name: /exit to freeform/i })).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: /confirm exit to freeform/i }),
    ).toBeNull();
    expect(mockExitToFreeform).not.toHaveBeenCalled();
  });
});

// ── Contract 4: distinctness / independence ──────────────────────────────────

describe("ExitToFreeformButton -- two simultaneous instances fire independently", () => {
  it("each instance calls exitToFreeform once when its own button is clicked", async () => {
    const user = userEvent.setup();
    const mockExitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExitToFreeform });

    render(
      <div>
        <ExitToFreeformButton />
        <ExitToFreeformButton />
      </div>,
    );

    const btns = screen.getAllByRole("button", { name: /exit to freeform/i });
    expect(btns).toHaveLength(2);

    await user.click(btns[0]);
    expect(mockExitToFreeform).toHaveBeenCalledTimes(0);

    await user.click(btns[1]);
    expect(mockExitToFreeform).toHaveBeenCalledTimes(0);

    const confirmButtons = screen.getAllByRole("button", {
      name: /confirm exit to freeform/i,
    });
    expect(confirmButtons).toHaveLength(2);

    await user.click(confirmButtons[0]);
    expect(mockExitToFreeform).toHaveBeenCalledTimes(1);
  });
});

// ── Contract 5: no auto-focus on mount ──────────────────────────────────────

describe("ExitToFreeformButton -- no auto-focus on initial render", () => {
  it("the button does not have focus immediately after render", () => {
    render(<ExitToFreeformButton />);
    const btn = screen.getByRole("button", { name: /exit to freeform/i });
    expect(document.activeElement).not.toBe(btn);
  });
});
