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
//   3. No-confirmation path -- ExitToFreeformButton fires immediately on click
//      (no intermediate confirm dialog).  This is intentionally simple for the
//      demo path; the plan notes an operator may add confirmation later.
//      This contract pins the *current* behaviour so a future "add confirmation"
//      refactor must first update these tests rather than silently changing UX.
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
  it("clicking the button calls useSessionStore.exitToFreeform once", async () => {
    const user = userEvent.setup();
    const mockExitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExitToFreeform });

    render(<ExitToFreeformButton />);
    await user.click(screen.getByRole("button", { name: /exit to freeform/i }));

    expect(mockExitToFreeform).toHaveBeenCalledTimes(1);
    expect(mockExitToFreeform).toHaveBeenCalledWith();
  });

  it("clicking the button a second time calls exitToFreeform a second time", async () => {
    const user = userEvent.setup();
    const mockExitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExitToFreeform });

    render(<ExitToFreeformButton />);
    const btn = screen.getByRole("button", { name: /exit to freeform/i });
    await user.click(btn);
    await user.click(btn);

    expect(mockExitToFreeform).toHaveBeenCalledTimes(2);
  });
});

// ── Contract 3: no-confirmation (fires immediately) ──────────────────────────

describe("ExitToFreeformButton -- no intermediate confirmation dialog", () => {
  it("does NOT render any confirmation dialog on initial render", () => {
    render(<ExitToFreeformButton />);
    // Common confirmation patterns: role="alertdialog", role="dialog", or text
    // containing "are you sure"/"confirm"/"cancel".
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByText(/are you sure/i)).toBeNull();
    expect(screen.queryByText(/cancel/i)).toBeNull();
  });

  it("does NOT render a confirmation dialog after clicking the button", async () => {
    const user = userEvent.setup();
    const mockExitToFreeform = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExitToFreeform });

    render(<ExitToFreeformButton />);
    await user.click(screen.getByRole("button", { name: /exit to freeform/i }));

    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
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
    expect(mockExitToFreeform).toHaveBeenCalledTimes(1);

    await user.click(btns[1]);
    expect(mockExitToFreeform).toHaveBeenCalledTimes(2);
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
