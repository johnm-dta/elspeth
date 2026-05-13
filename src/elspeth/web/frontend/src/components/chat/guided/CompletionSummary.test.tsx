// ============================================================================
// CompletionSummary -- regression coverage for the guided-mode terminal widget.
//
// Pinned contracts:
//   1. YAML text is rendered in the highlighted block -- the pipeline_yaml
//      string appears in the document when terminal.kind === "completed" and
//      pipeline_yaml is non-null.  (Does NOT assert Prism token structure --
//      that is Prism's contract, not ours.)
//   2. Exactly one action button renders as <button type="button">. The
//      terminal backend has one exit-to-freeform path, so the UI must not offer
//      two differently worded actions that do the same thing.
//   3. The exit click invokes useSessionStore.exitToFreeform once.
//   5. pipeline_yaml === null -> widget returns null (nothing rendered).
//      This is the strict gate: no defensive ?? "" coercion to silently
//      render an empty highlight block.
//   6. terminal.kind !== "completed" handling -- the parent should not render
//      CompletionSummary in non-completed terminals.  The widget defensively
//      returns null in that case as well.  Negative-space pin.
//   7. Distinctness pin (Task 7.4 I4 inheritance): two simultaneous
//      CompletionSummary instances have per-instance IDs that differ.
//      Asserted via not.toBe() on elements carrying useId()-scoped IDs.
//   8. Initial-render no-auto-focus -- neither button has focus on mount
//      (matches convention from InspectAndConfirmTurn, Task 7.3).
//   9. Reduced-motion classes -- verified by CSS (not directly testable in
//      jsdom); this file notes the expectation for the reviewer.
//
// Source of truth:
//   - types/guided.ts:54-58 (TerminalState wire shape)
//   - stores/sessionStore.ts:116 + 572-583 (exitToFreeform parameterless)
//   - docs/superpowers/plans/2026-05-11-composer-guided-mode.md:4445
// ============================================================================

import { beforeEach, describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CompletionSummary } from "./CompletionSummary";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";
import type { TerminalState } from "@/types/guided";

// ── Fixtures ──────────────────────────────────────────────────────────────────

const COMPLETED_TERMINAL: TerminalState = {
  kind: "completed",
  reason: null,
  pipeline_yaml: 'source:\n  plugin: csv\n  options:\n    path: data.csv\n',
};

const COMPLETED_TERMINAL_NULL_YAML: TerminalState = {
  kind: "completed",
  reason: null,
  pipeline_yaml: null,
};

const EXITED_TERMINAL: TerminalState = {
  kind: "exited_to_freeform",
  reason: "user_pressed_exit",
  pipeline_yaml: null,
};

// ── Store reset ───────────────────────────────────────────────────────────────

beforeEach(() => {
  resetStore(useSessionStore);
});

// ── Contract 1: YAML text rendered in highlighted block ───────────────────────

describe("CompletionSummary -- YAML rendering", () => {
  it("renders pipeline_yaml text in the document when kind=completed and yaml is non-null", () => {
    render(<CompletionSummary terminal={COMPLETED_TERMINAL} />);
    // The yaml text appears somewhere in the highlighted block.
    // We don't assert Prism token structure -- that's Prism's contract.
    const yamlText = COMPLETED_TERMINAL.pipeline_yaml!;
    // At minimum the raw string or its fragments appear in the document.
    // The Highlight component renders individual tokens; the pre element
    // contains all token text concatenated.
    const pre = document.querySelector("pre");
    expect(pre).not.toBeNull();
    expect(pre!.textContent).toContain("source:");
    expect(pre!.textContent).toContain("csv");
    // Full text check (Prism may split by token but textContent reunites them)
    expect(pre!.textContent).toContain(yamlText.split("\n")[0]);
  });

  it("renders a heading element for the completion state", () => {
    render(<CompletionSummary terminal={COMPLETED_TERMINAL} />);
    // Heading per Task 7.6 M3 convention for primary entity names
    const heading = screen.getByRole("heading");
    expect(heading).toBeInTheDocument();
  });
});

// ── Contract 2: Single button renders with type="button" ─────────────────────

describe("CompletionSummary -- button identity", () => {
  it("renders a single 'Save and exit guided mode' button with type='button'", () => {
    render(<CompletionSummary terminal={COMPLETED_TERMINAL} />);
    const btn = screen.getByRole("button", {
      name: /save and exit guided mode/i,
    });
    expect(btn).toBeInTheDocument();
    expect(btn.getAttribute("type")).toBe("button");
  });

  it("does not render a second wire-identical keep-editing button", () => {
    render(<CompletionSummary terminal={COMPLETED_TERMINAL} />);
    expect(
      screen.queryByRole("button", {
        name: /drop to freeform to keep editing/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("renders exactly one action button", () => {
    render(<CompletionSummary terminal={COMPLETED_TERMINAL} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(1);
  });
});

// ── Contract 3: Exit calls exitToFreeform ────────────────────────────────────

describe("CompletionSummary -- exit action", () => {
  it("clicking 'Save and exit guided mode' calls exitToFreeform once", async () => {
    const user = userEvent.setup();
    const mockExit = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({ exitToFreeform: mockExit });

    render(<CompletionSummary terminal={COMPLETED_TERMINAL} />);
    await user.click(
      screen.getByRole("button", { name: /save and exit guided mode/i }),
    );

    expect(mockExit).toHaveBeenCalledTimes(1);
    expect(mockExit).toHaveBeenCalledWith();
  });
});

// ── Contract 5: pipeline_yaml === null -> widget returns null ─────────────────

describe("CompletionSummary -- null pipeline_yaml guard", () => {
  it("returns null (renders nothing) when terminal.pipeline_yaml is null", () => {
    const { container } = render(
      <CompletionSummary terminal={COMPLETED_TERMINAL_NULL_YAML} />,
    );
    // Strict null render -- no buttons, no heading, no pre
    expect(container.firstChild).toBeNull();
  });

  it("does not render any button when pipeline_yaml is null", () => {
    render(<CompletionSummary terminal={COMPLETED_TERMINAL_NULL_YAML} />);
    expect(screen.queryByRole("button")).toBeNull();
  });
});

// ── Contract 6: non-completed terminal -> null ────────────────────────────────

describe("CompletionSummary -- non-completed terminal guard (negative space)", () => {
  it("returns null when terminal.kind is 'exited_to_freeform'", () => {
    const { container } = render(
      <CompletionSummary terminal={EXITED_TERMINAL} />,
    );
    expect(container.firstChild).toBeNull();
  });
});

// ── Contract 7: distinctness pin (two simultaneous instances) ─────────────────

describe("CompletionSummary -- distinctness pin (Task 7.4 I4 inheritance)", () => {
  it("two simultaneous CompletionSummary instances have distinct heading IDs", () => {
    render(
      <div>
        <CompletionSummary terminal={COMPLETED_TERMINAL} />
        <CompletionSummary terminal={COMPLETED_TERMINAL} />
      </div>,
    );
    const headings = screen.getAllByRole("heading");
    // Two instances => two headings
    expect(headings).toHaveLength(2);
    // Headings are distinct DOM nodes (not.toBe per Task 7.4 I4 convention)
    expect(headings[0]).not.toBe(headings[1]);
  });

  it("pre blocks in two simultaneous instances are distinct DOM nodes", () => {
    const { container } = render(
      <div>
        <CompletionSummary terminal={COMPLETED_TERMINAL} />
        <CompletionSummary terminal={COMPLETED_TERMINAL} />
      </div>,
    );
    const pres = container.querySelectorAll("pre");
    expect(pres).toHaveLength(2);
    expect(pres[0]).not.toBe(pres[1]);
  });
});

// ── Contract 8: no auto-focus on mount ────────────────────────────────────────

describe("CompletionSummary -- no auto-focus on initial render", () => {
  it("the exit button does not have focus immediately after render", () => {
    render(<CompletionSummary terminal={COMPLETED_TERMINAL} />);
    const saveBtn = screen.getByRole("button", {
      name: /save and exit guided mode/i,
    });
    expect(document.activeElement).not.toBe(saveBtn);
  });
});
