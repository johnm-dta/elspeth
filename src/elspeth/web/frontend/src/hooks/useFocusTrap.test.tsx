import { render, screen, fireEvent, act } from "@testing-library/react";
import { useRef } from "react";
import { describe, it, expect } from "vitest";
import { useFocusTrap } from "./useFocusTrap";

function Trap({ label, active = true }: { label: string; active?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useFocusTrap(ref, active);
  return (
    <div ref={ref} aria-label={label}>
      <button>{label}-first</button>
      <button>{label}-last</button>
    </div>
  );
}

function StackedTraps() {
  // The inner trap is a SIBLING of the outer (mirrors SecretsPanel rendering its
  // ConfirmDialog outside the panel's trap container). Both are active, so the
  // outer trap's document listener sees focus "outside itself" while focus lives
  // in the inner trap.
  const outerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  useFocusTrap(outerRef, true); // activated first
  useFocusTrap(innerRef, true); // activated last -> topmost
  return (
    <>
      <div ref={outerRef} aria-label="outer">
        <button>outer-first</button>
        <button>outer-last</button>
      </div>
      <div ref={innerRef} aria-label="inner">
        <button>inner-first</button>
        <button>inner-last</button>
      </div>
    </>
  );
}

describe("useFocusTrap recapture (WCAG 2.4.3 — modal focus containment)", () => {
  it("pulls focus back to the first element on Tab when focus has escaped the trap", () => {
    render(
      <>
        <button>outside</button>
        <Trap label="trap" />
      </>,
    );
    const outside = screen.getByRole("button", { name: "outside" });
    act(() => outside.focus());
    expect(outside).toHaveFocus();

    fireEvent.keyDown(document, { key: "Tab" });

    expect(screen.getByRole("button", { name: "trap-first" })).toHaveFocus();
  });

  it("pulls focus back to the last element on Shift+Tab when focus has escaped", () => {
    render(
      <>
        <button>outside</button>
        <Trap label="trap" />
      </>,
    );
    const outside = screen.getByRole("button", { name: "outside" });
    act(() => outside.focus());

    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });

    expect(screen.getByRole("button", { name: "trap-last" })).toHaveFocus();
  });

  it("still wraps from last to first within the trap (existing behaviour preserved)", () => {
    render(<Trap label="trap" />);
    const last = screen.getByRole("button", { name: "trap-last" });
    act(() => last.focus());

    fireEvent.keyDown(document, { key: "Tab" });

    expect(screen.getByRole("button", { name: "trap-first" })).toHaveFocus();
  });

  it("only the topmost trap acts when traps are stacked — no cross-trap focus thrash", () => {
    render(<StackedTraps />);
    const innerLast = screen.getByRole("button", { name: "inner-last" });
    act(() => innerLast.focus());

    // Count focus moves caused by the Tab: the topmost (inner) trap should make
    // exactly ONE move (last -> first within itself). If the non-topmost (outer)
    // trap also acted, it would first yank focus to an outer button (a second
    // move), thrashing focus out of the active dialog.
    const moves: string[] = [];
    const onFocusIn = (e: Event) => {
      const el = e.target as HTMLElement;
      moves.push(el.getAttribute("aria-label") ?? el.textContent ?? "");
    };
    document.addEventListener("focusin", onFocusIn);
    fireEvent.keyDown(document, { key: "Tab" });
    document.removeEventListener("focusin", onFocusIn);

    expect(screen.getByRole("button", { name: "inner-first" })).toHaveFocus();
    expect(moves).toEqual(["inner-first"]);
  });
});
