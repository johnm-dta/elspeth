import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Layout } from "./Layout";

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string): string | null => store[key] ?? null),
    setItem: vi.fn((key: string, val: string) => { store[key] = val; }),
    clear: () => { store = {}; },
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });
Object.defineProperty(window, "innerWidth", { value: 1600, writable: true });

describe("Layout", () => {
  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
  });

  it("uses approximately 50% of remaining space for inspector by default", () => {
    const { container } = render(
      <Layout
        sidebar={<div>Sidebar</div>}
        chat={<div>Chat</div>}
        inspector={<div>Inspector</div>}
      />,
    );
    const layoutDiv = container.querySelector(".app-layout") as HTMLElement;
    const columns = layoutDiv.style.gridTemplateColumns;
    const match = columns.match(/(\d+)px$/);
    expect(match).not.toBeNull();
    const inspectorWidth = Number(match![1]);
    // With 1600px viewport and 200px sidebar, half of remaining = 700px
    expect(inspectorWidth).toBeGreaterThanOrEqual(600);
    expect(inspectorWidth).toBeLessThanOrEqual(800);
  });

  it("restores persisted inspector width from localStorage", () => {
    localStorageMock.getItem.mockImplementation((key: string) => {
      if (key === "elspeth_inspector_width") return "500";
      return null;
    });
    const { container } = render(
      <Layout
        sidebar={<div>Sidebar</div>}
        chat={<div>Chat</div>}
        inspector={<div>Inspector</div>}
      />,
    );
    const layoutDiv = container.querySelector(".app-layout") as HTMLElement;
    const columns = layoutDiv.style.gridTemplateColumns;
    expect(columns).toContain("500px");
  });

  describe("Layout resize handle keyboard arrows", () => {
    function lastPersistedWidth(setItem: ReturnType<typeof vi.spyOn>): number {
      const calls = setItem.mock.calls.filter(
        ([k]: [string, string]) => k === "elspeth_inspector_width",
      );
      return Number(calls[calls.length - 1]?.[1]);
    }

    it("ArrowLeft decreases by 10px and ArrowRight increases by 10px (per-keypress step)", async () => {
      const user = userEvent.setup();
      const setItem = vi.spyOn(localStorageMock, "setItem");

      const { container } = render(
        <Layout
          sidebar={<div>Sidebar</div>}
          chat={<div>Chat</div>}
          inspector={<div>Inspector</div>}
        />,
      );

      // Read the initial mounted width directly from the grid template so
      // step-size assertions are anchored to a known baseline.
      const layoutDiv = container.querySelector(".app-layout") as HTMLElement;
      const initialMatch = layoutDiv.style.gridTemplateColumns.match(/(\d+)px$/);
      const initialWidth = Number(initialMatch![1]);

      const handle = screen.getByRole("separator", { name: /resize inspector/i });
      handle.focus();

      await user.keyboard("{ArrowRight}");
      expect(lastPersistedWidth(setItem)).toBe(initialWidth + 10);

      await user.keyboard("{ArrowLeft}");
      expect(lastPersistedWidth(setItem)).toBe(initialWidth);

      await user.keyboard("{ArrowLeft}");
      expect(lastPersistedWidth(setItem)).toBe(initialWidth - 10);
    });

    it("clamps inspector width at 50% of viewport on repeated ArrowRight", async () => {
      // window.innerWidth is fixed at 1600 for these tests, so the upper
      // clamp is 800.  Start near the upper bound so the clamp is reachable
      // in a few keypresses.
      const user = userEvent.setup();
      const setItem = vi.spyOn(localStorageMock, "setItem");

      localStorageMock.getItem.mockImplementation((key: string) => {
        if (key === "elspeth_inspector_width") return "780";
        return null;
      });

      render(
        <Layout
          sidebar={<div>Sidebar</div>}
          chat={<div>Chat</div>}
          inspector={<div>Inspector</div>}
        />,
      );

      const handle = screen.getByRole("separator", { name: /resize inspector/i });
      handle.focus();

      // 780 → 790 → 800 → 800 (clamp).  Three keypresses; clamp at 800.
      await user.keyboard("{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}");
      expect(lastPersistedWidth(setItem)).toBe(800);
    });

    it("clamps inspector width at MIN_INSPECTOR_WIDTH (240px) on repeated ArrowLeft", async () => {
      const user = userEvent.setup();
      const setItem = vi.spyOn(localStorageMock, "setItem");

      // Start narrow so the clamp is reachable in a few keypresses.
      localStorageMock.getItem.mockImplementation((key: string) => {
        if (key === "elspeth_inspector_width") return "260";
        return null;
      });

      render(
        <Layout
          sidebar={<div>Sidebar</div>}
          chat={<div>Chat</div>}
          inspector={<div>Inspector</div>}
        />,
      );

      const handle = screen.getByRole("separator", { name: /resize inspector/i });
      handle.focus();

      // 260 → 250 → 240 → 240 (clamp).  Three keypresses; clamp at 240.
      await user.keyboard("{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}");
      expect(lastPersistedWidth(setItem)).toBe(240);
    });

    it("exposes aria-valuenow/min/max so AT can announce current width", () => {
      render(
        <Layout
          sidebar={<div>Sidebar</div>}
          chat={<div>Chat</div>}
          inspector={<div>Inspector</div>}
        />,
      );

      const handle = screen.getByRole("separator", { name: /resize inspector/i });
      expect(handle.getAttribute("aria-valuenow")).not.toBeNull();
      expect(Number(handle.getAttribute("aria-valuenow"))).toBeGreaterThanOrEqual(240);
      expect(handle.getAttribute("aria-valuemin")).toBe("240");
      // valuemax = 50% of viewport (1600 / 2 = 800).
      expect(handle.getAttribute("aria-valuemax")).toBe("800");
    });
  });
});
