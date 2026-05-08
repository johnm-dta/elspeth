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
    it("ArrowLeft decreases inspector width and ArrowRight increases it", async () => {
      const user = userEvent.setup();
      const setItem = vi.spyOn(localStorageMock, "setItem");

      render(
        <Layout
          sidebar={<div>Sidebar</div>}
          chat={<div>Chat</div>}
          inspector={<div>Inspector</div>}
        />,
      );

      const handle = screen.getByRole("separator", { name: /resize inspector/i });
      handle.focus();

      await user.keyboard("{ArrowRight}");
      const widthAfterRight = Number(
        setItem.mock.calls.findLast(([k]) => k === "elspeth_inspector_width")?.[1],
      );

      await user.keyboard("{ArrowLeft}");
      const widthAfterLeft = Number(
        setItem.mock.calls.findLast(([k]) => k === "elspeth_inspector_width")?.[1],
      );

      expect(widthAfterLeft).toBeLessThan(widthAfterRight);
    });
  });
});
