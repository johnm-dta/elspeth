import { act, render, renderHook, screen } from "@testing-library/react";
import { createElement } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { GraphModal } from "@/components/sidebar/GraphModal";
import {
  OPEN_GRAPH_MODAL_EVENT,
  OPEN_YAML_MODAL_EVENT,
} from "@/lib/composer-events";
import { resetStore } from "@/test/store-helpers";
import { useSessionStore } from "@/stores/sessionStore";
import { useHashRouter } from "./useHashRouter";

vi.mock("@/components/inspector/GraphView", () => ({
  GraphView: () => createElement("div", { "data-testid": "graph-view-stub" }),
}));

describe("useHashRouter Phase 3B fragment migration", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    window.history.replaceState(null, "", window.location.pathname);
    useSessionStore.setState({
      sessions: [{ id: "sess-1", title: "Session 1" } as never],
      activeSessionId: null,
      selectSession: vi.fn(),
    } as never);
  });

  it("rewrites #/{id}/spec to #/{id}", () => {
    window.history.replaceState(null, "", "#/sess-1/spec");

    renderHook(() => useHashRouter());

    expect(window.location.hash).toBe("#/sess-1");
  });

  it("rewrites #/{id}/runs to #/{id}", () => {
    window.history.replaceState(null, "", "#/sess-1/runs");

    renderHook(() => useHashRouter());

    expect(window.location.hash).toBe("#/sess-1");
  });

  it("opens the graph modal event and rewrites #/{id}/graph", async () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
    window.history.replaceState(null, "", "#/sess-1/graph");

    renderHook(() => useHashRouter());
    await act(async () => {});

    expect(handler).toHaveBeenCalled();
    expect(window.location.hash).toBe("#/sess-1");
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
  });

  it("opens the yaml modal event and rewrites #/{id}/yaml", async () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);
    window.history.replaceState(null, "", "#/sess-1/yaml");

    renderHook(() => useHashRouter());
    await act(async () => {});

    expect(handler).toHaveBeenCalled();
    expect(window.location.hash).toBe("#/sess-1");
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
  });

  it("strips any unrecognized verb", () => {
    window.history.replaceState(null, "", "#/sess-1/nonsense");

    renderHook(() => useHashRouter());

    expect(window.location.hash).toBe("#/sess-1");
  });

  it("cold-loads the graph modal when the graph hash exists before mount", async () => {
    window.history.replaceState(null, "", "#/sess-1/graph");

    function HarnessTree() {
      useHashRouter();
      return createElement(GraphModal);
    }

    render(createElement(HarnessTree));
    await act(async () => {});

    expect(
      screen.getByRole("dialog", { name: /pipeline graph/i }),
    ).toBeInTheDocument();
  });
});
