import { act, fireEvent, render, renderHook, screen } from "@testing-library/react";
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

/** Minimal composition with content (one source) — export is meaningful. */
function nonEmptyCompositionState() {
  return {
    id: "state-1",
    version: 1,
    sources: { source: { plugin: "csv", options: {} } },
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

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

  it("opens the yaml modal event and rewrites #/{id}/yaml when the pipeline has content", async () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);
    window.history.replaceState(null, "", "#/sess-1/yaml");
    // Session already active with a KNOWN, non-empty composition — the
    // yaml verb is content-gated (elspeth-bff8043d33 residual).
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionStateLoaded: true,
      compositionState: nonEmptyCompositionState(),
    } as never);

    renderHook(() => useHashRouter());
    await act(async () => {});

    expect(handler).toHaveBeenCalled();
    expect(window.location.hash).toBe("#/sess-1");
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
  });

  it("does NOT open the yaml modal for #/{id}/yaml on an empty pipeline", async () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);
    window.history.replaceState(null, "", "#/sess-1/yaml");
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionStateLoaded: true,
      compositionState: null,
    } as never);

    renderHook(() => useHashRouter());
    await act(async () => {});

    expect(handler).not.toHaveBeenCalled();
    // The hash is still canonicalised — only the modal open is withheld.
    expect(window.location.hash).toBe("#/sess-1");
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
  });

  it("defers the yaml modal until the composition state loads, then gates on content", async () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);
    window.history.replaceState(null, "", "#/sess-1/yaml");
    // Fresh deep-link arrival: selectSession's fetch is still in flight, so
    // the composition is not yet known.
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionStateLoaded: false,
      compositionState: null,
    } as never);

    renderHook(() => useHashRouter());
    await act(async () => {});
    expect(handler).not.toHaveBeenCalled();

    // The fetch settles with content — the deferred dispatch fires.
    await act(async () => {
      useSessionStore.setState({
        compositionStateLoaded: true,
        compositionState: nonEmptyCompositionState(),
      } as never);
    });

    expect(handler).toHaveBeenCalledTimes(1);
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

  it("defers cold-load graph actions until enabled", async () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
    window.history.replaceState(null, "", "#/sess-1/graph");

    const { rerender } = renderHook(
      ({ enabled }) => useHashRouter({ enabled }),
      { initialProps: { enabled: false } },
    );
    await act(async () => {});

    expect(handler).not.toHaveBeenCalled();
    expect(window.location.hash).toBe("#/sess-1/graph");

    rerender({ enabled: true });
    await act(async () => {});

    expect(handler).toHaveBeenCalled();
    expect(window.location.hash).toBe("#/sess-1");
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
  });
});

describe("useHashRouter — Batch 2 fixes", () => {
  const TOAST_KEY = "elspeth_redirect_toast_dismissed";

  beforeEach(() => {
    resetStore(useSessionStore);
    window.history.replaceState(null, "", window.location.pathname);
    localStorage.removeItem(TOAST_KEY);
    useSessionStore.setState({
      sessions: [{ id: "sess-1", title: "Session 1" } as never],
      activeSessionId: null,
      selectSession: vi.fn(),
    } as never);
  });

  // ── Fix A: prototype-walk guard ─────────────────────────────────────────
  //
  // The old code used `verb in ACTION_VERBS` which walks the prototype chain.
  // For example, `"constructor" in ACTION_VERBS` returns true even though
  // "constructor" is not an own property of the ACTION_VERBS object —
  // `ACTION_VERBS["constructor"]` returns the Object constructor function, and
  // `new CustomEvent(fn)` coerces it to a string and dispatches a garbage event.
  //
  // The fix uses Object.hasOwn(ACTION_VERBS, verb) which only checks own
  // properties.  We use "constructor" as the test verb because it:
  //   1. Is all-lowercase (matches the regex [a-z]+)
  //   2. Is a prototype-inherited property of plain objects
  //   3. Was exploitable under the old `in` check

  it("prototype-walk guard: #/{id}/constructor does not dispatch any CustomEvent", async () => {
    window.history.replaceState(null, "", "#/sess-1/constructor");

    // Spy on dispatchEvent BEFORE rendering so we capture all calls
    const spy = vi.spyOn(window, "dispatchEvent");

    renderHook(() => useHashRouter());
    // Flush any queued microtasks
    await act(async () => {});

    // No CustomEvent should be dispatched; native events (e.g. popstate) are
    // not CustomEvent instances so we filter to CustomEvent only.
    const customEvents = spy.mock.calls
      .map(([e]) => e)
      .filter((e) => e instanceof CustomEvent);
    expect(customEvents).toHaveLength(0);

    spy.mockRestore();
  });

  it("prototype-walk guard: #/{id}/constructor rewrites to #/{id} (silent strip)", () => {
    window.history.replaceState(null, "", "#/sess-1/constructor");

    renderHook(() => useHashRouter());

    // Silently stripped — no event, no toast, just the canonical hash
    expect(window.location.hash).toBe("#/sess-1");
  });

  // ── Fix C: retired-verb redirect toast ─────────────────────────────────

  it("shows redirect toast on first visit to #/{id}/runs", () => {
    window.history.replaceState(null, "", "#/sess-1/runs");

    const { result } = renderHook(() => useHashRouter());

    expect(result.current.redirectToast).not.toBeNull();
    expect(result.current.redirectToast?.message).toMatch(/Runs tab was removed/i);
  });

  it("shows redirect toast on first visit to #/{id}/spec", () => {
    window.history.replaceState(null, "", "#/sess-1/spec");

    const { result } = renderHook(() => useHashRouter());

    expect(result.current.redirectToast).not.toBeNull();
    expect(result.current.redirectToast?.message).toMatch(/Spec tab was removed/i);
  });

  it("dismiss clears toast state and writes localStorage flag", async () => {
    window.history.replaceState(null, "", "#/sess-1/runs");

    const { result } = renderHook(() => useHashRouter());
    expect(result.current.redirectToast).not.toBeNull();

    await act(async () => {
      result.current.redirectToast!.dismiss();
    });

    expect(result.current.redirectToast).toBeNull();
    expect(localStorage.getItem(TOAST_KEY)).toBe("1");
  });

  it("does not show toast after dismiss flag is set in localStorage (cross-path)", () => {
    // Simulate: user dismissed on /runs, now arrives at /spec in a fresh hook
    localStorage.setItem(TOAST_KEY, "1");

    window.history.replaceState(null, "", "#/sess-1/spec");

    const { result } = renderHook(() => useHashRouter());

    expect(result.current.redirectToast).toBeNull();
  });

  it("unrecognized non-retired verb does NOT show a toast", () => {
    // "nonsense" is not a retired verb — it is silently stripped with no toast
    window.history.replaceState(null, "", "#/sess-1/nonsense");

    const { result } = renderHook(() => useHashRouter());

    expect(result.current.redirectToast).toBeNull();
  });

  // ── Fix C: redirect toast renders in the DOM via App integration ────────

  it("redirect toast banner renders above the alert region when visiting #/{id}/runs", async () => {
    window.history.replaceState(null, "", "#/sess-1/runs");

    // Render a minimal harness that surfaces the toast DOM
    function ToastHarness() {
      const { redirectToast } = useHashRouter();
      if (!redirectToast) return null;
      return createElement(
        "div",
        { role: "alert", "data-testid": "toast-banner" },
        createElement("span", null, redirectToast.message),
        createElement(
          "button",
          { type: "button", onClick: redirectToast.dismiss, "aria-label": "Dismiss" },
          "Dismiss",
        ),
      );
    }

    render(createElement(ToastHarness));

    const banner = screen.getByRole("alert");
    expect(banner).toHaveTextContent(/Runs tab was removed/i);

    // Dismiss via the button
    fireEvent.click(screen.getByRole("button", { name: /Dismiss/i }));

    await act(async () => {});

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(localStorage.getItem(TOAST_KEY)).toBe("1");
  });

  // ── Fix B: try/finally guard on applying.current ────────────────────────

  it("applying.current is cleared so URL-echo subscription works after selectSession throws", async () => {
    // The try/finally in applyHash guarantees applying.current = false even
    // when selectSession throws.  We verify this by:
    //   1. Mounting the hook with a normal selectSession.
    //   2. Triggering a hashchange to a new session where selectSession throws.
    //      JSDOM propagates the uncaught exception as a window "error" event;
    //      we intercept it to prevent Vitest from treating it as an unhandled error.
    //   3. After the throw, triggering the URL-echo subscription (useEffect #3)
    //      directly via a store mutation and verifying the URL updates.
    //      If applying.current had stayed true, the subscription would early-return
    //      and the URL would not change.

    useSessionStore.setState({
      sessions: [
        { id: "sess-1", title: "Session 1" } as never,
        { id: "sess-2", title: "Session 2" } as never,
      ],
      activeSessionId: "sess-1",
      selectSession: vi.fn(),
    } as never);

    window.history.replaceState(null, "", "#/sess-1");
    renderHook(() => useHashRouter());

    // Replace selectSession with a throwing variant
    useSessionStore.setState({
      selectSession: vi.fn(() => {
        throw new Error("selectSession boom");
      }) as never,
    } as never);

    // Intercept the uncaught exception that JSDOM propagates so Vitest
    // doesn't flag it as an unhandled error.  We assert it was thrown so
    // the test fails if the exception is unexpectedly swallowed.
    let caught: Event | null = null;
    function onError(e: Event) {
      e.preventDefault(); // Suppress "Uncaught Exception" in Vitest
      caught = e;
    }
    window.addEventListener("error", onError);

    await act(async () => {
      window.history.replaceState(null, "", "#/sess-2");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });

    window.removeEventListener("error", onError);

    // Confirm the error was thrown (proving the throwing code path ran)
    expect(caught).not.toBeNull();

    // Now verify applying.current was reset by try/finally.
    // The URL-echo subscription (useEffect #3) fires whenever activeSessionId
    // changes — but only if applying.current is false.  Drive it directly.
    //
    // IMPORTANT: we use "sess-3" here, not "sess-2".  After the failed
    // hashchange, lastWrittenHash.current === "#/sess-2" (set by
    // handleHashChange before calling applyHash).  If we then set
    // activeSessionId="sess-2", the subscription early-returns because
    // hash === lastWrittenHash.current — that's a false pass whether or not
    // applying.current was fixed.  Using "sess-3" forces the subscription to
    // actually pushState, which only happens if applying.current is false.
    //
    // Under the bug (no try/finally):
    //   applying.current stays true → subscription early-returns → hash stays "#/sess-2"
    // Under the fix (try/finally):
    //   applying.current resets to false → subscription pushes "#/sess-3"
    await act(async () => {
      useSessionStore.setState({ selectSession: vi.fn() } as never);
      useSessionStore.setState({
        sessions: [
          { id: "sess-1", title: "Session 1" } as never,
          { id: "sess-2", title: "Session 2" } as never,
          { id: "sess-3", title: "Session 3" } as never,
        ],
        activeSessionId: "sess-3",
      } as never);
    });

    expect(window.location.hash).toBe("#/sess-3");
  });

  // ── popstate triggers reapply ────────────────────────────────────────────

  it("popstate event triggers graph modal dispatch when hash is #/{id}/graph", async () => {
    window.history.replaceState(null, "", "#/sess-1");
    renderHook(() => useHashRouter());

    const handler = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handler);

    await act(async () => {
      // Simulate navigating back to a graph URL via popstate
      window.history.pushState(null, "", "#/sess-1/graph");
      window.dispatchEvent(new PopStateEvent("popstate"));
      // Flush queueMicrotask
      await Promise.resolve();
    });

    expect(handler).toHaveBeenCalled();
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
  });

  it("popstate to an empty hash clears the active session", async () => {
    useSessionStore.setState({
      sessions: [{ id: "sess-1", title: "Session 1" } as never],
      activeSessionId: "sess-1",
      selectSession: vi.fn(),
    } as never);
    window.history.replaceState(null, "", "#/sess-1");
    renderHook(() => useHashRouter());

    await act(async () => {
      window.history.replaceState(null, "", window.location.pathname);
      window.dispatchEvent(new PopStateEvent("popstate"));
    });

    expect(window.location.hash).toBe("");
    expect(useSessionStore.getState().activeSessionId).toBeNull();
  });

  // ── Two rapid hashchanges ────────────────────────────────────────────────

  it("two rapid hashchanges both fire their respective modal events in order", async () => {
    window.history.replaceState(null, "", "#/sess-1");
    // The yaml verb is content-gated: give the active session a KNOWN,
    // non-empty composition so its dispatch fires (elspeth-bff8043d33).
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionStateLoaded: true,
      compositionState: nonEmptyCompositionState(),
    } as never);
    renderHook(() => useHashRouter());

    const graphHandler = vi.fn();
    const yamlHandler = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, graphHandler);
    window.addEventListener(OPEN_YAML_MODAL_EVENT, yamlHandler);

    await act(async () => {
      // Fire two hashchanges synchronously; both handlers are queued as microtasks
      window.history.replaceState(null, "", "#/sess-1/graph");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
      window.history.replaceState(null, "", "#/sess-1/yaml");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
      // Flush all queued microtasks
      await Promise.resolve();
      await Promise.resolve();
    });

    // Both events should have fired (each hashchange triggers its own applyHash
    // which queues its own microtask; the hook processes each event independently)
    expect(graphHandler).toHaveBeenCalled();
    expect(yamlHandler).toHaveBeenCalled();

    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, graphHandler);
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, yamlHandler);
  });
});
