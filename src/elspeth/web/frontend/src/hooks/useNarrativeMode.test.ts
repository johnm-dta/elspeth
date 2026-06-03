import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import {
  useNarrativeMode,
  _resetNarrativeModeCacheForTesting,
} from "./useNarrativeMode";
import { useSessionStore } from "@/stores/sessionStore";
import * as apiClient from "@/api/client";

const TAGGED = { name: "batch_classifier_metrics", capability_tags: ["narrative-summary"] } as any;
const UNTAGGED = { name: "passthrough", capability_tags: [] } as any;

function _stubCatalog(transforms: any[], sources: any[] = [], sinks: any[] = []) {
  vi.spyOn(apiClient, "listTransforms").mockResolvedValue(transforms);
  vi.spyOn(apiClient, "listSources").mockResolvedValue(sources);
  vi.spyOn(apiClient, "listSinks").mockResolvedValue(sinks);
}

function _setComposition(state: any) {
  useSessionStore.setState({ compositionState: state } as never);
}

describe("useNarrativeMode", () => {
  beforeEach(() => {
    _resetNarrativeModeCacheForTesting();
    vi.restoreAllMocks();
    useSessionStore.setState({ compositionState: null } as never);
  });

  it("returns narrativeMode=false when no composition state exists", async () => {
    _stubCatalog([TAGGED]);
    const { result } = renderHook(() => useNarrativeMode());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.narrativeMode).toBe(false);
  });

  it("returns true when a composition node's plugin has the narrative-summary tag", async () => {
    _stubCatalog([TAGGED]);
    _setComposition({
      source: null,
      nodes: [
        { id: "n1", node_type: "transform", plugin: "batch_classifier_metrics", input: "src", on_success: null, on_error: null, options: {} },
      ],
      edges: [],
      outputs: [],
      metadata: { name: "demo", description: "" },
      version: 1,
    });
    const { result } = renderHook(() => useNarrativeMode());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.narrativeMode).toBe(true);
  });

  it("returns false when only untagged plugins are in the composition", async () => {
    _stubCatalog([UNTAGGED, TAGGED]);
    _setComposition({
      source: null,
      nodes: [
        { id: "n1", node_type: "transform", plugin: "passthrough", input: "src", on_success: null, on_error: null, options: {} },
      ],
      edges: [],
      outputs: [],
      metadata: { name: "demo", description: "" },
      version: 1,
    });
    const { result } = renderHook(() => useNarrativeMode());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.narrativeMode).toBe(false);
  });

  it("returns true when a SOURCE plugin has the narrative-summary tag", async () => {
    _stubCatalog([], [TAGGED]);
    _setComposition({
      source: { plugin: "batch_classifier_metrics", options: {} },
      nodes: [],
      edges: [],
      outputs: [],
      metadata: { name: "demo", description: "" },
      version: 1,
    });
    const { result } = renderHook(() => useNarrativeMode());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.narrativeMode).toBe(true);
  });

  it("returns true when a SINK plugin has the narrative-summary tag", async () => {
    _stubCatalog([], [], [TAGGED]);
    _setComposition({
      source: null,
      nodes: [],
      edges: [],
      outputs: [
        { name: "out", plugin: "batch_classifier_metrics", options: {} },
      ],
      metadata: { name: "demo", description: "" },
      version: 1,
    });
    const { result } = renderHook(() => useNarrativeMode());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.narrativeMode).toBe(true);
  });

  it("falls back to narrativeMode=false when the catalog fetch fails", async () => {
    vi.spyOn(apiClient, "listTransforms").mockRejectedValue(new Error("network"));
    vi.spyOn(apiClient, "listSources").mockResolvedValue([]);
    vi.spyOn(apiClient, "listSinks").mockResolvedValue([]);
    _setComposition({
      source: null,
      nodes: [{ id: "n1", node_type: "transform", plugin: "anything", input: "src", on_success: null, on_error: null, options: {} }],
      edges: [],
      outputs: [],
      metadata: { name: "demo", description: "" },
      version: 1,
    });
    const { result } = renderHook(() => useNarrativeMode());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.narrativeMode).toBe(false);
  });

  it("matches the tag string exact-case only", async () => {
    // Plan §Task 5 test case 7: the narrative-summary opt-in is an
    // exact-case string match against capability_tags. Neither the
    // plural ("narrative-summaries") nor a mixed-case variant
    // ("Narrative-Summary") should activate narrative mode — only
    // the canonical "narrative-summary" tag does. This pins the
    // contract: future maintainers who try to be "helpful" with a
    // case-insensitive or fuzzy match break the open-vocabulary
    // capability_tags discipline (ADR-022 D8).
    _stubCatalog([
      { name: "p1", capability_tags: ["narrative-summaries"] } as never,
      { name: "p2", capability_tags: ["Narrative-Summary"] } as never,
    ]);
    _setComposition({
      source: null,
      nodes: [
        { id: "n1", node_type: "transform", plugin: "p1", input: "src", on_success: null, on_error: null, options: {} },
        { id: "n2", node_type: "transform", plugin: "p2", input: "n1", on_success: null, on_error: null, options: {} },
      ],
      edges: [],
      outputs: [],
      metadata: { name: "demo", description: "" },
      version: 1,
    });
    const { result } = renderHook(() => useNarrativeMode());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.narrativeMode).toBe(false);
  });

  it("caches the catalog across renders (does not re-fetch)", async () => {
    const transformsSpy = vi.spyOn(apiClient, "listTransforms").mockResolvedValue([TAGGED]);
    vi.spyOn(apiClient, "listSources").mockResolvedValue([]);
    vi.spyOn(apiClient, "listSinks").mockResolvedValue([]);
    const { result: r1, rerender } = renderHook(() => useNarrativeMode());
    await waitFor(() => expect(r1.current.isLoading).toBe(false));
    expect(transformsSpy).toHaveBeenCalledTimes(1);
    rerender();
    expect(transformsSpy).toHaveBeenCalledTimes(1); // still 1
  });
});
