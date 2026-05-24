/**
 * useNarrativeMode — Phase 6B Task 5.
 *
 * Returns true when at least one of the current composition's transforms
 * (or any pipeline plugin) carries the ``"narrative-summary"`` capability
 * tag. The frontend's results view branches on this hook to render
 * `NarrativeResults` (Task 6) vs. the existing tabular preview.
 *
 * Reads from:
 *
 * * `useSessionStore.compositionState` — for the per-node `plugin` ids.
 * * The catalog (lists of transforms/sources/sinks fetched on app boot)
 *   — for each plugin's `capability_tags`.
 *
 * The catalog is loaded lazily via the existing `listTransforms` /
 * `listSources` / `listSinks` API calls. The hook caches the catalog
 * lookup on the first call and reuses it for subsequent renders; it
 * does NOT subscribe to a Zustand store because no catalog store
 * exists pre-Phase-6 (the catalog is rendered ad-hoc by CatalogDrawer
 * via local component state). Phase 6B introduces a minimal singleton
 * cache inside this hook rather than adding a new store.
 *
 * Per plan 19b §"Scope boundaries": the opt-in is binary at the plugin
 * level. There is no per-plugin tuning of narrative mode. If any
 * pipeline plugin opts in, the entire result view switches.
 */

import { useEffect, useMemo, useState } from "react";

import { listSinks, listSources, listTransforms } from "@/api/client";
import { useSessionStore } from "@/stores/sessionStore";
import { sortedSourceEntries } from "@/utils/compositionState";
import type { PluginSummary } from "@/types/api";

const NARRATIVE_SUMMARY_TAG = "narrative-summary";

interface CatalogTagMap {
  /** plugin_name → capability_tags (each tag is a string). */
  [pluginName: string]: readonly string[];
}

let _cachedCatalog: CatalogTagMap | null = null;
let _cachedCatalogPromise: Promise<CatalogTagMap> | null = null;

async function _loadCatalog(): Promise<CatalogTagMap> {
  if (_cachedCatalog !== null) return _cachedCatalog;
  if (_cachedCatalogPromise !== null) return _cachedCatalogPromise;
  _cachedCatalogPromise = (async () => {
    const [sources, transforms, sinks] = await Promise.all([
      listSources(),
      listTransforms(),
      listSinks(),
    ]);
    const out: CatalogTagMap = {};
    for (const plugin of [...sources, ...transforms, ...sinks] as PluginSummary[]) {
      // PluginSummary.capability_tags is typed as `string[]`; default to []
      // for plugins shipped before Phase 6 where the wire field was added.
      out[plugin.name] = plugin.capability_tags ?? [];
    }
    _cachedCatalog = out;
    return out;
  })();
  return _cachedCatalogPromise;
}

/** Test helper: drops the catalog cache so unit tests can stub
 *  listTransforms etc. with fresh data per test. */
export function _resetNarrativeModeCacheForTesting(): void {
  _cachedCatalog = null;
  _cachedCatalogPromise = null;
}

export interface UseNarrativeModeResult {
  /** True if any plugin in the current composition has the narrative tag. */
  narrativeMode: boolean;
  /** True while the catalog is still being fetched. False once cached. */
  isLoading: boolean;
}

export function useNarrativeMode(): UseNarrativeModeResult {
  const compositionState = useSessionStore((s) => s.compositionState);
  const [catalog, setCatalog] = useState<CatalogTagMap | null>(_cachedCatalog);
  const [isLoading, setIsLoading] = useState<boolean>(_cachedCatalog === null);

  useEffect(() => {
    if (_cachedCatalog !== null) {
      setCatalog(_cachedCatalog);
      setIsLoading(false);
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    _loadCatalog()
      .then((loaded) => {
        if (cancelled) return;
        setCatalog(loaded);
        setIsLoading(false);
      })
      .catch(() => {
        // Catalog fetch failed — fall back to non-narrative mode. The
        // results view's default rendering still works without the
        // catalog; only the narrative branch is gated.
        if (cancelled) return;
        setCatalog({});
        setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const narrativeMode = useMemo(() => {
    if (catalog === null || compositionState === null) return false;
    const plugins: string[] = [];
    for (const [, source] of sortedSourceEntries(compositionState)) {
      plugins.push(source.plugin);
    }
    for (const node of compositionState.nodes) {
      if (node.plugin !== null) plugins.push(node.plugin);
    }
    for (const output of compositionState.outputs) {
      plugins.push(output.plugin);
    }
    return plugins.some((plugin) => {
      const tags = catalog[plugin] ?? [];
      return tags.includes(NARRATIVE_SUMMARY_TAG);
    });
  }, [catalog, compositionState]);

  return { narrativeMode, isLoading };
}
