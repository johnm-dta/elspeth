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
 * The principal/fingerprint-scoped plugin catalog store owns the list and
 * schema caches. This hook consumes only that store, so auth or policy changes
 * cannot leave a module-global catalog from another principal in memory.
 *
 * Per plan 19b §"Scope boundaries": the opt-in is binary at the plugin
 * level. There is no per-plugin tuning of narrative mode. If any
 * pipeline plugin opts in, the entire result view switches.
 */

import { useEffect, useMemo } from "react";

import { useAuthStore } from "@/stores/authStore";
import { usePluginCatalogStore } from "@/stores/pluginCatalogStore";
import { useSessionStore } from "@/stores/sessionStore";
import { sortedSourceEntries } from "@/utils/compositionState";

const NARRATIVE_SUMMARY_TAG = "narrative-summary";

interface CatalogTagMap {
  /** plugin_name → capability_tags (each tag is a string). */
  [pluginName: string]: readonly string[];
}

/** Compatibility test helper retained for existing hook tests. */
export function _resetNarrativeModeCacheForTesting(): void {
  usePluginCatalogStore.getState().clear();
}

export interface UseNarrativeModeResult {
  /** True if any plugin in the current composition has the narrative tag. */
  narrativeMode: boolean;
  /** True while the catalog is still being fetched. False once cached. */
  isLoading: boolean;
}

export function useNarrativeMode(): UseNarrativeModeResult {
  const compositionState = useSessionStore((s) => s.compositionState);
  const principal = useAuthStore((s) => s.user?.user_id ?? null);
  const sources = usePluginCatalogStore((s) => s.sources);
  const transforms = usePluginCatalogStore((s) => s.transforms);
  const sinks = usePluginCatalogStore((s) => s.sinks);
  const storeLoading = usePluginCatalogStore((s) => s.isLoading);
  const catalogError = usePluginCatalogStore((s) => s.error);
  const loadCatalog = usePluginCatalogStore((s) => s.load);

  useEffect(() => {
    if (principal !== null) void loadCatalog({ principal });
  }, [loadCatalog, principal]);

  const catalog = useMemo<CatalogTagMap | null>(() => {
    if (sources === null || transforms === null || sinks === null) return null;
    const tags: CatalogTagMap = {};
    for (const plugin of [...sources, ...transforms, ...sinks]) {
      tags[plugin.name] = plugin.capability_tags ?? [];
    }
    return tags;
  }, [sources, transforms, sinks]);

  const isLoading =
    principal !== null && (storeLoading || (catalog === null && catalogError === null));

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
