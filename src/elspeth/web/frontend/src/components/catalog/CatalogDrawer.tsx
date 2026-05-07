// ============================================================================
// CatalogDrawer
//
// Slide-over drawer listing available plugins organised by tab
// (Sources, Transforms, Sinks). Opens from the right side of the inspector
// panel whose outermost container already carries position: relative.
//
// Features:
// - Fuzzy search across plugin names and descriptions
// - Tab-based filtering by plugin type with counts
// - Schema details fetched on demand and cached
//
// Data is fetched once on first open and cached in component state for the
// lifetime of the drawer instance.
// ============================================================================

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import {
  listSources,
  listTransforms,
  listSinks,
  getPluginSchema,
} from "@/api/client";
import type { PluginSummary, PluginSchemaInfo } from "@/types/index";
import { PluginCard } from "./PluginCard";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import {
  MIN_FUZZY_CONFIDENCE,
  confidenceFromScore,
  fuzzyMatch,
} from "@/utils/fuzzyScore";

type CatalogTab = "sources" | "transforms" | "sinks";

/**
 * Score a plugin's name+description against the search query.
 *
 * Returns the raw fuzzy score (lower is better, -1 for no match) when the
 * normalized confidence clears the noise floor, otherwise -1. Centralising
 * this keeps the list and the per-tab counts in lockstep — without it, the
 * tab badges and the visible results can disagree on what "matches" means.
 */
function scorePlugin(query: string, plugin: PluginSummary): number {
  const target = `${plugin.name} ${plugin.description ?? ""}`;
  const score = fuzzyMatch(query, target);
  if (score < 0) return -1;
  if (confidenceFromScore(score, target.length) < MIN_FUZZY_CONFIDENCE) {
    return -1;
  }
  return score;
}

interface CatalogDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CatalogDrawer({ isOpen, onClose }: CatalogDrawerProps) {
  const [activeTab, setActiveTab] = useState<CatalogTab>("sources");
  const [sources, setSources] = useState<PluginSummary[] | null>(null);
  const [transforms, setTransforms] = useState<PluginSummary[] | null>(null);
  const [sinks, setSinks] = useState<PluginSummary[] | null>(null);
  const [schemaCache, setSchemaCache] = useState<Map<string, PluginSchemaInfo>>(
    new Map(),
  );
  const [loadingSchemas, setLoadingSchemas] = useState<Set<string>>(new Set());
  const [schemaErrors, setSchemaErrors] = useState<Set<string>>(new Set());
  const [fetchError, setFetchError] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);
  const drawerRef = useRef<HTMLDivElement>(null);
  useFocusTrap(drawerRef, isOpen);

  // Fetch all three lists in parallel on first open.
  // On failure: set fetchError, don't retry until drawer is closed and reopened.
  useEffect(() => {
    if (!isOpen || sources !== null || isFetching || fetchError) return;

    setIsFetching(true);
    Promise.all([listSources(), listTransforms(), listSinks()])
      .then(([src, xfm, snk]) => {
        setSources(src);
        setTransforms(xfm);
        setSinks(snk);
      })
      .catch(() => {
        setFetchError(true);
      })
      .finally(() => {
        setIsFetching(false);
      });
  }, [isOpen, sources, isFetching, fetchError]);

  // Clear error on close so next open retries
  useEffect(() => {
    if (!isOpen && fetchError) {
      setFetchError(false);
    }
  }, [isOpen, fetchError]);

  // Keyboard: Escape closes, / focuses search
  useEffect(() => {
    if (!isOpen) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      // "/" focuses search unless already in an editable element
      const active = document.activeElement;
      const isEditable =
        active?.tagName === "INPUT" ||
        active?.tagName === "TEXTAREA" ||
        (active as HTMLElement)?.isContentEditable;
      if (e.key === "/" && !isEditable) {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  // Clear search when drawer closes
  useEffect(() => {
    if (!isOpen) {
      setSearchQuery("");
    }
  }, [isOpen]);

  const handleExpand = useCallback(
    (plugin: PluginSummary) => {
      const cacheKey = `${plugin.plugin_type}:${plugin.name}`;
      if (schemaCache.has(cacheKey) || loadingSchemas.has(cacheKey)) return;
      // Clear previous error on retry
      setSchemaErrors((prev) => {
        if (!prev.has(cacheKey)) return prev;
        const next = new Set(prev);
        next.delete(cacheKey);
        return next;
      });

      setLoadingSchemas((prev) => new Set(prev).add(cacheKey));

      getPluginSchema(plugin.plugin_type, plugin.name)
        .then((info) => {
          setSchemaCache((prev) => {
            const next = new Map(prev);
            next.set(cacheKey, info);
            return next;
          });
          setSchemaErrors((prev) => {
            if (!prev.has(cacheKey)) return prev;
            const next = new Set(prev);
            next.delete(cacheKey);
            return next;
          });
        })
        .catch(() => {
          setSchemaErrors((prev) => new Set(prev).add(cacheKey));
        })
        .finally(() => {
          setLoadingSchemas((prev) => {
            const next = new Set(prev);
            next.delete(cacheKey);
            return next;
          });
        });
    },
    [schemaCache, loadingSchemas, schemaErrors],
  );

  // Get plugins for current tab and apply search filter
  const allPluginsForTab: PluginSummary[] =
    activeTab === "sources"
      ? (sources ?? [])
      : activeTab === "transforms"
        ? (transforms ?? [])
        : (sinks ?? []);

  const pluginList = useMemo(() => {
    const query = searchQuery.trim();
    if (!query) {
      return allPluginsForTab;
    }
    // Score, drop noise, then sort best-match first (ascending = lower score
    // = closer match). Stable sort keeps original tab order for ties.
    return allPluginsForTab
      .map((plugin) => ({ plugin, score: scorePlugin(query, plugin) }))
      .filter((item) => item.score >= 0)
      .sort((a, b) => a.score - b.score)
      .map((item) => item.plugin);
  }, [allPluginsForTab, searchQuery]);

  // Counts for tab badges (filtered counts when searching). Uses the same
  // scored predicate so the per-tab badge totals match the rendered list.
  const counts = useMemo(() => {
    const query = searchQuery.trim();
    const filterFn = query
      ? (p: PluginSummary) => scorePlugin(query, p) >= 0
      : () => true;

    return {
      sources: (sources ?? []).filter(filterFn).length,
      transforms: (transforms ?? []).filter(filterFn).length,
      sinks: (sinks ?? []).filter(filterFn).length,
    };
  }, [sources, transforms, sinks, searchQuery]);

  const isLoading =
    (activeTab === "sources" && sources === null) ||
    (activeTab === "transforms" && transforms === null) ||
    (activeTab === "sinks" && sinks === null);

  if (!isOpen) return <div style={{ display: "none" }} />;

  return (
    <>
      {/* Backdrop */}
      <div
        data-testid="catalog-backdrop"
        className="catalog-backdrop"
        onClick={onClose}
      />

      {/* Drawer panel */}
      <div ref={drawerRef} className="catalog-drawer">
        {/* Header */}
        <div className="catalog-header">
          <span className="catalog-header-title">Plugin Catalog</span>
          <button
            onClick={onClose}
            aria-label="Close plugin catalog"
            className="btn catalog-close-btn"
          >
            ×
          </button>
        </div>

        {/* Search input */}
        <div className="catalog-search-wrapper">
          <div className="catalog-search-container">
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search plugins... (press /)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              aria-label="Search plugins"
              className="catalog-search-input"
            />
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery("");
                  searchInputRef.current?.focus();
                }}
                aria-label="Clear search"
                className="catalog-search-clear"
              >
                ×
              </button>
            )}
          </div>
        </div>

        {/* Tab strip with counts */}
        <div
          role="tablist"
          aria-label="Plugin type tabs"
          className="catalog-tab-strip"
        >
          {(["sources", "transforms", "sinks"] as CatalogTab[]).map((tab) => {
            const label =
              tab === "sources"
                ? "Sources"
                : tab === "transforms"
                  ? "Transforms"
                  : "Sinks";
            const count = counts[tab];
            const isActive = activeTab === tab;
            return (
              <button
                key={tab}
                role="tab"
                aria-selected={isActive}
                onClick={() => setActiveTab(tab)}
                className={`tab-strip-tab catalog-tab ${isActive ? "tab-strip-tab-active" : ""}`}
              >
                {label}
                {sources !== null && (
                  <span
                    className={`catalog-tab-count ${isActive ? "catalog-tab-count--active" : "catalog-tab-count--inactive"}`}
                  >
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Scrollable plugin list */}
        <div className="catalog-list">
          {fetchError ? (
            <div className="catalog-status-message catalog-status-message--error">
              Failed to load plugin catalog. Close and reopen to retry.
            </div>
          ) : isLoading || isFetching ? (
            <div
              role="status"
              aria-live="polite"
              className="catalog-status-message"
            >
              Loading...
            </div>
          ) : pluginList.length === 0 ? (
            <div className="catalog-status-message catalog-status-message--center">
              {searchQuery.trim()
                ? `No plugins matching "${searchQuery}"`
                : "No plugins available."}
            </div>
          ) : (
            pluginList.map((plugin) => {
              const cacheKey = `${plugin.plugin_type}:${plugin.name}`;
              const schema = schemaCache.get(cacheKey) ?? null;
              const hasSchemaError = schemaErrors.has(cacheKey);
              return (
                <PluginCard
                  key={cacheKey}
                  plugin={plugin}
                  schema={schema}
                  schemaError={hasSchemaError}
                  onExpand={() => handleExpand(plugin)}
                />
              );
            })
          )}
        </div>
      </div>
    </>
  );
}
