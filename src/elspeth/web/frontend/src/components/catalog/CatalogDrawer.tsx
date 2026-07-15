// ============================================================================
// CatalogDrawer
//
// Slide-over drawer listing available plugins organised by tab
// (Sources, Transforms, Sinks). Opens from the right side of the inspector
// panel whose outermost container already carries position: relative.
//
// Features:
// - Fuzzy search across plugin names, descriptions, usage prose, and tags
// - Per-tab filter chips for capability tags and audit characteristics
// - Tab-based filtering by plugin type with counts
// - Principal/fingerprint-isolated list and schema caches
// - Sources tab pins an InlineChatSourceEntry as the first row — a
//   synthetic affordance that is unaffected by filters or search and is
//   always visible alongside the empty-state message when filters
//   eliminate every real plugin.
//
// Filter state is **per-tab** (one CatalogFilters record per CatalogTab),
// so an active capability filter on Sources does NOT silently hide every
// plugin on Transforms when the user switches tabs.
//
// Data is owned by the plugin catalog store; component lifetime is not a
// cache boundary.
// ============================================================================

import {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import type { PluginPolicyFinding, PluginSummary } from "@/types/index";
import { PluginCard, PREFILL_CHAT_INPUT_EVENT } from "./PluginCard";
import { FilterChipStrip, type CatalogFilters } from "./FilterChipStrip";
import { InlineChatSourceEntry } from "./InlineChatSourceEntry";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import {
  MIN_FUZZY_CONFIDENCE,
  confidenceFromScore,
  fuzzyMatch,
} from "@/utils/fuzzyScore";
import { pluginDisplayName } from "./pluginDisplayName";
import { useAuthStore } from "@/stores/authStore";
import { usePluginCatalogStore } from "@/stores/pluginCatalogStore";
import { useSessionStore } from "@/stores/sessionStore";

type CatalogTab = "sources" | "transforms" | "sinks";
const CATALOG_TABS: readonly CatalogTab[] = ["sources", "transforms", "sinks"];
const EMPTY_POLICY_FINDINGS: PluginPolicyFinding[] = [];

function catalogTabLabel(tab: CatalogTab): string {
  return tab === "sources" ? "Sources" : tab === "transforms" ? "Transforms" : "Sinks";
}

function catalogTabId(tab: CatalogTab): string {
  return `catalog-tab-${tab}`;
}

function catalogPanelId(tab: CatalogTab): string {
  return `catalog-panel-${tab}`;
}

/**
 * Score a plugin against the search query.
 *
 * Fuzzy-matches across name, description, usage prose (when_to_use,
 * when_not_to_use), and capability tags. Returns the raw fuzzy score
 * (lower is better, -1 for no match) when the normalized confidence
 * clears the noise floor, otherwise -1. Centralising this keeps the
 * list and the per-tab counts in lockstep — without it, the tab badges
 * and the visible results can disagree on what "matches" means.
 */
function scorePlugin(query: string, plugin: PluginSummary): number {
  const target = [
    plugin.name,
    // The card's primary label is the human display name — searching for
    // "blob storage" must find azure_blob (elspeth-5ee1f76e39).
    pluginDisplayName(plugin.name),
    plugin.description ?? "",
    plugin.usage_when_to_use ?? "",
    plugin.usage_when_not_to_use ?? "",
    plugin.capability_tags.join(" "),
  ].join(" ");
  const score = fuzzyMatch(query, target);
  if (score < 0) return -1;
  if (confidenceFromScore(score, target.length) < MIN_FUZZY_CONFIDENCE) {
    return -1;
  }
  return score;
}

/**
 * Predicate: does a plugin satisfy the active filter sets?
 *
 * AND composition across groups, OR composition within a group.
 * Empty group = no constraint (matches all).
 */
function matchesFilters(plugin: PluginSummary, filters: CatalogFilters): boolean {
  if (filters.capabilityTags.size > 0) {
    const has = plugin.capability_tags.some((t) => filters.capabilityTags.has(t));
    if (!has) return false;
  }
  if (filters.auditCharacteristics.size > 0) {
    const has = plugin.audit_characteristics.some((a) =>
      filters.auditCharacteristics.has(a),
    );
    if (!has) return false;
  }
  return true;
}

function hasActiveFilters(f: CatalogFilters): boolean {
  return f.capabilityTags.size > 0 || f.auditCharacteristics.size > 0;
}

function emptyFilters(): CatalogFilters {
  return {
    capabilityTags: new Set(),
    auditCharacteristics: new Set(),
  };
}

function emptyFiltersByTab(): Record<CatalogTab, CatalogFilters> {
  return {
    sources: emptyFilters(),
    transforms: emptyFilters(),
    sinks: emptyFilters(),
  };
}

function unavailableReasonLabel(reason: PluginPolicyFinding["reason_code"]): string {
  return {
    plugin_not_enabled: "Not enabled",
    plugin_not_installed: "Not installed",
    plugin_unavailable: "Unavailable",
    credential_unavailable: "Credential unavailable",
    profile_unavailable: "Profile unavailable",
  }[reason];
}

function repairTab(pluginId: string): CatalogTab {
  if (pluginId.startsWith("source:")) return "sources";
  if (pluginId.startsWith("sink:")) return "sinks";
  return "transforms";
}

interface CatalogDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CatalogDrawer({ isOpen, onClose }: CatalogDrawerProps) {
  const [activeTab, setActiveTab] = useState<CatalogTab>("sources");
  const principal = useAuthStore((state) => state.user?.user_id ?? null);
  const sources = usePluginCatalogStore((state) => state.sources);
  const transforms = usePluginCatalogStore((state) => state.transforms);
  const sinks = usePluginCatalogStore((state) => state.sinks);
  const schemaCache = usePluginCatalogStore((state) => state.schemas);
  const schemaErrors = usePluginCatalogStore((state) => state.schemaErrors);
  const fetchError = usePluginCatalogStore((state) => state.error);
  const isFetching = usePluginCatalogStore((state) => state.isLoading);
  const catalogFingerprint = usePluginCatalogStore((state) => state.fingerprint);
  const load = usePluginCatalogStore((state) => state.load);
  const loadSchema = usePluginCatalogStore((state) => state.loadSchema);
  const storedPolicyFindings = useSessionStore(
    (state) => state.compositionState?.plugin_policy_findings ?? EMPTY_POLICY_FINDINGS,
  );
  const policyFindings = useMemo(
    () =>
      catalogFingerprint === null
        ? EMPTY_POLICY_FINDINGS
        : storedPolicyFindings.filter(
            (finding) => finding.snapshot_fingerprint === catalogFingerprint,
          ),
    [catalogFingerprint, storedPolicyFindings],
  );
  const [searchQuery, setSearchQuery] = useState("");
  // Per-tab filter state — switching tabs reveals the user's filter set
  // for that tab. Avoids the "active capability filter on Sources hides
  // every Transform on tab switch" UX trap.
  const [filtersByTab, setFiltersByTab] = useState<
    Record<CatalogTab, CatalogFilters>
  >(emptyFiltersByTab);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const drawerRef = useRef<HTMLDivElement>(null);
  useFocusTrap(drawerRef, isOpen);

  const filters = filtersByTab[activeTab];
  const setFilters = useCallback(
    (next: CatalogFilters) =>
      setFiltersByTab((prev) => ({ ...prev, [activeTab]: next })),
    [activeTab],
  );

  const handleTabKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLButtonElement>, index: number) => {
      let nextIndex: number | null = null;
      if (event.key === "ArrowRight") {
        nextIndex = (index + 1) % CATALOG_TABS.length;
      } else if (event.key === "ArrowLeft") {
        nextIndex = (index - 1 + CATALOG_TABS.length) % CATALOG_TABS.length;
      } else if (event.key === "Home") {
        nextIndex = 0;
      } else if (event.key === "End") {
        nextIndex = CATALOG_TABS.length - 1;
      }

      if (nextIndex === null) return;
      event.preventDefault();
      const nextTab = CATALOG_TABS[nextIndex];
      setActiveTab(nextTab);
      event.currentTarget
        .parentElement
        ?.querySelector<HTMLButtonElement>(`#${catalogTabId(nextTab)}`)
        ?.focus();
    },
    [],
  );

  const loadCatalog = useCallback(() => {
    if (principal === null || isFetching) return;
    void load({ principal, force: fetchError !== null });
  }, [fetchError, isFetching, load, principal]);

  // Fetch all three lists in parallel on first open.
  useEffect(() => {
    if (
      !isOpen ||
      principal === null ||
      sources !== null ||
      isFetching ||
      fetchError !== null
    ) return;

    loadCatalog();
  }, [isOpen, principal, sources, isFetching, fetchError, loadCatalog]);

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
        return;
      }
      // Alt+1 / Alt+2 / Alt+3: Switch catalog tab while drawer is open.
      // Binding is drawer-scoped — the handler only runs when isOpen is true.
      // Does NOT dispatch a custom event; calls setActiveTab directly so
      // App.test.tsx's "does not dispatch retired inspector tab shortcuts on
      // Alt+digit" regression guard continues to pass unchanged.
      if (e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
        if (e.key === "1") { e.preventDefault(); setActiveTab("sources"); return; }
        if (e.key === "2") { e.preventDefault(); setActiveTab("transforms"); return; }
        if (e.key === "3") { e.preventDefault(); setActiveTab("sinks"); return; }
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  // Clear filters when drawer closes. Search persists across close/reopen so
  // operators can inspect a plugin, leave the drawer, then continue the same
  // catalog lookup without retyping.
  useEffect(() => {
    if (!isOpen) {
      setFiltersByTab(emptyFiltersByTab());
    }
  }, [isOpen]);

  const handleExpand = useCallback(
    (plugin: PluginSummary) => {
      void loadSchema(plugin.plugin_type, plugin.name);
    },
    [loadSchema],
  );

  const handleRemoveDisabled = useCallback(
    (finding: PluginPolicyFinding) => {
      window.dispatchEvent(
        new CustomEvent(PREFILL_CHAT_INPUT_EVENT, {
          detail: `Remove disabled component ${finding.component_id} (${finding.plugin_id}) from this pipeline.`,
        }),
      );
      onClose();
    },
    [onClose],
  );

  const handleReplaceDisabled = useCallback((finding: PluginPolicyFinding) => {
    setActiveTab(repairTab(finding.plugin_id));
    setSearchQuery("");
    queueMicrotask(() => searchInputRef.current?.focus());
  }, []);

  // Get plugins for current tab — used as the input for both the
  // filter-chip strip (which derives chip values from this set) and the
  // composed filter+search predicate.
  //
  // Memoized so downstream useMemo hooks (pluginList,
  // availableCapabilityTags, availableAuditCharacteristics) don't see a
  // freshly-built reference on every render — without memoization those
  // hooks tripped `react-hooks/exhaustive-deps` warnings (their dep array
  // depended on a conditional expression rebuilt each render).
  const allPluginsForTab = useMemo<PluginSummary[]>(
    () =>
      activeTab === "sources"
        ? (sources ?? [])
        : activeTab === "transforms"
          ? (transforms ?? [])
          : (sinks ?? []),
    [activeTab, sources, transforms, sinks],
  );

  // Composed list: filter first, then search. AND composition between
  // filter groups and the search predicate; OR composition within a group.
  const pluginList = useMemo(() => {
    const query = searchQuery.trim();
    const filtered = allPluginsForTab.filter((p) => matchesFilters(p, filters));
    if (!query) return filtered;
    return filtered
      .map((plugin) => ({ plugin, score: scorePlugin(query, plugin) }))
      .filter((item) => item.score >= 0)
      .sort((a, b) => a.score - b.score)
      .map((item) => item.plugin);
  }, [allPluginsForTab, searchQuery, filters]);

  // Derive available chip values from the loaded plugins on the visible
  // tab. Chips for tags/characteristics that aren't actually present in
  // the catalog would be dead-ends — show only what's available.
  const availableCapabilityTags = useMemo(() => {
    const set = new Set<string>();
    for (const p of allPluginsForTab) {
      for (const t of p.capability_tags) set.add(t);
    }
    return [...set].sort();
  }, [allPluginsForTab]);

  const availableAuditCharacteristics = useMemo(() => {
    const set = new Set<string>();
    for (const p of allPluginsForTab) {
      for (const a of p.audit_characteristics) set.add(a);
    }
    return [...set].sort();
  }, [allPluginsForTab]);

  // Per-tab counts apply each tab's own filters AND the global search.
  const counts = useMemo(() => {
    const query = searchQuery.trim();
    const passes = (p: PluginSummary, tab: CatalogTab) =>
      matchesFilters(p, filtersByTab[tab]) &&
      (query ? scorePlugin(query, p) >= 0 : true);
    return {
      sources: (sources ?? []).filter((p) => passes(p, "sources")).length,
      transforms: (transforms ?? []).filter((p) => passes(p, "transforms")).length,
      sinks: (sinks ?? []).filter((p) => passes(p, "sinks")).length,
    };
  }, [sources, transforms, sinks, searchQuery, filtersByTab]);

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
      <div
        ref={drawerRef}
        className="catalog-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="catalog-drawer-title"
      >
        {/* Header */}
        <div className="catalog-header">
          <div className="catalog-header-copy">
            <span className="catalog-header-eyebrow">Reference</span>
            <span id="catalog-drawer-title" className="catalog-header-title">
              Plugin Catalog
            </span>
            <span className="catalog-header-subtitle">
              Browse available sources, transforms, and sinks before asking the
              composer to apply them.
            </span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close plugin catalog"
            className="btn catalog-close-btn"
          >
            ×
          </button>
        </div>

        {policyFindings.length > 0 && (
          <section
            role="region"
            aria-labelledby="catalog-disabled-components-title"
            className="validation-banner validation-banner-fail"
          >
            <div
              id="catalog-disabled-components-title"
              className="validation-banner-fail-title"
            >
              Unavailable saved components
            </div>
            <p>
              These historical components remain visible, but must be removed
              or replaced before the pipeline can run.
            </p>
            <ul className="validation-banner-fail-list">
              {policyFindings.map((finding) => (
                <li
                  key={`${finding.component_id}:${finding.plugin_id}`}
                  className="validation-banner-error-item"
                >
                  <div>
                    <strong>{finding.component_id}</strong>{" "}
                    <code>{finding.plugin_id}</code> —{" "}
                    {unavailableReasonLabel(finding.reason_code)}
                  </div>
                  <div className="import-yaml-actions">
                    <button
                      type="button"
                      className="btn btn-small"
                      aria-label={`Remove disabled component ${finding.component_id} (${finding.plugin_id})`}
                      onClick={() => handleRemoveDisabled(finding)}
                    >
                      Remove
                    </button>
                    <button
                      type="button"
                      className="btn btn-small"
                      aria-label={`Replace disabled component ${finding.component_id} (${finding.plugin_id}) with an available ${repairTab(finding.plugin_id).slice(0, -1)}`}
                      onClick={() => handleReplaceDisabled(finding)}
                    >
                      Replace
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}

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

        {/* Filter chip strip — between search and tab strip per plan. */}
        <FilterChipStrip
          availableCapabilityTags={availableCapabilityTags}
          availableAuditCharacteristics={availableAuditCharacteristics}
          filters={filters}
          onChange={setFilters}
        />

        {/* Tab strip with counts */}
        <div
          role="tablist"
          aria-label="Plugin type tabs"
          className="catalog-tab-strip"
        >
          {CATALOG_TABS.map((tab, index) => {
            const label = catalogTabLabel(tab);
            const count = counts[tab];
            const isActive = activeTab === tab;
            return (
              <button
                key={tab}
                role="tab"
                id={catalogTabId(tab)}
                aria-controls={catalogPanelId(tab)}
                aria-label={label}
                aria-selected={isActive}
                tabIndex={isActive ? 0 : -1}
                onKeyDown={(event) => handleTabKeyDown(event, index)}
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

        {/* Scrollable plugin list.
            The InlineChatSourceEntry is a pinned affordance: it sits OUTSIDE
            the fetchError → loading → empty → list conditional ladder so it
            remains visible even when filters eliminate every real plugin.
            The empty-state message applies to the plugin list, not the
            Sources tab as a whole — the synthetic entry stays usable. */}
        <div
          id={catalogPanelId(activeTab)}
          role="tabpanel"
          aria-labelledby={catalogTabId(activeTab)}
          className="catalog-list"
        >
          {activeTab === "sources" && (
            <InlineChatSourceEntry onCloseDrawer={onClose} />
          )}

          {fetchError !== null ? (
            // role="alert" so the load failure is announced assertively the
            // moment it appears (WCAG 4.1.3) — the loading branch below is
            // polite, but an error the user did not initiate is urgent.
            <div
              role="alert"
              className="catalog-status-message catalog-status-message--error"
            >
              <span>Failed to load plugin catalog.</span>
              <button
                type="button"
                className="btn btn-small"
                onClick={loadCatalog}
                aria-label="Retry loading plugin catalog"
              >
                Retry
              </button>
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
            // Polite live region: a screen-reader user hears the empty state
            // when search/filter eliminates every plugin (WCAG 4.1.3).
            <div
              role="status"
              aria-live="polite"
              className="catalog-status-message catalog-status-message--center"
            >
              {hasActiveFilters(filters)
                ? "No plugins match the active filters."
                : "No plugins available."}
            </div>
          ) : (
            <>
              {/* Polite live region announcing the result COUNT so the
                  number of plugins surviving the search/filter pass is
                  spoken (WCAG 4.1.3). Sits OUTSIDE the role="list" below so
                  the list's only direct children are role="listitem". */}
              <div role="status" aria-live="polite" className="sr-only">
                {`${pluginList.length} ${pluginList.length === 1 ? "plugin" : "plugins"}`}
              </div>
              {/* List semantics (WCAG 1.3.1): each plugin card is a listitem
                  so assistive tech announces "list, N items" and per-item
                  position instead of a flat run of unrelated regions. */}
              <div role="list" className="catalog-plugin-list">
                {pluginList.map((plugin) => {
                  const cacheKey = `${plugin.plugin_type}:${plugin.name}`;
                  const schema = schemaCache[cacheKey] ?? null;
                  const hasSchemaError = schemaErrors[cacheKey] === true;
                  return (
                    <div role="listitem" key={cacheKey}>
                      <PluginCard
                        plugin={plugin}
                        schema={schema}
                        schemaError={hasSchemaError}
                        onExpand={() => handleExpand(plugin)}
                        onRetrySchema={() => handleExpand(plugin)}
                      />
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
