// ============================================================================
// InspectorPanel
//
// Right panel with two-row header and tab-driven content area.
//
// Row 1: VersionSelector (custom dropdown with inline revert) + ValidationDot
//         on the left; Validate + Execute buttons on the right.
// Row 2: Tab strip (Spec, Graph, YAML, Runs) navigable by arrow keys.
//
// Validation result banner renders between header and tab content.
// Tab content area is wrapped in an ARIA live region.
// ============================================================================

import { useState, useCallback, useEffect, useRef } from "react";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { SWITCH_TAB_EVENT } from "@/components/common/CommandPalette";
import { TAB_CHANGED_EVENT } from "@/hooks/useHashRouter";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { SpecView } from "./SpecView";
import { GraphView } from "./GraphView";
import { YamlView } from "./YamlView";
import { RunsView } from "./RunsView";
import { ValidationResultBanner } from "@/components/execution/ValidationResult";
import { CatalogDrawer } from "@/components/catalog/CatalogDrawer";
import type { CompositionStateVersion } from "@/types/index";
import { relativeTime } from "@/utils/time";

type TabId = "spec" | "graph" | "yaml" | "runs";

const TABS: { id: TabId; label: string }[] = [
  { id: "spec", label: "Spec" },
  { id: "graph", label: "Graph" },
  { id: "yaml", label: "YAML" },
  { id: "runs", label: "Runs" },
];

// ---------------------------------------------------------------------------
// VersionSelector — custom dropdown with inline revert
// ---------------------------------------------------------------------------

interface VersionSelectorProps {
  currentVersion: number;
  stateVersions: CompositionStateVersion[];
  isLoadingVersions: boolean;
  onOpen: () => void;
  onRevert: (stateId: string, version: number) => void;
}

function VersionSelector({
  currentVersion,
  stateVersions,
  isLoadingVersions,
  onOpen,
  onRevert,
}: VersionSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const [revertTarget, setRevertTarget] = useState<CompositionStateVersion | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  // Build the ordered list: current version first, then others descending.
  // If the current version isn't in the fetched page, synthesize an entry
  // so it always appears as the anchor in the dropdown.
  const sortedVersions: CompositionStateVersion[] = [];
  const currentEntry = stateVersions.find(
    (v) => v.version === currentVersion,
  );
  if (currentEntry) {
    sortedVersions.push(currentEntry);
  } else {
    sortedVersions.push({
      id: "",
      version: currentVersion,
      created_at: new Date().toISOString(),
      node_count: 0,
    });
  }
  stateVersions
    .filter((v) => v.version !== currentVersion)
    .sort((a, b) => b.version - a.version)
    .forEach((v) => sortedVersions.push(v));

  const toggle = useCallback(() => {
    setIsOpen((prev) => {
      const next = !prev;
      if (next) {
        onOpen();
        setFocusedIndex(-1);
      }
      return next;
    });
  }, [onOpen]);

  const close = useCallback(() => {
    setIsOpen(false);
    setFocusedIndex(-1);
    triggerRef.current?.focus();
  }, []);

  // Click-outside close
  useEffect(() => {
    if (!isOpen) return;
    function handleMouseDown(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
        setFocusedIndex(-1);
      }
    }
    document.addEventListener("mousedown", handleMouseDown);
    return () => document.removeEventListener("mousedown", handleMouseDown);
  }, [isOpen]);

  // Focus the listbox when dropdown opens so keyboard navigation works
  useEffect(() => {
    if (isOpen) {
      listRef.current?.focus();
    }
  }, [isOpen]);

  // Scroll focused item into view
  useEffect(() => {
    if (!isOpen || focusedIndex < 0) return;
    const items = listRef.current?.querySelectorAll("[role='option']");
    items?.[focusedIndex]?.scrollIntoView({ block: "nearest" });
  }, [isOpen, focusedIndex]);

  function handleTriggerKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (!isOpen) {
        toggle();
      }
    }
  }

  function handleListKeyDown(e: React.KeyboardEvent) {
    const count = sortedVersions.length;
    if (e.key === "Escape") {
      e.preventDefault();
      close();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (count > 0) setFocusedIndex((prev) => (prev + 1) % count);
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (count > 0) setFocusedIndex((prev) => (prev - 1 + count) % count);
      return;
    }
    if (e.key === "Enter" && focusedIndex >= 0) {
      e.preventDefault();
      const entry = sortedVersions[focusedIndex];
      if (entry && entry.version !== currentVersion) {
        handleRevert(entry);
      }
    }
  }

  function handleRevert(entry: CompositionStateVersion) {
    setRevertTarget(entry);
  }

  function confirmRevert() {
    if (revertTarget) {
      onRevert(revertTarget.id, revertTarget.version);
      setRevertTarget(null);
      close();
    }
  }

  return (
    <div ref={containerRef} className="version-selector">
      <button
        ref={triggerRef}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-label={`Version ${currentVersion}`}
        onClick={toggle}
        onKeyDown={handleTriggerKeyDown}
        className="btn version-selector-trigger"
      >
        v{currentVersion} ▾
      </button>

      {isOpen && (
        <div className="version-selector-dropdown">
          <ul
            ref={listRef}
            role="listbox"
            aria-label="Version history"
            aria-activedescendant={focusedIndex >= 0 && sortedVersions[focusedIndex] ? `version-option-${sortedVersions[focusedIndex].version}` : undefined}
            onKeyDown={handleListKeyDown}
            tabIndex={0}
            className="version-selector-list"
          >
            {isLoadingVersions && sortedVersions.length === 0 && (
              <li className="version-selector-loading">
                Loading versions...
              </li>
            )}
            {sortedVersions.map((v, i) => {
              const isCurrent = v.version === currentVersion;
              const isFocused = focusedIndex === i;
              return (
                <li
                  key={v.version}
                  id={`version-option-${v.version}`}
                  role="option"
                  aria-selected={isCurrent}
                  aria-label={`Version ${v.version}${isCurrent ? " (current)" : ""}`}
                  className={`version-selector-item${isFocused ? " version-selector-item--focused" : ""}${isCurrent ? " version-selector-item--current" : ""}`}
                  onMouseEnter={() => setFocusedIndex(i)}
                >
                  <span className="version-selector-item-info">
                    <span className="version-selector-item-label">
                      v{v.version}
                      {isCurrent && (
                        <span className="version-selector-item-tag">
                          (current)
                        </span>
                      )}
                    </span>
                    <span className="version-selector-item-meta">
                      {v.node_count} nodes
                    </span>
                    <span className="version-selector-item-meta">
                      {relativeTime(v.created_at)}
                    </span>
                  </span>
                  {!isCurrent && (
                    <button
                      aria-label={`Revert to version ${v.version}`}
                      className="btn version-selector-revert-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRevert(v);
                      }}
                    >
                      Revert
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {revertTarget && (
        <ConfirmDialog
          title="Revert pipeline"
          message={`Revert pipeline to version ${revertTarget.version}? This will replace the current composition.`}
          confirmLabel="Revert"
          variant="danger"
          onConfirm={confirmRevert}
          onCancel={() => setRevertTarget(null)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// InspectorPanel
// ---------------------------------------------------------------------------

export function InspectorPanel() {
  const [activeTab, setActiveTab] = useState<TabId>("spec");
  const [catalogOpen, setCatalogOpen] = useState(false);

  // Listen for tab switch requests from the command palette or hash router.
  useEffect(() => {
    function handleSwitchTab(e: Event) {
      const tab = (e as CustomEvent<string>).detail;
      if (tab === "spec" || tab === "graph" || tab === "yaml" || tab === "runs") {
        setActiveTab(tab);
      }
    }
    window.addEventListener(SWITCH_TAB_EVENT, handleSwitchTab);
    return () => window.removeEventListener(SWITCH_TAB_EVENT, handleSwitchTab);
  }, []);

  // Notify hash router when tab changes (for URL sync).
  useEffect(() => {
    window.dispatchEvent(
      new CustomEvent(TAB_CHANGED_EVENT, { detail: activeTab }),
    );
  }, [activeTab]);

  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const stateVersions = useSessionStore((s) => s.stateVersions);
  const isLoadingVersions = useSessionStore((s) => s.isLoadingVersions);
  const revertToVersion = useSessionStore((s) => s.revertToVersion);
  const loadStateVersions = useSessionStore((s) => s.loadStateVersions);
  const injectSystemMessage = useSessionStore((s) => s.injectSystemMessage);
  const sendValidationFeedback = useSessionStore((s) => s.sendValidationFeedback);
  const selectNode = useSessionStore((s) => s.selectNode);

  const validationResult = useExecutionStore((s) => s.validationResult);
  const isValidating = useExecutionStore((s) => s.isValidating);
  const isExecuting = useExecutionStore((s) => s.isExecuting);
  const validate = useExecutionStore((s) => s.validate);
  const execute = useExecutionStore((s) => s.execute);
  const progress = useExecutionStore((s) => s.progress);
  const error = useExecutionStore((s) => s.error);

  const canExecute =
    validationResult?.is_valid === true &&
    !isExecuting &&
    progress?.status !== "running";

  // A composition "has content" when it has a source, nodes, or outputs.
  // Source→sink pipelines have zero nodes but are still valid compositions.
  const hasCompositionContent =
    !!compositionState &&
    (compositionState.source !== null ||
      compositionState.nodes.length > 0 ||
      compositionState.outputs.length > 0);

  const canValidate =
    !!activeSessionId &&
    hasCompositionContent &&
    !isValidating &&
    !isExecuting;

  const handleValidate = useCallback(async () => {
    if (!activeSessionId || !canValidate) return;

    // Store handles the API call and stores the result.
    await validate(activeSessionId);

    // Read the result and orchestrate side effects at the component level.
    // This keeps the store focused on state and the component in control
    // of cross-store interactions.
    const result = useExecutionStore.getState().validationResult;
    if (!result) return;

    const VALIDATION_MSG_ID = "system-validation-current";

    if (!result.is_valid && result.errors.length > 0) {
      const lines = ["**Validation failed** — the following errors were sent to the agent:"];
      for (const err of result.errors) {
        lines.push(`- **[${err.component_type ?? "unknown"}] ${err.component_id ?? "unknown"}:** ${err.message}`);
      }
      injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);

      // Send to the LLM so it can attempt fixes.
      // Await so errors are surfaced, not silently swallowed.
      try {
        await sendValidationFeedback(result);
      } catch {
        // Feedback send failed — user still sees the system message,
        // but the agent didn't receive it. The error banner from
        // sendMessage's catch block will surface this.
      }
    } else if (result.is_valid && result.warnings && result.warnings.length > 0) {
      const lines = ["**Validation passed with warnings:**"];
      for (const warn of result.warnings) {
        lines.push(`- **[${warn.component_type ?? "unknown"}] ${warn.component_id ?? "unknown"}:** ${warn.message}`);
      }
      injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
    }
  }, [activeSessionId, canValidate, validate, injectSystemMessage, sendValidationFeedback]);

  const handleExecute = useCallback(async () => {
    if (activeSessionId && canExecute) {
      const runId = await execute(activeSessionId);
      if (runId) {
        setActiveTab("runs");
      }
    }
  }, [activeSessionId, canExecute, execute]);

  const handleVersionDropdownOpen = useCallback(() => {
    loadStateVersions();
  }, [loadStateVersions]);

  const handleRevert = useCallback(
    (stateId: string, _version: number) => {
      revertToVersion(stateId);
    },
    [revertToVersion],
  );

  // Handle click on validation error/warning — select node and switch to Spec tab
  // Only navigate for actual nodes; source/sink entries can't be highlighted in SpecView
  const handleValidationComponentClick = useCallback(
    (componentId: string) => {
      const isNode = compositionState?.nodes.some((n) => n.id === componentId);
      if (isNode) {
        selectNode(componentId);
        setActiveTab("spec");
      }
      // For source/sink, don't switch tabs — SpecView only renders nodes
    },
    [selectNode, compositionState],
  );

  // Tab navigation with arrow keys (left/right wrapping)
  function handleTabKeyDown(e: React.KeyboardEvent, tabIndex: number) {
    let nextIndex: number | null = null;
    if (e.key === "ArrowRight") {
      nextIndex = (tabIndex + 1) % TABS.length;
    } else if (e.key === "ArrowLeft") {
      nextIndex = (tabIndex - 1 + TABS.length) % TABS.length;
    }
    if (nextIndex !== null) {
      e.preventDefault();
      setActiveTab(TABS[nextIndex].id);
      const tabButton = document.getElementById(
        `inspector-tab-${TABS[nextIndex].id}`,
      );
      tabButton?.focus();
    }
  }

  return (
    <div className="inspector-content">
      {/* Header — z-index above tab content but below catalog backdrop.
          When the catalog drawer is open, backdrop covers the header;
          close via backdrop click, Escape, or the drawer's X button. */}
      <div className="inspector-header">
        {/* Row 1: Version selector + validation dot | Validate + Execute */}
        <div className="inspector-header-row">
          {/* Left: VersionSelector + ValidationDot */}
          <div className="inspector-header-left">
            {compositionState && (
              <VersionSelector
                key={activeSessionId}
                currentVersion={compositionState.version}
                stateVersions={stateVersions}
                isLoadingVersions={isLoadingVersions}
                onOpen={handleVersionDropdownOpen}
                onRevert={handleRevert}
              />
            )}

            {/* Validation status indicator — three-state (A7) */}
            {hasCompositionContent && (() => {
              const hasWarnings = validationResult?.warnings && validationResult.warnings.length > 0;
              const status: string =
                validationResult === null
                  ? "unchecked"
                  : !validationResult.is_valid
                    ? "invalid"
                    : hasWarnings
                      ? "warning"
                      : "valid";

              const labels: Record<string, string> = {
                unchecked: "Not validated",
                valid: "Validation passed",
                warning: "Validation passed with warnings",
                invalid: "Validation failed",
              };

              const colors: Record<string, string> = {
                unchecked: "var(--color-warning)",
                valid: "var(--color-success)",
                warning: "var(--color-warning)",
                invalid: "var(--color-error)",
              };

              const symbols: Record<string, string> = {
                unchecked: "\u25CB",  // ○ hollow circle
                valid: "\u2713",      // ✓ checkmark
                warning: "\u26A0",    // ⚠ warning triangle
                invalid: "\u2717",    // ✗ cross mark
              };

              return (
                <span
                  aria-label={labels[status]}
                  title={labels[status]}
                  className="inspector-validation-dot"
                  style={{ color: colors[status] }}
                >
                  {symbols[status]}
                </span>
              );
            })()}
          </div>

          {/* Right: Catalog + Validate + Execute */}
          <div className="inspector-header-right">
            {/* Catalog toggle */}
            <button
              onClick={() => setCatalogOpen(!catalogOpen)}
              className="btn inspector-action-btn"
            >
              Catalog
            </button>

            {/* Validate button with spinner */}
            <button
              onClick={handleValidate}
              disabled={!canValidate}
              aria-label={isValidating ? "Validating" : "Validate pipeline"}
              className="btn inspector-action-btn"
            >
              {isValidating ? (
                <span
                  className="spinner"
                  role="status"
                  aria-label="Validating"
                />
              ) : (
                "Validate"
              )}
            </button>

            {/* Execute button with spinner */}
            <button
              onClick={handleExecute}
              disabled={isExecuting || !canExecute}
              aria-label={isExecuting ? "Starting pipeline" : "Execute pipeline"}
              className={`btn inspector-action-btn ${canExecute && !isExecuting ? "btn-primary" : ""}`}
            >
              {isExecuting ? (
                <>
                  <span
                    className="spinner"
                    role="status"
                    aria-label="Starting pipeline"
                  />
                  Starting...
                </>
              ) : (
                "Execute"
              )}
            </button>
          </div>
        </div>

        {/* Row 2: Tab strip */}
        <div
          role="tablist"
          aria-label="Inspector tabs"
          className="tab-strip"
        >
          {TABS.map((tab, i) => (
            <button
              key={tab.id}
              id={`inspector-tab-${tab.id}`}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`inspector-tabpanel-${tab.id}`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={(e) => handleTabKeyDown(e, i)}
              className={`tab-strip-tab ${activeTab === tab.id ? "tab-strip-tab-active" : ""}`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Error banner from execution store */}
      {error && (
        <div
          role="alert"
          className="validation-banner validation-banner-fail inspector-error-banner"
        >
          {error}
        </div>
      )}

      {/* Validation result banner */}
      {validationResult && (
        <ValidationResultBanner
          result={validationResult}
          nodes={compositionState?.nodes}
          onComponentClick={handleValidationComponentClick}
        />
      )}

      {/* Tab content area with ARIA live region */}
      <div
        role="tabpanel"
        aria-live="polite"
        id={`inspector-tabpanel-${activeTab}`}
        aria-labelledby={`inspector-tab-${activeTab}`}
        className="inspector-tab-content"
      >
        {activeTab === "spec" && (
          <ErrorBoundary label="Spec view">
            <SpecView />
          </ErrorBoundary>
        )}
        {activeTab === "graph" && (
          <ErrorBoundary label="Graph view">
            <GraphView />
          </ErrorBoundary>
        )}
        {activeTab === "yaml" && (
          <ErrorBoundary label="YAML view">
            <YamlView />
          </ErrorBoundary>
        )}
        {activeTab === "runs" && (
          <ErrorBoundary label="Runs view">
            <RunsView />
          </ErrorBoundary>
        )}
      </div>

      <CatalogDrawer isOpen={catalogOpen} onClose={() => setCatalogOpen(false)} />
    </div>
  );
}
