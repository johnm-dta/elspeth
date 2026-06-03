import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { useSessionStore } from "@/stores/sessionStore";
import type { CompositionStateVersion } from "@/types/index";
import { relativeTime } from "@/utils/time";

export function HeaderVersionSelector(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const stateVersions = useSessionStore((s) => s.stateVersions);
  const isLoadingVersions = useSessionStore((s) => s.isLoadingVersions);
  const loadStateVersions = useSessionStore((s) => s.loadStateVersions);
  const revertToVersion = useSessionStore((s) => s.revertToVersion);

  const [isOpen, setIsOpen] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [revertTarget, setRevertTarget] =
    useState<CompositionStateVersion | null>(null);
  const listboxId = useId();
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const currentVersion = compositionState?.version ?? null;

  const sortedVersions: CompositionStateVersion[] = [];
  if (currentVersion !== null) {
    const currentEntry = stateVersions.find(
      (version) => version.version === currentVersion,
    );
    if (currentEntry) {
      sortedVersions.push(currentEntry);
    } else {
      sortedVersions.push({
        id: "",
        version: currentVersion,
        created_at: new Date().toISOString(),
        node_count: compositionState?.nodes.length ?? 0,
      });
    }
    stateVersions
      .filter((version) => version.version !== currentVersion)
      .sort((left, right) => right.version - left.version)
      .forEach((version) => sortedVersions.push(version));
  }

  const toggle = useCallback(() => {
    setIsOpen((prev) => {
      const next = !prev;
      if (next) {
        void loadStateVersions();
        setFocusedIndex(0);
        setSelectedIndex(0);
      }
      return next;
    });
  }, [loadStateVersions]);

  const close = useCallback(() => {
    setIsOpen(false);
    setFocusedIndex(-1);
    triggerRef.current?.focus();
  }, []);

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

  useEffect(() => {
    if (isOpen) {
      listRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || focusedIndex < 0) return;
    const items = listRef.current?.querySelectorAll("[role='option']");
    items?.[focusedIndex]?.scrollIntoView?.({ block: "nearest" });
  }, [isOpen, focusedIndex]);

  useEffect(() => {
    if (selectedIndex < sortedVersions.length) return;
    setSelectedIndex(0);
    setFocusedIndex(sortedVersions.length > 0 ? 0 : -1);
  }, [selectedIndex, sortedVersions.length]);

  if (!activeSessionId || currentVersion === null) {
    return null;
  }

  function handleTriggerKeyDown(e: KeyboardEvent<HTMLButtonElement>) {
    if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (!isOpen) {
        toggle();
      }
    }
  }

  function handleListKeyDown(e: KeyboardEvent<HTMLUListElement>) {
    const count = sortedVersions.length;
    if (e.key === "Escape") {
      e.preventDefault();
      close();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (count > 0) {
        setFocusedIndex((prev) => {
          const next = (prev + 1) % count;
          setSelectedIndex(next);
          return next;
        });
      }
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (count > 0) {
        setFocusedIndex((prev) => {
          const next = (prev - 1 + count) % count;
          setSelectedIndex(next);
          return next;
        });
      }
      return;
    }
    if ((e.key === "Enter" || e.key === " ") && focusedIndex >= 0) {
      e.preventDefault();
      setSelectedIndex(focusedIndex);
      const focusedVersion = sortedVersions[focusedIndex];
      if (focusedVersion && focusedVersion.version !== currentVersion) {
        setRevertTarget(focusedVersion);
      }
    }
  }

  function confirmRevert() {
    if (!revertTarget) return;
    void revertToVersion(revertTarget.id);
    setRevertTarget(null);
    close();
  }

  const selectedVersion = sortedVersions[selectedIndex] ?? null;
  const canRevertSelected =
    !!selectedVersion && selectedVersion.version !== currentVersion;

  return (
    <div ref={containerRef} className="version-selector header-version-selector">
      <button
        ref={triggerRef}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={listboxId}
        aria-label={`Composition history (currently v${currentVersion})`}
        onClick={toggle}
        onKeyDown={handleTriggerKeyDown}
        className="btn version-selector-trigger"
      >
        v{currentVersion} <span aria-hidden="true">▾</span>
      </button>

      {isOpen && (
        <div className="version-selector-dropdown">
          <ul
            ref={listRef}
            id={listboxId}
            role="listbox"
            aria-label="Composition history"
            aria-activedescendant={
              focusedIndex >= 0 && sortedVersions[focusedIndex]
                ? `${listboxId}-option-${sortedVersions[focusedIndex].version}`
                : undefined
            }
            onKeyDown={handleListKeyDown}
            tabIndex={0}
            className="version-selector-list"
          >
            {isLoadingVersions && sortedVersions.length === 0 && (
              <li className="version-selector-loading">Loading versions...</li>
            )}
            {sortedVersions.map((version, index) => {
              const isCurrent = version.version === currentVersion;
              const isFocused = focusedIndex === index;
              return (
                <li
                  key={version.version}
                  id={`${listboxId}-option-${version.version}`}
                  role="option"
                  aria-selected={selectedIndex === index}
                  aria-label={`Version ${version.version}${
                    isCurrent ? " (current)" : ""
                  }`}
                  className={`version-selector-item${
                    isFocused ? " version-selector-item--focused" : ""
                  }${isCurrent ? " version-selector-item--current" : ""}`}
                  onClick={() => {
                    setFocusedIndex(index);
                    setSelectedIndex(index);
                  }}
                  onMouseEnter={() => setFocusedIndex(index)}
                >
                  <span className="version-selector-item-info">
                    <span className="version-selector-item-label">
                      v{version.version}
                      {isCurrent && (
                        <span className="version-selector-item-tag">
                          (current)
                        </span>
                      )}
                    </span>
                    <span className="version-selector-item-meta">
                      {version.node_count} nodes
                    </span>
                    <span className="version-selector-item-meta">
                      {relativeTime(version.created_at)}
                    </span>
                  </span>
                </li>
              );
            })}
          </ul>
          <div className="version-selector-actions">
            <button
              type="button"
              className="btn version-selector-revert-btn"
              disabled={!canRevertSelected}
              aria-label={
                canRevertSelected
                  ? `Revert to version ${selectedVersion.version}`
                  : "Select a previous version to revert"
              }
              onClick={() => {
                if (canRevertSelected) {
                  setRevertTarget(selectedVersion);
                }
              }}
            >
              {canRevertSelected
                ? `Revert to v${selectedVersion.version}`
                : "Current version selected"}
            </button>
          </div>
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
