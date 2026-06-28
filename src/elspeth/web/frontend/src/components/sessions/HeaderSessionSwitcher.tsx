// ============================================================================
// HeaderSessionSwitcher
//
// Top-left header dropdown listing every session, plus a "New session" verb.
// Primary session switcher for the app header.
// ============================================================================

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import type { Session } from "@/types/index";

const MENU_ID = "header-session-switcher-menu";

export function HeaderSessionSwitcher(): JSX.Element {
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const selectSession = useSessionStore((s) => s.selectSession);
  const createSession = useSessionStore((s) => s.createSession);
  const renameSession = useSessionStore((s) => s.renameSession);
  const archiveSession = useSessionStore((s) => s.archiveSession);
  const [open, setOpen] = useState(false);
  const [focusIndex, setFocusIndex] = useState(0);
  const [archiveTarget, setArchiveTarget] = useState<Session | null>(null);
  const [archiveError, setArchiveError] = useState<string | null>(null);
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");
  const [renamePending, setRenamePending] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const [filterText, setFilterText] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLElement | null)[]>([]);
  const renameInputRef = useRef<HTMLInputElement | null>(null);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const triggerLabel = activeSession?.title || "Untitled";
  const filteredSessions = sessions.filter(
    (s) =>
      (showArchived || !s.archived) &&
      s.title.toLowerCase().includes(filterText.toLowerCase()),
  );
  const itemCount = 1 + filteredSessions.length * 3;

  const closeAndReturnFocus = useCallback(() => {
    setOpen(false);
    triggerRef.current?.focus();
  }, []);

  useEffect(() => {
    if (open) {
      setFocusIndex(0);
    }
  }, [open]);

  useEffect(() => {
    if (renamingSessionId !== null) {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }
  }, [renamingSessionId]);

  useEffect(() => {
    if (!open) {
      return;
    }
    itemRefs.current[focusIndex]?.focus();
  }, [open, focusIndex]);

  useEffect(() => {
    if (!open) {
      return;
    }
    function handleMouseDown(e: MouseEvent) {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleMouseDown);
    return () => document.removeEventListener("mousedown", handleMouseDown);
  }, [open]);

  const onNewSession = useCallback(() => {
    closeAndReturnFocus();
    void createSession();
  }, [closeAndReturnFocus, createSession]);

  const onSelect = useCallback(
    (id: string) => {
      closeAndReturnFocus();
      void selectSession(id);
    },
    [closeAndReturnFocus, selectSession],
  );

  const startRename = useCallback((session: Session) => {
    setRenamingSessionId(session.id);
    setRenameText(session.title);
  }, []);

  const cancelRename = useCallback(() => {
    setRenamingSessionId(null);
    setRenameText("");
    setRenamePending(false);
    setRenameError(null);
  }, []);

  const saveRename = useCallback(async () => {
    if (renamingSessionId === null || renamePending) return;
    const trimmed = renameText.trim();
    if (!trimmed) return;
    setRenamePending(true);
    try {
      await renameSession(renamingSessionId, trimmed);
      setRenameError(null);
      setRenamingSessionId(null);
      setRenameText("");
      closeAndReturnFocus();
    } catch (err) {
      // Mirror confirmArchive: preserve the backend's diagnostic message
      // when the rejection is an Error, fall back to a friendly message
      // otherwise.  Inline rather than relying on the composer-level
      // error region so the alert is co-located with the rename form.
      const detail = err instanceof Error && err.message ? err.message : null;
      setRenameError(
        detail !== null
          ? `Could not rename session: ${detail}`
          : "Could not rename session. Please try again.",
      );
    } finally {
      setRenamePending(false);
    }
  }, [closeAndReturnFocus, renamePending, renameSession, renameText, renamingSessionId]);

  const confirmArchive = useCallback(async () => {
    if (!archiveTarget) return;
    const targetId = archiveTarget.id;
    setArchiveTarget(null);
    closeAndReturnFocus();
    try {
      await archiveSession(targetId);
      setArchiveError(null);
    } catch (err) {
      // Preserve the backend's diagnostic message when available — an
      // auditable system shouldn't drop the actual failure reason.  Fall
      // back to a friendly message when the rejection isn't an Error
      // (e.g. a string thrown manually somewhere in the call chain).
      const detail = err instanceof Error && err.message ? err.message : null;
      setArchiveError(detail !== null ? `Could not archive session: ${detail}` : "Could not archive session. Please try again.");
    }
  }, [archiveSession, archiveTarget, closeAndReturnFocus]);

  const activateMenuIndex = useCallback(
    (index: number) => {
      if (index === 0) {
        onNewSession();
        return;
      }
      const offset = index - 1;
      const session = filteredSessions[Math.floor(offset / 3)];
      if (!session) return;
      const action = offset % 3;
      if (action === 0) {
        onSelect(session.id);
      } else if (action === 1) {
        startRename(session);
      } else {
        setArchiveTarget(session);
      }
    },
    [onNewSession, onSelect, filteredSessions, startRename],
  );

  const onMenuKeyDown = useCallback(
    (e: KeyboardEvent<HTMLUListElement>) => {
      switch (e.key) {
        case "Escape":
          e.preventDefault();
          closeAndReturnFocus();
          break;
        case "Tab":
          // Shift+Tab must move BACKWARD into the filter controls (the filter
          // input + "Show archived" toggle sit before the menu in the DOM and
          // are natively tabbable). Previously this branch closed the menu on
          // BOTH directions, leaving those controls keyboard-unreachable
          // (WCAG 2.1.1 — elspeth-0730f27017). Let Shift+Tab flow naturally;
          // plain Tab still dismisses the menu forward.
          if (e.shiftKey) {
            break;
          }
          e.preventDefault();
          closeAndReturnFocus();
          break;
        case "ArrowDown":
          e.preventDefault();
          setFocusIndex((i) => (i + 1) % itemCount);
          break;
        case "ArrowUp":
          e.preventDefault();
          setFocusIndex((i) => (i - 1 + itemCount) % itemCount);
          break;
        case "Home":
          e.preventDefault();
          setFocusIndex(0);
          break;
        case "End":
          e.preventDefault();
          setFocusIndex(itemCount - 1);
          break;
        case "Enter":
        case " ":
          // Skip menu-level activation when the event originates inside the
          // rename form. Save/Cancel buttons don't stopPropagation themselves,
          // so without this guard their native Enter/Space activation is
          // suppressed by preventDefault here and activateMenuIndex(focusIndex)
          // fires the wrong command (typically "+ New session" at index 0).
          if ((e.target as HTMLElement).closest("form")) {
            return;
          }
          e.preventDefault();
          activateMenuIndex(focusIndex);
          break;
      }
    },
    [activateMenuIndex, closeAndReturnFocus, focusIndex, itemCount],
  );

  return (
    <>
      <div ref={wrapperRef} className="header-session-switcher">
        <button
          ref={triggerRef}
          type="button"
          aria-haspopup="menu"
          aria-expanded={open}
          aria-controls={MENU_ID}
          aria-label={`Session switcher: ${triggerLabel}`}
          onClick={() => {
            setOpen((v) => {
              // Reopening the menu clears stale inline alerts (archive,
              // rename) so the user doesn't see a previous failure
              // persisting next to a fresh menu interaction.  The alerts
              // still render when their state is set; this just stops
              // them sticking around after dismiss-and-reopen.
              if (!v) {
                setArchiveError(null);
                setRenameError(null);
              }
              return !v;
            });
          }}
          className="btn-compact header-session-switcher-trigger"
        >
          <span aria-hidden="true">Session:</span>{" "}
          <strong>{triggerLabel}</strong>
          <span aria-hidden="true"> ▾</span>
        </button>
        {archiveError !== null && (
          <div role="alert" className="header-session-switcher-archive-error">
            {archiveError}
          </div>
        )}
        {open && (
          <>
            {/*
              Filter input and show-archived toggle are NOT menu items —
              they control the menu's contents.  They previously sat
              inside ``<ul role="menu">`` and axe-core flagged this as
              ``aria-required-children`` (a menu can only contain
              menuitem/menuitemcheckbox/menuitemradio children).  Hoisted
              out into a sibling controls strip so the menu only contains
              real menuitems.  The strip carries its own
              ``role="group"`` + ``aria-label`` so screen readers
              announce it as the filter-controls group rather than as
              part of the session list.
            */}
            <div
              role="group"
              aria-label="Filter sessions"
              className="header-session-switcher-controls"
            >
              <input
                type="text"
                aria-label="Find a session…"
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                onKeyDown={(e) => e.stopPropagation()}
                className="header-session-switcher-filter"
                placeholder="Find a session…"
              />
              <label className="header-session-switcher-show-archived">
                <input
                  type="checkbox"
                  aria-label="Show archived"
                  checked={showArchived}
                  onChange={(e) => setShowArchived(e.target.checked)}
                />
                {" "}Show archived
              </label>
            </div>
            <ul
              id={MENU_ID}
              role="menu"
              aria-label="Sessions"
              className="header-session-switcher-menu"
              onKeyDown={onMenuKeyDown}
            >
            <li
              ref={(el) => {
                itemRefs.current[0] = el;
              }}
              role="menuitem"
              tabIndex={focusIndex === 0 ? 0 : -1}
              onClick={onNewSession}
              className="header-session-switcher-item header-session-switcher-item-new"
            >
              + New session
            </li>
            {filteredSessions.map((session, idx) => {
              const title = session.title || `Session ${session.id.slice(0, 8)}`;
              const selectIndex = 1 + idx * 3;
              const renameIndex = selectIndex + 1;
              const archiveIndex = selectIndex + 2;
              const isRenaming = session.id === renamingSessionId;
              if (isRenaming) {
                const trimmedRename = renameText.trim();
                return (
                  <li key={session.id} role="none" className="header-session-switcher-row">
                    <form
                      className="header-session-switcher-rename-form"
                      onSubmit={(e) => {
                        e.preventDefault();
                        void saveRename();
                      }}
                    >
                      <input
                        ref={renameInputRef}
                        type="text"
                        value={renameText}
                        onChange={(e) => setRenameText(e.target.value)}
                        onKeyDown={(e) => {
                          e.stopPropagation();
                          if (e.key === "Escape") {
                            e.preventDefault();
                            cancelRename();
                          }
                        }}
                        aria-label="Rename session"
                        disabled={renamePending}
                      />
                      <button
                        type="submit"
                        className="btn-compact"
                        aria-label="Save session name"
                        disabled={renamePending || !trimmedRename}
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        className="btn-compact"
                        aria-label="Cancel session rename"
                        onClick={cancelRename}
                        disabled={renamePending}
                      >
                        Cancel
                      </button>
                    </form>
                    {renameError !== null && (
                      <div
                        role="alert"
                        className="header-session-switcher-rename-error"
                      >
                        {renameError}
                      </div>
                    )}
                  </li>
                );
              }
              return (
                <li key={session.id} role="none" className="header-session-switcher-row">
                  <button
                    ref={(el) => {
                      itemRefs.current[selectIndex] = el;
                    }}
                    type="button"
                    role="menuitem"
                    tabIndex={focusIndex === selectIndex ? 0 : -1}
                    aria-current={session.id === activeSessionId ? "page" : undefined}
                    onClick={() => onSelect(session.id)}
                    className="header-session-switcher-item header-session-switcher-item-session"
                  >
                    {title}
                  </button>
                  <button
                    ref={(el) => {
                      itemRefs.current[renameIndex] = el;
                    }}
                    type="button"
                    role="menuitem"
                    tabIndex={focusIndex === renameIndex ? 0 : -1}
                    aria-label={`Rename ${title}`}
                    onClick={() => startRename(session)}
                    className="header-session-switcher-action"
                  >
                    Rename
                  </button>
                  <button
                    ref={(el) => {
                      itemRefs.current[archiveIndex] = el;
                    }}
                    type="button"
                    role="menuitem"
                    tabIndex={focusIndex === archiveIndex ? 0 : -1}
                    aria-label={`Archive ${title}`}
                    onClick={() => setArchiveTarget(session)}
                    className="header-session-switcher-action"
                  >
                    Archive
                  </button>
                </li>
              );
            })}
            </ul>
          </>
        )}
      </div>
      {archiveTarget && (
        <ConfirmDialog
          title="Archive session"
          message={`Archive session "${archiveTarget.title}"? You can restore it later.`}
          confirmLabel="Archive"
          variant="danger"
          onConfirm={confirmArchive}
          onCancel={() => setArchiveTarget(null)}
        />
      )}
    </>
  );
}
