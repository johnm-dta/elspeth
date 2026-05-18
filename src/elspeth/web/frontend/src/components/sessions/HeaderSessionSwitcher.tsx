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
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");
  const [renamePending, setRenamePending] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLElement | null)[]>([]);
  const renameInputRef = useRef<HTMLInputElement | null>(null);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const triggerLabel = activeSession?.title || "Untitled";
  const itemCount = 1 + sessions.length * 3;

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
  }, []);

  const saveRename = useCallback(async () => {
    if (renamingSessionId === null || renamePending) return;
    const trimmed = renameText.trim();
    if (!trimmed) return;
    setRenamePending(true);
    try {
      await renameSession(renamingSessionId, trimmed);
      setRenamingSessionId(null);
      setRenameText("");
      closeAndReturnFocus();
    } finally {
      setRenamePending(false);
    }
  }, [closeAndReturnFocus, renamePending, renameSession, renameText, renamingSessionId]);

  const confirmArchive = useCallback(() => {
    if (!archiveTarget) return;
    const targetId = archiveTarget.id;
    setArchiveTarget(null);
    closeAndReturnFocus();
    void archiveSession(targetId);
  }, [archiveSession, archiveTarget, closeAndReturnFocus]);

  const activateMenuIndex = useCallback(
    (index: number) => {
      if (index === 0) {
        onNewSession();
        return;
      }
      const offset = index - 1;
      const session = sessions[Math.floor(offset / 3)];
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
    [onNewSession, onSelect, sessions, startRename],
  );

  const onMenuKeyDown = useCallback(
    (e: KeyboardEvent<HTMLUListElement>) => {
      switch (e.key) {
        case "Escape":
          e.preventDefault();
          closeAndReturnFocus();
          break;
        case "Tab":
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
          onClick={() => setOpen((v) => !v)}
          className="header-session-switcher-trigger"
        >
          <span aria-hidden="true">Session:</span>{" "}
          <strong>{triggerLabel}</strong>
          <span aria-hidden="true"> ▾</span>
        </button>
        {open && (
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
            {sessions.map((session, idx) => {
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
                        aria-label="Save session name"
                        disabled={renamePending || !trimmedRename}
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        aria-label="Cancel session rename"
                        onClick={cancelRename}
                        disabled={renamePending}
                      >
                        Cancel
                      </button>
                    </form>
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
