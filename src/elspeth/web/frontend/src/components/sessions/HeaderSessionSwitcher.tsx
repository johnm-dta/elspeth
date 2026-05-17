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

const MENU_ID = "header-session-switcher-menu";

export function HeaderSessionSwitcher(): JSX.Element {
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const selectSession = useSessionStore((s) => s.selectSession);
  const createSession = useSessionStore((s) => s.createSession);
  const [open, setOpen] = useState(false);
  const [focusIndex, setFocusIndex] = useState(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLLIElement | null)[]>([]);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const triggerLabel = activeSession?.title || "Untitled";
  const itemCount = 1 + sessions.length;

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
          e.preventDefault();
          if (focusIndex === 0) {
            onNewSession();
          } else {
            onSelect(sessions[focusIndex - 1].id);
          }
          break;
      }
    },
    [closeAndReturnFocus, focusIndex, itemCount, onNewSession, onSelect, sessions],
  );

  return (
    <div ref={wrapperRef} className="header-session-switcher">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={MENU_ID}
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
            const itemIndex = idx + 1;
            return (
              <li
                key={session.id}
                ref={(el) => {
                  itemRefs.current[itemIndex] = el;
                }}
                role="menuitem"
                tabIndex={focusIndex === itemIndex ? 0 : -1}
                aria-current={session.id === activeSessionId ? "page" : undefined}
                onClick={() => onSelect(session.id)}
                className="header-session-switcher-item"
              >
                {session.title || `Session ${session.id.slice(0, 8)}`}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
