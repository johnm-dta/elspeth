import { useCallback, useEffect, useRef, useState } from "react";

interface UserMenuProps {
  onOpenSettings: () => void;
  onSignOut: () => void;
}

/**
 * Account dropdown in the sidebar toolbar. Click-outside, Escape-to-close
 * with focus return to the trigger, and Tab/Shift+Tab navigation (the
 * project convention; CommandPalette.tsx uses the same pattern).
 */
export function UserMenu({
  onOpenSettings,
  onSignOut,
}: UserMenuProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // Click-outside closes
  useEffect(() => {
    if (!open) return;
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

  // Escape closes and returns focus to trigger
  useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open]);

  const onSettings = useCallback(() => {
    setOpen(false);
    onOpenSettings();
  }, [onOpenSettings]);

  const onSignOutClick = useCallback(() => {
    setOpen(false);
    onSignOut();
  }, [onSignOut]);

  return (
    <div ref={wrapperRef} className="user-menu" style={{ position: "relative" }}>
      <button
        ref={triggerRef}
        type="button"
        aria-label="account menu"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        Account
      </button>
      {open && (
        <ul
          role="menu"
          style={{
            position: "absolute",
            top: "100%",
            right: 0,
            margin: 0,
            padding: "4px 0",
            listStyle: "none",
            backgroundColor: "var(--color-surface, #fff)",
            border: "1px solid var(--color-border)",
            borderRadius: 4,
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            minWidth: 140,
            zIndex: 50,
          }}
        >
          <li
            role="menuitem"
            tabIndex={0}
            onClick={onSettings}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSettings();
              }
            }}
            style={{ padding: "6px 12px", cursor: "pointer" }}
          >
            Settings
          </li>
          <li
            role="menuitem"
            tabIndex={0}
            onClick={onSignOutClick}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSignOutClick();
              }
            }}
            style={{ padding: "6px 12px", cursor: "pointer" }}
          >
            Sign out
          </li>
        </ul>
      )}
    </div>
  );
}
