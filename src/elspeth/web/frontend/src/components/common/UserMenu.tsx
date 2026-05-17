import { useCallback, useEffect, useRef, useState } from "react";
import { useTheme } from "@/hooks/useTheme";

interface UserMenuProps {
  onOpenSettings: () => void;
  onSignOut: () => void;
}

/**
 * Account dropdown in the app header. Click-outside, Escape-to-close
 * with focus return to the trigger, and Tab/Shift+Tab navigation (the
 * project convention; CommandPalette.tsx uses the same pattern).
 *
 * Role contract: this is a disclosure/popover of actions, NOT a WAI-ARIA
 * `menu` widget. The earlier `role="menu"` / `role="menuitem"` assertion
 * (with `aria-haspopup="menu"`) promised the full menu keyboard
 * contract (arrow keys, Home/End, type-ahead) which we don't implement.
 * Per the Phase 1B accessibility-audit panel finding, the correct fix
 * is to drop the menu role rather than add arrow keys: this component
 * is already a correct disclosure (Tab + Escape + focus-return).
 * Trigger uses `aria-haspopup="true"` (the "no specific popup role
 * promise" value); the dropdown is a plain `<ul>` of `<button>`
 * elements with their implicit roles.
 *
 * Item naming: "Composer preferences" rather than "Settings" because
 * the pane today only contains composer preferences — the broader
 * "Settings" framing was the UX panel's "absorb into a hub later"
 * placeholder which would mis-label a single-pane experience.
 */
export function UserMenu({
  onOpenSettings,
  onSignOut,
}: UserMenuProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const { resolvedTheme, toggleTheme } = useTheme();
  const themeLabel =
    resolvedTheme === "dark" ? "Switch to light theme" : "Switch to dark theme";

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

  const onThemeToggle = useCallback(() => {
    toggleTheme();
    setOpen(false);
  }, [toggleTheme]);

  return (
    <div ref={wrapperRef} className="user-menu" style={{ position: "relative" }}>
      <button
        ref={triggerRef}
        type="button"
        aria-label="account menu"
        aria-haspopup="true"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        Account
      </button>
      {open && (
        <ul
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
            minWidth: 180,
            zIndex: 50,
          }}
        >
          <li style={{ margin: 0 }}>
            <button
              type="button"
              onClick={onThemeToggle}
              aria-label={themeLabel}
              title={themeLabel}
              style={{
                display: "block",
                width: "100%",
                padding: "6px 12px",
                textAlign: "left",
                background: "transparent",
                border: 0,
                cursor: "pointer",
                font: "inherit",
                minHeight: 24,
              }}
            >
              <span aria-hidden="true">
                {resolvedTheme === "dark" ? "\u2600" : "\u263E"}
              </span>{" "}
              {themeLabel}
            </button>
          </li>
          <li style={{ margin: 0 }}>
            <button
              type="button"
              onClick={onSettings}
              style={{
                display: "block",
                width: "100%",
                padding: "6px 12px",
                textAlign: "left",
                background: "transparent",
                border: 0,
                cursor: "pointer",
                font: "inherit",
                minHeight: 24,
              }}
            >
              Composer preferences
            </button>
          </li>
          <li style={{ margin: 0 }}>
            <button
              type="button"
              onClick={onSignOutClick}
              style={{
                display: "block",
                width: "100%",
                padding: "6px 12px",
                textAlign: "left",
                background: "transparent",
                border: 0,
                cursor: "pointer",
                font: "inherit",
                minHeight: 24,
              }}
            >
              Sign out
            </button>
          </li>
        </ul>
      )}
    </div>
  );
}
