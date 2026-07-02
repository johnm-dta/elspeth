import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FocusEvent,
} from "react";
import { useTheme } from "@/hooks/useTheme";

interface UserMenuProps {
  onOpenSettings: () => void;
  onSignOut: () => void;
}

/**
 * Target for the "Help & documentation" entry. The deployment serves no
 * user-facing docs site of its own, so this points at the repository docs
 * directory named by the package metadata (pyproject [project.urls]).
 * Exported so tests pin the single honest destination.
 */
export const HELP_DOCS_URL = "https://github.com/johnm-dta/elspeth/tree/main/docs";

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

  // Focus leaving the menu subtree closes it (elspeth-83eb51334f): a
  // keyboard user could previously Tab past the trigger while the popup
  // stayed visually open. Only acts when relatedTarget is a real element
  // outside the wrapper — a null relatedTarget (window blur, or browsers
  // that don't focus buttons on mousedown) is left to the click-outside
  // handler so an in-menu click is never swallowed mid-flight.
  const onWrapperBlur = useCallback((e: FocusEvent<HTMLDivElement>) => {
    const next = e.relatedTarget;
    if (
      next instanceof Node &&
      wrapperRef.current !== null &&
      !wrapperRef.current.contains(next)
    ) {
      setOpen(false);
    }
  }, []);

  return (
    <div ref={wrapperRef} className="user-menu" onBlur={onWrapperBlur}>
      <button
        ref={triggerRef}
        type="button"
        aria-label="account menu"
        aria-haspopup="true"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="user-menu-trigger"
      >
        Account
      </button>
      {open && (
        <ul className="user-menu-list">
          <li className="user-menu-item">
            <button
              type="button"
              onClick={onThemeToggle}
              aria-label={themeLabel}
              title={themeLabel}
              className="user-menu-action"
            >
              <span aria-hidden="true">
                {resolvedTheme === "dark" ? "\u2600" : "\u263E"}
              </span>{" "}
              {themeLabel}
            </button>
          </li>
          <li className="user-menu-item">
            <button
              type="button"
              onClick={onSettings}
              className="user-menu-action"
            >
              Composer preferences
            </button>
          </li>
          <li className="user-menu-item">
            {/* One honest help entry (elspeth-8225736807): the project's
                documentation directory, opened in a new tab. Not a help
                centre — the deployment serves no docs of its own. */}
            <a
              href={HELP_DOCS_URL}
              target="_blank"
              rel="noreferrer"
              className="user-menu-action user-menu-action--link"
              onClick={() => setOpen(false)}
            >
              Help &amp; documentation
            </a>
          </li>
          <li className="user-menu-item">
            <button
              type="button"
              onClick={onSignOutClick}
              className="user-menu-action user-menu-action--danger"
            >
              Sign out
            </button>
          </li>
        </ul>
      )}
    </div>
  );
}
