import { useEffect, useRef } from "react";

export const FOCUSABLE_SELECTOR =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

// Stack of currently-active trap containers, most-recently-activated last.
// Each active trap installs its own document-level keydown listener, so when
// modals stack (e.g. a SecretsPanel that opens a ConfirmDialog rendered as a
// sibling of its trap container), every trap's listener fires on the same Tab.
// The recapture rule below ("focus left my container -> pull it back") makes
// that actively harmful: the outer trap, seeing focus inside the inner dialog as
// "outside itself", would yank it back out. To prevent that, only the TOPMOST
// (last-activated) trap responds; the rest stand down until it unmounts and pops
// off the stack.
const trapStack: HTMLElement[] = [];

/**
 * Traps keyboard focus within a container element.
 *
 * When activated:
 * - Saves the currently focused element
 * - Moves focus to the first focusable child (or a specific element via initialFocusSelector)
 * - On Tab/Shift+Tab, recaptures focus that has escaped the container (these
 *   drawer/dialog surfaces have no backdrop/inerting, so focus can otherwise
 *   land behind an aria-modal surface) and wraps at the first/last edge
 *
 * When deactivated or unmounted:
 * - Restores focus to the previously focused element
 *
 * Only the most-recently-activated trap is live at any moment (see trapStack),
 * so stacked/nested modals do not fight over focus.
 */
export function useFocusTrap(
  containerRef: React.RefObject<HTMLElement | null>,
  active: boolean = true,
  initialFocusSelector?: string,
): void {
  const previouslyFocused = useRef<Element | null>(null);

  useEffect(() => {
    if (!active) return;
    const container = containerRef.current;
    if (!container) return;

    // Save current focus
    previouslyFocused.current = document.activeElement;

    // Register as the topmost active trap.
    trapStack.push(container);

    // Move focus into the container
    const initialTarget = initialFocusSelector
      ? container.querySelector<HTMLElement>(initialFocusSelector)
      : container.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
    initialTarget?.focus();

    // Tab trap handler
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key !== "Tab" || !container) return;
      // Only the topmost trap acts; nested/stacked traps stand down so they do
      // not recapture focus out of the dialog currently on top.
      if (trapStack[trapStack.length - 1] !== container) return;

      const focusable = container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR);
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const activeElement = document.activeElement;
      const escaped = !container.contains(activeElement);

      if (e.shiftKey && (activeElement === first || escaped)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (activeElement === last || escaped)) {
        e.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      // Pop this trap off the stack (by identity, robust to unmount ordering).
      const index = trapStack.lastIndexOf(container);
      if (index !== -1) trapStack.splice(index, 1);
      // Restore focus to the element that was focused before the trap
      if (previouslyFocused.current instanceof HTMLElement) {
        previouslyFocused.current.focus();
      }
    };
  }, [active, containerRef, initialFocusSelector]);
}
