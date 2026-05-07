import { useEffect, useRef } from "react";

const FOCUSABLE_SELECTOR =
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

/**
 * Traps keyboard focus within a container element.
 *
 * When activated:
 * - Saves the currently focused element
 * - Moves focus to the first focusable child (or a specific element via initialFocusSelector)
 * - Wraps Tab/Shift+Tab within the container
 *
 * When deactivated or unmounted:
 * - Restores focus to the previously focused element
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

    // Move focus into the container
    const initialTarget = initialFocusSelector
      ? container.querySelector<HTMLElement>(initialFocusSelector)
      : container.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
    initialTarget?.focus();

    // Tab trap handler
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key !== "Tab" || !container) return;
      const focusable = container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR);
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      // Restore focus to the element that was focused before the trap
      if (previouslyFocused.current instanceof HTMLElement) {
        previouslyFocused.current.focus();
      }
    };
  }, [active, containerRef, initialFocusSelector]);
}
