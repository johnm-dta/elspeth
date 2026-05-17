import {
  useState,
  useRef,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import { DefaultModeChangedBanner } from "./DefaultModeChangedBanner";

const SIDERAIL_WIDTH_KEY = "elspeth_inspector_width";

const MIN_SIDERAIL_WIDTH = 240;

/** Breakpoint below which the side rail becomes an overlay sheet. */
const OVERLAY_BREAKPOINT = 900;

/**
 * Compute the default side-rail width as ~50% of the viewport.
 * This gives an even chat/side-rail split (A4).
 */
function defaultSideRailWidth(): number {
  const half = Math.round(window.innerWidth / 2);
  return Math.max(MIN_SIDERAIL_WIDTH, half);
}

function loadPersistedNumber(key: string, fallback: number): number {
  const raw = localStorage.getItem(key);
  if (raw === null) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

interface LayoutProps {
  chat: ReactNode;
  siderail: ReactNode;
}

/**
 * Two-panel CSS grid layout with responsive breakpoints.
 *
 * Desktop (>900px):
 *   - Chat panel: flex (1fr, takes remaining space)
 *   - Side-rail panel: resizable via drag handle (persisted)
 *
 * Overlay (<= 900px):
 *   - Side rail becomes a slide-over overlay sheet with backdrop
 *   - Toggle button appears in the chat header area
 */
export function Layout({
  chat,
  siderail,
}: LayoutProps) {
  const [sideRailWidth, setSideRailWidth] = useState(() =>
    loadPersistedNumber(SIDERAIL_WIDTH_KEY, defaultSideRailWidth())
  );
  const [sideRailVisible, setSideRailVisible] = useState(true);
  const [isOverlayMode, setIsOverlayMode] = useState(
    () => window.innerWidth <= OVERLAY_BREAKPOINT,
  );
  // Tracks viewport width so aria-valuemax (= 50% of viewport) on the resize
  // separator stays in sync with the live clamp inside handleMouseDown /
  // onKeyDown.  Without this, AT announces a max captured at mount time and
  // the user can keep increasing past the announced max after a window resize.
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth);
  const isResizing = useRef(false);

  // Respond to viewport width changes for responsive breakpoints.
  useEffect(() => {
    const overlayMq = window.matchMedia(`(max-width: ${OVERLAY_BREAKPOINT}px)`);

    function handleOverlay(e: MediaQueryListEvent) {
      setIsOverlayMode(e.matches);
      if (e.matches) {
        // Hide side rail when entering overlay mode
        setSideRailVisible(false);
      } else {
        // Always show side rail when leaving overlay mode
        setSideRailVisible(true);
      }
    }

    // Apply initial state
    if (overlayMq.matches) {
      setIsOverlayMode(true);
      setSideRailVisible(false);
    }

    overlayMq.addEventListener("change", handleOverlay);

    // Keep viewportWidth in sync so aria-valuemax stays accurate.  Window
    // resize fires frequently; debounce-via-rAF keeps the state churn cheap.
    let rafHandle = 0;
    function handleResize() {
      if (rafHandle) return;
      rafHandle = window.requestAnimationFrame(() => {
        rafHandle = 0;
        setViewportWidth(window.innerWidth);
      });
    }
    window.addEventListener("resize", handleResize);

    return () => {
      overlayMq.removeEventListener("change", handleOverlay);
      window.removeEventListener("resize", handleResize);
      if (rafHandle) {
        window.cancelAnimationFrame(rafHandle);
      }
    };
  }, []);

  // Persist side-rail width to localStorage when it changes. The storage key
  // intentionally preserves its pre-rename value for existing preferences.
  useEffect(() => {
    localStorage.setItem(SIDERAIL_WIDTH_KEY, String(sideRailWidth));
  }, [sideRailWidth]);

  const handleToggleSideRail = useCallback(() => {
    setSideRailVisible((prev) => !prev);
  }, []);

  const handleCloseOverlay = useCallback(() => {
    setSideRailVisible(false);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;

    function handleMouseMove(ev: MouseEvent) {
      if (!isResizing.current) return;
      const newWidth = window.innerWidth - ev.clientX;
      const maxWidth = window.innerWidth * 0.5;
      setSideRailWidth(Math.max(MIN_SIDERAIL_WIDTH, Math.min(newWidth, maxWidth)));
    }

    function handleMouseUp() {
      isResizing.current = false;
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const handleTouchStart = useCallback((_e: React.TouchEvent) => {
    isResizing.current = true;

    function handleTouchMove(ev: TouchEvent) {
      if (!isResizing.current) return;
      const touch = ev.touches[0];
      if (!touch) return;
      const newWidth = window.innerWidth - touch.clientX;
      const maxWidth = window.innerWidth * 0.5;
      setSideRailWidth(Math.max(MIN_SIDERAIL_WIDTH, Math.min(newWidth, maxWidth)));
    }

    function handleTouchEnd() {
      isResizing.current = false;
      document.removeEventListener("touchmove", handleTouchMove);
      document.removeEventListener("touchend", handleTouchEnd);
    }

    document.addEventListener("touchmove", handleTouchMove, { passive: true });
    document.addEventListener("touchend", handleTouchEnd);
  }, []);

  // In overlay mode, the grid only has chat (side rail floats).
  const gridColumns = isOverlayMode
    ? "1fr"
    : sideRailVisible
      ? `1fr ${sideRailWidth}px`
      : "1fr";

  return (
    <div
      className={`app-layout${isOverlayMode ? " app-layout--overlay" : ""}`}
      style={{ gridTemplateColumns: gridColumns }}
    >
      {/* Chat panel */}
      <div className="layout-chat">
        {/* Phase 1B — opt-out banner. Mounted inside the chat column so it
            consumes the same height budget as the chat scrollback rather
            than adding above-Layout vertical space that the grid wouldn't
            account for. Self-gates on visibility (returns null when not
            applicable). */}
        <DefaultModeChangedBanner />
        {/* Side-rail toggle button — visible when hidden or in overlay mode */}
        {(!sideRailVisible || isOverlayMode) && (
          <button
            className="inspector-toggle-btn"
            onClick={handleToggleSideRail}
            aria-label={sideRailVisible ? "Hide side rail" : "Show side rail"}
            title={sideRailVisible ? "Hide side rail" : "Show side rail"}
          >
            {sideRailVisible ? "\u25B6" : "\u25C0"} Side rail
          </button>
        )}
        <ErrorBoundary label="Chat panel">
          {chat}
        </ErrorBoundary>
      </div>

      {/* Side-rail panel — inline in desktop, overlay sheet in narrow viewports.
          Always mounted so the transitional inspector preserves state across
          overlay toggles; hidden via display:none instead of unmounting. */}
      {sideRailVisible && isOverlayMode && (
        <div
          className="siderail-overlay-backdrop"
          onClick={handleCloseOverlay}
          aria-hidden="true"
        />
      )}
      <div
        className={`layout-siderail${isOverlayMode ? " layout-siderail--overlay" : ""}`}
        style={
          !sideRailVisible
            ? { display: "none" }
            : isOverlayMode
              ? { width: Math.min(sideRailWidth, window.innerWidth - 48) }
              : undefined
        }
      >
        {/* Drag-to-resize handle — hidden in overlay mode.
            Keyboard convention: ArrowLeft decreases the value (panel narrows),
            ArrowRight increases it (panel widens) — per WAI-ARIA APG for
            role=separator. Note the visual paradox: the handle sits on the
            inspector's LEFT edge, so ArrowLeft moves the handle visually right
            (toward the chat) when the panel shrinks. The convention is
            value-direction, not spatial-drag-direction.
            aria-valuenow/min/max are required for focusable separators so AT
            can announce the current width. */}
        {!isOverlayMode && (
          <div
            className="resize-handle"
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize side rail"
            aria-valuenow={sideRailWidth}
            aria-valuemin={MIN_SIDERAIL_WIDTH}
            aria-valuemax={Math.round(viewportWidth * 0.5)}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft") {
                e.preventDefault();
                setSideRailWidth((w) => Math.max(w - 10, MIN_SIDERAIL_WIDTH));
              } else if (e.key === "ArrowRight") {
                e.preventDefault();
                setSideRailWidth((w) =>
                  Math.min(w + 10, window.innerWidth * 0.5)
                );
              }
            }}
          />
        )}

        {/* Close button in overlay mode */}
        {isOverlayMode && (
          <button
            className="siderail-overlay-close"
            onClick={handleCloseOverlay}
            aria-label="Close side rail"
            title="Close side rail"
          >
            &#x2715;
          </button>
        )}

        <ErrorBoundary label="Side rail">
          {siderail}
        </ErrorBoundary>
      </div>
    </div>
  );
}
