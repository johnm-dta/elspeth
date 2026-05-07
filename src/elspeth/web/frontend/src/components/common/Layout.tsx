import {
  useState,
  useRef,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { useTheme } from "@/hooks/useTheme";
import { ErrorBoundary } from "./ErrorBoundary";

const INSPECTOR_WIDTH_KEY = "elspeth_inspector_width";
const SIDEBAR_COLLAPSED_KEY = "elspeth_sidebar_collapsed";

const MIN_INSPECTOR_WIDTH = 240;
const SIDEBAR_EXPANDED_WIDTH = 200;
const SIDEBAR_COLLAPSED_WIDTH = 40;

/** Breakpoint below which the sidebar auto-collapses. */
const NARROW_BREAKPOINT = 1024;

/** Breakpoint below which the inspector becomes an overlay sheet. */
const OVERLAY_BREAKPOINT = 900;

/**
 * Compute the default inspector width as ~50% of the space remaining
 * after the sidebar. This gives an even chat/inspector split (A4).
 * Falls back to 50% of viewport if called before layout.
 */
function defaultInspectorWidth(): number {
  const available = window.innerWidth - SIDEBAR_EXPANDED_WIDTH;
  const half = Math.round(available / 2);
  return Math.max(MIN_INSPECTOR_WIDTH, half);
}

function loadPersistedNumber(key: string, fallback: number): number {
  const raw = localStorage.getItem(key);
  if (raw === null) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function loadPersistedBoolean(key: string, fallback: boolean): boolean {
  const raw = localStorage.getItem(key);
  if (raw === null) return fallback;
  return raw === "true";
}

interface LayoutProps {
  sidebar: ReactNode;
  chat: ReactNode;
  inspector: ReactNode;
}

/**
 * Three-panel CSS grid layout with responsive breakpoints.
 *
 * Desktop (>1024px):
 *   - Sessions sidebar: 200px fixed, collapsible to 40px (persisted)
 *   - Chat panel: flex (1fr, takes remaining space)
 *   - Inspector panel: resizable via drag handle (persisted)
 *
 * Narrow (<=1024px):
 *   - Sidebar auto-collapses (user can still expand manually)
 *
 * Overlay (<= 900px):
 *   - Inspector becomes a slide-over overlay sheet with backdrop
 *   - Toggle button appears in the chat header area
 */
export function Layout({ sidebar, chat, inspector }: LayoutProps) {
  const [inspectorWidth, setInspectorWidth] = useState(() =>
    loadPersistedNumber(INSPECTOR_WIDTH_KEY, defaultInspectorWidth())
  );
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() =>
    loadPersistedBoolean(SIDEBAR_COLLAPSED_KEY, false)
  );
  const [inspectorVisible, setInspectorVisible] = useState(true);
  const [isOverlayMode, setIsOverlayMode] = useState(
    () => window.innerWidth <= OVERLAY_BREAKPOINT,
  );
  const isResizing = useRef(false);
  const { resolvedTheme, toggleTheme } = useTheme();

  // Respond to viewport width changes for responsive breakpoints.
  useEffect(() => {
    const narrowMq = window.matchMedia(`(max-width: ${NARROW_BREAKPOINT}px)`);
    const overlayMq = window.matchMedia(`(max-width: ${OVERLAY_BREAKPOINT}px)`);

    function handleNarrow(e: MediaQueryListEvent) {
      if (e.matches) {
        setSidebarCollapsed(true);
      }
    }

    function handleOverlay(e: MediaQueryListEvent) {
      setIsOverlayMode(e.matches);
      if (e.matches) {
        // Hide inspector when entering overlay mode
        setInspectorVisible(false);
      } else {
        // Always show inspector when leaving overlay mode
        setInspectorVisible(true);
      }
    }

    // Apply initial state
    if (narrowMq.matches) {
      setSidebarCollapsed(true);
    }
    if (overlayMq.matches) {
      setIsOverlayMode(true);
      setInspectorVisible(false);
    }

    narrowMq.addEventListener("change", handleNarrow);
    overlayMq.addEventListener("change", handleOverlay);
    return () => {
      narrowMq.removeEventListener("change", handleNarrow);
      overlayMq.removeEventListener("change", handleOverlay);
    };
  }, []);

  // Persist inspector width to localStorage when it changes.
  useEffect(() => {
    localStorage.setItem(INSPECTOR_WIDTH_KEY, String(inspectorWidth));
  }, [inspectorWidth]);

  // Persist sidebar collapsed state to localStorage when it changes.
  useEffect(() => {
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  const handleToggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => !prev);
  }, []);

  const handleToggleInspector = useCallback(() => {
    setInspectorVisible((prev) => !prev);
  }, []);

  const handleCloseOverlay = useCallback(() => {
    setInspectorVisible(false);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;

    function handleMouseMove(ev: MouseEvent) {
      if (!isResizing.current) return;
      const newWidth = window.innerWidth - ev.clientX;
      const maxWidth = window.innerWidth * 0.5;
      setInspectorWidth(Math.max(MIN_INSPECTOR_WIDTH, Math.min(newWidth, maxWidth)));
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
      setInspectorWidth(Math.max(MIN_INSPECTOR_WIDTH, Math.min(newWidth, maxWidth)));
    }

    function handleTouchEnd() {
      isResizing.current = false;
      document.removeEventListener("touchmove", handleTouchMove);
      document.removeEventListener("touchend", handleTouchEnd);
    }

    document.addEventListener("touchmove", handleTouchMove, { passive: true });
    document.addEventListener("touchend", handleTouchEnd);
  }, []);

  const sidebarWidth = sidebarCollapsed
    ? SIDEBAR_COLLAPSED_WIDTH
    : SIDEBAR_EXPANDED_WIDTH;

  // In overlay mode, the grid only has sidebar + chat (inspector floats).
  const gridColumns = isOverlayMode
    ? `${sidebarWidth}px 1fr`
    : inspectorVisible
      ? `${sidebarWidth}px 1fr ${inspectorWidth}px`
      : `${sidebarWidth}px 1fr`;

  return (
    <div
      className={`app-layout${isOverlayMode ? " app-layout--overlay" : ""}`}
      style={{ gridTemplateColumns: gridColumns }}
    >
      {/* Sidebar panel */}
      <div className="layout-sidebar" style={{ width: sidebarWidth }}>
        {/* Sidebar toolbar: collapse toggle + theme toggle */}
        <div
          className={`layout-sidebar-toolbar${sidebarCollapsed ? " layout-sidebar-toolbar--collapsed" : ""}`}
        >
          {/* Collapse toggle */}
          <button
            className="sidebar-toggle"
            onClick={handleToggleSidebar}
            aria-label={
              sidebarCollapsed ? "Expand sessions sidebar" : "Collapse sessions sidebar"
            }
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? "\u25B6" : "\u25C0"}
          </button>

          {/* Theme toggle — visible when sidebar is expanded */}
          {!sidebarCollapsed && (
            <button
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label={
                resolvedTheme === "dark"
                  ? "Switch to light theme"
                  : "Switch to dark theme"
              }
              title={
                resolvedTheme === "dark"
                  ? "Switch to light theme"
                  : "Switch to dark theme"
              }
            >
              {/* Sun for light theme, moon for dark */}
              {resolvedTheme === "dark" ? "\u2600" : "\u263E"}
            </button>
          )}
        </div>
        {/* Sidebar content — hidden when collapsed */}
        <div
          className={`layout-sidebar-content${sidebarCollapsed ? " layout-sidebar-content--hidden" : ""}`}
        >
          <ErrorBoundary label="Session sidebar">
            {sidebar}
          </ErrorBoundary>
        </div>
      </div>

      {/* Chat panel */}
      <div className="layout-chat">
        {/* Inspector toggle button — visible when inspector is hidden or in overlay mode */}
        {(!inspectorVisible || isOverlayMode) && (
          <button
            className="inspector-toggle-btn"
            onClick={handleToggleInspector}
            aria-label={inspectorVisible ? "Hide inspector" : "Show inspector"}
            title={inspectorVisible ? "Hide inspector" : "Show inspector"}
          >
            {inspectorVisible ? "\u25B6" : "\u25C0"} Inspector
          </button>
        )}
        <ErrorBoundary label="Chat panel">
          {chat}
        </ErrorBoundary>
      </div>

      {/* Inspector panel — inline in desktop, overlay sheet in narrow viewports.
          Always mounted so InspectorPanel preserves state (active tab) across
          overlay toggles; hidden via display:none instead of unmounting. */}
      {inspectorVisible && isOverlayMode && (
        <div
          className="inspector-overlay-backdrop"
          onClick={handleCloseOverlay}
          aria-hidden="true"
        />
      )}
      <div
        className={`layout-inspector${isOverlayMode ? " layout-inspector--overlay" : ""}`}
        style={
          !inspectorVisible
            ? { display: "none" }
            : isOverlayMode
              ? { width: Math.min(inspectorWidth, window.innerWidth - 48) }
              : undefined
        }
      >
        {/* Drag-to-resize handle — hidden in overlay mode */}
        {!isOverlayMode && (
          <div
            className="resize-handle"
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize inspector panel"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft") {
                e.preventDefault();
                setInspectorWidth((w) =>
                  Math.min(w + 10, window.innerWidth * 0.5)
                );
              } else if (e.key === "ArrowRight") {
                e.preventDefault();
                setInspectorWidth((w) => Math.max(w - 10, MIN_INSPECTOR_WIDTH));
              }
            }}
          />
        )}

        {/* Close button in overlay mode */}
        {isOverlayMode && (
          <button
            className="inspector-overlay-close"
            onClick={handleCloseOverlay}
            aria-label="Close inspector"
            title="Close inspector"
          >
            &#x2715;
          </button>
        )}

        <ErrorBoundary label="Inspector panel">
          {inspector}
        </ErrorBoundary>
      </div>
    </div>
  );
}
