import { useEffect, useState, useCallback, useRef } from "react";
import "./App.css";
import * as api from "./api/client";
import { AuthGuard } from "./components/common/AuthGuard";
import { AppHeader } from "./components/common/AppHeader";
import { Layout } from "./components/common/Layout";
import { SideRail } from "./components/sidebar/SideRail";
import { ExecuteButton } from "./components/sidebar/ExecuteButton";
import { GraphMiniView } from "./components/sidebar/GraphMiniView";
import { GraphModal } from "./components/sidebar/GraphModal";
import { ExportYamlButton } from "./components/sidebar/ExportYamlButton";
import { ExportYamlModal } from "./components/sidebar/ExportYamlModal";
import {
  CatalogButton,
  OPEN_CATALOG_EVENT,
} from "./components/sidebar/CatalogButton";
import { CommandPalette } from "./components/common/CommandPalette";
import { ConfirmDialog } from "./components/common/ConfirmDialog";
import { ShortcutsHelp } from "./components/common/ShortcutsHelp";
import { ChatPanel } from "./components/chat/ChatPanel";
import { CatalogDrawer } from "./components/catalog/CatalogDrawer";
import { AuditReadinessPanel } from "./components/audit/AuditReadinessPanel";
import { RecoveryPanel } from "./components/recovery/RecoveryPanel";
import { SecretsPanel } from "./components/settings/SecretsPanel";
import { ComposerPreferencesPanel } from "./components/settings/ComposerPreferencesPanel";
import { SideRailValidationBanner } from "./components/sidebar/SideRailValidationBanner";
import { useAuthStore } from "./stores/authStore";
import { initStoreSubscriptions } from "./stores/subscriptions";
import { useSessionStore } from "./stores/sessionStore";
import { useExecutionStore } from "./stores/executionStore";
import { usePreferencesStore } from "./stores/preferencesStore";
import { useHashRouter } from "./hooks/useHashRouter";
import { useAuth } from "./hooks/useAuth";
import { useSessionLifecycle } from "./hooks/useSession";
import {
  OPEN_GRAPH_MODAL_EVENT,
  OPEN_YAML_MODAL_EVENT,
} from "./lib/composer-events";
import type { SystemStatus } from "./types/index";

// Health check interval in milliseconds (30 seconds)
const HEALTH_CHECK_INTERVAL = 30_000;
// NOTE (P3A-003 — operator-gated retention, see CLAUDE.md "No Legacy Code Policy"):
// This key cleans up localStorage state from the old SessionSidebar collapse widget
// which was deleted in Phase 3A. Retained (not deleted) because elspeth.foundryside.dev
// holds the operator's own browser state; clearing it is an operator-gated action.
// Safe to remove once the operator confirms staging localStorage has been cleared.
const RETIRED_SIDEBAR_COLLAPSED_KEY = "elspeth_sidebar_collapsed";

// Wire up cross-store subscriptions once at module load time.
// This must run before any component renders so that version-change
// auto-clear is active from the first render.
initStoreSubscriptions();

/**
 * Top-level application component.
 *
 * Single composition root: AuthGuard gates the entire app behind authentication,
 * then AppHeader and Layout render the composer shell with ChatPanel and
 * SideRail. No router in v1 -- the entire application is a single page.
 */
function App() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [showSecrets, setShowSecrets] = useState(false);
  const [showPalette, setShowPalette] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showComposerSettings, setShowComposerSettings] = useState(false);
  const [catalogOpen, setCatalogOpen] = useState(false);
  const logout = useAuthStore((s) => s.logout);
  const openComposerSettings = useCallback(
    () => setShowComposerSettings(true),
    [],
  );
  const closeComposerSettings = useCallback(
    () => setShowComposerSettings(false),
    [],
  );
  const healthCheckRef = useRef<number | null>(null);

  // Sync URL hash ↔ session/tab state for deep linking & back/forward
  const { redirectToast } = useHashRouter();
  useSessionLifecycle();

  const createSession = useSessionStore((s) => s.createSession);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const recoveryError = useSessionStore((s) => s.recoveryError);
  const applyRecoveredState = useSessionStore((s) => s.applyRecoveredState);
  const discardRecovery = useSessionStore((s) => s.discardRecovery);
  const pendingFanoutGuard = useExecutionStore((s) => s.pendingFanoutGuard);
  const { isAuthenticated } = useAuth();
  const bootstrapPrefs = usePreferencesStore((s) => s.bootstrap);

  useEffect(() => {
    localStorage.removeItem(RETIRED_SIDEBAR_COLLAPSED_KEY);
  }, []);

  useEffect(() => {
    function handleOpenCatalog() {
      setCatalogOpen(true);
    }

    window.addEventListener(OPEN_CATALOG_EVENT, handleOpenCatalog);
    return () => window.removeEventListener(OPEN_CATALOG_EVENT, handleOpenCatalog);
  }, []);

  // Phase 1B: load account-level composer preferences once authenticated.
  // Failure is non-fatal — the store stays at its initial state (guided,
  // not-dismissed) so the UI degrades to the default behaviour rather than
  // blocking the user from creating sessions.
  useEffect(() => {
    if (!isAuthenticated) return;
    bootstrapPrefs().catch((err) => {
      console.error("[preferences] bootstrap failed:", err);
    });
  }, [isAuthenticated, bootstrapPrefs]);

  const openSecrets = useCallback(() => setShowSecrets(true), []);
  const closeSecrets = useCallback(() => setShowSecrets(false), []);
  const closePalette = useCallback(() => setShowPalette(false), []);
  const confirmFanoutExecution = useCallback(async () => {
    await useExecutionStore.getState().confirmFanoutExecution();
  }, []);
  const dismissFanoutGuard = useCallback(() => {
    useExecutionStore.getState().dismissFanoutGuard();
  }, []);

  // Check backend health
  const checkHealth = useCallback(async () => {
    try {
      const status = await api.fetchSystemStatus();
      setSystemStatus(status);
      setBackendAvailable(true);
    } catch (err) {
      // Preserve diagnostic detail (network vs CORS vs auth vs 5xx) for
      // operators inspecting DevTools when Retry keeps failing.  The
      // user-visible signal is the role=alert banner; this is the only
      // channel that exposes the underlying cause.
      console.error("[health-check] fetchSystemStatus failed:", err);
      setSystemStatus(null);
      setBackendAvailable(false);
    }
  }, []);

  // Global keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Ctrl+K / Cmd+K: Open command palette
      if (e.key === "k" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setShowPalette(true);
        return;
      }

      // Ctrl+Shift+P / Cmd+Shift+P: Open plugin catalog
      if (
        e.key.toLowerCase() === "p" &&
        e.shiftKey &&
        (e.ctrlKey || e.metaKey)
      ) {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent(OPEN_CATALOG_EVENT));
        return;
      }

      // Ctrl+Shift+G / Cmd+Shift+G: Open graph modal
      if (
        e.key.toLowerCase() === "g" &&
        e.shiftKey &&
        (e.ctrlKey || e.metaKey)
      ) {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
        return;
      }

      // Ctrl+Shift+Y / Cmd+Shift+Y: Open YAML export modal
      if (
        e.key.toLowerCase() === "y" &&
        e.shiftKey &&
        (e.ctrlKey || e.metaKey)
      ) {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT));
        return;
      }

      // Ctrl+N / Cmd+N: New session
      if (e.key === "n" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        createSession();
        return;
      }

      // Ctrl+/ / Cmd+/: Focus chat input
      if (e.key === "/" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        const input = document.querySelector<HTMLTextAreaElement>(
          "[data-chat-input]",
        );
        input?.focus();
        return;
      }

      // Ctrl+Shift+V / Cmd+Shift+V: Validate pipeline
      if (
        e.key === "V" &&
        e.shiftKey &&
        (e.ctrlKey || e.metaKey) &&
        activeSessionId &&
        compositionState
      ) {
        e.preventDefault();
        useExecutionStore.getState().validate(activeSessionId);
        return;
      }

      // Ctrl+E / Cmd+E: Execute pipeline
      if (e.key === "e" && (e.ctrlKey || e.metaKey) && activeSessionId) {
        e.preventDefault();
        const execStore = useExecutionStore.getState();
        const canExec =
          execStore.validationResult?.is_valid === true &&
          !execStore.isExecuting &&
          execStore.progress?.status !== "running";
        if (canExec) {
          execStore.execute(activeSessionId);
        }
        return;
      }

      // ?: Show keyboard shortcuts (only when not typing in an input)
      if (e.key === "?" && !e.ctrlKey && !e.metaKey) {
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        e.preventDefault();
        setShowShortcuts(true);
        return;
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [createSession, activeSessionId, compositionState]);

  // Initial health check and periodic polling
  useEffect(() => {
    checkHealth();

    // Set up periodic health checks
    healthCheckRef.current = window.setInterval(checkHealth, HEALTH_CHECK_INTERVAL);

    return () => {
      if (healthCheckRef.current !== null) {
        window.clearInterval(healthCheckRef.current);
      }
    };
  }, [checkHealth]);

  return (
    <AuthGuard>
      <div className="app-root">
        <a href="#chat-main" className="skip-to-content">
          Skip to main content
        </a>
        <h1 className="sr-only">ELSPETH Pipeline Composer</h1>

        {redirectToast && (
          <div role="alert" className="alert-banner alert-banner--info">
            <span>{redirectToast.message}</span>
            <button
              onClick={redirectToast.dismiss}
              aria-label="Dismiss redirect notice"
              title="Dismiss"
              className="alert-banner-action"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Backend unavailable banner.
            role=alert (assertive) because backend-down is a hard outage that
            blocks every feature in the app; the composer-unavailable banner
            below is role=status (polite) because it's a soft degradation. */}
        {backendAvailable === false && (
          <div role="alert" className="alert-banner">
            <span>
              <strong>Backend unavailable</strong> — Cannot connect to the
              ELSPETH server. Check that the backend is running.
            </span>
            <button
              onClick={checkHealth}
              aria-label="Retry connection"
              title="Retry connection"
              className="alert-banner-action"
            >
              Retry
            </button>
          </div>
        )}

        {/* Composer unavailable banner (backend is up but LLM not configured) */}
        {backendAvailable && systemStatus && !systemStatus.composer_available && (
          <div role="status" className="alert-banner">
            <span>
              Service unavailable:{" "}
              {systemStatus.composer_reason ??
                "The composer cannot reach a usable LLM right now."}
            </span>
            <button
              onClick={openSecrets}
              aria-label="Open secrets settings"
              title="Configure API keys"
              className="alert-banner-action"
            >
              ⚙ API Keys
            </button>
          </div>
        )}
        <AppHeader
          onOpenSettings={openComposerSettings}
          onSignOut={logout}
        />
        <div className="app-main" role="main">
          <Layout
            chat={
              <ChatPanel
                onOpenSecrets={openSecrets}
                onOpenComposerPreferences={openComposerSettings}
              />
            }
            siderail={
              <SideRail
                auditReadinessSlot={<AuditReadinessPanel />}
                validationBannerSlot={<SideRailValidationBanner />}
                graphMiniSlot={<GraphMiniView />}
                catalogSlot={<CatalogButton />}
                exportYamlSlot={<ExportYamlButton />}
                executeButtonSlot={<ExecuteButton />}
                completionBarSlot={null}
              />
            }
          />
        </div>

        {showSecrets && <SecretsPanel onClose={closeSecrets} />}
        <GraphModal />
        <ExportYamlModal />
        <CatalogDrawer
          isOpen={catalogOpen}
          onClose={() => setCatalogOpen(false)}
        />
        {showComposerSettings && (
          <ComposerPreferencesPanel onClose={closeComposerSettings} />
        )}
        <CommandPalette isOpen={showPalette} onClose={closePalette} />
        {showShortcuts && (
          <ShortcutsHelp onClose={() => setShowShortcuts(false)} />
        )}
        <RecoveryPanel
          activeSessionId={activeSessionId}
          currentState={compositionState}
          recoveryError={recoveryError}
          onApply={applyRecoveredState}
          onDiscard={discardRecovery}
        />
        {pendingFanoutGuard && (
          <ConfirmDialog
            title="Review LLM provider calls"
            message={pendingFanoutGuard.summary}
            confirmLabel="Execute"
            cancelLabel="Cancel"
            variant="danger"
            onConfirm={confirmFanoutExecution}
            onCancel={dismissFanoutGuard}
          />
        )}
      </div>
    </AuthGuard>
  );
}

export default App;
