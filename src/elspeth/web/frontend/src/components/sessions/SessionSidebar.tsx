// src/components/sessions/SessionSidebar.tsx
//
// Session list sidebar. Always renders at full width -- collapse/expand
// is controlled by Layout.tsx via CSS grid column sizing.
import { useState, useCallback } from "react";
import { useSession } from "@/hooks/useSession";
import { useAuth } from "@/hooks/useAuth";
import { useExecutionStore } from "@/stores/executionStore";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { isTerminalRunStatus, type Session } from "@/types/index";
import { relativeTime } from "@/utils/time";

export function SessionSidebar() {
  const { sessions, activeSessionId, createSession, selectSession, archiveSession } =
    useSession();
  const { user, logout } = useAuth();
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const progress = useExecutionStore((s) => s.progress);
  const hasActiveRun =
    !!activeRunId &&
    !!progress &&
    !isTerminalRunStatus(progress.status);
  const [isCreating, setIsCreating] = useState(false);
  const [archiveTarget, setArchiveTarget] = useState<Session | null>(null);
  const [filter, setFilter] = useState("");

  const handleCreateSession = useCallback(async () => {
    if (isCreating) return;
    setIsCreating(true);
    try {
      await createSession();
    } finally {
      setIsCreating(false);
    }
  }, [isCreating, createSession]);

  return (
    <aside
      className="session-sidebar session-sidebar-container"
      aria-label="Sessions sidebar"
    >
      {/* Header */}
      <div className="session-sidebar-header">
        <span className="session-sidebar-title">
          Sessions
        </span>
      </div>

      {/* Session filter — visible when there are enough sessions to warrant it */}
      {sessions.length > 3 && (
        <div className="session-sidebar-filter">
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter sessions..."
            aria-label="Filter sessions"
            className="session-sidebar-filter-input"
          />
        </div>
      )}

      {/* Session list */}
      <nav
        className="session-list"
        aria-label="Session list"
      >
        {sessions.length === 0 ? (
          <div className="session-list-empty">
            No sessions yet. Click the button below to start.
          </div>
        ) : (
          <ul className="session-list-ul">
            {sessions
              .filter((s) =>
                !filter || s.title.toLowerCase().includes(filter.toLowerCase()),
              )
              .map((session) => {
              const isActive = session.id === activeSessionId;
              return (
                <li
                  key={session.id}
                  className="session-list-item"
                >
                  <button
                    onClick={() => selectSession(session.id)}
                    aria-current={isActive ? "page" : undefined}
                    aria-label={`Session: ${session.title}, ${relativeTime(session.updated_at)}`}
                    className={`session-list-btn ${isActive ? "session-list-btn--active" : "session-list-btn--inactive"}`}
                  >
                    <div
                      className={`session-list-btn-title ${isActive ? "session-list-btn-title--active" : ""}`}
                    >
                      {session.forked_from_session_id && (
                        <span
                          title="Forked session"
                          aria-label="Forked session"
                          className="session-list-fork-icon"
                        >
                          &#x2442;
                        </span>
                      )}
                      <span className="session-title-overflow">
                        {session.title}
                      </span>
                      {isActive && hasActiveRun && (
                        <span
                          className="run-indicator-dot"
                          title="Pipeline running"
                          aria-label="Pipeline running"
                        />
                      )}
                    </div>
                    <div className="session-list-btn-time">
                      {relativeTime(session.updated_at)}
                    </div>
                  </button>
                  <button
                    onClick={() => setArchiveTarget(session)}
                    aria-label={`Archive session: ${session.title}`}
                    title="Archive session"
                    className="session-archive-btn"
                  >
                    {"\u00D7"}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </nav>

      {/* New session button */}
      <div className="session-new-btn-wrap">
        <button
          onClick={handleCreateSession}
          disabled={isCreating}
          aria-label={isCreating ? "Creating session" : "Create new session"}
          className="session-new-btn"
        >
          {isCreating ? "Creating..." : "+ New Session"}
        </button>
      </div>

      {/* User identity + logout */}
      {user && (
        <div className="session-user-bar">
          <span
            className="session-user-name"
            title={user.username}
          >
            {user.display_name || user.username}
          </span>
          <button
            onClick={logout}
            aria-label="Sign out"
            className="session-signout-btn"
          >
            Sign out
          </button>
        </div>
      )}
      {archiveTarget && (
        <ConfirmDialog
          title="Archive session"
          message={`Archive session "${archiveTarget.title}"? You can restore it later.`}
          confirmLabel="Archive"
          variant="danger"
          onConfirm={() => {
            archiveSession(archiveTarget.id);
            setArchiveTarget(null);
          }}
          onCancel={() => setArchiveTarget(null)}
        />
      )}
    </aside>
  );
}
