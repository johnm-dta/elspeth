/**
 * Command Palette (Ctrl+K)
 *
 * Keyboard-first navigation and action execution overlay.
 * Fuzzy search over actions, sessions, and navigation targets.
 */

import {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
} from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { fuzzyMatch } from "@/utils/fuzzyScore";

// ── Types ────────────────────────────────────────────────────────────────────

interface Command {
  id: string;
  title: string;
  category: "action" | "session" | "navigation";
  shortcut?: string;
  action: () => void;
  /** Whether command is currently available */
  enabled?: boolean;
}

/** Custom event name used to request an inspector tab switch. */
export const SWITCH_TAB_EVENT = "elspeth-switch-tab";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

// ── Component ────────────────────────────────────────────────────────────────

export function CommandPalette({
  isOpen,
  onClose,
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Store hooks
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const createSession = useSessionStore((s) => s.createSession);
  const selectSession = useSessionStore((s) => s.selectSession);
  const compositionState = useSessionStore((s) => s.compositionState);

  const validate = useExecutionStore((s) => s.validate);
  const execute = useExecutionStore((s) => s.execute);
  const validationResult = useExecutionStore((s) => s.validationResult);

  // Build command list
  const commands = useMemo<Command[]>(() => {
    const cmds: Command[] = [];

    // Actions
    cmds.push({
      id: "new-session",
      title: "New Session",
      category: "action",
      shortcut: "Ctrl+N",
      action: () => {
        createSession();
        onClose();
      },
    });

    cmds.push({
      id: "validate",
      title: "Validate Pipeline",
      category: "action",
      shortcut: "Ctrl+Shift+V",
      enabled: !!compositionState && !!activeSessionId,
      action: () => {
        if (activeSessionId) {
          validate(activeSessionId);
        }
        onClose();
      },
    });

    cmds.push({
      id: "execute",
      title: "Execute Pipeline",
      category: "action",
      shortcut: "Ctrl+E",
      enabled: validationResult?.is_valid === true && !!activeSessionId,
      action: () => {
        if (activeSessionId) {
          execute(activeSessionId);
        }
        onClose();
      },
    });

    cmds.push({
      id: "focus-chat",
      title: "Focus Chat Input",
      category: "action",
      shortcut: "Ctrl+/",
      action: () => {
        const input = document.querySelector<HTMLTextAreaElement>(
          "[data-chat-input]",
        );
        input?.focus();
        onClose();
      },
    });

    // Navigation (inspector tabs) — dispatched via custom DOM event so
    // InspectorPanel can listen without prop threading through Layout.
    const switchTab = (tab: string) => {
      window.dispatchEvent(
        new CustomEvent(SWITCH_TAB_EVENT, { detail: tab }),
      );
      onClose();
    };

    cmds.push({
      id: "tab-spec",
      title: "Switch to Spec Tab",
      category: "navigation",
      shortcut: "Alt+1",
      action: () => switchTab("spec"),
    });

    cmds.push({
      id: "tab-graph",
      title: "Switch to Graph Tab",
      category: "navigation",
      shortcut: "Alt+2",
      action: () => switchTab("graph"),
    });

    cmds.push({
      id: "tab-yaml",
      title: "Switch to YAML Tab",
      category: "navigation",
      shortcut: "Alt+3",
      action: () => switchTab("yaml"),
    });

    cmds.push({
      id: "tab-runs",
      title: "Switch to Runs Tab",
      category: "navigation",
      shortcut: "Alt+4",
      action: () => switchTab("runs"),
    });

    // Sessions (up to 10 recent)
    const recentSessions = sessions
      .filter((s) => s.id !== activeSessionId)
      .slice(0, 10);

    for (const session of recentSessions) {
      cmds.push({
        id: `session-${session.id}`,
        title: session.title || `Session ${session.id.slice(0, 8)}`,
        category: "session",
        action: () => {
          selectSession(session.id);
          onClose();
        },
      });
    }

    return cmds;
  }, [
    sessions,
    activeSessionId,
    compositionState,
    validationResult,
    createSession,
    selectSession,
    validate,
    execute,
    onClose,
  ]);

  // Filter and sort commands by fuzzy match
  const filteredCommands = useMemo(() => {
    if (!query.trim()) {
      // Show all commands, actions first
      return commands.filter((c) => c.enabled !== false);
    }

    return commands
      .map((cmd) => ({
        cmd,
        score: fuzzyMatch(query, cmd.title),
      }))
      .filter((item) => item.score >= 0 && item.cmd.enabled !== false)
      .sort((a, b) => a.score - b.score)
      .map((item) => item.cmd);
  }, [commands, query]);

  // Reset selection when query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Focus input when palette opens
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      // Small delay to ensure modal is mounted
      requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    }
  }, [isOpen]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((i) =>
            Math.min(i + 1, filteredCommands.length - 1)
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((i) => Math.max(i - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    },
    [filteredCommands, selectedIndex, onClose]
  );

  // Scroll selected item into view
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const selected = list.querySelector(`[data-index="${selectedIndex}"]`);
    selected?.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  if (!isOpen) return null;

  // Group commands by category for display
  const grouped = {
    action: filteredCommands.filter((c) => c.category === "action"),
    navigation: filteredCommands.filter((c) => c.category === "navigation"),
    session: filteredCommands.filter((c) => c.category === "session"),
  };

  let globalIndex = -1;

  return (
    <>
      {/* Backdrop */}
      <div
        className="command-palette-backdrop"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Palette */}
      <div
        className="command-palette"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onKeyDown={handleKeyDown}
      >
        {/* Search input */}
        <div className="command-palette-input-wrapper">
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            className="command-palette-input"
            placeholder="Type a command or search..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search commands"
            aria-expanded={filteredCommands.length > 0}
            aria-controls="command-palette-listbox"
            aria-activedescendant={
              filteredCommands[selectedIndex]
                ? `cmd-${filteredCommands[selectedIndex].id}`
                : undefined
            }
            aria-autocomplete="list"
          />
        </div>

        {/* Results */}
        <div
          ref={listRef}
          id="command-palette-listbox"
          className="command-palette-list"
          role="listbox"
        >
          {filteredCommands.length === 0 ? (
            <div className="command-palette-empty">
              No commands found
            </div>
          ) : (
            <>
              {grouped.action.length > 0 && (
                <div className="command-palette-group">
                  <div className="command-palette-group-header">Actions</div>
                  {grouped.action.map((cmd) => {
                    globalIndex++;
                    const idx = globalIndex;
                    return (
                      <div
                        key={cmd.id}
                        id={`cmd-${cmd.id}`}
                        data-index={idx}
                        className={`command-palette-item ${
                          idx === selectedIndex ? "command-palette-item-selected" : ""
                        }`}
                        role="option"
                        aria-selected={idx === selectedIndex}
                        onClick={() => cmd.action()}
                        onMouseEnter={() => setSelectedIndex(idx)}
                      >
                        <span className="command-palette-item-title">
                          {cmd.title}
                        </span>
                        {cmd.shortcut && (
                          <kbd className="command-palette-kbd">
                            {cmd.shortcut}
                          </kbd>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {grouped.navigation.length > 0 && (
                <div className="command-palette-group">
                  <div className="command-palette-group-header">Navigation</div>
                  {grouped.navigation.map((cmd) => {
                    globalIndex++;
                    const idx = globalIndex;
                    return (
                      <div
                        key={cmd.id}
                        id={`cmd-${cmd.id}`}
                        data-index={idx}
                        className={`command-palette-item ${
                          idx === selectedIndex ? "command-palette-item-selected" : ""
                        }`}
                        role="option"
                        aria-selected={idx === selectedIndex}
                        onClick={() => cmd.action()}
                        onMouseEnter={() => setSelectedIndex(idx)}
                      >
                        <span className="command-palette-item-title">
                          {cmd.title}
                        </span>
                        {cmd.shortcut && (
                          <kbd className="command-palette-kbd">
                            {cmd.shortcut}
                          </kbd>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {grouped.session.length > 0 && (
                <div className="command-palette-group">
                  <div className="command-palette-group-header">Sessions</div>
                  {grouped.session.map((cmd) => {
                    globalIndex++;
                    const idx = globalIndex;
                    return (
                      <div
                        key={cmd.id}
                        id={`cmd-${cmd.id}`}
                        data-index={idx}
                        className={`command-palette-item ${
                          idx === selectedIndex ? "command-palette-item-selected" : ""
                        }`}
                        role="option"
                        aria-selected={idx === selectedIndex}
                        onClick={() => cmd.action()}
                        onMouseEnter={() => setSelectedIndex(idx)}
                      >
                        <span className="command-palette-item-title">
                          {cmd.title}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer hint */}
        <div className="command-palette-footer">
          <span><kbd>↑↓</kbd> navigate</span>
          <span><kbd>Enter</kbd> select</span>
          <span><kbd>Esc</kbd> close</span>
        </div>
      </div>
    </>
  );
}
