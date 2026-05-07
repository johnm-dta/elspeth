import { useRef } from "react";
import { useFocusTrap } from "@/hooks/useFocusTrap";

interface ShortcutsHelpProps {
  onClose: () => void;
}

const SHORTCUTS = [
  { keys: "Ctrl+K", action: "Command palette" },
  { keys: "Ctrl+N", action: "New session" },
  { keys: "Ctrl+/", action: "Focus chat input" },
  { keys: "Ctrl+Shift+V", action: "Validate pipeline" },
  { keys: "Ctrl+E", action: "Execute pipeline" },
  { keys: "Alt+1-4", action: "Switch inspector tab (Spec/Graph/YAML/Runs)" },
  { keys: "?", action: "Keyboard shortcuts" },
  { keys: "Escape", action: "Close dialog or drawer" },
];

export function ShortcutsHelp({ onClose }: ShortcutsHelpProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef);

  return (
    <>
      <div
        className="confirm-dialog-backdrop"
        onClick={onClose}
        role="presentation"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        className="confirm-dialog"
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            onClose();
          }
        }}
      >
        <h2 className="confirm-dialog-title">Keyboard Shortcuts</h2>
        <dl className="shortcuts-list">
          {SHORTCUTS.map(({ keys, action }) => (
            <div key={keys} className="shortcuts-list-item">
              <dt>
                <kbd className="command-palette-kbd">{keys}</kbd>
              </dt>
              <dd>{action}</dd>
            </div>
          ))}
        </dl>
        <div className="confirm-dialog-actions">
          <button
            onClick={onClose}
            className="btn confirm-dialog-btn"
          >
            Close
          </button>
        </div>
      </div>
    </>
  );
}
