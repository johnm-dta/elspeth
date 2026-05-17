import { useRef } from "react";
import { useFocusTrap } from "@/hooks/useFocusTrap";

interface ShortcutsHelpProps {
  onClose: () => void;
}

interface ShortcutEntry {
  keys: string;
  action: string;
}

// Phase 7B regroup: the flat list becomes two sections so the catalog
// shortcut visually moves out of the "Actions" gravity well and into
// "Reference," signalling its reshape from interactive toolkit to
// searchable system-capability reference. Per design doc 08-§Keyboard
// shortcut placement and roadmap open question D1.
//
// Alt+1-3 is drawer-scoped: the handler lives in CatalogDrawer and runs
// only while isOpen is true. It is harmless from the main canvas.
const ACTION_SHORTCUTS: ShortcutEntry[] = [
  { keys: "Ctrl+K", action: "Command palette" },
  { keys: "Ctrl+N", action: "New session" },
  { keys: "Ctrl+/", action: "Focus chat input" },
  { keys: "Ctrl+Shift+V", action: "Validate pipeline" },
  { keys: "Ctrl+E", action: "Execute pipeline" },
  { keys: "Ctrl/Cmd+Shift+G", action: "Open graph view" },
  { keys: "Ctrl/Cmd+Shift+Y", action: "Export YAML" },
  { keys: "Alt+1-3", action: "Switch catalog tab (Sources / Transforms / Sinks)" },
  { keys: "Escape", action: "Close dialog or drawer" },
];

const REFERENCE_SHORTCUTS: ShortcutEntry[] = [
  { keys: "Ctrl/Cmd+Shift+P", action: "Open plugin catalog" },
  { keys: "?", action: "Keyboard shortcuts" },
];

function ShortcutList({ shortcuts }: { shortcuts: ShortcutEntry[] }) {
  return (
    <dl className="shortcuts-list">
      {shortcuts.map(({ keys, action }) => (
        <div key={keys} className="shortcuts-list-item">
          <dt>
            <kbd className="command-palette-kbd">{keys}</kbd>
          </dt>
          <dd>{action}</dd>
        </div>
      ))}
    </dl>
  );
}

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
        <h3 className="shortcuts-subheading">Actions</h3>
        <ShortcutList shortcuts={ACTION_SHORTCUTS} />
        <h3 className="shortcuts-subheading">Reference</h3>
        <ShortcutList shortcuts={REFERENCE_SHORTCUTS} />
        <div className="confirm-dialog-actions">
          <button onClick={onClose} className="btn confirm-dialog-btn">
            Close
          </button>
        </div>
      </div>
    </>
  );
}
