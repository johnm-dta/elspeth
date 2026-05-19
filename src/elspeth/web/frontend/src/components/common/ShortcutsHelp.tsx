import { useRef } from "react";
import { useFocusTrap } from "@/hooks/useFocusTrap";

interface ShortcutsHelpProps {
  onClose: () => void;
}

interface ShortcutEntry {
  keys: string;
  action: string;
}

interface ShortcutGroup {
  name: string;
  items: ShortcutEntry[];
}

// Phase 8c-5 regroup: four semantic sections replace the two-section (Actions
// / Reference) layout from Phase 7B. Distribution rationale:
//
//   Actions    — things that change state (new session, run, validate).
//   Navigation — things that move focus or switch view (palette, chat, catalog tabs).
//   Reference  — things that surface static information (catalog, this dialog).
//   Editing    — modal-management gestures (Escape).
//
// Ctrl+Shift+V (Validate) is kept because requestValidate has live consumers
// in CommandPalette and CompletionSummary (probe outcome: Phase 8c-5 Step 2).
//
// Alt+1-3 is drawer-scoped: the handler lives in CatalogDrawer and runs only
// while the catalog drawer is open. It is documented here for discoverability.
//
// SWITCH_TAB_EVENT (Alt+1-4 inspector tabs) was never added to App.tsx after
// Phase 3 removed the inspector tabs — no deletion needed (probe: no matches).
const GROUPS: ShortcutGroup[] = [
  {
    name: "Actions",
    items: [
      { keys: "Ctrl+N", action: "New session" },
      { keys: "Ctrl+E", action: "Run pipeline" },
      { keys: "Ctrl+Shift+V", action: "Validate pipeline" },
      { keys: "Ctrl/Cmd+Shift+G", action: "Open graph view" },
      { keys: "Ctrl/Cmd+Shift+Y", action: "Export YAML" },
    ],
  },
  {
    name: "Navigation",
    items: [
      { keys: "Ctrl+K", action: "Command palette" },
      { keys: "Ctrl+/", action: "Focus chat input" },
      { keys: "Alt+1-3", action: "Switch catalog tab (Sources / Transforms / Sinks)" },
    ],
  },
  {
    name: "Reference",
    items: [
      { keys: "Ctrl/Cmd+Shift+P", action: "Open plugin catalog" },
      { keys: "?", action: "Keyboard shortcuts" },
    ],
  },
  {
    name: "Editing",
    items: [{ keys: "Escape", action: "Close dialog or drawer" }],
  },
];

function ShortcutList({ items }: { items: ShortcutEntry[] }) {
  return (
    <dl className="shortcuts-list">
      {items.map(({ keys, action }) => (
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
        {GROUPS.map((group) => (
          <section
            key={group.name}
            aria-label={group.name}
            className="shortcuts-group"
          >
            <h3 className="shortcuts-subheading">{group.name}</h3>
            <ShortcutList items={group.items} />
          </section>
        ))}
        <div className="confirm-dialog-actions">
          <button onClick={onClose} className="btn confirm-dialog-btn">
            Close
          </button>
        </div>
      </div>
    </>
  );
}
