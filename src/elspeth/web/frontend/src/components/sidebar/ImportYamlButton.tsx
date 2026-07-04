import { useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { ImportYamlModal } from "./ImportYamlModal";

/**
 * Import-YAML trigger + its modal (elspeth-24c56585f9 T-1) -- the missing
 * half of the export/import round-trip for the "compose, export, hand-edit,
 * re-import" audience.
 *
 * Unlike ExportYamlButton (a bare trigger that dispatches a window event an
 * App.tsx-mounted `<ExportYamlModal />` listens for), this component owns
 * its modal directly: it renders `<ImportYamlModal />` as a conditional
 * child instead of dispatching a global event for an App-level listener.
 * That is a deliberate deviation, not an oversight -- App.tsx and
 * CompletionBar.tsx (which mounts ExportYamlButton) are outside this
 * change's file ownership, so a self-contained button+modal pair is the
 * only shape that renders anything without touching either file. SideRail
 * mounts this component directly (not via a slot prop) for the same
 * reason.
 */
export function ImportYamlButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const [isOpen, setIsOpen] = useState(false);

  if (!activeSessionId) return null;

  return (
    <>
      <button
        type="button"
        className="btn side-rail-import-yaml-btn"
        onClick={() => setIsOpen(true)}
        aria-label="Import YAML"
      >
        Import YAML
      </button>
      {isOpen && <ImportYamlModal onClose={() => setIsOpen(false)} />}
    </>
  );
}
