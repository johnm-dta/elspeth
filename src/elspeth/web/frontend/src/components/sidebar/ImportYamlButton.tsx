import { OPEN_IMPORT_YAML_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";

/**
 * Import-YAML trigger (elspeth-24c56585f9 T-1) -- the missing
 * half of the export/import round-trip for the "compose, export, hand-edit,
 * re-import" audience.
 *
 * Matches ExportYamlButton: this is only a side-rail trigger. The modal is
 * mounted at app-root level by ImportYamlModalHost so the full-screen dialog
 * never lives inside the `.layout-siderail` subtree.
 */
export function ImportYamlButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);

  if (!activeSessionId) return null;

  return (
    <button
      type="button"
      className="btn side-rail-import-yaml-btn"
      onClick={() =>
        window.dispatchEvent(new CustomEvent(OPEN_IMPORT_YAML_MODAL_EVENT))
      }
      aria-label="Import YAML"
    >
      Import YAML
    </button>
  );
}
