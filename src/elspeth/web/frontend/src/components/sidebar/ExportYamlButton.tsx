import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";

export function ExportYamlButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);

  if (!activeSessionId) return null;

  return (
    <button
      type="button"
      className="btn side-rail-export-yaml-btn"
      onClick={() => window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT))}
      aria-label="Export YAML"
    >
      Export YAML
    </button>
  );
}
