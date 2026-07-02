import { useId } from "react";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";
import { hasCompositionContent } from "@/utils/compositionState";

/**
 * Tooltip / description text shown while the pipeline has no components.
 * Exported so the tests pin the exact user-facing string.
 */
export const EXPORT_YAML_EMPTY_PIPELINE_TITLE =
  "Add pipeline components before exporting YAML.";

/**
 * Export-YAML trigger.
 *
 * Disabled while the composition has no content (no sources, nodes, or
 * outputs) — an empty pipeline has nothing to export, and opening the
 * modal onto a one-line placeholder was the observed live defect
 * (elspeth-bff8043d33). Validation state deliberately does NOT gate this
 * button (plan 19b:229, AC 4): exporting the YAML of an invalid pipeline
 * for review is a supported flow.
 *
 * Disabled-with-reason idiom mirrors ExecuteButton: `disabled` +
 * `aria-disabled` + `title` + `aria-describedby` pointing at a
 * visually-hidden span (a `title` attribute alone is not reliably
 * announced by screen readers).
 */
export function ExportYamlButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const reactId = useId();
  const describedById = `${reactId}-export-empty-reason`;

  if (!activeSessionId) return null;

  const hasContent = hasCompositionContent(compositionState);

  return (
    <>
      <button
        type="button"
        className="btn side-rail-export-yaml-btn"
        onClick={() => window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT))}
        disabled={!hasContent}
        aria-disabled={!hasContent ? true : undefined}
        aria-label="Export YAML"
        title={!hasContent ? EXPORT_YAML_EMPTY_PIPELINE_TITLE : undefined}
        aria-describedby={!hasContent ? describedById : undefined}
      >
        Export YAML
      </button>
      {!hasContent && (
        <span id={describedById} className="sr-only">
          {EXPORT_YAML_EMPTY_PIPELINE_TITLE}
        </span>
      )}
    </>
  );
}
