/**
 * ReadinessRowDetail (Phase 2C)
 *
 * Drawer/popover content for one row of the audit-readiness panel.
 * Renders the row's detail string (multi-line preserved) and offers a
 * jump-to-component button for each entry in component_ids that
 * resolves to a node in the current composition. Unresolvable ids are
 * shown as plain text — they may refer to source/sink names or YAML
 * fragments the user can grep for.
 *
 * Phase 8 will add a telemetry emit here for audit-row-click. No emit yet.
 */
import { useId } from "react";

import { SWITCH_TAB_EVENT } from "@/components/common/CommandPalette";
import { useSessionStore } from "../../stores/sessionStore";
import type { ReadinessRow } from "../../types/api";

export interface ReadinessRowDetailProps {
  row: ReadinessRow;
  onClose: () => void;
}

export function ReadinessRowDetail({ row, onClose }: ReadinessRowDetailProps) {
  const compositionState = useSessionStore((s) => s.compositionState);
  const selectNode = useSessionStore((s) => s.selectNode);
  const labelId = useId();

  const nodeIds = new Set(compositionState?.nodes.map((n) => n.id) ?? []);

  function handleJump(componentId: string) {
    selectNode(componentId);
    window.dispatchEvent(new CustomEvent(SWITCH_TAB_EVENT, { detail: "spec" }));
    // Phase 8 deferral: emit telemetry here.
    onClose();
  }

  return (
    <div
      role="dialog"
      aria-labelledby={labelId}
      aria-modal="false"
      className="readiness-row-detail"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          onClose();
        }
      }}
    >
      <header className="readiness-row-detail-header">
        <h3 id={labelId} className="readiness-row-detail-title">
          {row.label}
        </h3>
        <button
          type="button"
          className="readiness-row-detail-close"
          onClick={onClose}
          aria-label="Close detail"
        >
          ×
        </button>
      </header>

      <p className="readiness-row-detail-summary">{row.summary}</p>

      {row.detail && (
        <pre className="readiness-row-detail-body">{row.detail}</pre>
      )}

      {row.component_ids.length > 0 && (
        <section
          aria-label="Components implicated"
          className="readiness-row-detail-components"
        >
          <h4 className="readiness-row-detail-components-heading">Components</h4>
          <ul className="readiness-row-detail-components-list">
            {row.component_ids.map((id) => {
              const resolvable = nodeIds.has(id);
              return (
                <li key={id}>
                  {resolvable ? (
                    <button
                      type="button"
                      className="btn readiness-row-detail-jump-btn"
                      onClick={() => handleJump(id)}
                      aria-label={`Jump to ${id}`}
                    >
                      Jump to {id}
                    </button>
                  ) : (
                    <span className="readiness-row-detail-component-id">{id}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
