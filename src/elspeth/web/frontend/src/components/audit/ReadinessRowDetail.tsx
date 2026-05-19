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
import { useEffect, useId, useRef } from "react";

import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
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
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);

  const nodeIds = new Set(compositionState?.nodes.map((n) => n.id) ?? []);

  // P0.4(b): on mount, move focus to the Close button. Close is
  // always present in this drawer; Jump may be absent when
  // component_ids is empty, so Close is the safer focus target.
  // Without an explicit mount-time focus, the Escape handler bound
  // on the root <div role="dialog"> would not fire if focus stays
  // on a sibling element outside the dialog tree.
  useEffect(() => {
    closeBtnRef.current?.focus();
  }, []);

  function handleJump(componentId: string) {
    selectNode(componentId);
    // P0.3: GraphModal is mounted unconditionally at App.tsx near the
    // app root, so this CustomEvent always reaches its listener — the
    // fire-and-forget shape is safe under current architecture. If a
    // future change conditionally unmounts GraphModal, this dispatch
    // would silently no-op; that change must add a guard here (track
    // listener presence in graphStore or open the modal via a store
    // action before dispatching the selection event).
    window.dispatchEvent(new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
    // P0.2: do NOT close the drawer here. The user clicked Jump to
    // see the highlighted component IN context; the drawer must
    // remain open so they can verify the target. Dismissal is
    // explicit (Escape or the Close button).
    // Phase 8 deferral: emit telemetry here.
  }

  return (
    <div
      role="dialog"
      aria-labelledby={labelId}
      aria-modal="false"
      tabIndex={-1}
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
          ref={closeBtnRef}
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
        // P0.4(c): prose, not code. <pre> announces as preformatted /
        // code in screen readers; the row detail is narrative text
        // with embedded linebreaks. whiteSpace: pre-line preserves
        // the linebreaks while keeping <p>'s prose semantics.
        <p
          className="readiness-row-detail-body"
          style={{ whiteSpace: "pre-line" }}
        >
          {row.detail}
        </p>
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
