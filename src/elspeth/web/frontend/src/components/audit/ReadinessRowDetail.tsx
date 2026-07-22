/**
 * ReadinessRowDetail (Phase 2C)
 *
 * Drawer/popover content for one row of the audit-readiness panel.
 *
 * For the `validation` row it re-humanises the structured findings through the
 * SAME humaniser the rail strip uses (humaniseValidationMessage) so that
 * engine-grade dumps — raw pydantic "Field required" traces, schema-contract
 * violations, interpretation-review-pending strings — never render verbatim on
 * this novice surface (elspeth-901a404926). Each finding shows its plain
 * headline; any raw text is preserved behind a "Technical details" disclosure
 * for the engineer read. Other rows render their backend detail string
 * (multi-line preserved) unchanged.
 *
 * Jump-to-component buttons name the target by its plain step phrase ("the
 * 'rate each row' step") rather than the internal node id (e.g.
 * `guided_xform_1`), which is itself a lingo leak. Unresolvable ids are shown
 * as plain text — they may refer to source/sink names the user can grep for.
 *
 * Phase 8 will add a telemetry emit here for audit-row-click. No emit yet.
 */
import { useEffect, useId, useMemo, useRef } from "react";

import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "../../stores/sessionStore";
import type { ReadinessRow, ValidationError } from "../../types/api";
import { resolveNodePlugin, stepLabelForPlugin } from "../chat/interpretationStepLabel";
import {
  humaniseValidationMessage,
  makePhraseFor,
} from "@/lib/validationHumaniser";
import { UNKNOWN_COMPONENT_PHRASE } from "../chat/guided/pipelineGloss";

export interface ReadinessRowDetailProps {
  row: ReadinessRow;
  /** For the `validation` row: the structured findings, re-humanised here so
   *  engine-grade dumps never render raw on this novice surface. Absent for
   *  every other row (which carry display-ready backend prose). */
  validationErrors?: readonly ValidationError[];
  onClose: () => void;
}

export function ReadinessRowDetail({ row, validationErrors, onClose }: ReadinessRowDetailProps) {
  const compositionState = useSessionStore((s) => s.compositionState);
  const selectNode = useSessionStore((s) => s.selectNode);
  const labelId = useId();
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);

  const nodeIds = new Set(compositionState?.nodes.map((n) => n.id) ?? []);

  // Shared with the rail strip: component_id → plain phrase, and the
  // acknowledgement-card step label (for the review-pending humaniser case).
  // Memoised (elspeth-40d6efac2b): avoids rebuilding + re-tokenising the
  // phrase map on every render of this drawer.
  const phraseFor = useMemo(() => makePhraseFor(compositionState), [compositionState]);
  const stepLabelFor = (componentId: string): string | null => {
    const plugin = resolveNodePlugin(compositionState, componentId);
    return plugin === null ? null : stepLabelForPlugin(plugin);
  };

  // Humanise the validation row's findings; leave other rows' prose untouched.
  const humanisedFindings =
    validationErrors && validationErrors.length > 0
      ? validationErrors.map((err) =>
          humaniseValidationMessage(err.message, phraseFor, stepLabelFor),
        )
      : null;
  const technicalDetails =
    humanisedFindings?.map((f) => f.raw).filter((raw): raw is string => raw !== null) ?? [];

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

  // Name a jump target by its plain step phrase, never the internal id. A
  // resolvable node id maps through the gloss; the generic fallback reads
  // "this step" rather than echoing an id.
  function jumpTargetLabel(componentId: string): string {
    const phrase = phraseFor(componentId);
    return phrase === UNKNOWN_COMPONENT_PHRASE ? "this step" : `the "${phrase}" step`;
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

      {humanisedFindings !== null ? (
        <>
          {/* Plain-language findings, one per line. The raw engine text (if
              any) moves behind the Technical-details disclosure below. */}
          <ul className="readiness-row-detail-findings">
            {humanisedFindings.map((finding, index) => (
              <li key={index} className="readiness-row-detail-body">
                {finding.headline}
              </li>
            ))}
          </ul>
          {technicalDetails.length > 0 && (
            <details className="readiness-row-detail-raw">
              <summary>Technical details</summary>
              <pre className="readiness-row-detail-raw-text">
                {technicalDetails.join("\n\n")}
              </pre>
            </details>
          )}
        </>
      ) : (
        row.detail && (
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
        )
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
                      aria-label={`Jump to ${jumpTargetLabel(id)}`}
                    >
                      Jump to {jumpTargetLabel(id)}
                    </button>
                  ) : (
                    // Not a composition node: a source/sink name or YAML
                    // fragment. Show its plain phrase when the gloss resolves
                    // it; otherwise keep the raw name (the user can grep for
                    // it) rather than collapsing to a generic "this step".
                    <span className="readiness-row-detail-component-id">
                      {phraseFor(id) === UNKNOWN_COMPONENT_PHRASE
                        ? id
                        : `the "${phraseFor(id)}" step`}
                    </span>
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
