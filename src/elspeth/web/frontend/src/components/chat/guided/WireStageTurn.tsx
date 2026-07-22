import { useId, useState } from "react";
import type { GuidedEditTarget, ProposalFlow, ProposalNodeBehavior, WireRowCardinality, WireStageData } from "@/types/guided";
import { focusAcknowledgementCard } from "../AcknowledgementCard";
import { stepLabelForPlugin } from "../interpretationStepLabel";
import { WireReviewList } from "./WireReviewList";

/**
 * One named blocker behind a disabled "Confirm wiring" — a pending
 * acknowledgement the user must resolve first. `id` is the interpretation
 * event id; clicking the entry scrolls to + focuses the blocking card
 * (`focusAcknowledgementCard`). Part of the shared dead-end fix: a disabled
 * primary action must name each pending item and give a direct path to it
 * (elspeth-3b35abf148 variant 1).
 */
export interface WireBlockerLink {
  id: string;
  label: string;
}

export interface WireEdge {
  stable_id: string;
  from: string;
  to: string;
  label: string;
  flow: ProposalFlow;
  satisfied: boolean | null;
  missing_fields: string[];
}

export function reconstructWireEdges(data: WireStageData): WireEdge[] {
  return data.connections.map((connection) => ({
    stable_id: connection.stable_id,
    from: connection.from_endpoint.stable_id,
    to: connection.to_endpoint.kind === "discard" ? "discard" : connection.to_endpoint.stable_id,
    label: connection.flow.kind,
    flow: connection.flow,
    satisfied: connection.schema_contract?.satisfied ?? null,
    missing_fields: connection.schema_contract?.missing_fields ?? [],
  }));
}

export interface WireStageTurnProps {
  data: WireStageData;
  onConfirm: () => void;
  confirmDisabled: boolean;
  /** Exit to freeform remains available without changing proposal authority. */
  onExitToFreeform?: () => void;
  /** Pending acknowledgements blocking this confirm. Each renders as a named
   *  jump link under the disabled button (scrolls to + focuses the card). */
  pendingAcknowledgements?: WireBlockerLink[];
  /** Client-known validation blockers (the persisted composition is invalid).
   *  Non-empty DISABLES confirm — a confirm the server must reject is never
   *  offered as a live button (elspeth-3b35abf148 variant 3, client side). */
  validationIssues?: string[];
  onCorrect?: (target: GuidedEditTarget, feedback: string) => void;
}

/**
 * Human names for every topology entity, keyed by its internal id
 * (elspeth-016f463ff0: internal node and output identifiers must not reach
 * first-run copy). Transforms reuse the acknowledgement cards' step-label
 * mapping (stepLabelForPlugin, e.g. llm → "Summarise") so the wiring list and
 * the cards name a step identically; a plugin-less structural node falls back
 * to its node_type through the same humaniser. Sources read as "Source" (or
 * "<name> source" for a named source); outputs read as "<sink_name> output" —
 * both names are user-meaningful, not internal ids.
 */
export function buildEntityNames(data: WireStageData): Map<string, string> {
  const names = new Map<string, string>();
  for (const source of data.sources) {
    names.set(source.stable_id, source.label);
  }
  for (const node of data.nodes) {
    names.set(node.stable_id, `${node.label} (${stepLabelForPlugin(node.plugin ?? node.node_type)})`);
  }
  for (const output of data.outputs) {
    names.set(output.stable_id, output.label);
  }
  names.set("discard", "Discard");
  return names;
}

/** Plain-language connection state: "(contract unchecked)" is engineer
 *  register; a first-run user reads "not yet checked". */
function edgeStatus(edge: WireEdge): string {
  if (edge.satisfied === true) return "connected";
  if (edge.satisfied === false) return "not connected correctly";
  return "not yet checked";
}

/** Chip variant for a route's contract state (WireReviewList status). */
function edgeStatusKind(edge: WireEdge): "connected" | "warning" | "unchecked" {
  if (edge.satisfied === true) return "connected";
  if (edge.satisfied === false) return "warning";
  return "unchecked";
}

/**
 * One-line route roll-up ("9 routes — 2 connected, 7 not yet checked") so the
 * list's overall state reads once, instead of every row trailing the same
 * "— not yet checked" clause (the operator-reported debug-dump read).
 * Zero-count categories are elided.
 */
export function routesSummaryText(edges: WireEdge[]): string {
  const connected = edges.filter((edge) => edge.satisfied === true).length;
  const broken = edges.filter((edge) => edge.satisfied === false).length;
  const unchecked = edges.length - connected - broken;
  const parts = [
    ...(connected > 0 ? [`${connected} connected`] : []),
    ...(broken > 0 ? [`${broken} not connected correctly`] : []),
    ...(unchecked > 0 ? [`${unchecked} not yet checked`] : []),
  ];
  const heading = edges.length === 1 ? "1 route" : `${edges.length} routes`;
  return parts.length > 0 ? `${heading} — ${parts.join(", ")}` : heading;
}

function humanToken(value: string): string {
  return value.replace(/_/g, " ");
}

function cardinalityText(cardinality: WireRowCardinality): string {
  const expected = cardinality.expected_output_count === null
    ? ""
    : ` (expected ${cardinality.expected_output_count})`;
  return `Cardinality: ${humanToken(cardinality.input)} → ${humanToken(cardinality.output)}${expected}`;
}

function fieldsText(label: "Required" | "Guaranteed", fields: string[]): string {
  return `${label} fields: ${fields.length > 0 ? fields.join(", ") : "none"}`;
}

function flowText(flow: ProposalFlow): string {
  switch (flow.kind) {
    case "source_success":
      return flow.branch === null ? "Source success" : `Source success on ${flow.branch}`;
    case "source_validation_failure":
      return "Source validation failure";
    case "node_success":
      return flow.branch === null ? "Node success" : `Node success on ${flow.branch}`;
    case "node_error":
      return "Node failure";
    case "gate_route":
      return flow.branch === null ? `Gate route ${flow.route}` : `Gate route ${flow.route} on ${flow.branch}`;
    case "gate_fork":
      return `Gate fork ${flow.routes.join(", ")} as ${flow.branch}`;
    case "queue_continue":
      return flow.branch === null ? "Queue continuation" : `Queue continuation on ${flow.branch}`;
    case "coalesce_success":
      return flow.branch === null ? "Coalesce success" : `Coalesce success on ${flow.branch}`;
    case "output_write_failure":
      return "Output write failure";
  }
}

function behaviorDetails(behavior: ProposalNodeBehavior): string[] {
  switch (behavior.kind) {
    case "transform":
      return ["Policy: transform each input row"];
    case "queue":
      return ["Policy: continue queued items individually"];
    case "gate":
      return [
        `Routes: ${behavior.route_aliases.join(", ")}`,
        ...behavior.fork_branches.map((fork) => `Fork branch ${fork.branch}: ${fork.routes.join(", ")}`),
      ];
    case "aggregation":
      return [
        `Triggers: ${behavior.trigger_kinds.join(", ")}`,
        ...(behavior.count === null ? [] : [`Count: ${behavior.count}`]),
        ...(behavior.timeout_seconds === null ? [] : [`Timeout: ${behavior.timeout_seconds} seconds`]),
        `Output mode: ${humanToken(behavior.output_mode)}`,
        ...(behavior.expected_output_count === null ? [] : [`Expected output count: ${behavior.expected_output_count}`]),
      ];
    case "coalesce":
      return [
        `Branches: ${behavior.branch_aliases.join(", ")}`,
        `Policy: ${humanToken(behavior.policy)}`,
        `Merge: ${humanToken(behavior.merge)}`,
      ];
  }
}

/** The verbatim engineer-grade row, preserved behind the Technical details
 *  expander for operators (same idiom as the validation summary's raw dump). */
function rawEdgeRow(edge: WireEdge): string {
  const status =
    edge.satisfied === true
      ? "(connected)"
      : edge.satisfied === false
        ? "(not satisfied)"
        : "(contract unchecked)";
  const missing =
    edge.missing_fields.length > 0
      ? ` Missing fields: ${edge.missing_fields.join(", ")}`
      : "";
  return `[${edge.stable_id}] ${edge.from} -> ${edge.to} via ${flowText(edge.flow)} ${status}${missing}`;
}

function warningText(warning: Record<string, unknown>): string {
  const message = warning.message;
  if (typeof message === "string") return message;
  return JSON.stringify(warning);
}

export function WireStageTurn({
  data,
  onConfirm,
  confirmDisabled,
  onExitToFreeform,
  pendingAcknowledgements,
  validationIssues,
  onCorrect,
}: WireStageTurnProps) {
  const edges = reconstructWireEdges(data);
  const entityNames = buildEntityNames(data);
  const nameFor = (id: string): string => entityNames.get(id) ?? id;
  const blockersId = useId();
  const routesHeadingId = useId();
  const correctionSelectId = useId();
  const correctionFeedbackId = useId();
  const correctionTargets: Array<{ target: GuidedEditTarget; label: string }> = [
    ...data.sources.map((source) => ({ target: { kind: "source" as const, stable_id: source.stable_id }, label: source.label })),
    ...data.nodes.map((node) => ({ target: { kind: "node" as const, stable_id: node.stable_id }, label: node.label })),
    ...data.connections.map((connection, index) => ({
      target: { kind: "edge" as const, stable_id: connection.stable_id },
      label: `Route ${index + 1}`,
    })),
    ...data.outputs.map((output) => ({ target: { kind: "output" as const, stable_id: output.stable_id }, label: output.label })),
  ];
  const [correctionTarget, setCorrectionTarget] = useState(correctionTargets[0]?.target.stable_id ?? "");
  const [correctionFeedback, setCorrectionFeedback] = useState("");

  const acknowledgements = pendingAcknowledgements ?? [];
  const blockingValidationIssues = validationIssues ?? [];
  const hasBlockers =
    acknowledgements.length > 0 || blockingValidationIssues.length > 0 || data.blockers.length > 0;

  const confirmButton = (
    <button
      type="button"
      className="guided-turn-primary"
      onClick={onConfirm}
      disabled={confirmDisabled || !data.can_confirm || data.blockers.length > 0 || blockingValidationIssues.length > 0}
      aria-describedby={hasBlockers ? blockersId : undefined}
    >
      Confirm wiring
    </button>
  );

  // Named-blocker panel: renders directly under the (possibly disabled)
  // confirm button so the unblock path is never buried in another column
  // (elspeth-3b35abf148 variant 1). Acknowledgement entries are jump links to
  // the blocking card; validation issues are the concrete "what's invalid".
  const blockersPanel = hasBlockers ? (
    <div id={blockersId} className="wire-stage__blockers">
      {acknowledgements.length > 0 && (
        <>
          <p className="wire-stage__blockers-heading">
            {acknowledgements.length === 1
              ? "1 acknowledgement pending — resolve it to enable Confirm wiring:"
              : `${acknowledgements.length} acknowledgements pending — resolve each to enable Confirm wiring:`}
          </p>
          <ul className="wire-stage__blockers-list">
            {acknowledgements.map((blocker) => (
              <li key={blocker.id}>
                <button
                  type="button"
                  className="wire-stage__blocker-link"
                  onClick={() => focusAcknowledgementCard(blocker.id)}
                >
                  {blocker.label}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
      {blockingValidationIssues.length > 0 && (
        <>
          <p className="wire-stage__blockers-heading">
            The pipeline isn't ready to confirm:
          </p>
          <ul className="wire-stage__blockers-list wire-stage__blockers-list--issues">
            {blockingValidationIssues.map((issue, index) => (
              <li key={index}>{issue}</li>
            ))}
          </ul>
        </>
      )}
      {data.blockers.length > 0 ? (
        <ul className="wire-stage__blockers-list wire-stage__blockers-list--issues">
          {data.blockers.map((blocker, index) => <li key={index}>{warningText(blocker)}</li>)}
        </ul>
      ) : null}
    </div>
  ) : null;

  const exitButton = (
    <button
      type="button"
      className="guided-turn-secondary"
      onClick={() => onExitToFreeform?.()}
    >
      Exit to freeform
    </button>
  );

  return (
    <div className="guided-turn wire-stage">
      <h3>Review wiring</h3>

      {data.warnings.length > 0 ? (
        <ul className="wire-stage__warnings">
          {data.warnings.map((warning, index) => (
            <li key={index}>{warningText(warning)}</li>
          ))}
        </ul>
      ) : null}

      <section className="wire-stage__components" aria-label="Reviewed components">
        {data.sources.length > 0 ? (
          <div>
            <h4>Sources</h4>
            <ul>
              {data.sources.map((source) => (
                <li key={source.stable_id}>
                  <strong>{source.label}</strong> <span>({source.plugin})</span>
                  <p>{cardinalityText(source.row_cardinality)}</p>
                  <p>{fieldsText("Guaranteed", source.guaranteed_fields)}</p>
                  <p>Validation failure: {humanToken(source.on_validation_failure)}</p>
                  <details><summary>Stable ID</summary><code>{source.stable_id}</code></details>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {data.nodes.length > 0 ? (
          <div>
            <h4>Processing nodes</h4>
            <ul>
              {data.nodes.map((node) => (
                <li key={node.stable_id}>
                  <strong>{node.label}</strong> <span>({node.plugin ?? humanToken(node.node_type)})</span>
                  <p>{cardinalityText(node.row_cardinality)}</p>
                  <p>{fieldsText("Required", node.required_fields)}</p>
                  <p>{fieldsText("Guaranteed", node.guaranteed_fields)}</p>
                  {behaviorDetails(node.behavior).map((detail) => <p key={detail}>{detail}</p>)}
                  {node.structured_output_fields.length > 0 ? (
                    <ul aria-label={`${node.label} structured output fields`}>
                      {node.structured_output_fields.map((field) => (
                        <li key={`${field.query}:${field.field}`}>
                          {`${field.field} (${field.type}) from ${field.query}${
                            field.enum_values.length > 0 ? `; values: ${field.enum_values.join(", ")}` : ""
                          }`}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                  <details><summary>Stable ID</summary><code>{node.stable_id}</code></details>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {data.outputs.length > 0 ? (
          <div>
            <h4>Outputs</h4>
            <ul>
              {data.outputs.map((output) => (
                <li key={output.stable_id}>
                  <strong>{output.label}</strong> <span>({output.plugin})</span>
                  <p>{fieldsText("Required", output.required_fields)}</p>
                  <p>Write failure: {humanToken(output.on_write_failure)}</p>
                  <p>Schema mode: {humanToken(output.business_schema.mode)}</p>
                  {output.business_schema.fields.length > 0 ? (
                    <ul aria-label={`${output.label} business schema fields`}>
                      {output.business_schema.fields.map((field) => (
                        <li key={field.name}>
                          {`${field.name}: ${field.type} — ${field.required ? "required" : "optional"}, ${
                            field.nullable ? "nullable" : "non-null"
                          }`}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                  <p>{fieldsText("Guaranteed", output.business_schema.guaranteed_fields)}</p>
                  <p>{fieldsText("Required", output.business_schema.required_fields)}</p>
                  <details><summary>Stable ID</summary><code>{output.stable_id}</code></details>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      {/* Human step names + plain-language connection state
          (elspeth-016f463ff0). Status renders as a per-row chip plus one
          count roll-up (never per-row trailing "— …" prose) and is folded
          into each row's accessible name — the li aria-label overrides its
          text content, so a chip outside it is invisible to screen readers.
          The raw ids / connection labels stay available verbatim behind the
          Technical details expander. */}
      {edges.length > 0 ? (
        <section className="wire-stage__routes" aria-labelledby={routesHeadingId}>
          <h4 id={routesHeadingId}>Routes</h4>
          <p className="wire-stage__routes-summary">{routesSummaryText(edges)}</p>
          <WireReviewList
            className="wire-stage__edges"
            ariaLabel="Wiring routes"
            items={edges.map((edge) => ({
              id: `${edge.from}\u0000${edge.label}\u0000${edge.to}`,
              from: nameFor(edge.from),
              to: nameFor(edge.to),
              summary: flowText(edge.flow),
              status: edgeStatusKind(edge),
              detail:
                edge.missing_fields.length > 0
                  ? `Missing fields: ${edge.missing_fields.join(", ")}`
                  : null,
              ariaLabel: `${nameFor(edge.from)} to ${nameFor(edge.to)} — ${edgeStatus(edge)}`,
            }))}
          />
          <details className="wire-stage__raw">
            <summary>Technical details</summary>
            <pre className="wire-stage__raw-text">
              {edges.map(rawEdgeRow).join("\n")}
            </pre>
          </details>
        </section>
      ) : null}

      {onCorrect !== undefined && correctionTargets.length > 0 ? (
        <form
          className="wire-stage__correction"
          onSubmit={(event) => {
            event.preventDefault();
            const selected = correctionTargets.find((item) => item.target.stable_id === correctionTarget);
            if (selected !== undefined && correctionFeedback.trim().length > 0) {
              onCorrect(selected.target, correctionFeedback.trim());
            }
          }}
        >
          {/* Proper form-group idiom (operator-reported soup: wrapping labels
              overlapped the bare native select and the tiny off-baseline
              textarea). Explicit for/id association + the schema form's field
              classes; keep the accessible names verbatim ("Component" /
              "What should change?"). */}
          <h4>Request a wiring correction</h4>
          <div className="guided-schema-field-row">
            <label className="guided-schema-label" htmlFor={correctionSelectId}>
              Component
            </label>
            <select
              id={correctionSelectId}
              className="guided-schema-select"
              value={correctionTarget}
              onChange={(event) => setCorrectionTarget(event.target.value)}
            >
              {correctionTargets.map((item) => (
                <option key={`${item.target.kind}:${item.target.stable_id}`} value={item.target.stable_id}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>
          <div className="guided-schema-field-row">
            <label className="guided-schema-label" htmlFor={correctionFeedbackId}>
              What should change?
            </label>
            <textarea
              id={correctionFeedbackId}
              className="wire-stage__correction-input"
              rows={2}
              value={correctionFeedback}
              maxLength={4096}
              onChange={(event) => setCorrectionFeedback(event.target.value)}
            />
          </div>
          <div className="wire-stage__correction-actions">
            <button type="submit" className="guided-turn-secondary" disabled={correctionFeedback.trim().length === 0}>
              Re-plan wiring
            </button>
          </div>
        </form>
      ) : null}

      <div className="wire-stage__actions">
        {confirmButton}
        {onExitToFreeform !== undefined ? exitButton : null}
      </div>
      {blockersPanel}
    </div>
  );
}
