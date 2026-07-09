import { useId } from "react";
import type { WireStageData } from "@/types/guided";
import { focusAcknowledgementCard } from "../AcknowledgementCard";
import { stepLabelForPlugin } from "../interpretationStepLabel";

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
  from: string;
  to: string;
  label: string;
  satisfied: boolean | null;
  missing_fields: string[];
}

function pairKey(from: string, to: string): string {
  return `${from}\u0000${to}`;
}

function branchConnectionLabels(
  branches: string[] | Record<string, string> | null,
): string[] {
  if (branches === null) return [];
  if (Array.isArray(branches)) return branches;
  return Object.values(branches);
}

export function reconstructWireEdges(data: WireStageData): WireEdge[] {
  const consumerByLabel = new Map<string, string>();
  for (const node of data.topology.nodes) {
    if (node.node_type === "coalesce") {
      for (const label of branchConnectionLabels(node.branches)) {
        consumerByLabel.set(label, node.id);
      }
    } else if (node.input !== null) {
      consumerByLabel.set(node.input, node.id);
    }
  }
  for (const output of data.topology.outputs) {
    consumerByLabel.set(output.sink_name, output.id);
  }

  const contractByPair = new Map(
    data.edge_contracts.map((contract) => [
      pairKey(contract.from, contract.to),
      contract,
    ]),
  );

  const edges: WireEdge[] = [];
  const pushEdge = (from: string, label: string | null) => {
    if (label === null) return;
    const to = consumerByLabel.get(label);
    if (to === undefined) return;
    const contract = contractByPair.get(pairKey(from, to));
    edges.push({
      from,
      to,
      label,
      satisfied: contract?.satisfied ?? null,
      missing_fields: contract?.missing_fields ?? [],
    });
  };

  for (const source of Object.values(data.topology.sources)) {
    pushEdge(source.id, source.on_success);
    pushEdge(source.id, source.on_validation_failure);
  }
  for (const node of data.topology.nodes) {
    pushEdge(node.id, node.on_success);
    pushEdge(node.id, node.on_error);
    for (const label of Object.values(node.routes ?? {})) {
      pushEdge(node.id, label);
    }
    for (const label of node.fork_to ?? []) {
      pushEdge(node.id, label);
    }
  }
  for (const output of data.topology.outputs) {
    pushEdge(output.id, output.on_write_failure);
  }

  return edges;
}

export interface WireStageTurnProps {
  data: WireStageData;
  onConfirm: () => void;
  confirmDisabled: boolean;
  /** Explicit "Ask advisor" re-ask. Spends one sign-off pass; rendered only on
   *  the REVISE outcome (a FLAG with budget remaining). */
  onAskAdvisor?: () => void;
  /** Exit to freeform — the always-available escape on every flag/block/unknown
   *  outcome so the wire stage is never a dead-end. */
  onExitToFreeform?: () => void;
  /** Complete WITHOUT sign-off. The server honours this ONLY when the outcome is
   *  escape_unavailable (advisor unreachable + budget exhausted), so it is
   *  rendered nowhere else — a FLAG can never be acknowledged into a bypass. */
  onCompleteWithoutSignoff?: () => void;
  /** Pending acknowledgements blocking this confirm. Each renders as a named
   *  jump link under the disabled button (scrolls to + focuses the card). */
  pendingAcknowledgements?: WireBlockerLink[];
  /** Client-known validation blockers (the persisted composition is invalid).
   *  Non-empty DISABLES confirm — a confirm the server must reject is never
   *  offered as a live button (elspeth-3b35abf148 variant 3, client side). */
  invalidChainIssues?: string[];
}

/**
 * Human names for every topology entity, keyed by its internal id
 * (elspeth-016f463ff0: `guided_xform_0` / `output:main` must not reach
 * first-run copy). Transforms reuse the acknowledgement cards' step-label
 * mapping (stepLabelForPlugin, e.g. llm → "Summarise") so the wiring list and
 * the cards name a step identically; a plugin-less structural node falls back
 * to its node_type through the same humaniser. Sources read as "Source" (or
 * "<name> source" for a named source); outputs read as "<sink_name> output" —
 * both names are user-meaningful, not internal ids.
 */
export function buildEntityNames(data: WireStageData): Map<string, string> {
  const names = new Map<string, string>();
  const sourceEntries = Object.entries(data.topology.sources);
  for (const [name, source] of sourceEntries) {
    names.set(
      source.id,
      sourceEntries.length === 1 ? "Source" : `${name} source`,
    );
  }
  for (const node of data.topology.nodes) {
    names.set(node.id, `${stepLabelForPlugin(node.plugin ?? node.node_type)} step`);
  }
  for (const output of data.topology.outputs) {
    names.set(output.id, `${output.sink_name} output`);
  }
  return names;
}

/** Plain-language connection state: "(contract unchecked)" is engineer
 *  register; a first-run user reads "not yet checked". */
function edgeStatus(edge: WireEdge): string {
  if (edge.satisfied === true) return "connected";
  if (edge.satisfied === false) return "not connected correctly";
  return "not yet checked";
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
  return `${edge.from} -> ${edge.to} via ${edge.label} ${status}${missing}`;
}

function warningText(warning: Record<string, unknown>): string {
  const message = warning.message;
  if (typeof message === "string") return message;
  return JSON.stringify(warning);
}

// Short in-context guidance for the outcomes whose advisor_findings alone do not
// tell the user what to do next. The findings block carries the "why"; these
// lines carry the "what now". Public-service register, no marketing.
const BLOCKED_UNAVAILABLE_COPY =
  "Advisor sign-off could not run, so this pipeline cannot be completed here. Exit to freeform to finish it manually.";
const ESCAPE_UNAVAILABLE_COPY =
  "The advisor is unreachable and the review budget is exhausted. You can complete without sign-off (recorded as advisor-unreachable) or exit to freeform.";
const UNKNOWN_OUTCOME_COPY =
  "This wiring review returned a status this version does not recognise. Exit to freeform to continue.";

export function WireStageTurn({
  data,
  onConfirm,
  confirmDisabled,
  onAskAdvisor,
  onExitToFreeform,
  onCompleteWithoutSignoff,
  pendingAcknowledgements,
  invalidChainIssues,
}: WireStageTurnProps) {
  const edges = reconstructWireEdges(data);
  const entityNames = buildEntityNames(data);
  const nameFor = (id: string): string => entityNames.get(id) ?? id;
  const outcome = data.signoff_outcome;
  const passesRemaining = data.passes_remaining;
  const blockersId = useId();

  const acknowledgements = pendingAcknowledgements ?? [];
  const chainIssues = invalidChainIssues ?? [];
  const hasBlockers = acknowledgements.length > 0 || chainIssues.length > 0;

  const confirmButton = (
    <button
      type="button"
      className="guided-turn-primary"
      onClick={onConfirm}
      disabled={confirmDisabled || chainIssues.length > 0}
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
      {chainIssues.length > 0 && (
        <>
          <p className="wire-stage__blockers-heading">
            The pipeline isn't ready to confirm:
          </p>
          <ul className="wire-stage__blockers-list wire-stage__blockers-list--issues">
            {chainIssues.map((issue, index) => (
              <li key={index}>{issue}</li>
            ))}
          </ul>
        </>
      )}
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

  // Action area gated on the sign-off outcome. Every branch is closed and the
  // default is a safe escape (never an empty area, never a bypassing confirm on
  // a flag/block). See the outcome -> affordance contract in the slice-B plan.
  function renderActions() {
    switch (outcome) {
      // Initial turn (not yet checked) OR a re-emitted CLEAN verdict. The
      // backend never auto-completes on a clean re-emit, so confirm stays the
      // actionable finalize step — not a dead-end.
      case undefined:
      case "complete":
        return (
          <>
            <div className="wire-stage__actions">
              {confirmButton}
              {chainIssues.length > 0 && onExitToFreeform !== undefined
                ? exitButton
                : null}
            </div>
            {blockersPanel}
          </>
        );

      // FLAGGED, budget remains: re-ask (with disclosed cost) or exit. No bare
      // confirm — that is the silent repeat-burn this slice removes.
      case "revise": {
        const costCopy =
          passesRemaining !== undefined ? ` (spends 1 of ${passesRemaining})` : "";
        return (
          <div className="wire-stage__actions">
            <button
              type="button"
              className="guided-turn-secondary"
              onClick={() => onAskAdvisor?.()}
              // Disable-at-0 is a defensive guard: REVISE is only emitted while
              // budget remains, so 0 is live-unreachable here.
              disabled={passesRemaining === 0}
            >
              {`Ask advisor${costCopy}`}
            </button>
            {exitButton}
          </div>
        );
      }

      // FLAGGED, exhausted: fail-closed terminal. No budget-burning button.
      case "blocked_flagged":
        return <div className="wire-stage__actions">{exitButton}</div>;

      // Service/budget missing: explanation + exit.
      case "blocked_unavailable":
        return (
          <>
            <p className="wire-stage__guidance">{BLOCKED_UNAVAILABLE_COPY}</p>
            <div className="wire-stage__actions">{exitButton}</div>
          </>
        );

      // Advisor unreachable + exhausted: the ONLY outcome that offers
      // complete-without-sign-off (server-enforced; defense-in-depth here).
      case "escape_unavailable":
        return (
          <>
            <p className="wire-stage__guidance">{ESCAPE_UNAVAILABLE_COPY}</p>
            <div className="wire-stage__actions">
              <button
                type="button"
                className="guided-turn-primary"
                onClick={() => onCompleteWithoutSignoff?.()}
              >
                Complete without sign-off
              </button>
              {exitButton}
            </div>
          </>
        );

      // Any unknown/forward outcome: never a dead-end, never a bypassing confirm.
      default:
        return (
          <>
            <p className="wire-stage__guidance">{UNKNOWN_OUTCOME_COPY}</p>
            <div className="wire-stage__actions">{exitButton}</div>
          </>
        );
    }
  }

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

      {/* Human step names + plain-language connection state
          (elspeth-016f463ff0). The raw ids / connection labels stay available
          verbatim behind the Technical details expander below. */}
      <ul className="wire-stage__edges">
        {edges.map((edge) => (
          <li
            key={`${edge.from}\u0000${edge.label}\u0000${edge.to}`}
            aria-label={`${nameFor(edge.from)} to ${nameFor(edge.to)}`}
          >
            <span>{nameFor(edge.from)}</span>
            <span aria-hidden="true">{" → "}</span>
            <span>{nameFor(edge.to)}</span>
            <span aria-hidden="true">{" — "}</span>
            <span>{edgeStatus(edge)}</span>
            {edge.missing_fields.length > 0 ? (
              <span> Missing fields: {edge.missing_fields.join(", ")}</span>
            ) : null}
          </li>
        ))}
      </ul>
      {edges.length > 0 ? (
        <details className="wire-stage__raw">
          <summary>Technical details</summary>
          <pre className="wire-stage__raw-text">
            {edges.map(rawEdgeRow).join("\n")}
          </pre>
        </details>
      ) : null}

      {data.advisor_findings !== undefined ? (
        <div className="wire-stage__findings">
          <h4>Advisor review</h4>
          <p>{data.advisor_findings}</p>
        </div>
      ) : null}

      {renderActions()}
    </div>
  );
}
