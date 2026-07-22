import { useEffect, useMemo, useRef } from "react";

import type {
  GuidedEditTarget,
  GuidedProposalReviewState,
  GuidedProposalRetryAction,
  GuidedRespondAction,
  ProposalFlow,
  ProposalNodeBehavior,
  ProposePipelinePayload,
} from "@/types/guided";
import {
  ReadOnlyPipelineGraph,
  type ReadOnlyPipelineGraphEdge,
  type ReadOnlyPipelineGraphNode,
} from "./ReadOnlyPipelineGraph";
import { WireReviewList, type WireReviewItem } from "./WireReviewList";

interface ProposePipelineTurnProps {
  payload: ProposePipelinePayload;
  reviewState: GuidedProposalReviewState;
  onSubmit: (body: GuidedRespondAction) => void;
  disabled?: boolean;
  isTutorial?: boolean;
}

const DISCARD_NODE_ID = "guided-proposal-discard";

const BLOCKER_COPY: Record<ProposePipelinePayload["blockers"][number]["code"], string> = {
  pipeline_invalid: "The proposed pipeline has validation problems that must be revised.",
  policy_review_required: "A policy review is required before this pipeline can advance.",
  plugin_unavailable: "A required plugin is unavailable and must be replaced.",
  interpretation_required: "A pending interpretation must be resolved before wiring review.",
};

function proposalBindingMatches(
  payload: ProposePipelinePayload,
  state: GuidedProposalReviewState,
): boolean {
  return state.proposal_id === payload.proposal_id && state.draft_hash === payload.draft_hash;
}

function sameRetryAction(
  retained: GuidedProposalRetryAction,
  candidate: GuidedProposalRetryAction,
): boolean {
  if (retained.kind !== candidate.kind) return false;
  if (retained.kind !== "revise" || candidate.kind !== "revise") return true;
  return (
    retained.edit_target.kind === candidate.edit_target.kind &&
    retained.edit_target.stable_id === candidate.edit_target.stable_id
  );
}

function flowLabel(flow: ProposalFlow): string {
  switch (flow.kind) {
    case "source_success":
      return flow.branch === null ? "on source success" : `on source success in ${flow.branch}`;
    case "source_validation_failure":
      return "on validation failure";
    case "node_success":
      return flow.branch === null ? "on success" : `on success in ${flow.branch}`;
    case "node_error":
      return "on error";
    case "gate_route":
      return flow.branch === null
        ? `${flow.route} route`
        : `${flow.route} route in ${flow.branch}`;
    case "gate_fork":
      return `${flow.routes.join(" + ")} forks to ${flow.branch}`;
    case "queue_continue":
      return flow.branch === null ? "queue continues" : `queue continues in ${flow.branch}`;
    case "coalesce_success":
      return flow.branch === null ? "after join" : `after join in ${flow.branch}`;
    case "output_write_failure":
      return "on write failure";
  }
}

function behaviorSummary(behavior: ProposalNodeBehavior): string {
  switch (behavior.kind) {
    case "transform":
      return "Transforms each incoming item.";
    case "gate":
      return `Routes ${behavior.route_aliases.join(", ")}; ${behavior.fork_branches.length} fork branches.`;
    case "aggregation": {
      const triggers: string[] = [];
      if (behavior.count !== null) triggers.push(`count ${behavior.count}`);
      if (behavior.timeout_seconds !== null) triggers.push(`timeout ${behavior.timeout_seconds}s`);
      if (behavior.trigger_kinds.includes("condition")) triggers.push("condition");
      return `Collects until ${triggers.join(" or ")}; ${behavior.output_mode} output.`;
    }
    case "queue":
      return "Queue continues in sequence without correlating records.";
    case "coalesce":
      return `Joins ${behavior.branch_aliases.join(", ")} using ${behavior.policy} / ${behavior.merge}.`;
  }
}

function reviewStatusCopy(
  state: GuidedProposalReviewState,
  isCurrentBinding: boolean,
): { role: "status" | "alert"; message: string } | null {
  if (!isCurrentBinding) {
    return {
      role: "status",
      message: "The previous proposal became stale. Review this refreshed proposal before taking action.",
    };
  }
  switch (state.status) {
    case "active":
      return null;
    case "submitting":
      return { role: "status", message: "Submitting this proposal decision…" };
    case "reloading":
      return { role: "status", message: "This proposal changed. Reloading the authoritative proposal…" };
    case "stale":
      return {
        role: "status",
        message: "This proposal is stale and cannot be used. Ask the assistant to regenerate it.",
      };
    case "error":
      return { role: "alert", message: state.message };
  }
}

export function ProposePipelineTurn({
  payload,
  reviewState,
  onSubmit,
  disabled = false,
  isTutorial = false,
}: ProposePipelineTurnProps): JSX.Element {
  const statusRef = useRef<HTMLParagraphElement | null>(null);
  const labelById = useMemo(() => {
    const labels = new Map<string, string>();
    for (const source of payload.graph.sources) labels.set(source.stable_id, source.label);
    for (const node of payload.nodes) labels.set(node.stable_id, node.label);
    for (const output of payload.outputs) labels.set(output.stable_id, output.label);
    return labels;
  }, [payload]);
  const hasDiscard = payload.graph.edges.some((edge) => edge.to_endpoint.kind === "discard");
  const graphNodes = useMemo<ReadOnlyPipelineGraphNode[]>(() => [
    ...payload.graph.sources.map((source) => ({
      id: source.stable_id,
      label: source.label,
      kind: "source" as const,
      subtitle: source.plugin.id,
    })),
    ...payload.nodes.map((node) => ({
      id: node.stable_id,
      label: node.label,
      kind: node.node_type,
      subtitle: node.plugin?.id ?? null,
    })),
    ...payload.outputs.map((output) => ({
      id: output.stable_id,
      label: output.label,
      kind: "output" as const,
      subtitle: output.plugin.id,
    })),
    ...(hasDiscard
      ? [{ id: DISCARD_NODE_ID, label: "discard", kind: "discard" as const, subtitle: null }]
      : []),
  ], [hasDiscard, payload]);
  const graphEdges = useMemo<ReadOnlyPipelineGraphEdge[]>(() =>
    payload.graph.edges.map((edge) => {
      const targetId = edge.to_endpoint.kind === "discard"
        ? DISCARD_NODE_ID
        : edge.to_endpoint.stable_id;
      const from = labelById.get(edge.from_endpoint.stable_id) ?? edge.from_endpoint.stable_id;
      const to = edge.to_endpoint.kind === "discard"
        ? "discard"
        : (labelById.get(edge.to_endpoint.stable_id) ?? edge.to_endpoint.stable_id);
      return {
        id: edge.stable_id,
        source: edge.from_endpoint.stable_id,
        target: targetId,
        label: `${from} ${flowLabel(edge.flow)} → ${to}`,
        isError: ["source_validation_failure", "node_error", "output_write_failure"].includes(edge.flow.kind),
      };
    }), [labelById, payload.graph.edges]);
  const routeItems = useMemo<WireReviewItem[]>(() =>
    payload.graph.edges.map((edge) => ({
      id: edge.stable_id,
      from: labelById.get(edge.from_endpoint.stable_id) ?? edge.from_endpoint.stable_id,
      to: edge.to_endpoint.kind === "discard"
        ? "discard"
        : (labelById.get(edge.to_endpoint.stable_id) ?? edge.to_endpoint.stable_id),
      summary: flowLabel(edge.flow),
    })), [labelById, payload.graph.edges]);
  const currentBinding = proposalBindingMatches(payload, reviewState);
  const status = reviewStatusCopy(reviewState, currentBinding);
  const controlsLocked =
    disabled ||
    !currentBinding ||
    ["submitting", "reloading", "stale"].includes(reviewState.status) ||
    (reviewState.status === "error" && !reviewState.retryable);
  const actionEnabled = (candidate: GuidedProposalRetryAction): boolean => {
    if (controlsLocked) return false;
    if (reviewState.status !== "error") return true;
    if (!reviewState.retryable) return false;
    return sameRetryAction(reviewState.retry_action, candidate);
  };

  useEffect(() => {
    if (status !== null && reviewState.status !== "submitting") {
      statusRef.current?.focus({ preventScroll: true });
    }
  }, [reviewState.status, reviewState.proposal_id, reviewState.draft_hash, status]);

  const targetLabel = (target: GuidedEditTarget): string => {
    if (target.kind !== "edge") return labelById.get(target.stable_id) ?? `${target.kind} component`;
    const edge = payload.graph.edges.find((candidate) => candidate.stable_id === target.stable_id);
    if (edge === undefined) return "route";
    const from = labelById.get(edge.from_endpoint.stable_id) ?? "component";
    const to = edge.to_endpoint.kind === "discard"
      ? "discard"
      : (labelById.get(edge.to_endpoint.stable_id) ?? "component");
    return `route from ${from} to ${to}`;
  };

  return (
    <article className="guided-turn guided-proposal" aria-labelledby="guided-proposal-heading">
      <header className="guided-proposal__header">
        <h3 id="guided-proposal-heading">Review pipeline proposal</h3>
        <p>A complete pipeline is ready for review.</p>
        <p>Review its structure, routes, and blockers before checking the detailed wiring.</p>
        <p className="guided-proposal__counts">
          {payload.component_counts.sources} sources · {payload.component_counts.nodes} nodes ·{" "}
          {payload.component_counts.edges} routes · {payload.component_counts.outputs} outputs
        </p>
      </header>

      {status !== null ? (
        <p
          ref={statusRef}
          tabIndex={-1}
          role={status.role}
          className={`guided-proposal__status guided-proposal__status--${reviewState.status}`}
        >
          {status.message}
        </p>
      ) : null}

      {payload.blockers.length > 0 ? (
        <section className="guided-proposal__blockers" aria-labelledby="guided-proposal-blockers">
          <h4 id="guided-proposal-blockers">Before wiring review</h4>
          <ul>
            {payload.blockers.map((blocker, index) => (
              <li key={`${blocker.code}-${blocker.edit_target?.stable_id ?? index}`}>
                {BLOCKER_COPY[blocker.code]}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <ReadOnlyPipelineGraph
        nodes={graphNodes}
        edges={graphEdges}
        ariaLabel={`Pipeline proposal graph with ${graphNodes.length} components and ${graphEdges.length} routes`}
      />

      <section className="guided-proposal__components" aria-labelledby="guided-proposal-components">
        <h4 id="guided-proposal-components">Components</h4>
        <ul>
          {payload.graph.sources.map((source) => (
            <li key={source.stable_id}>{source.label} · {source.plugin.id}</li>
          ))}
          {payload.nodes.map((node) => (
            <li key={node.stable_id}>
              <strong>{node.label} · {node.node_type}{node.plugin === null ? "" : ` · ${node.plugin.id}`}</strong>
              <span> {behaviorSummary(node.behavior)}</span>
            </li>
          ))}
          {payload.outputs.map((output) => (
            <li key={output.stable_id}>{output.label} · {output.plugin.id}</li>
          ))}
        </ul>
      </section>

      <section className="guided-proposal__routes" aria-labelledby="guided-proposal-routes">
        <h4 id="guided-proposal-routes">Routes</h4>
        <WireReviewList items={routeItems} ariaLabel="Proposed pipeline routes" />
      </section>

      {/* Tutorial mode follows the same pattern as the other leaf widgets
          (InspectAndConfirmTurn keeps "Looks right" and hides "Edit columns…";
          SchemaFormTurn keeps Continue and hides Edit): keep the PRIMARY
          advance, hide the off-script affordances. Post-7.1 the tutorial's
          transforms phase produces a REAL planner proposal — there is no canned
          recipe exhibit any more (planning.py is the only propose_pipeline
          producer) — so "Review wiring" must dispatch for the learner to reach
          the wire stage at all. Reject (destructive, discards the planner
          build) and Revise (re-enters planner rounds off-script) stay hidden
          for the passive learner. Tutorial mode therefore PRESUPPOSES the
          frozen-prompt proposal arrives unblocked: revise — the only in-turn
          affordance that clears a blocker — is withheld, and the one blocker
          the tutorial realistically sees (interpretation_required) clears via
          the Accept cards outside this turn. */}
      {isTutorial ? (
        <p className="guided-proposal__tutorial-note">
          The assistant planned this pipeline from your prompt. Review how its sources, processing steps, routes, and outputs fit together, then press Review wiring to continue.
        </p>
      ) : null}
      <div className="guided-proposal__controls">
        <div className="guided-proposal__primary-actions">
          <button
            type="button"
            className="guided-turn-primary"
            disabled={!actionEnabled({ kind: "review_wiring" }) || payload.blockers.length > 0}
            onClick={() => onSubmit({
              chosen: ["review_wiring"],
              edited_values: null,
              custom_inputs: null,
              edit_target: null,
              control_signal: null,
              proposal_id: payload.proposal_id,
              draft_hash: payload.draft_hash,
            } satisfies GuidedRespondAction)}
          >
            Review wiring
          </button>
          {!isTutorial && (
            <button
              type="button"
              className="guided-turn-secondary"
              disabled={!actionEnabled({ kind: "reject" })}
              onClick={() => onSubmit({
                chosen: null,
                edited_values: null,
                custom_inputs: null,
                edit_target: null,
                control_signal: "reject",
                proposal_id: payload.proposal_id,
                draft_hash: payload.draft_hash,
              } satisfies GuidedRespondAction)}
            >
              Reject proposal
            </button>
          )}
        </div>
        {!isTutorial && (
          <fieldset className="guided-proposal__revise" disabled={controlsLocked}>
            <legend>Revise a component</legend>
            {payload.edit_targets.map((target) => (
              <button
                type="button"
                className="guided-turn-secondary"
                key={`${target.kind}-${target.stable_id}`}
                disabled={!actionEnabled({ kind: "revise", edit_target: target })}
                onClick={() => onSubmit({
                  chosen: null,
                  edited_values: null,
                  custom_inputs: null,
                  edit_target: target,
                  control_signal: null,
                  proposal_id: payload.proposal_id,
                  draft_hash: payload.draft_hash,
                } satisfies GuidedRespondAction)}
              >
                Revise {targetLabel(target)}
              </button>
            ))}
          </fieldset>
        )}
      </div>
    </article>
  );
}
