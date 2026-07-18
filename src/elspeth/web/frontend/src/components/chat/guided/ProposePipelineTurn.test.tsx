import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type {
  GuidedProposalReviewState,
  ProposePipelinePayload,
} from "@/types/guided";
import { ProposePipelineTurn } from "./ProposePipelineTurn";

const IDS = {
  proposal: "00000000-0000-4000-8000-000000000401",
  sourceOne: "00000000-0000-4000-8000-000000000402",
  sourceTwo: "00000000-0000-4000-8000-000000000403",
  gate: "00000000-0000-4000-8000-000000000404",
  queue: "00000000-0000-4000-8000-000000000405",
  aggregation: "00000000-0000-4000-8000-000000000406",
  coalesce: "00000000-0000-4000-8000-000000000407",
  outputOne: "00000000-0000-4000-8000-000000000408",
  outputTwo: "00000000-0000-4000-8000-000000000409",
} as const;

function edgeId(index: number): string {
  return `00000000-0000-4000-8000-${String(500 + index).padStart(12, "0")}`;
}

function payload(): ProposePipelinePayload {
  const edges: ProposePipelinePayload["graph"]["edges"] = [
    {
      stable_id: edgeId(1),
      from_endpoint: { kind: "source", stable_id: IDS.sourceOne },
      to_endpoint: { kind: "node", stable_id: IDS.gate },
      flow: { kind: "source_success", branch: null },
    },
    {
      stable_id: edgeId(2),
      from_endpoint: { kind: "source", stable_id: IDS.sourceOne },
      to_endpoint: { kind: "discard" },
      flow: { kind: "source_validation_failure" },
    },
    {
      stable_id: edgeId(3),
      from_endpoint: { kind: "source", stable_id: IDS.sourceTwo },
      to_endpoint: { kind: "node", stable_id: IDS.queue },
      flow: { kind: "source_success", branch: null },
    },
    {
      stable_id: edgeId(4),
      from_endpoint: { kind: "source", stable_id: IDS.sourceTwo },
      to_endpoint: { kind: "discard" },
      flow: { kind: "source_validation_failure" },
    },
    {
      stable_id: edgeId(5),
      from_endpoint: { kind: "node", stable_id: IDS.queue },
      to_endpoint: { kind: "node", stable_id: IDS.gate },
      flow: { kind: "queue_continue", branch: null },
    },
    {
      stable_id: edgeId(6),
      from_endpoint: { kind: "node", stable_id: IDS.gate },
      to_endpoint: { kind: "node", stable_id: IDS.aggregation },
      flow: { kind: "gate_fork", routes: ["route-1"], branch: "branch-1" },
    },
    {
      stable_id: edgeId(7),
      from_endpoint: { kind: "node", stable_id: IDS.gate },
      to_endpoint: { kind: "node", stable_id: IDS.coalesce },
      flow: { kind: "gate_fork", routes: ["route-1"], branch: "branch-2" },
    },
    {
      stable_id: edgeId(8),
      from_endpoint: { kind: "node", stable_id: IDS.gate },
      to_endpoint: { kind: "output", stable_id: IDS.outputTwo },
      flow: { kind: "gate_route", route: "route-2", branch: null },
    },
    {
      stable_id: edgeId(9),
      from_endpoint: { kind: "node", stable_id: IDS.aggregation },
      to_endpoint: { kind: "node", stable_id: IDS.coalesce },
      flow: { kind: "node_success", branch: "branch-1" },
    },
    {
      stable_id: edgeId(10),
      from_endpoint: { kind: "node", stable_id: IDS.aggregation },
      to_endpoint: { kind: "discard" },
      flow: { kind: "node_error" },
    },
    {
      stable_id: edgeId(11),
      from_endpoint: { kind: "node", stable_id: IDS.coalesce },
      to_endpoint: { kind: "output", stable_id: IDS.outputOne },
      flow: { kind: "coalesce_success", branch: null },
    },
    {
      stable_id: edgeId(12),
      from_endpoint: { kind: "output", stable_id: IDS.outputOne },
      to_endpoint: { kind: "discard" },
      flow: { kind: "output_write_failure" },
    },
    {
      stable_id: edgeId(13),
      from_endpoint: { kind: "output", stable_id: IDS.outputTwo },
      to_endpoint: { kind: "discard" },
      flow: { kind: "output_write_failure" },
    },
  ];
  return {
    proposal_id: IDS.proposal,
    draft_hash: "d".repeat(64),
    summary: "guided.proposal.summary.full_graph.v1",
    rationale: "guided.proposal.rationale.review_required.v1",
    component_counts: { sources: 2, nodes: 4, edges: edges.length, outputs: 2 },
    blockers: [],
    graph: {
      sources: [
        {
          stable_id: IDS.sourceOne,
          label: "source-1",
          plugin: { kind: "source", id: "csv" },
        },
        {
          stable_id: IDS.sourceTwo,
          label: "source-2",
          plugin: { kind: "source", id: "json" },
        },
      ],
      edges,
    },
    nodes: [
      {
        stable_id: IDS.gate,
        label: "node-1",
        node_type: "gate",
        plugin: null,
        behavior: {
          kind: "gate",
          route_aliases: ["route-1", "route-2"],
          fork_branches: [
            { routes: ["route-1"], branch: "branch-1" },
            { routes: ["route-1"], branch: "branch-2" },
          ],
        },
      },
      {
        stable_id: IDS.queue,
        label: "node-2",
        node_type: "queue",
        plugin: null,
        behavior: { kind: "queue" },
      },
      {
        stable_id: IDS.aggregation,
        label: "node-3",
        node_type: "aggregation",
        plugin: { kind: "transform", id: "batch" },
        behavior: {
          kind: "aggregation",
          trigger_kinds: ["count", "timeout"],
          count: "50",
          timeout_seconds: 10,
          output_mode: "passthrough",
          expected_output_count: "2",
        },
      },
      {
        stable_id: IDS.coalesce,
        label: "node-4",
        node_type: "coalesce",
        plugin: null,
        behavior: {
          kind: "coalesce",
          branch_aliases: ["branch-1", "branch-2"],
          policy: "quorum",
          merge: "nested",
        },
      },
    ],
    outputs: [
      {
        stable_id: IDS.outputOne,
        label: "output-1",
        plugin: { kind: "sink", id: "json" },
      },
      {
        stable_id: IDS.outputTwo,
        label: "output-2",
        plugin: { kind: "sink", id: "csv" },
      },
    ],
    edit_targets: [
      { kind: "source", stable_id: IDS.sourceOne },
      { kind: "node", stable_id: IDS.gate },
      { kind: "edge", stable_id: edgeId(6) },
      { kind: "output", stable_id: IDS.outputOne },
    ],
  };
}

function activeReview(): GuidedProposalReviewState {
  return {
    status: "active",
    proposal_id: IDS.proposal,
    draft_hash: "d".repeat(64),
  };
}

describe("ProposePipelineTurn", () => {
  it("renders the full DAG, stable edge identities, virtual discard, routes, queues, aggregations, fan-in, sources, and outputs", () => {
    const { container } = render(
      <ProposePipelineTurn
        payload={payload()}
        reviewState={activeReview()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getByRole("img", { name: /pipeline proposal graph/i })).toBeVisible();
    expect(container.querySelector(`[data-edge-id="${edgeId(6)}"]`)).not.toBeNull();
    expect(container.querySelector('[data-node-kind="discard"]')).not.toBeNull();
    expect(screen.getByText("2 sources · 4 nodes · 13 routes · 2 outputs")).toBeVisible();
    expect(screen.getByText(/routes route-1, route-2/i)).toBeVisible();
    expect(screen.getByText(/queue continues in sequence/i)).toBeVisible();
    expect(screen.getByText(/count 50 or timeout 10s/i)).toBeVisible();
    expect(screen.getByText(/joins branch-1, branch-2/i)).toBeVisible();
    expect(screen.getAllByText(/route-1 forks to branch-1/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/on error → discard/i).length).toBeGreaterThan(0);
    expect(screen.getByText("source-2 · json")).toBeVisible();
    expect(screen.getByText("output-2 · csv")).toBeVisible();
  });

  it("uses fixed local copy for server template ids and never renders template ids as rationale", () => {
    render(
      <ProposePipelineTurn
        payload={payload()}
        reviewState={activeReview()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getByText("A complete pipeline is ready for review.")).toBeVisible();
    expect(
      screen.getByText("Review its structure, routes, and blockers before accepting it."),
    ).toBeVisible();
    expect(screen.queryByText("guided.proposal.rationale.review_required.v1")).toBeNull();
  });

  it("submits exact proposal-bound accept, reject, and target-only revise actions", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <ProposePipelineTurn
        payload={payload()}
        reviewState={activeReview()}
        onSubmit={onSubmit}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Accept pipeline" }));
    await user.click(screen.getByRole("button", { name: "Reject proposal" }));
    await user.click(screen.getByRole("button", { name: "Revise node-1" }));

    expect(onSubmit.mock.calls).toEqual([
      [{
        chosen: ["accept"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: IDS.proposal,
        draft_hash: "d".repeat(64),
        edit_target: null,
        control_signal: null,
      }],
      [{
        chosen: null,
        edited_values: null,
        custom_inputs: null,
        proposal_id: IDS.proposal,
        draft_hash: "d".repeat(64),
        edit_target: null,
        control_signal: "reject",
      }],
      [{
        chosen: null,
        edited_values: null,
        custom_inputs: null,
        proposal_id: IDS.proposal,
        draft_hash: "d".repeat(64),
        edit_target: { kind: "node", stable_id: IDS.gate },
        control_signal: null,
      }],
    ]);
  });

  it("supports keyboard activation and moves focus to proposal status changes", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const { rerender } = render(
      <ProposePipelineTurn
        payload={payload()}
        reviewState={activeReview()}
        onSubmit={onSubmit}
      />,
    );
    const accept = screen.getByRole("button", { name: "Accept pipeline" });
    accept.focus();
    await user.keyboard("{Enter}");
    expect(onSubmit).toHaveBeenCalledTimes(1);

    rerender(
      <ProposePipelineTurn
        payload={payload()}
        reviewState={{
          status: "stale",
          proposal_id: IDS.proposal,
          draft_hash: "d".repeat(64),
        }}
        onSubmit={onSubmit}
      />,
    );
    expect(screen.getByRole("status")).toHaveFocus();
    expect(accept).toBeDisabled();
  });

  it("renders allowlisted blockers, disables acceptance, and keeps stable revise controls available", () => {
    const blocked = payload();
    blocked.blockers = [
      {
        code: "policy_review_required",
        category: "policy",
        summary: "guided.proposal.blocker.policy_review_required.v1",
        edit_target: { kind: "node", stable_id: IDS.gate },
      },
    ];
    render(
      <ProposePipelineTurn
        payload={blocked}
        reviewState={activeReview()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getByText("A policy review is required before this pipeline can be accepted.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Accept pipeline" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Revise node-1" })).toBeEnabled();
  });

  it.each(["submitting", "reloading", "stale"] as const)(
    "disables the exact proposal controls while review state is %s",
    (status) => {
      render(
        <ProposePipelineTurn
          payload={payload()}
          reviewState={{
            status,
            proposal_id: IDS.proposal,
            draft_hash: "d".repeat(64),
          }}
          onSubmit={vi.fn()}
        />,
      );
      expect(screen.getByRole("button", { name: "Accept pipeline" })).toBeDisabled();
      expect(screen.getByRole("button", { name: "Reject proposal" })).toBeDisabled();
      expect(screen.getByRole("button", { name: "Revise node-1" })).toBeDisabled();
    },
  );

  it.each([
    ["accept", { kind: "accept" }, "Accept pipeline"],
    ["reject", { kind: "reject" }, "Reject proposal"],
    [
      "one exact revise target",
      { kind: "revise", edit_target: { kind: "node", stable_id: IDS.gate } },
      "Revise node-1",
    ],
  ] as const)(
    "enables only the retained %s action after an ambiguous transport failure",
    (_label, retryAction, enabledName) => {
      render(
        <ProposePipelineTurn
          payload={payload()}
          reviewState={{
            status: "error",
            proposal_id: IDS.proposal,
            draft_hash: "d".repeat(64),
            message: "The response was not received. Retry the same action.",
            retryable: true,
            retry_action: retryAction,
          } as never}
          onSubmit={vi.fn()}
        />,
      );

      expect(screen.getByRole("alert")).toHaveTextContent(/response was not received/i);
      const controls = screen.getAllByRole("button");
      expect(controls.filter((control) => !control.hasAttribute("disabled"))).toHaveLength(1);
      expect(screen.getByRole("button", { name: enabledName })).toBeEnabled();
    },
  );

  it("locks the stale proposal controls when an authoritative reload fails", () => {
    render(
      <ProposePipelineTurn
        payload={payload()}
        reviewState={{
          status: "error",
          proposal_id: IDS.proposal,
          draft_hash: "d".repeat(64),
          message: "The authoritative replacement could not be loaded.",
          retryable: false,
          retry_action: null,
        }}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/could not be loaded/i);
    expect(screen.getByRole("button", { name: "Accept pipeline" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reject proposal" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Revise node-1" })).toBeDisabled();
  });

  it("renders the same closed proposal passively for tutorials without action controls", () => {
    render(
      <ProposePipelineTurn
        payload={payload()}
        reviewState={activeReview()}
        onSubmit={vi.fn()}
        isTutorial
      />,
    );
    expect(screen.getByRole("img", { name: /pipeline proposal graph/i })).toBeVisible();
    expect(screen.getByText("source-1 · csv")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Accept pipeline" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Reject proposal" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Revise/ })).toBeNull();
  });
});
