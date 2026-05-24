import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ToolCallCard } from "./ToolCallCard";
import type { CompositionProposal, ToolCall } from "@/types/api";

const toolCall: ToolCall = {
  id: "call-1",
  type: "function",
  function: {
    name: "set_pipeline",
    arguments: "{\"source\":{\"plugin\":\"csv\"}}",
  },
};

const proposal: CompositionProposal = {
  id: "proposal-1",
  session_id: "session-1",
  tool_call_id: "call-1",
  tool_name: "set_pipeline",
  status: "pending",
  summary: "Replace the pipeline with csv input, 1 transform, and 1 output.",
  rationale: "Requested by the current composer turn.",
  affects: ["graph", "validation", "yaml"],
  arguments_redacted_json: { source: { plugin: "csv" } },
  base_state_id: null,
  committed_state_id: null,
  audit_event_id: "event-1",
  created_at: "2026-05-14T00:00:00Z",
  updated_at: "2026-05-14T00:00:00Z",
};

describe("ToolCallCard", () => {
  it("renders pending write proposals with balanced accept and reject actions", () => {
    render(
      <ToolCallCard
        toolCall={toolCall}
        proposal={proposal}
        onAccept={vi.fn()}
        onReject={vi.fn()}
      />,
    );

    expect(screen.getByText("Proposed: set_pipeline")).toBeInTheDocument();
    expect(screen.getByText(proposal.summary)).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: `Accept proposal: ${proposal.summary}`,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: `Reject proposal: ${proposal.summary}`,
      }),
    ).toBeInTheDocument();
  });

  it("renders read-only tools as ribbons", () => {
    render(
      <ToolCallCard
        toolCall={{
          id: "read-1",
          type: "function",
          function: { name: "get_pipeline_state", arguments: "{}" },
        }}
        proposal={null}
        onAccept={vi.fn()}
        onReject={vi.fn()}
      />,
    );

    expect(screen.getByText("Looked up: get_pipeline_state")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Accept proposal/ }),
    ).not.toBeInTheDocument();
  });

  it("keeps the tool-call info button at or above the 24px target-size threshold", () => {
    const css = readFileSync("src/components/chat/chat.css", "utf8");
    const rule = /\.tool-call-info-trigger\s*\{(?<body>[\s\S]*?)\n\}/.exec(css);
    expect(rule?.groups?.body).toContain("width: 24px");
    expect(rule?.groups?.body).toContain("height: 24px");
  });

  it("calls accept and reject handlers", async () => {
    const user = userEvent.setup();
    const onAccept = vi.fn();
    const onReject = vi.fn();
    render(
      <ToolCallCard
        toolCall={toolCall}
        proposal={proposal}
        onAccept={onAccept}
        onReject={onReject}
      />,
    );

    await user.click(
      screen.getByRole("button", {
        name: `Accept proposal: ${proposal.summary}`,
      }),
    );
    // S3.5 (button-audit): Reject now opens a ConfirmDialog. Click the
    // dialog's primary action ("Reject proposal", exactly — distinct from
    // the original card button whose accessible name is "Reject proposal:
    // {summary}") to actually invoke onReject.
    await user.click(
      screen.getByRole("button", {
        name: `Reject proposal: ${proposal.summary}`,
      }),
    );
    await user.click(
      screen.getByRole("button", { name: /^reject proposal$/i }),
    );

    expect(onAccept).toHaveBeenCalledWith("proposal-1");
    expect(onReject).toHaveBeenCalledWith("proposal-1");
  });

  it("renders stale proposals without actionable accept or reject buttons", () => {
    render(
      <ToolCallCard
        toolCall={toolCall}
        proposal={proposal}
        isStale={true}
        isBusy={false}
        onAccept={vi.fn()}
        onReject={vi.fn()}
      />,
    );

    expect(screen.getByText(/Stale proposal/)).toBeInTheDocument();
    expect(
      screen.getByText(/Ask the composer to rebase or revise this proposal/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Accept proposal/ }),
    ).not.toBeInTheDocument();
  });
});
