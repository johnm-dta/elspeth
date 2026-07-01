// GuidedHistory -- always-visible plain-language decision summary.

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { GuidedHistory } from "./GuidedHistory";
import type { TurnRecord } from "@/types/guided";

const TURN_1: TurnRecord = {
  step: "step_1_source",
  turn_type: "single_select",
  payload_hash: "aabbcc001122",
  response_hash: "ddeeff334455",
  emitter: "server",
  summary: "Source selected: csv",
};

const TURN_2: TurnRecord = {
  step: "step_2_sink",
  turn_type: "schema_form",
  payload_hash: "112233aabbcc",
  response_hash: null,
  emitter: "llm",
  summary: "Sink configured: jsonl",
};

describe("GuidedHistory", () => {
  it("renders an always-visible decision summary heading", () => {
    render(<GuidedHistory history={[TURN_1, TURN_2]} currentStep="step_4_wire" />);
    expect(
      screen.getByRole("heading", { name: /decisions so far/i }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /show steps/i })).toBeNull();
  });

  it("renders one list item per completed decision", () => {
    render(<GuidedHistory history={[TURN_1, TURN_2]} currentStep="step_4_wire" />);
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByText("Source selected: csv")).toBeInTheDocument();
    expect(screen.getByText("Sink configured: jsonl")).toBeInTheDocument();
  });

  it("uses step labels and hides emitter details from the default surface", () => {
    render(<GuidedHistory history={[TURN_1]} currentStep="step_2_sink" />);
    expect(screen.getByText("Source")).toBeInTheDocument();
    expect(screen.queryByText("server")).toBeNull();
  });

  it("omits a step that has not yet recorded a summary (in progress)", () => {
    // currentStep is elsewhere, so the only thing keeping step_1_source out is
    // the missing summary. The whole card collapses to nothing — no "Decided".
    const { container } = render(
      <GuidedHistory
        history={[{ ...TURN_1, summary: null }]}
        currentStep="step_2_sink"
      />,
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByText("Decided")).toBeNull();
  });

  it("does not render the current step even when it already carries a summary", () => {
    // Shot 02: still on Source, the answered single_select recorded a summary,
    // but the source schema_form is not yet submitted — Source is not decided.
    const { container } = render(
      <GuidedHistory
        history={[{ ...TURN_1, summary: "Configured: json" }]}
        currentStep="step_1_source"
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("collapses multiple turns of the same step to one row (most-recent summary wins)", () => {
    render(
      <GuidedHistory
        history={[
          { ...TURN_1, summary: null },
          { ...TURN_1, turn_type: "schema_form", summary: "Source configured: csv" },
        ]}
        currentStep="step_2_sink"
      />,
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    expect(screen.getByText("Source configured: csv")).toBeInTheDocument();
  });

  it("does not let a trailing unsummarised next-turn mask the step's summary", () => {
    // Real shape after a chat-resolve: the entry record carries the decision
    // summary, then an unanswered next-turn record (summary: null) is emitted.
    render(
      <GuidedHistory
        history={[
          { ...TURN_1, turn_type: "single_select", summary: "Configured: web_scrape" },
          { ...TURN_1, turn_type: "schema_form", summary: null },
        ]}
        currentStep="step_2_sink"
      />,
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    expect(screen.getByText("Configured: web_scrape")).toBeInTheDocument();
    expect(screen.queryByText("Decided")).toBeNull();
  });

  it("renders nothing when history is empty", () => {
    const { container } = render(<GuidedHistory history={[]} currentStep="step_1_source" />);
    expect(container.firstChild).toBeNull();
  });
});
