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
    render(<GuidedHistory history={[TURN_1, TURN_2]} />);
    expect(
      screen.getByRole("heading", { name: /decisions so far/i }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /show steps/i })).toBeNull();
  });

  it("renders one list item per completed decision", () => {
    render(<GuidedHistory history={[TURN_1, TURN_2]} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByText("Source selected: csv")).toBeInTheDocument();
    expect(screen.getByText("Sink configured: jsonl")).toBeInTheDocument();
  });

  it("uses step labels and hides emitter details from the default surface", () => {
    render(<GuidedHistory history={[TURN_1]} />);
    expect(screen.getByText("Source")).toBeInTheDocument();
    expect(screen.queryByText("server")).toBeNull();
  });

  it("falls back to plain turn labels for legacy records without summaries", () => {
    render(<GuidedHistory history={[{ ...TURN_1, summary: null }]} />);
    expect(screen.getByText("Single select")).toBeInTheDocument();
  });

  it("renders nothing when history is empty", () => {
    const { container } = render(<GuidedHistory history={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
