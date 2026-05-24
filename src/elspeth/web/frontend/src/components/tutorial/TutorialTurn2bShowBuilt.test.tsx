import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { resetStore } from "@/test/store-helpers";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import type { InterpretationEvent } from "@/types/interpretation";
import { TutorialTurn2bShowBuilt } from "./TutorialTurn2bShowBuilt";
import type { TutorialBuiltSummary } from "./tutorialMachine";

vi.mock("@/api/client", () => ({
  optOutOfInterpretations: vi.fn(),
  resolveInterpretation: vi.fn(),
}));

const summary: TutorialBuiltSummary = {
  sourceLabel: "Inline blob",
  urls: ["https://dta.gov.au"],
  transforms: ["LLM rating"],
  sinkLabel: "JSONL",
};

function interpretationEvent(
  id: string,
  userTerm: string,
  createdAt: string,
): InterpretationEvent {
  return {
    id,
    session_id: "session-1",
    composition_state_id: "state-1",
    affected_node_id: "rate",
    tool_call_id: `call-${id}`,
    user_term: userTerm,
    kind: "vague_term",
    llm_draft: `${userTerm} interpretation`,
    accepted_value: null,
    choice: "pending",
    created_at: createdAt,
    resolved_at: null,
    actor: "composer-llm",
    interpretation_source: "user_approved",
    model_identifier: "openrouter/openai/gpt-5.4-mini",
    model_version: "openai/gpt-5.4-mini-20260317",
    provider: "openrouter",
    composer_skill_hash: "a".repeat(64),
    arguments_hash: null,
    hash_domain_version: null,
    runtime_model_identifier_at_resolve: null,
    runtime_model_version_at_resolve: null,
    resolved_prompt_template_hash: null,
  };
}

describe("TutorialTurn2bShowBuilt", () => {
  beforeEach(() => {
    resetStore(useInterpretationEventsStore);
    vi.clearAllMocks();
  });

  it("renders every pending tutorial assumption and disables continuing until they are approved", () => {
    const latest = interpretationEvent(
      "event-3",
      "latest term",
      "2026-05-19T12:00:04Z",
    );
    const earliest = interpretationEvent(
      "event-1",
      "earliest term",
      "2026-05-19T12:00:02Z",
    );
    const middle = interpretationEvent(
      "event-2",
      "middle term",
      "2026-05-19T12:00:03Z",
    );
    useInterpretationEventsStore.setState({
      pendingBySession: {
        "session-1": {
          [latest.id]: latest,
          [earliest.id]: earliest,
          [middle.id]: middle,
        },
      },
    });

    render(
      <TutorialTurn2bShowBuilt
        sessionId="session-1"
        summary={summary}
        onContinue={vi.fn()}
        onBack={vi.fn()}
      />,
    );

    expect(screen.getByText("3 assumptions to review")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        name: "Here is what the composer drafted - review its assumptions.",
      }),
    ).toBeInTheDocument();

    const continueButton = screen.getByRole("button", { name: "Looks good" });
    expect(continueButton).toBeDisabled();
    expect(continueButton).toHaveAttribute("aria-disabled", "true");
    expect(continueButton).toHaveAttribute(
      "title",
      "Approve the assumptions above first",
    );
    expect(
      screen
        .getAllByRole("status")
        .some((status) =>
          status.textContent?.includes("Approve the assumptions above first"),
        ),
    ).toBe(true);

    const earliestNode = screen.getByText("earliest term");
    const middleNode = screen.getByText("middle term");
    const latestNode = screen.getByText("latest term");
    expect(
      earliestNode.compareDocumentPosition(middleNode) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      middleNode.compareDocumentPosition(latestNode) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
