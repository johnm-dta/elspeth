import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { resetStore } from "@/test/store-helpers";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import type { InterpretationEvent } from "@/types/interpretation";
import { GuidedInterpretationReviews } from "./GuidedInterpretationReviews";

// InterpretationReviewTurn resolves through interpretationEventsStore actions,
// which call into @/api/client at resolve time.  This render-only test never
// resolves, but mirror the tutorial test's mock so no real HTTP surface loads.
vi.mock("@/api/client", () => ({
  optOutOfInterpretations: vi.fn(),
  resolveInterpretation: vi.fn(),
}));

const SID = "11111111-1111-1111-1111-111111111111";

function pendingEvent(id: string): InterpretationEvent {
  // Every required field of InterpretationEvent (types/interpretation.ts:122-165),
  // so no `as` cast is needed and a future field addition fails the typecheck.
  return {
    id,
    session_id: SID,
    composition_state_id: "22222222-2222-2222-2222-222222222222",
    affected_node_id: "rate_node",
    tool_call_id: "backend_auto_surface:abc",
    user_term: "llm_model_choice:rate_node",
    kind: "llm_model_choice",
    llm_draft: "anthropic/claude-sonnet-4.6",
    accepted_value: null,
    choice: "pending",
    created_at: "2026-06-22T00:00:00Z",
    resolved_at: null,
    actor: "system:composer",
    interpretation_source: "user_approved",
    model_identifier: "anthropic/claude-opus-4-7",
    model_version: "anthropic/claude-opus-4-7",
    provider: "anthropic",
    composer_skill_hash: "0".repeat(64),
    arguments_hash: null,
    hash_domain_version: null,
    runtime_model_identifier_at_resolve: null,
    runtime_model_version_at_resolve: null,
    resolved_prompt_template_hash: null,
  };
}

describe("GuidedInterpretationReviews", () => {
  beforeEach(() => {
    resetStore(useInterpretationEventsStore);
    vi.clearAllMocks();
    useInterpretationEventsStore.setState({
      pendingBySession: { [SID]: { e1: pendingEvent("e1") } },
    });
  });

  it("renders a review affordance for each pending user_approved event", () => {
    render(<GuidedInterpretationReviews sessionId={SID} />);
    // With one event there are TWO regions: the wrapper <section
    // aria-label="Assumptions to review"> AND the inner InterpretationReviewTurn's
    // own <section role="region"> — so getByRole("region") would throw "Found
    // multiple elements", and InterpretationReviewTurn also carries role="status",
    // so that role is ambiguous too. Assert via the wrapper count line instead,
    // the same way the sibling TutorialTurn2bShowBuilt.test.tsx does.
    expect(screen.getByText("1 assumption to review")).toBeInTheDocument();
  });

  it("renders nothing when there are no pending events", () => {
    useInterpretationEventsStore.setState({ pendingBySession: { [SID]: {} } });
    const { container } = render(<GuidedInterpretationReviews sessionId={SID} />);
    expect(container).toBeEmptyDOMElement();
  });
});
