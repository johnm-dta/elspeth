import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { NarrativeResults } from "./NarrativeResults";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";

describe("NarrativeResults", () => {
  beforeEach(() => {
    useSessionStore.setState({ activeSessionId: null } as never);
    resetStore(useInterpretationEventsStore);
  });

  it("renders the no-summary placeholder when summaryOverride is undefined (live mode without an aggregated summary)", () => {
    render(<NarrativeResults />);
    expect(screen.getByTestId("narrative-results")).toBeInTheDocument();
    expect(screen.getByTestId("narrative-results-no-summary")).toBeInTheDocument();
    expect(screen.queryByTestId("narrative-results-summary")).toBeNull();
  });

  it("renders the supplied summary when summaryOverride is provided", () => {
    render(<NarrativeResults summaryOverride="The pipeline achieved an F1 of 0.87." />);
    expect(screen.getByTestId("narrative-results-summary")).toHaveTextContent(
      "The pipeline achieved an F1 of 0.87.",
    );
    expect(screen.queryByTestId("narrative-results-no-summary")).toBeNull();
  });

  it("renders the no-summary placeholder when summaryOverride is explicitly null", () => {
    render(<NarrativeResults summaryOverride={null} />);
    expect(screen.getByTestId("narrative-results-no-summary")).toBeInTheDocument();
  });

  it("renders the no-summary placeholder when summaryOverride is the empty string", () => {
    render(<NarrativeResults summaryOverride="" />);
    expect(screen.getByTestId("narrative-results-no-summary")).toBeInTheDocument();
  });

  it("does not render the interpretation overlay when no session is active", () => {
    render(<NarrativeResults summaryOverride="anything" />);
    expect(screen.queryByTestId("narrative-results-interpretations")).toBeNull();
  });

  it("does not render the interpretation overlay when the session has not opted out", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useInterpretationEventsStore.setState({
      optedOutBySession: { "sess-1": false },
    } as never);
    render(<NarrativeResults summaryOverride="anything" />);
    expect(screen.queryByTestId("narrative-results-interpretations")).toBeNull();
  });

  it("renders the interpretation overlay when the active session has opted out of LLM surfacing", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    useInterpretationEventsStore.setState({
      optedOutBySession: { "sess-1": true },
    } as never);
    render(<NarrativeResults summaryOverride="anything" />);
    expect(screen.getByTestId("narrative-results-interpretations")).toBeInTheDocument();
    expect(screen.getByTestId("narrative-results-interpretations")).toHaveTextContent(
      /opt-out/i,
    );
  });
});
