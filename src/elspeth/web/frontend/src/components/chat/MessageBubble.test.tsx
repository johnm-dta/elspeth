import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MessageBubble } from "./MessageBubble";
import type { ChatMessage, CompositionProposal } from "@/types/api";

function makeMessage(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: "msg-1",
    session_id: "session-1",
    role: "user",
    content: "Hello world",
    tool_calls: null,
    created_at: new Date().toISOString(),
    ...overrides,
  };
}

function makeProposal(
  overrides: Partial<CompositionProposal> = {},
): CompositionProposal {
  return {
    id: "proposal-1",
    session_id: "session-1",
    tool_call_id: "tc-1",
    tool_name: "set_pipeline",
    status: "pending",
    summary: "Replace the pipeline.",
    rationale: "Requested by the current composer turn.",
    affects: ["graph"],
    arguments_redacted_json: {},
    base_state_id: null,
    committed_state_id: null,
    audit_event_id: "event-1",
    created_at: "2026-05-14T00:00:00Z",
    updated_at: "2026-05-14T00:00:00Z",
    ...overrides,
  };
}

describe("MessageBubble", () => {
  describe("send-state suppression", () => {
    it("shows Sending... when pending and not composing", () => {
      render(
        <MessageBubble
          message={makeMessage({ local_status: "pending" })}
          isComposing={false}
        />,
      );
      expect(screen.getByText("Sending...")).toBeInTheDocument();
    });

    it("hides Sending... when pending and composing is active", () => {
      render(
        <MessageBubble
          message={makeMessage({ local_status: "pending" })}
          isComposing={true}
        />,
      );
      expect(screen.queryByText("Sending...")).not.toBeInTheDocument();
    });

    it("shows failed state with retry regardless of composing", () => {
      const onRetry = vi.fn();
      render(
        <MessageBubble
          message={makeMessage({ local_status: "failed", local_error: "The AI service is temporarily unavailable. Please try again in a moment." })}
          isComposing={false}
          onRetry={onRetry}
        />,
      );
      expect(screen.getByText("Retry")).toBeInTheDocument();
      expect(screen.getByText("The AI service is temporarily unavailable. Please try again in a moment.")).toBeInTheDocument();
    });

    it("shows default error when local_error is absent", () => {
      const onRetry = vi.fn();
      render(
        <MessageBubble
          message={makeMessage({ local_status: "failed" })}
          isComposing={false}
          onRetry={onRetry}
        />,
      );
      expect(screen.getByText("Failed to send message. Please try again.")).toBeInTheDocument();
    });
  });

  describe("copy button", () => {
    it("renders a copy button on user messages", () => {
      render(<MessageBubble message={makeMessage()} />);
      expect(screen.getByLabelText("Copy message")).toBeInTheDocument();
    });

    it("renders a copy button on assistant messages", () => {
      render(
        <MessageBubble message={makeMessage({ role: "assistant" })} />,
      );
      expect(screen.getByLabelText("Copy message")).toBeInTheDocument();
    });

    it("does not render a copy button on system messages", () => {
      render(
        <MessageBubble message={makeMessage({ role: "system" })} />,
      );
      expect(screen.queryByLabelText("Copy message")).not.toBeInTheDocument();
    });

    it("copies message content to clipboard", async () => {
      const user = userEvent.setup();
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText },
        writable: true,
        configurable: true,
      });

      render(
        <MessageBubble message={makeMessage({ content: "Test copy" })} />,
      );
      await user.click(screen.getByLabelText("Copy message"));

      expect(writeText).toHaveBeenCalledWith("Test copy");
      expect(screen.getByText("Copied!")).toBeInTheDocument();
    });
  });

  describe("tool call exclusion from copy", () => {
    it("copies only message.content, not tool call details", async () => {
      const user = userEvent.setup();
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText },
        writable: true,
        configurable: true,
      });

      const message = makeMessage({
        role: "assistant",
        content: "I'll set that up.",
        tool_calls: [
          {
            id: "tc-1",
            type: "function",
            function: { name: "set_source", arguments: '{"plugin":"csv"}' },
          },
        ],
      });

      render(<MessageBubble message={message} />);
      await user.click(screen.getByLabelText("Copy message"));

      expect(writeText).toHaveBeenCalledWith("I'll set that up.");
    });

    it("renders proposal cards for matching tool calls", async () => {
      const user = userEvent.setup();
      const onAcceptProposal = vi.fn();
      const onRejectProposal = vi.fn();
      const proposal = makeProposal();
      const message = makeMessage({
        role: "assistant",
        content: "I'll prepare that change.",
        tool_calls: [
          {
            id: "tc-1",
            type: "function",
            function: { name: "set_pipeline", arguments: "{}" },
          },
        ],
      });

      render(
        <MessageBubble
          message={message}
          proposalsByToolCallId={new Map([["tc-1", proposal]])}
          onAcceptProposal={onAcceptProposal}
          onRejectProposal={onRejectProposal}
        />,
      );

      expect(screen.getByText("Proposed: set_pipeline")).toBeInTheDocument();
      await user.click(
        screen.getByRole("button", {
          name: `Accept proposal: ${proposal.summary}`,
        }),
      );

      expect(onAcceptProposal).toHaveBeenCalledWith("proposal-1");
      expect(onRejectProposal).not.toHaveBeenCalled();
    });
  });

  // ------------------------------------------------------------------
  // Sources-created in-bubble group
  // ------------------------------------------------------------------
  // The bubble surfaces dynamic-source events created by the LLM as a
  // second collapsible group below the tool-calls group, separated by a
  // horizontal ruler. Pattern mirrors tool calls so users only learn one
  // disclosure affordance. Starts collapsed; the inner widget (with its
  // own audit-info disclosure) renders only when expanded.
  describe("sources-created group", () => {
    function makeSummary(overrides = {}) {
      return {
        filename: "rows.csv",
        mimeType: "text/csv",
        rowCount: 5,
        contentHash: "a".repeat(64),
        blobId: "blob-1",
        provenance: "llm-generated" as const,
        contentPreview: "row 1\nrow 2",
        ...overrides,
      };
    }

    it("renders no Sources-created group when sourcesCreated is empty/undefined", () => {
      const message = makeMessage({ role: "assistant", content: "Done." });
      render(<MessageBubble message={message} />);
      expect(screen.queryByLabelText(/Sources created/)).not.toBeInTheDocument();
    });

    it("renders Sources created (N) toggle button when summaries are supplied", () => {
      const message = makeMessage({ role: "assistant", content: "Done." });
      render(
        <MessageBubble
          message={message}
          sourcesCreated={[makeSummary()]}
        />,
      );
      expect(
        screen.getByRole("button", { name: "Sources created (1)" }),
      ).toBeInTheDocument();
    });

    it("starts collapsed — the inner widget is not in the DOM until clicked", () => {
      const message = makeMessage({ role: "assistant", content: "Done." });
      render(
        <MessageBubble
          message={message}
          sourcesCreated={[makeSummary()]}
        />,
      );
      expect(
        screen.queryByTestId("inline-source-created-turn"),
      ).not.toBeInTheDocument();
    });

    it("expands to reveal the InlineSourceCreatedTurn when the toggle is clicked", async () => {
      const user = userEvent.setup();
      const message = makeMessage({ role: "assistant", content: "Done." });
      render(
        <MessageBubble
          message={message}
          sourcesCreated={[makeSummary()]}
        />,
      );
      await user.click(
        screen.getByRole("button", { name: "Sources created (1)" }),
      );
      expect(
        screen.getByTestId("inline-source-created-turn"),
      ).toBeInTheDocument();
    });

    it("renders a horizontal ruler between Tool calls and Sources created when BOTH are present", () => {
      const message = makeMessage({
        role: "assistant",
        content: "Done.",
        tool_calls: [
          {
            id: "tc-1",
            type: "function",
            function: { name: "list_models", arguments: "{}" },
          },
        ],
      });
      const { container } = render(
        <MessageBubble
          message={message}
          sourcesCreated={[makeSummary()]}
        />,
      );
      expect(container.querySelector("hr.message-group-separator")).not.toBeNull();
    });

    it("omits the horizontal ruler when only Sources created is present (no tool calls)", () => {
      // First-message hello-world shape: the LLM produces a dynamic source
      // from the user's prompt text directly, without invoking any tools.
      // A ruler in that case would float above nothing.
      const message = makeMessage({ role: "assistant", content: "Done." });
      const { container } = render(
        <MessageBubble
          message={message}
          sourcesCreated={[makeSummary()]}
        />,
      );
      expect(container.querySelector("hr.message-group-separator")).toBeNull();
    });

    it("omits the horizontal ruler when only Tool calls are present (no sources)", () => {
      const message = makeMessage({
        role: "assistant",
        content: "Done.",
        tool_calls: [
          {
            id: "tc-1",
            type: "function",
            function: { name: "list_models", arguments: "{}" },
          },
        ],
      });
      const { container } = render(<MessageBubble message={message} />);
      expect(container.querySelector("hr.message-group-separator")).toBeNull();
    });

    it("aria-expanded reflects the disclosure state", async () => {
      const user = userEvent.setup();
      const message = makeMessage({ role: "assistant", content: "Done." });
      render(
        <MessageBubble
          message={message}
          sourcesCreated={[makeSummary()]}
        />,
      );
      const toggle = screen.getByRole("button", { name: "Sources created (1)" });
      expect(toggle).toHaveAttribute("aria-expanded", "false");
      await user.click(toggle);
      expect(toggle).toHaveAttribute("aria-expanded", "true");
    });
  });
});
