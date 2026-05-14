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
});
