// ============================================================================
// InlineSourceFallbackPrompt.test.tsx — Phase 5a Task 5
//
// Widget tests for the LLM-skip safety-net prompt. Mirrors the test shape of
// InlineSourceCreatedTurn.test.tsx and InlineSourceDisambiguationTurn.test.tsx
// — same role="region" surface, same load-bearing accessible-name pattern.
//
// The widget renders the fallback affordance ABOVE the chat input when the
// ChatPanel predicate determines the user's recent typed text looks source-
// shaped but the LLM has failed to propose an inline-blob source for it. The
// predicate itself (and the F-20 dismiss-persistence gate) lives in
// ChatPanel.tsx; the widget is a dumb render of a boolean + a candidate
// string + two callbacks.
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InlineSourceFallbackPrompt } from "./InlineSourceFallbackPrompt";

describe("InlineSourceFallbackPrompt", () => {
  it("does not render when the predicate is false", () => {
    const { container } = render(
      <InlineSourceFallbackPrompt
        shouldRender={false}
        candidateText="https://example.com"
        onAccept={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders the call-to-action when the predicate is true", () => {
    render(
      <InlineSourceFallbackPrompt
        shouldRender={true}
        candidateText="https://example.com"
        onAccept={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByText(/looks like source data/i)).toBeInTheDocument();
  });

  it("exposes the load-bearing region role + aria-label", () => {
    render(
      <InlineSourceFallbackPrompt
        shouldRender={true}
        candidateText="https://example.com"
        onAccept={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    // The ChatPanel wiring tests query this region by name; keep
    // "Inline source fallback prompt" stable.
    expect(
      screen.getByRole("region", { name: /inline source fallback prompt/i }),
    ).toBeInTheDocument();
  });

  it("calls onAccept with the candidate text when the user accepts", () => {
    const onAccept = vi.fn();
    render(
      <InlineSourceFallbackPrompt
        shouldRender={true}
        candidateText="https://example.com"
        onAccept={onAccept}
        onDismiss={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /create source/i }));
    expect(onAccept).toHaveBeenCalledWith("https://example.com");
  });

  it("calls onDismiss when the user dismisses", () => {
    const onDismiss = vi.fn();
    render(
      <InlineSourceFallbackPrompt
        shouldRender={true}
        candidateText="x"
        onAccept={vi.fn()}
        onDismiss={onDismiss}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalled();
  });
});
