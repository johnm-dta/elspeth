// ============================================================================
// GuidedChatHistory -- per-step chat log regression coverage (Phase A slice 6).
//
// Pins SEVEN contracts:
//   1. Empty state: returns null (no DOM contribution) when chat_history is [].
//   2. Non-empty render: one <li> per ChatTurn in seq order.
//   3. Live region: role="log", aria-live="polite", aria-relevant="additions" —
//      new chat turns are announced to screen readers when appended.  Distinct
//      from ChatPanel's wizard log region; the two coexist.
//   4. Role-class hook: each <li> carries a CSS class scoped to the role
//      (`guided-chat-history-item--user` / `--assistant`) so styling can
//      differentiate without coupling to text content.
//   5. Step badge: each entry shows the step it was produced at (load-bearing
//      for Phase A.5 when chat can span step changes after back-button).
//   6. Screen-reader role prefix: each content paragraph carries a visually
//      hidden "You said: " / "Assistant said: " span so SR users get the
//      conversation structure even though the visual badge is aria-hidden.
//   7. Out-of-order seq defense: rendering sorts by `seq` rather than array
//      index, so a backend returning entries out of order still produces a
//      stable, monotonic chat log.  Slice 5 guarantees monotonic seq; the
//      sort is belt-and-braces against future bugs.
//
// Source of truth:
//   - types/guided.ts ChatTurn (role / content / seq / step / ts_iso)
//   - state_machine.py GuidedSession.chat_history (server-authoritative)
//   - GuidedHistory.test.tsx (sibling component test pattern)
// ============================================================================

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GuidedChatHistory } from "./GuidedChatHistory";
import type { ChatTurn } from "@/types/guided";

// ── Fixtures ──────────────────────────────────────────────────────────────────

const TURN_USER: ChatTurn = {
  role: "user",
  content: "what columns are in this CSV?",
  seq: 0,
  step: "step_1_source",
  ts_iso: "2026-05-13T12:00:00+00:00",
};

const TURN_ASSISTANT: ChatTurn = {
  role: "assistant",
  content: "The CSV has price, quantity, and timestamp columns.",
  seq: 1,
  step: "step_1_source",
  ts_iso: "2026-05-13T12:00:00+00:00",
};

const TWO_TURNS: ChatTurn[] = [TURN_USER, TURN_ASSISTANT];

// ── 1. Empty state ───────────────────────────────────────────────────────────

describe("GuidedChatHistory empty state", () => {
  it("returns null and contributes no DOM when chat_history is []", () => {
    const { container } = render(<GuidedChatHistory chatHistory={[]} />);

    // No live region, no list, no entries.
    expect(screen.queryByRole("log")).not.toBeInTheDocument();
    expect(container).toBeEmptyDOMElement();
  });
});

// ── 2-3. Live region + ordered render ────────────────────────────────────────

describe("GuidedChatHistory with chat turns", () => {
  it("renders a role=log live region with aria-live=polite + aria-relevant=additions", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    const log = screen.getByRole("log", { name: "Step chat history" });
    expect(log).toBeInTheDocument();
    expect(log).toHaveAttribute("aria-live", "polite");
    expect(log).toHaveAttribute("aria-relevant", "additions");
  });

  it("renders one list item per ChatTurn", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
  });

  it("shows the literal content of each turn", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    expect(screen.getByText("what columns are in this CSV?")).toBeInTheDocument();
    expect(
      screen.getByText("The CSV has price, quantity, and timestamp columns."),
    ).toBeInTheDocument();
  });
});

// ── 4. Role-class hook ────────────────────────────────────────────────────────

describe("GuidedChatHistory role classes", () => {
  it("scopes a role-specific CSS class to each <li>", () => {
    const { container } = render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    const items = container.querySelectorAll(".guided-chat-history-item");
    expect(items).toHaveLength(2);
    expect(items[0].classList.contains("guided-chat-history-item--user")).toBe(true);
    expect(items[1].classList.contains("guided-chat-history-item--assistant")).toBe(true);
  });
});

// ── 5. Step badge ────────────────────────────────────────────────────────────

describe("GuidedChatHistory step badge", () => {
  it("renders the wizard step label for each entry", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    // Both turns are at step_1_source -> "Source"; both badges render.
    const sourceBadges = screen.getAllByText("Source");
    expect(sourceBadges.length).toBeGreaterThanOrEqual(2);
  });

  it("renders distinct step labels when turns span multiple steps", () => {
    const crossStep: ChatTurn[] = [
      { ...TURN_USER, seq: 0, step: "step_1_source" },
      { ...TURN_ASSISTANT, seq: 1, step: "step_1_source" },
      { ...TURN_USER, seq: 2, step: "step_2_sink", content: "what about sinks?" },
      { ...TURN_ASSISTANT, seq: 3, step: "step_2_sink", content: "ack" },
    ];

    render(<GuidedChatHistory chatHistory={crossStep} />);

    // Both step labels appear.
    expect(screen.getAllByText("Source").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Sink").length).toBeGreaterThanOrEqual(2);
  });
});

// ── 6. Screen-reader role prefix ─────────────────────────────────────────────

describe("GuidedChatHistory SR role prefix", () => {
  it("renders a visually hidden role prefix on each content paragraph", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    // "You said: " — present in DOM for SR, visually hidden by class.
    const userPrefix = screen.getByText("You said:", { exact: false });
    expect(userPrefix).toBeInTheDocument();
    expect(userPrefix).toHaveClass("visually-hidden");

    const assistantPrefix = screen.getByText("Assistant said:", { exact: false });
    expect(assistantPrefix).toBeInTheDocument();
    expect(assistantPrefix).toHaveClass("visually-hidden");
  });
});

// ── 7. Out-of-order seq defense ──────────────────────────────────────────────

describe("GuidedChatHistory seq ordering", () => {
  it("renders entries in seq order even when the input array is shuffled", () => {
    const shuffled: ChatTurn[] = [TURN_ASSISTANT, TURN_USER]; // seq 1, then seq 0
    const { container } = render(<GuidedChatHistory chatHistory={shuffled} />);

    // First DOM <li> has the lower seq (user message); second has the higher.
    const items = container.querySelectorAll(".guided-chat-history-item");
    expect(items[0].getAttribute("data-seq")).toBe("0");
    expect(items[1].getAttribute("data-seq")).toBe("1");
  });
});
