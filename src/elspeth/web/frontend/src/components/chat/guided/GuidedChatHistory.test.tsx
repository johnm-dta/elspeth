// ============================================================================
// GuidedChatHistory -- per-step chat log regression coverage.
//
// One rendering idiom (the tutorial-workspace bubbles, promoted to the only
// guided transcript when the workspace became the one guided layout —
// the flat <ol> list died with the pre-workspace flat layout):
//
//   1. Empty state: returns null (no DOM contribution) when chat_history
//      is [].
//   2. Live region: role="log", aria-live="polite",
//      aria-relevant="additions" — new chat turns are announced to screen
//      readers when appended.  Distinct from ChatPanel's wizard log region;
//      the two coexist.
//   3. Rows compose freeform's CSS classes: .message-row--user (right) /
//      .message-row--assistant (left); bubbles compose .bubble-user /
//      .bubble-assistant + .message-bubble-content(--user).
//   4. Assistant content renders through MarkdownRenderer (**bold** →
//      <strong>); user content stays PLAIN TEXT (the tutorial's locked
//      prompt's authored newlines/URLs must survive pre-wrap, and literal
//      markdown characters must not be re-rendered as formatting).
//   5. sr-only author prefixes use freeform's register: "You said:" /
//      "ELSPETH said:".
//   6. Stage dividers: one per step-change boundary (including the
//      transcript start) — .bubble-system--stage rows with accessible text
//      "<Label> stage", NOT aria-hidden, no per-turn step badges. "Output"
//      vocabulary, "Sink" absent (shared stepLabels map).
//   7. Out-of-order seq defense: rendering sorts by `seq` rather than array
//      index, so a backend returning entries out of order still produces a
//      stable, monotonic chat log.  Slice 5 guarantees monotonic seq; the
//      sort is belt-and-braces against future bugs.
//   8. The flat-list markup is GONE: no .guided-chat-history* classes leak
//      out of history.
//
// Source of truth:
//   - types/guided.ts ChatTurn (role / content / seq / step / ts_iso)
//   - state_machine.py GuidedSession.chat_history (server-authoritative)
//   - stepLabels.ts GUIDED_STEP_LABELS (shared wizard-step vocabulary)
//   - MessageBubble.tsx / chat.css (bubble idiom + sr-only author register)
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
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

/** Four turns spanning two steps → exactly one step-change boundary plus the
    transcript-start boundary. */
const CROSS_STEP: ChatTurn[] = [
  { ...TURN_USER, seq: 0, step: "step_1_source" },
  { ...TURN_ASSISTANT, seq: 1, step: "step_1_source" },
  { ...TURN_USER, seq: 2, step: "step_2_sink", content: "what about outputs?" },
  { ...TURN_ASSISTANT, seq: 3, step: "step_2_sink", content: "ack" },
];

// ── 1. Empty state ───────────────────────────────────────────────────────────

describe("GuidedChatHistory empty state", () => {
  it("returns null and contributes no DOM when chat_history is []", () => {
    const { container } = render(<GuidedChatHistory chatHistory={[]} />);

    expect(screen.queryByRole("log")).not.toBeInTheDocument();
    expect(container).toBeEmptyDOMElement();
  });
});

// ── 2. Live-region contract ──────────────────────────────────────────────────

describe("GuidedChatHistory live region", () => {
  it("renders a role=log live region with aria-live=polite + aria-relevant=additions", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    const log = screen.getByRole("log", { name: "Step chat history" });
    expect(log).toBeInTheDocument();
    expect(log).toHaveAttribute("aria-live", "polite");
    expect(log).toHaveAttribute("aria-relevant", "additions");
  });

  it("shows the literal content of each turn", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    expect(screen.getByText("what columns are in this CSV?")).toBeInTheDocument();
    expect(
      screen.getByText("The CSV has price, quantity, and timestamp columns."),
    ).toBeInTheDocument();
  });
});

// ── 3. Freeform bubble idiom ─────────────────────────────────────────────────

describe("GuidedChatHistory bubble markup", () => {
  it("composes freeform's row + bubble classes per role", () => {
    const { container } = render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    const userRow = container.querySelector(".message-row--user");
    const assistantRow = container.querySelector(".message-row--assistant");
    expect(userRow).not.toBeNull();
    expect(assistantRow).not.toBeNull();

    // User bubble: .bubble-user + the --user content cap (right-aligned,
    // pre-wrap — preserves the tutorial locked prompt's authored newlines).
    const userBubble = userRow!.querySelector(".bubble");
    expect(userBubble).not.toBeNull();
    expect(userBubble!.classList.contains("bubble-user")).toBe(true);
    expect(userBubble!.classList.contains("message-bubble-content")).toBe(true);
    expect(userBubble!.classList.contains("message-bubble-content--user")).toBe(true);

    // Assistant bubble: .bubble-assistant, content cap without the --user
    // pre-wrap modifier (markdown structures its own breaks).
    const assistantBubble = assistantRow!.querySelector(".bubble");
    expect(assistantBubble).not.toBeNull();
    expect(assistantBubble!.classList.contains("bubble-assistant")).toBe(true);
    expect(assistantBubble!.classList.contains("message-bubble-content")).toBe(true);
    expect(assistantBubble!.classList.contains("message-bubble-content--user")).toBe(false);
  });

  it("renders no flat-list markup (died with the pre-workspace layout)", () => {
    const { container } = render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    expect(container.querySelector(".guided-chat-history")).toBeNull();
    expect(container.querySelector(".guided-chat-history-item")).toBeNull();
    // Per-turn step badges are gone too — stage dividers replace them.
    expect(container.querySelector(".guided-chat-history-step")).toBeNull();
  });
});

// ── 4. Markdown: assistant only ──────────────────────────────────────────────

describe("GuidedChatHistory markdown", () => {
  it("renders assistant content through markdown but keeps user content literal", () => {
    const turns: ChatTurn[] = [
      { ...TURN_USER, content: "make **this** a pipeline" },
      { ...TURN_ASSISTANT, content: "I built **three** transforms." },
    ];
    const { container } = render(<GuidedChatHistory chatHistory={turns} />);

    // Assistant: **three** became <strong> inside a .markdown-body wrapper
    // (the wrapper also carries chat.css's white-space reset — load-bearing).
    const assistantBubble = container.querySelector(".bubble-assistant");
    expect(assistantBubble!.querySelector(".markdown-body")).not.toBeNull();
    const strong = assistantBubble!.querySelector("strong");
    expect(strong).not.toBeNull();
    expect(strong!.textContent).toBe("three");

    // User: literal text, asterisks intact, no markdown rendering.
    const userBubble = container.querySelector(".bubble-user");
    expect(userBubble!.querySelector("strong")).toBeNull();
    expect(userBubble!.querySelector(".markdown-body")).toBeNull();
    expect(userBubble!.textContent).toContain("make **this** a pipeline");
  });
});

// ── 5. sr-only author prefixes (freeform register) ───────────────────────────

describe("GuidedChatHistory SR prefixes", () => {
  it("prefixes each bubble with freeform's sr-only author register", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    const userPrefix = screen.getByText("You said:", { exact: false });
    expect(userPrefix).toBeInTheDocument();
    expect(userPrefix).toHaveClass("sr-only");

    // Freeform's register is "ELSPETH said:", not the dead flat list's
    // "Assistant said:".
    const assistantPrefix = screen.getByText("ELSPETH said:", { exact: false });
    expect(assistantPrefix).toBeInTheDocument();
    expect(assistantPrefix).toHaveClass("sr-only");
    expect(screen.queryByText("Assistant said:", { exact: false })).toBeNull();
  });
});

// ── 6. Stage dividers ────────────────────────────────────────────────────────

describe("GuidedChatHistory stage dividers", () => {
  it("renders one divider per step-change boundary, in seq order", () => {
    const { container } = render(<GuidedChatHistory chatHistory={CROSS_STEP} />);

    // Two boundaries: transcript start (Source stage) + the step_1→step_2
    // change (Output stage). NOT one per turn.
    const dividers = container.querySelectorAll(".bubble-system--stage");
    expect(dividers).toHaveLength(2);
    expect(dividers[0].textContent).toBe("Source stage");
    expect(dividers[1].textContent).toBe("Output stage");

    // Divider rows reuse the centred system-row visual.
    for (const divider of dividers) {
      const row = divider.closest(".message-row");
      expect(row).not.toBeNull();
      expect(row!.classList.contains("message-row--system")).toBe(true);
      // Announced once via the log's additions semantics — never aria-hidden.
      expect(divider.getAttribute("aria-hidden")).toBeNull();
      expect(row!.getAttribute("aria-hidden")).toBeNull();
    }

    // The " stage" suffix is sr-only; the visible label is just the step name.
    const suffix = dividers[0].querySelector(".sr-only");
    expect(suffix).not.toBeNull();
    expect(suffix!.textContent).toBe(" stage");
  });

  it("uses the shared 'Output' vocabulary — 'Sink' never appears", () => {
    render(<GuidedChatHistory chatHistory={CROSS_STEP} />);

    expect(screen.getByText(/Output/)).toBeInTheDocument();
    expect(screen.queryByText(/Sink/)).toBeNull();
  });

  it("does not repeat the divider while consecutive turns share a step", () => {
    const { container } = render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    // Single-step history → exactly one divider (the transcript start).
    const dividers = container.querySelectorAll(".bubble-system--stage");
    expect(dividers).toHaveLength(1);
    expect(dividers[0].textContent).toBe("Source stage");
  });
});

// ── 7. seq order on bubble rows ──────────────────────────────────────────────

describe("GuidedChatHistory seq ordering", () => {
  it("renders turn rows in seq order even when the input array is shuffled", () => {
    const shuffled: ChatTurn[] = [TURN_ASSISTANT, TURN_USER]; // seq 1, then seq 0
    const { container } = render(<GuidedChatHistory chatHistory={shuffled} />);

    const rows = container.querySelectorAll("[data-seq]");
    expect(rows).toHaveLength(2);
    expect(rows[0].getAttribute("data-seq")).toBe("0");
    expect(rows[1].getAttribute("data-seq")).toBe("1");
    // The lower-seq row is the user's turn.
    expect(rows[0].classList.contains("message-row--user")).toBe(true);
    expect(rows[1].classList.contains("message-row--assistant")).toBe(true);
  });
});

// ── 8. Synthetic-failure turns (C-2) ─────────────────────────────────────────

const TURN_SYNTHETIC_FAILURE: ChatTurn = {
  role: "assistant",
  content: "I'm unavailable right now; you can still use the wizard controls.",
  seq: 1,
  step: "step_1_source",
  ts_iso: "2026-05-13T12:00:00+00:00",
  assistant_message_kind: "synthetic_failure",
};

describe("GuidedChatHistory synthetic-failure turns", () => {
  it("never renders the ELSPETH-said assistant bubble for a synthetic-failure turn", () => {
    const { container } = render(
      <GuidedChatHistory chatHistory={[TURN_USER, TURN_SYNTHETIC_FAILURE]} />,
    );

    // No "ELSPETH said:" prefix anywhere — the synthetic turn must not read
    // as a real assistant reply.
    expect(screen.queryByText("ELSPETH said:", { exact: false })).toBeNull();
    expect(container.querySelector(".bubble-assistant")).toBeNull();
  });

  it("renders a distinct error bubble carrying the server's message, without an assertive alert role", () => {
    const { container } = render(
      <GuidedChatHistory chatHistory={[TURN_SYNTHETIC_FAILURE]} />,
    );

    // The error bubble is visually and semantically distinct (bubble-error +
    // sr-only "Error:" prefix), but it is NOT role="alert": these turns are
    // persisted history, so an assertive alert would re-announce every past
    // failure on each remount. Live announcement of a new failure is the
    // parent transcript log's job (aria-live="polite").
    expect(screen.queryByRole("alert")).toBeNull();
    const bubble = container.querySelector(".bubble-error");
    expect(bubble).not.toBeNull();
    expect(bubble).toHaveTextContent(
      "I'm unavailable right now; you can still use the wizard controls.",
    );
    expect(bubble).toHaveTextContent("Error:");
  });

  it("omits the Retry button when no onRetrySyntheticFailure handler is supplied", () => {
    render(<GuidedChatHistory chatHistory={[TURN_SYNTHETIC_FAILURE]} />);

    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("Retry calls the handler with the failed turn", () => {
    const onRetry = vi.fn();
    render(
      <GuidedChatHistory
        chatHistory={[TURN_SYNTHETIC_FAILURE]}
        onRetrySyntheticFailure={onRetry}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onRetry).toHaveBeenCalledWith(TURN_SYNTHETIC_FAILURE);
  });

  it("disables Retry while retryDisabled is set (no race with an in-flight resend)", () => {
    render(
      <GuidedChatHistory
        chatHistory={[TURN_SYNTHETIC_FAILURE]}
        onRetrySyntheticFailure={vi.fn()}
        retryDisabled
      />,
    );

    expect(screen.getByRole("button", { name: "Retry" })).toBeDisabled();
  });

  it("still emits the step-change stage divider around a synthetic-failure turn", () => {
    const { container } = render(
      <GuidedChatHistory chatHistory={[TURN_SYNTHETIC_FAILURE]} />,
    );

    expect(container.querySelector(".bubble-system--stage")).not.toBeNull();
  });
});
