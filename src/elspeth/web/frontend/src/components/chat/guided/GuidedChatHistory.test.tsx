// ============================================================================
// GuidedChatHistory -- per-step chat log regression coverage.
//
// The component renders TWO variants over the same wire data:
//
// FLAT (default; live guided) -- seven contracts:
//   1. Empty state: returns null (no DOM contribution) when chat_history is [].
//   2. Non-empty render: one <li> per ChatTurn in seq order.
//   3. Live region: role="log", aria-live="polite", aria-relevant="additions" —
//      new chat turns are announced to screen readers when appended.  Distinct
//      from ChatPanel's wizard log region; the two coexist.
//   4. Role-class hook: each <li> carries a CSS class scoped to the role
//      (`guided-chat-history-item--user` / `--assistant`) so styling can
//      differentiate without coupling to text content.
//   5. Step badge: each entry shows the step it was produced at, using the
//      SHARED stepLabels vocabulary — step_2_sink reads "Output" (the stepper's
//      word), never the old "Sink" copy.
//   6. Screen-reader role prefix: each content paragraph carries a visually
//      hidden "You said: " / "Assistant said: " span so SR users get the
//      conversation structure even though the visual badge is aria-hidden.
//   7. Out-of-order seq defense: rendering sorts by `seq` rather than array
//      index, so a backend returning entries out of order still produces a
//      stable, monotonic chat log.  Slice 5 guarantees monotonic seq; the
//      sort is belt-and-braces against future bugs.
//
// BUBBLES (tutorial workspace) -- same wire data + a11y contract, freeform's
// bubble idiom on new lightweight markup:
//   8.  Same role=log / aria-live / aria-relevant / accessible-name contract.
//   9.  Rows compose freeform's CSS classes: .message-row--user (right) /
//       .message-row--assistant (left); bubbles compose .bubble-user /
//       .bubble-assistant + .message-bubble-content(--user).
//   10. Assistant content renders through MarkdownRenderer (**bold** →
//       <strong>); user content stays PLAIN TEXT (the locked tutorial prompt's
//       authored newlines/URLs must survive pre-wrap, and literal markdown
//       characters must not be re-rendered as formatting).
//   11. sr-only author prefixes use freeform's register: "You said:" /
//       "ELSPETH said:".
//   12. Stage dividers: one per step-change boundary (including the transcript
//       start) — .bubble-system--stage rows with accessible text
//       "<Label> stage", NOT aria-hidden, no per-turn step badges. "Output"
//       vocabulary, "Sink" absent.
//   13. seq-order defense holds on the [data-seq] rows.
//   14. Empty state returns null.
//   15. Flat/bubbles never leak into each other: default (no variant prop)
//       renders the flat markup EXACTLY (live parity); bubbles renders no
//       .guided-chat-history* markup.
//
// Source of truth:
//   - types/guided.ts ChatTurn (role / content / seq / step / ts_iso)
//   - state_machine.py GuidedSession.chat_history (server-authoritative)
//   - stepLabels.ts GUIDED_STEP_LABELS (shared wizard-step vocabulary)
//   - MessageBubble.tsx / chat.css (bubble idiom + sr-only author register)
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

/** Four turns spanning two steps → exactly one step-change boundary plus the
    transcript-start boundary. */
const CROSS_STEP: ChatTurn[] = [
  { ...TURN_USER, seq: 0, step: "step_1_source" },
  { ...TURN_ASSISTANT, seq: 1, step: "step_1_source" },
  { ...TURN_USER, seq: 2, step: "step_2_sink", content: "what about outputs?" },
  { ...TURN_ASSISTANT, seq: 3, step: "step_2_sink", content: "ack" },
];

// ════════════════════════════════════════════════════════════════════════════
// FLAT VARIANT (default — live guided parity)
// ════════════════════════════════════════════════════════════════════════════

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

  it("flat variant renders assistant markdown, not raw markup (Explain replies carry headings/tables)", () => {
    const { container } = render(
      <GuidedChatHistory
        chatHistory={[
          {
            role: "user",
            content: "**not markdown** for user turns",
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-07-03T00:00:00Z",
          },
          {
            role: "assistant",
            content: "Settings **explained** here.",
            seq: 2,
            step: "step_1_source",
            ts_iso: "2026-07-03T00:00:01Z",
          },
        ]}
      />,
    );

    // Assistant: markdown rendered (strong element, no literal asterisks).
    const strong = container.querySelector(".guided-chat-history-item--assistant strong");
    expect(strong).not.toBeNull();
    expect(strong!.textContent).toBe("explained");
    // User: plain text preserved verbatim (never markdown-ised).
    expect(
      screen.getByText("**not markdown** for user turns"),
    ).toBeInTheDocument();
  });

  it("keeps the flat list markup when no variant prop is given (live parity)", () => {
    const { container } = render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    // The default is the live-guided flat idiom: .guided-chat-history wrapper,
    // <ol>, per-turn badges — and NONE of the tutorial bubble markup.
    expect(container.querySelector(".guided-chat-history")).not.toBeNull();
    expect(container.querySelector("ol.guided-chat-history-list")).not.toBeNull();
    expect(container.querySelector(".message-row")).toBeNull();
    expect(container.querySelector(".bubble")).toBeNull();
    expect(container.querySelector(".bubble-system--stage")).toBeNull();
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

// ── 5. Step badge (shared vocabulary) ────────────────────────────────────────

describe("GuidedChatHistory step badge", () => {
  it("renders the wizard step label for each entry", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} />);

    // Both turns are at step_1_source -> "Source"; both badges render.
    const sourceBadges = screen.getAllByText("Source");
    expect(sourceBadges.length).toBeGreaterThanOrEqual(2);
  });

  it("labels step_2_sink 'Output' (shared stepper vocabulary), never 'Sink'", () => {
    render(<GuidedChatHistory chatHistory={CROSS_STEP} />);

    // Both step labels appear, from the SHARED map: the flat variant's old
    // "Sink" copy diverged from the stepper and is gone.
    expect(screen.getAllByText("Source").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Output").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("Sink")).toBeNull();
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

// ════════════════════════════════════════════════════════════════════════════
// BUBBLES VARIANT (tutorial workspace)
// ════════════════════════════════════════════════════════════════════════════

// ── 8. Live-region contract + empty state ────────────────────────────────────

describe("GuidedChatHistory bubbles live region", () => {
  it("keeps the role=log contract (name, polite, additions)", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} variant="bubbles" />);

    const log = screen.getByRole("log", { name: "Step chat history" });
    expect(log).toBeInTheDocument();
    expect(log).toHaveAttribute("aria-live", "polite");
    expect(log).toHaveAttribute("aria-relevant", "additions");
  });

  it("returns null and contributes no DOM when chat_history is []", () => {
    const { container } = render(
      <GuidedChatHistory chatHistory={[]} variant="bubbles" />,
    );

    expect(screen.queryByRole("log")).not.toBeInTheDocument();
    expect(container).toBeEmptyDOMElement();
  });
});

// ── 9. Freeform bubble idiom ─────────────────────────────────────────────────

describe("GuidedChatHistory bubbles markup", () => {
  it("composes freeform's row + bubble classes per role", () => {
    const { container } = render(
      <GuidedChatHistory chatHistory={TWO_TURNS} variant="bubbles" />,
    );

    const userRow = container.querySelector(".message-row--user");
    const assistantRow = container.querySelector(".message-row--assistant");
    expect(userRow).not.toBeNull();
    expect(assistantRow).not.toBeNull();

    // User bubble: .bubble-user + the --user content cap (right-aligned,
    // pre-wrap — preserves the locked prompt's authored newlines).
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

  it("renders no flat-variant markup", () => {
    const { container } = render(
      <GuidedChatHistory chatHistory={TWO_TURNS} variant="bubbles" />,
    );

    expect(container.querySelector(".guided-chat-history")).toBeNull();
    expect(container.querySelector(".guided-chat-history-item")).toBeNull();
    // Per-turn step badges die in the bubbles variant — dividers replace them.
    expect(container.querySelector(".guided-chat-history-step")).toBeNull();
  });
});

// ── 10. Markdown: assistant only ─────────────────────────────────────────────

describe("GuidedChatHistory bubbles markdown", () => {
  it("renders assistant content through markdown but keeps user content literal", () => {
    const turns: ChatTurn[] = [
      { ...TURN_USER, content: "make **this** a pipeline" },
      { ...TURN_ASSISTANT, content: "I built **three** transforms." },
    ];
    const { container } = render(
      <GuidedChatHistory chatHistory={turns} variant="bubbles" />,
    );

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

// ── 11. sr-only author prefixes (freeform register) ──────────────────────────

describe("GuidedChatHistory bubbles SR prefixes", () => {
  it("prefixes each bubble with freeform's sr-only author register", () => {
    render(<GuidedChatHistory chatHistory={TWO_TURNS} variant="bubbles" />);

    const userPrefix = screen.getByText("You said:", { exact: false });
    expect(userPrefix).toBeInTheDocument();
    expect(userPrefix).toHaveClass("sr-only");

    // Freeform's register is "ELSPETH said:", not the flat variant's
    // "Assistant said:".
    const assistantPrefix = screen.getByText("ELSPETH said:", { exact: false });
    expect(assistantPrefix).toBeInTheDocument();
    expect(assistantPrefix).toHaveClass("sr-only");
    expect(screen.queryByText("Assistant said:", { exact: false })).toBeNull();
  });
});

// ── 12. Stage dividers ───────────────────────────────────────────────────────

describe("GuidedChatHistory bubbles stage dividers", () => {
  it("renders one divider per step-change boundary, in seq order", () => {
    const { container } = render(
      <GuidedChatHistory chatHistory={CROSS_STEP} variant="bubbles" />,
    );

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
    render(<GuidedChatHistory chatHistory={CROSS_STEP} variant="bubbles" />);

    expect(screen.getByText(/Output/)).toBeInTheDocument();
    expect(screen.queryByText(/Sink/)).toBeNull();
  });

  it("does not repeat the divider while consecutive turns share a step", () => {
    const { container } = render(
      <GuidedChatHistory chatHistory={TWO_TURNS} variant="bubbles" />,
    );

    // Single-step history → exactly one divider (the transcript start).
    const dividers = container.querySelectorAll(".bubble-system--stage");
    expect(dividers).toHaveLength(1);
    expect(dividers[0].textContent).toBe("Source stage");
  });
});

// ── 13. seq order on bubble rows ─────────────────────────────────────────────

describe("GuidedChatHistory bubbles seq ordering", () => {
  it("renders turn rows in seq order even when the input array is shuffled", () => {
    const shuffled: ChatTurn[] = [TURN_ASSISTANT, TURN_USER]; // seq 1, then seq 0
    const { container } = render(
      <GuidedChatHistory chatHistory={shuffled} variant="bubbles" />,
    );

    const rows = container.querySelectorAll("[data-seq]");
    expect(rows).toHaveLength(2);
    expect(rows[0].getAttribute("data-seq")).toBe("0");
    expect(rows[1].getAttribute("data-seq")).toBe("1");
    // The lower-seq row is the user's turn.
    expect(rows[0].classList.contains("message-row--user")).toBe(true);
    expect(rows[1].classList.contains("message-row--assistant")).toBe(true);
  });
});
