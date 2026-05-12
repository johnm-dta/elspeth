import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPanel } from "./ChatPanel";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";
import { useComposer } from "@/hooks/useComposer";
import type { ChatMessage, ComposerProgressSnapshot, Session } from "@/types/api";
import type {
  GuidedSession,
  SingleSelectPayload,
  TerminalState,
  TurnPayload,
  TurnRecord,
} from "@/types/guided";

vi.mock("@/hooks/useComposer", () => ({
  useComposer: vi.fn(),
}));

vi.mock("./MessageBubble", () => ({
  MessageBubble: ({ message }: { message: ChatMessage }) => (
    <div data-testid="message-bubble">{message.content}</div>
  ),
}));

vi.mock("./ChatInput", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}));

vi.mock("./TemplateCards", () => ({
  TemplateCards: () => <div data-testid="template-cards" />,
}));

vi.mock("@/components/blobs/BlobManager", () => ({
  BlobManager: () => <div data-testid="blob-manager" />,
}));

describe("ChatPanel", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: true,
      compositionState: null,
      error: null,
    });
  });

  it("passes backend composer progress to the composing indicator", () => {
    const session: Session = {
      id: "session-1",
      title: "Composer session",
      created_at: "2026-04-26T10:00:00Z",
      updated_at: "2026-04-26T10:00:00Z",
    };
    const userMessage: ChatMessage = {
      id: "message-1",
      session_id: "session-1",
      role: "user",
      content: "Exploit this HTML into JSON",
      tool_calls: null,
      created_at: "2026-04-26T10:00:01Z",
      local_status: "pending",
    };
    const progress: ComposerProgressSnapshot = {
      session_id: "session-1",
      request_id: "message-1",
      phase: "using_tools",
      headline: "The model requested plugin schemas.",
      evidence: ["Checking available source, transform, and sink tools."],
      likely_next: "ELSPETH will use the schemas to choose a pipeline shape.",
      reason: null,
      updated_at: "2026-04-26T10:00:02Z",
    };

    useSessionStore.setState({
      activeSessionId: "session-1",
      sessions: [session],
      messages: [userMessage],
      composerProgress: progress,
    });

    render(<ChatPanel />);

    expect(screen.getByText("The model requested plugin schemas.")).toBeInTheDocument();
    expect(screen.getByText("Checking available source, transform, and sink tools.")).toBeInTheDocument();
    expect(screen.queryByText("Working on: convert HTML into JSON")).not.toBeInTheDocument();
  });
});

// ── Mode discriminator tests (Task 8.1) ─────────────────────────────────────────
//
// These cover the top-level mode switch added to ChatPanel:
//   1. completed terminal  -> CompletionSummary surface
//   2. guided-active       -> GuidedTurn + ExitToFreeformButton surface
//   3. exited_to_freeform  -> falls through to freeform body
//   4. defensive (guidedSession set, guidedNextTurn null) -> falls through
//   5. no guided session   -> freeform body (existing behaviour)
//
// Fixtures use the exact wire shapes from src/types/guided.ts.  GuidedTurn
// is exercised via SingleSelectTurn (which renders <fieldset> with implicit
// role="group").
describe("ChatPanel mode discriminator", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: null,
    });
  });

  const guidedSessionFixture: Session = {
    id: "session-guided",
    title: "Guided composer session",
    created_at: "2026-05-12T10:00:00Z",
    updated_at: "2026-05-12T10:00:00Z",
  };

  function activeGuidedSession(): GuidedSession {
    return {
      step: "step_1_source",
      history: [],
      terminal: null,
    };
  }

  function singleSelectTurn(): TurnPayload {
    const payload: SingleSelectPayload = {
      question: "Which source plugin should we use?",
      options: [
        { id: "csv", label: "CSV", hint: null },
        { id: "api", label: "API", hint: null },
      ],
      allow_custom: false,
    };
    return { type: "single_select", step_index: 0, payload };
  }

  it("renders guided-active surface (GuidedTurn + ExitToFreeformButton) when guidedSession is active and next turn is present", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(<ChatPanel />);

    // GuidedTurn → SingleSelectTurn renders a <fieldset> (implicit role="group")
    // with the question as the accessible name (via <legend>).
    expect(
      screen.getByRole("group", { name: "Which source plugin should we use?" }),
    ).toBeInTheDocument();

    // ExitToFreeformButton present.
    expect(
      screen.getByRole("button", { name: "Exit to freeform" }),
    ).toBeInTheDocument();

    // Discriminator-anchored class assertion: pins the guided-active branch to
    // its own output (the `chat-panel--guided` class on `#chat-main`), parallel
    // to how the completed-branch test asserts `chat-panel--completed`. Without
    // this, a future change to `SingleSelectTurn`'s DOM (e.g., swap to
    // `<div role="radiogroup">`) would break the `getByRole("group", …)` check
    // and the failure would point at the discriminator test rather than at the
    // widget. The class assertion fails only when the discriminator itself
    // misroutes.
    const chatMain = container.querySelector("#chat-main");
    expect(chatMain).not.toBeNull();
    expect(chatMain?.classList.contains("chat-panel--guided")).toBe(true);

    // Freeform surface suppressed.
    expect(screen.queryByTestId("chat-input")).not.toBeInTheDocument();
  });

  it("renders CompletionSummary surface when terminal.kind === 'completed'", () => {
    const completedHistory: TurnRecord[] = [
      {
        step: "step_1_source",
        turn_type: "single_select",
        payload_hash: "h1",
        response_hash: "r1",
        emitter: "server",
      },
    ];
    const terminal: TerminalState = {
      kind: "completed",
      reason: null,
      pipeline_yaml: "source:\n  plugin: csv\n",
    };

    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        step: "step_3_transforms",
        history: completedHistory,
        terminal,
      },
      guidedTerminal: terminal,
    });

    const { container } = render(<ChatPanel />);

    // CompletionSummary renders both action buttons.
    expect(
      screen.getByRole("button", { name: "Save and exit" }),
    ).toBeInTheDocument();

    // Container carries the per-branch CSS hook AND preserves the skip-link anchor.
    const chatMain = container.querySelector("#chat-main");
    expect(chatMain).not.toBeNull();
    expect(chatMain?.classList.contains("chat-panel--completed")).toBe(true);

    // Freeform surface suppressed.
    expect(screen.queryByTestId("chat-input")).not.toBeInTheDocument();
  });

  it("does not render ExitToFreeformButton on the completed surface (regression pin for elspeth-obs-0a1002de6d)", () => {
    // Regression pin for observation `elspeth-obs-0a1002de6d`. The completed
    // branch renders `CompletionSummary` alone — it must NOT also render
    // `<ExitToFreeformButton />`. If a future change forgets the if/else split
    // and rehoists the button above the discriminator, a button with label
    // "Exit to freeform" will appear on the completed surface alongside
    // `CompletionSummary`'s "Save and exit" and "Drop to freeform to keep
    // editing" buttons (which have wire-identical semantics but different UX
    // framing). This `queryByRole` predicate catches that regression — it
    // returns `null` when the button is absent (correct state) and a non-null
    // element when present (regression).
    const terminal: TerminalState = {
      kind: "completed",
      reason: null,
      pipeline_yaml: "source:\n  plugin: csv\n",
    };

    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        step: "step_3_transforms",
        history: [],
        terminal,
      },
      guidedTerminal: terminal,
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("button", { name: "Exit to freeform" }),
    ).toBeNull();
  });

  it("renders the existing freeform body when guidedSession is null", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: null,
      guidedNextTurn: null,
    });

    render(<ChatPanel />);

    // ChatInput is mocked at file scope with data-testid="chat-input".
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    // Neither guided nor completed surface should be present.
    expect(
      screen.queryByRole("button", { name: "Save and exit" }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Exit to freeform" }),
    ).toBeNull();
  });

  it("falls through to the freeform body when terminal.kind === 'exited_to_freeform'", () => {
    const terminal: TerminalState = {
      kind: "exited_to_freeform",
      reason: "user_pressed_exit",
      pipeline_yaml: null,
    };

    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        step: "step_1_source",
        history: [],
        terminal,
      },
      guidedNextTurn: null,
      guidedTerminal: terminal,
    });

    render(<ChatPanel />);

    // Freeform surface visible.
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    // CompletionSummary NOT rendered (its guard fails on kind !== "completed").
    expect(
      screen.queryByRole("button", { name: "Save and exit" }),
    ).toBeNull();
  });

  it("wraps the guided turn surface in a role=log aria-live=polite region (Task 8.2 a11y)", () => {
    // Task 8.2 a11y fix: the guided branch must wrap the live turn surface in
    // role="log" aria-live="polite" so screen readers are notified when a new
    // turn arrives.  InspectAndConfirmTurn.tsx:39-46 documents the dependency
    // (its warnings <aside> omits its own live region in favour of the parent).
    // This test pins the contract on the parent side.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(<ChatPanel />);

    // The log region is the inner wrapper around <GuidedTurn>.
    const logRegion = container.querySelector<HTMLElement>(
      '[role="log"][aria-live="polite"]',
    );
    expect(logRegion).not.toBeNull();
    // Co-attributes that complete the contract (parallel to chat-panel-messages
    // in the freeform body).
    expect(logRegion?.getAttribute("aria-relevant")).toBe("additions");
    expect(logRegion?.getAttribute("aria-label")).toBe("Guided wizard step");

    // The log region contains the turn surface (GuidedTurn → SingleSelectTurn's
    // <fieldset> with the question as accessible name).
    const fieldset = logRegion?.querySelector("fieldset");
    expect(fieldset).not.toBeNull();
    expect(fieldset?.textContent).toContain("Which source plugin should we use?");

    // ExitToFreeformButton lives OUTSIDE the log region (persistent affordance,
    // not "new content" on turn change).  Document the layout decision via test.
    const exitButton = screen.getByRole("button", { name: "Exit to freeform" });
    expect(logRegion?.contains(exitButton)).toBe(false);
  });

  it("does not add a log region on the completed surface (regression pin for Task 8.2 a11y scope)", () => {
    // The completed branch shows a static summary — no new turns ever arrive,
    // so there must be no aria-live log region.  This test prevents an
    // over-zealous future refactor from rehoisting the log wrapper above the
    // discriminator and announcing the completion summary as if it were a
    // turn arrival event.
    const terminal: TerminalState = {
      kind: "completed",
      reason: null,
      pipeline_yaml: "source:\n  plugin: csv\n",
    };

    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        step: "step_3_transforms",
        history: [],
        terminal,
      },
      guidedTerminal: terminal,
    });

    render(<ChatPanel />);

    // No live region on the completed surface.
    expect(screen.queryByRole("log")).toBeNull();
  });

  it("falls through to the freeform body when guidedSession is active but guidedNextTurn is null", () => {
    // Defensive: this transient invariant should not crash the surface by passing
    // null into GuidedTurn — it should fall through to freeform.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: null,
    });

    render(<ChatPanel />);

    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    // No guided-mode chrome rendered.
    expect(
      screen.queryByRole("button", { name: "Exit to freeform" }),
    ).toBeNull();
  });
});
