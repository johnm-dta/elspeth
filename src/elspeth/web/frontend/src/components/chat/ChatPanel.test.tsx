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

    render(<ChatPanel />);

    // GuidedTurn → SingleSelectTurn renders a <fieldset> (implicit role="group")
    // with the question as the accessible name (via <legend>).
    expect(
      screen.getByRole("group", { name: "Which source plugin should we use?" }),
    ).toBeInTheDocument();

    // ExitToFreeformButton present.
    expect(
      screen.getByRole("button", { name: "Exit to freeform" }),
    ).toBeInTheDocument();

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
    // Same fixture as the completed-surface test, plus an asserted absence.
    // The button label "Exit to freeform" is identical to ExitToFreeformButton.tsx.
    // If the discriminator forgot to suppress ExitToFreeformButton in the completed
    // branch, three identical-action buttons would coexist on the completed surface
    // (Save and exit, Drop to freeform to keep editing, Exit to freeform).
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
