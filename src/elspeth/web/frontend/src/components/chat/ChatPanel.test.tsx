import { act, render, screen, waitFor } from "@testing-library/react";
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

// Mock surfaces both the placeholder and the onSend wiring as inspectable
// DOM properties so per-step / per-mode tests can assert without depending
// on the real ChatInput's textarea internals. Phase A slice 4 needs:
//   - per-step placeholder visible (guided-active branch)
//   - onSend callback wired so the test can simulate user submit
vi.mock("./ChatInput", () => ({
  ChatInput: ({
    placeholder,
    onSend,
    disabled,
  }: {
    placeholder?: string;
    onSend?: (content: string) => void;
    disabled?: boolean;
  }) => (
    <button
      type="button"
      data-testid="chat-input"
      data-placeholder={placeholder ?? ""}
      data-disabled={disabled ? "true" : "false"}
      onClick={() => onSend?.("test-chat-message")}
    >
      {placeholder ?? ""}
    </button>
  ),
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
      chat_history: [],
      chat_turn_seq: 0,
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

    // Phase A slice 4: ChatInput is rendered INSIDE the guided-active
    // branch (below GuidedTurn + ExitToFreeformButton) so the user can
    // ask scoped advisory questions of the LLM.  Previously this branch
    // suppressed all ChatInput surfaces; the assertion is now positive.
    // Per-step placeholder + onSend wiring are exercised in the two
    // dedicated tests below.
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
  });

  it("renders a persistent guided workflow stepper with the active step marked", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    const workflow = screen.getByRole("list", { name: /guided workflow/i });
    for (const label of ["Source", "Output", "Recipe", "Transforms", "Ready"]) {
      expect(workflow).toHaveTextContent(label);
    }
    expect(screen.getByRole("listitem", { current: "step" })).toHaveTextContent(
      "Source",
    );
  });

  it("renders the active turn in a current-decision panel with step purpose copy", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    expect(
      screen.getByRole("heading", { name: /current decision/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/choose the input and confirm what elspeth can read/i),
    ).toBeInTheDocument();
  });

  it("visually separates guided sidecar chat as ask about this step", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    const chatRegion = screen.getByRole("region", {
      name: /ask about this step/i,
    });
    expect(chatRegion).toContainElement(screen.getByTestId("chat-input"));
  });

  it("renders guided errors with the same alert banner as freeform mode", () => {
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: "Failed to submit guided response. Please try again.",
    });
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Failed to submit guided response. Please try again.",
    );
  });

  it("disables guided turn buttons while a guided response is pending", async () => {
    const respondGuidedSpy = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedResponsePending: true,
      respondGuided: respondGuidedSpy,
    });

    render(<ChatPanel />);

    const csvButton = screen.getByRole("button", { name: "CSV" });
    expect(csvButton).toBeDisabled();
    await act(async () => {
      csvButton.click();
    });
    expect(respondGuidedSpy).not.toHaveBeenCalled();
  });

  it("renders the per-step placeholder for STEP_1_SOURCE", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    // The mocked ChatInput renders `placeholder` as both its data attr
    // and its text content.  The text must match GUIDED_CHAT_PLACEHOLDERS
    // in ChatPanel.tsx; if a step's playbook is reworded, this expected
    // string updates here.
    const chatInput = screen.getByTestId("chat-input");
    expect(chatInput.dataset.placeholder).toBe(
      "Ask about source options, columns, or paste a sample row…",
    );
  });

  it("renders the per-step placeholder for STEP_2_SINK", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: { ...activeGuidedSession(), step: "step_2_sink" },
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    expect(screen.getByTestId("chat-input").dataset.placeholder).toBe(
      "Ask about sink config, outputs, or schema mode…",
    );
  });

  it("invokes sessionStore.chatGuided when the guided ChatInput onSend fires", async () => {
    const chatGuidedSpy = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      chatGuided: chatGuidedSpy,
    });

    render(<ChatPanel />);

    // The mocked ChatInput is a <button>; clicking it triggers the
    // onSend("test-chat-message") wired in the mock factory above.
    await act(async () => {
      screen.getByTestId("chat-input").click();
    });

    await waitFor(() => {
      expect(chatGuidedSpy).toHaveBeenCalledWith("test-chat-message");
    });
  });

  it("disables the guided ChatInput while guidedChatPending=true", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedChatPending: true,
    });

    render(<ChatPanel />);

    expect(screen.getByTestId("chat-input").dataset.disabled).toBe("true");
  });

  it("renders CompletionSummary surface when terminal.kind === 'completed'", () => {
    const completedHistory: TurnRecord[] = [
      {
        step: "step_1_source",
        turn_type: "single_select",
        payload_hash: "h1",
        response_hash: "r1",
        summary: "Source selected: csv",
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
        chat_history: [],
        chat_turn_seq: 0,
      },
      guidedTerminal: terminal,
    });

    const { container } = render(<ChatPanel />);

    // CompletionSummary renders task-oriented terminal actions.
    expect(
      screen.getByRole("button", { name: "Open freeform editor" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Review YAML" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Validate pipeline" }),
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
    // `CompletionSummary`'s task-oriented terminal actions. This
    // `queryByRole` predicate catches that regression — it
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
        chat_history: [],
        chat_turn_seq: 0,
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
      screen.queryByRole("button", { name: "Open freeform editor" }),
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
        chat_history: [],
        chat_turn_seq: 0,
      },
      guidedNextTurn: null,
      guidedTerminal: terminal,
    });

    render(<ChatPanel />);

    // Freeform surface visible.
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    // CompletionSummary NOT rendered (its guard fails on kind !== "completed").
    expect(
      screen.queryByRole("button", { name: "Open freeform editor" }),
    ).toBeNull();
  });

  it("wraps the guided turn surface in a role=log aria-live=polite region (Task 8.2 a11y)", () => {
    // Task 8.2 a11y fix: the guided branch must wrap the live turn surface in
    // role="log" aria-live="polite" so screen readers are notified when a new
    // turn arrives.  InspectAndConfirmTurn.tsx ("Warnings accessibility"
    // comment block) documents the dependency (its warnings <aside> omits its
    // own live region in favour of the parent).
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
        chat_history: [],
        chat_turn_seq: 0,
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

// ── Focus-on-step-advance tests (Phase 8 fix-up, spec §7.4) ─────────────────
//
// Verifies the useEffect in ChatPanel that moves focus to the first interactive
// element inside chat-panel-guided-log when the guided wizard advances to a
// new step.  Without this, a step-advancing button click unmounts the button
// before the browser returns focus, so focus falls to <body>.
//
// GuidedTurn is NOT mocked — tests use the real SingleSelectTurn widget (rendered
// by the singleSelectTurn() fixture) which produces genuine <button> children
// inside a <fieldset>.
describe("ChatPanel guided step-advance focus (spec §7.4)", () => {
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
    id: "session-focus",
    title: "Guided focus test session",
    created_at: "2026-05-12T10:00:00Z",
    updated_at: "2026-05-12T10:00:00Z",
  };

  function activeGuidedSession(): GuidedSession {
    return { step: "step_1_source", history: [], terminal: null, chat_history: [], chat_turn_seq: 0 };
  }

  // Options are intentionally distinct per step so that test 2's assertion at
  // step 1 can only pass if the effect actually re-fired (i.e. the button it
  // checks — "Database" — only exists in step 1's render, not step 0's).
  // Step 0 → "CSV" / "API"; step 1 → "Database" / "Streaming".
  function singleSelectTurn(stepIndex: number): TurnPayload {
    const options: SingleSelectPayload["options"] =
      stepIndex === 0
        ? [
            { id: "csv", label: "CSV", hint: null },
            { id: "api", label: "API", hint: null },
          ]
        : [
            { id: "database", label: "Database", hint: null },
            { id: "streaming", label: "Streaming", hint: null },
          ];
    const payload: SingleSelectPayload = {
      question: "Which source plugin should we use?",
      options,
      allow_custom: false,
    };
    return { type: "single_select", step_index: stepIndex, payload };
  }

  it("moves focus to the first button in chat-panel-guided-log when the turn first renders (step 0)", async () => {
    useSessionStore.setState({
      activeSessionId: "session-focus",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(0),
    });

    render(<ChatPanel />);

    // SingleSelectTurn renders one button per option; the first should receive
    // focus via the step-advance effect.
    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "CSV" }),
      );
    });
  });

  it("moves focus to the first button again when step_index advances (step 0 → step 1)", async () => {
    useSessionStore.setState({
      activeSessionId: "session-focus",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(0),
    });

    render(<ChatPanel />);

    // Wait for initial focus (step 0).
    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "CSV" }),
      );
    });

    // Advance to step 1 with a new turn payload.
    act(() => {
      useSessionStore.setState({
        guidedNextTurn: singleSelectTurn(1),
      });
    });

    // Focus should land on the first button of the step-1 turn ("Database").
    // The label only exists in step 1's options, so this assertion can only pass
    // if the focus effect re-fired after step_index changed (not if focus simply
    // persisted at the same DOM position as step 0).
    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Database" }),
      );
    });
  });

  it("does NOT re-focus when step_index is unchanged (same-step store mutation)", async () => {
    useSessionStore.setState({
      activeSessionId: "session-focus",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(0),
    });

    render(<ChatPanel />);

    // Wait for the initial focus (step 0, first button = "CSV").
    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "CSV" }),
      );
    });

    // Manually move focus to the second button as the user would after interacting
    // with the widget.
    const apiButton = screen.getByRole("button", { name: "API" });
    apiButton.focus();
    expect(document.activeElement).toBe(apiButton);

    // Issue a same-step store mutation: new object reference, same step_index.
    // The effect dep [guidedNextTurn?.step_index] should NOT re-fire.
    act(() => {
      useSessionStore.setState({
        guidedNextTurn: { ...singleSelectTurn(0) },
      });
    });

    // Flush React updates — give the effect a chance to incorrectly re-fire.
    await Promise.resolve();

    // Focus must remain on the API button; effect did not pull it back to CSV.
    expect(document.activeElement).toBe(apiButton);
  });
});
