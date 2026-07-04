import { readFileSync } from "node:fs";
import { join } from "node:path";

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ChatPanel,
  deriveRowCount,
  findOriginatingMessageId,
  hasExistingCompositionContent,
  hasSafeInlineSourceDisambiguationBase,
  isAmbiguousInlineProposal,
  looksLikeData,
  parseProposedRowsFromUserInput,
} from "./ChatPanel";
import { GUIDED_EXPLAIN_MESSAGE } from "./guided/explainPrompt";
import { useSessionStore } from "@/stores/sessionStore";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useInlineSourceStore } from "@/stores/inlineSourceStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useExecutionStore } from "@/stores/executionStore";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { resetStore } from "@/test/store-helpers";
import { useComposer } from "@/hooks/useComposer";
import { makeComposition } from "@/test/composerFixtures";
import * as apiClient from "@/api/client";
import type {
  BlobMetadata,
  ChatMessage,
  ComposerProgressSnapshot,
  CompositionProposal,
  Session,
} from "@/types/api";
import type {
  GuidedSession,
  SingleSelectPayload,
  TerminalState,
  TurnPayload,
  TurnRecord,
} from "@/types/guided";
import type { InterpretationEvent } from "@/types/interpretation";

vi.mock("@/hooks/useComposer", () => ({
  useComposer: vi.fn(),
}));

// Spy-style mock of the blob-fetch surface so the inline-source-projection
// effect can be driven from the test. The actual module is preserved
// (`...actual`) so we don't accidentally stub other exports the file uses.
vi.mock("@/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/client")>();
  return {
    ...actual,
    getBlobMetadata: vi.fn(),
    previewBlobContent: vi.fn(),
    // ModelChip's data source. Reset (undefined resolution) in most tests →
    // the chip renders nothing; the model-chip tests set a resolved value.
    fetchSystemStatus: vi.fn(),
  };
});

vi.mock("./MessageBubble", () => ({
  MessageBubble: ({
    message,
    proposalsByToolCallId,
    staleProposalIds,
  }: {
    message: ChatMessage;
    proposalsByToolCallId?: Map<string, CompositionProposal>;
    staleProposalIds?: string[];
  }) => {
    const toolCallId = message.tool_calls?.[0]?.id ?? null;
    const proposal = toolCallId
      ? proposalsByToolCallId?.get(toolCallId) ?? null
      : null;
    const isStale = proposal
      ? staleProposalIds?.includes(proposal.id) ?? false
      : false;
    return (
      <div data-testid="message-bubble">
        <div>{message.content}</div>
        {proposal && <div>{proposal.summary}</div>}
        {isStale && <div>Stale proposal</div>}
      </div>
    );
  },
}));

// Mock surfaces both the placeholder and the onSend wiring as inspectable
// DOM properties so per-step / per-mode tests can assert without depending
// on the real ChatInput's textarea internals. Phase A slice 4 needs:
//   - per-step placeholder visible (guided-active branch)
//   - onSend callback wired so the test can simulate user submit
// data-value / data-read-only expose the tutorial locked-prompt contract
// (value prefilled per stage + readOnly) so the bare-composer tests can pin
// it — before these attrs the lock was unpinned at the ChatPanel level.
vi.mock("./ChatInput", () => ({
  ChatInput: ({
    placeholder,
    onSend,
    onCancel,
    disabled,
    maxLength,
    value,
    readOnly,
  }: {
    placeholder?: string;
    onSend?: (content: string) => void;
    onCancel?: () => void;
    disabled?: boolean;
    maxLength?: number;
    value?: string;
    readOnly?: boolean;
  }) => (
    <button
      type="button"
      data-testid="chat-input"
      data-placeholder={placeholder ?? ""}
      data-disabled={disabled ? "true" : "false"}
      data-has-cancel={onCancel ? "true" : "false"}
      data-max-length={maxLength ?? ""}
      data-value={value ?? ""}
      data-read-only={readOnly ? "true" : "false"}
      onClick={() => onSend?.("test-chat-message")}
    >
      {placeholder ?? ""}
    </button>
  ),
}));

vi.mock("./TemplateCards", () => ({
  TemplateCards: ({
    onSelectTemplate: _onSelectTemplate,
  }: {
    onSelectTemplate: (
      seedPrompt: string,
      recommendedStartingPoint: "dynamic_source_from_chat" | "csv_upload" | "api_source",
    ) => void;
  }) => (
    <div data-testid="template-cards">
      Template cards
    </div>
  ),
}));

vi.mock("@/components/blobs/BlobManager", () => ({
  BlobManager: () => <div data-testid="blob-manager" />,
}));

vi.mock("@/components/execution/InlineRunResults", () => ({
  InlineRunResults: () => <div data-testid="inline-run-results" />,
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

  it("keeps terminal composer progress visible after composing ends", () => {
    const session: Session = {
      id: "session-1",
      title: "Composer session",
      created_at: "2026-04-26T10:00:00Z",
      updated_at: "2026-04-26T10:00:00Z",
    };
    const progress: ComposerProgressSnapshot = {
      session_id: "session-1",
      request_id: "message-1",
      phase: "cancelled",
      headline: "Composition stopped before saving.",
      evidence: ["The request ended before a valid pipeline was saved."],
      likely_next: "Revise the request and send it again.",
      reason: "client_cancelled",
      updated_at: "2026-04-26T10:00:02Z",
    };

    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      cancelComposition: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: null,
    });
    useSessionStore.setState({
      activeSessionId: "session-1",
      sessions: [session],
      messages: [],
      composerProgress: progress,
    });

    render(<ChatPanel />);

    expect(screen.getByText("Last composer update")).toBeInTheDocument();
    expect(screen.getByText("Composition stopped before saving.")).toBeInTheDocument();
    expect(screen.getByText("Revise the request and send it again.")).toBeInTheDocument();
  });

  it("renders template cards as a static gallery without sending a template prompt", () => {
    const sendMessage = vi.fn();
    const session: Session = {
      id: "session-templates",
      title: "Template session",
      created_at: "2026-04-26T10:00:00Z",
      updated_at: "2026-04-26T10:00:00Z",
    };

    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage,
      retryMessage: vi.fn(),
      cancelComposition: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: null,
    });
    useSessionStore.setState({
      activeSessionId: session.id,
      sessions: [session],
      messages: [],
    });

    render(<ChatPanel />);

    expect(screen.getByTestId("template-cards")).toBeInTheDocument();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("coalesces consecutive assistant rows into one agent turn bubble", () => {
    // Reproduces the live shape at session a8afd33e: one user prompt + seven
    // assistant rows (six with single tool_calls, one orphan empty) + a final
    // answer row. The backend persists each LLM round-trip as its own row
    // (Tier-1 audit doctrine), but the chat panel must render the user-visible
    // turn — one user bubble + one agent bubble carrying the aggregated tool
    // calls and the final answer.
    const session: Session = {
      id: "session-coalesce",
      title: "Coalesce session",
      created_at: "2026-05-19T00:00:00Z",
      updated_at: "2026-05-19T00:00:00Z",
    };
    const mkMsg = (overrides: Partial<ChatMessage> & { id: string; role: ChatMessage["role"] }): ChatMessage => ({
      session_id: session.id,
      content: "",
      tool_calls: null,
      created_at: "2026-05-19T00:00:00Z",
      ...overrides,
    }) as ChatMessage;
    const tc = (name: string, id = name) => ({ id, type: "function", function: { name, arguments: "{}" } });

    const messages: ChatMessage[] = [
      mkMsg({ id: "u1", role: "user", content: "create a list of 5 government web pages..." }),
      mkMsg({ id: "a1", role: "assistant", tool_calls: [tc("list_models"), tc("get_plugin_schema", "g1")] }),
      mkMsg({ id: "a2", role: "assistant", tool_calls: [tc("create_blob")] }),
      mkMsg({ id: "a3", role: "assistant", tool_calls: [tc("set_pipeline")] }),
      mkMsg({ id: "a4", role: "assistant", tool_calls: [tc("patch_node_options")] }),
      mkMsg({ id: "a5", role: "assistant", tool_calls: [tc("preview_pipeline", "p1")] }),
      mkMsg({ id: "a6", role: "assistant" }),
      mkMsg({ id: "a7", role: "assistant", tool_calls: [tc("preview_pipeline", "p2")] }),
      mkMsg({ id: "a8", role: "assistant", content: "Built a workflow that fetches each page and rates it." }),
    ];

    useSessionStore.setState({
      activeSessionId: session.id,
      sessions: [session],
      messages,
    });

    render(<ChatPanel />);

    // One bubble per turn: 1 user + 1 agent = 2 — not 9 (the audit row count).
    const bubbles = screen.getAllByTestId("message-bubble");
    expect(bubbles).toHaveLength(2);

    // Agent turn carries the final content from the LAST row in the turn,
    // not from any intermediate empty-content row.
    expect(
      screen.getByText("Built a workflow that fetches each page and rates it."),
    ).toBeInTheDocument();

    // User turn still renders its own bubble.
    expect(
      screen.getByText("create a list of 5 government web pages..."),
    ).toBeInTheDocument();
  });

  it("passes matching and stale proposal state to message bubbles", () => {
    const session: Session = {
      id: "session-1",
      title: "Proposal session",
      created_at: "2026-05-14T00:00:00Z",
      updated_at: "2026-05-14T00:00:00Z",
    };
    const assistantMessage: ChatMessage = {
      id: "assistant-1",
      session_id: "session-1",
      role: "assistant",
      content: "I need approval.",
      tool_calls: [
        {
          id: "call-1",
          type: "function",
          function: { name: "set_pipeline", arguments: "{}" },
        },
      ],
      created_at: "2026-05-14T00:00:01Z",
    };
    const proposal: CompositionProposal = {
      id: "proposal-1",
      session_id: "session-1",
      tool_call_id: "call-1",
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
    };

    useSessionStore.setState({
      activeSessionId: "session-1",
      sessions: [session],
      messages: [assistantMessage],
      compositionProposals: [proposal],
      staleProposalIds: ["proposal-1"],
    });

    render(<ChatPanel />);

    expect(screen.getByText("Replace the pipeline.")).toBeInTheDocument();
    expect(screen.getByText("Stale proposal")).toBeInTheDocument();
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
    // P3.6: the guided-branch tests seed pendingBySession to exercise the
    // interpretation-card block; reset it so a seeded card does not leak into
    // sibling tests and spuriously disable their guided turn.
    resetStore(useInterpretationEventsStore);
    // Slice C: the verification-panel tests seed validationResult; reset the
    // execution store so a seeded result does not leak into sibling tests.
    resetStore(useExecutionStore);
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
      profile: null,
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

    // The intent ChatInput is rendered INSIDE the guided-active branch. For a
    // non-tutorial session it now docks at the BOTTOM (below the decision), as
    // the primary chat affordance; DOM order is asserted in the dedicated
    // "docks ... BELOW the decision" test above. This test asserts presence
    // only; per-step placeholder + onSend wiring are exercised below.
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    expect(screen.getByTestId("inline-run-results")).toBeInTheDocument();
  });

  it("shows the composer model chip in the GUIDED chat header too (elspeth-e9f7678de8)", async () => {
    // Freeform's header already carries the chip; guided authoring must name
    // its model the same way (same chip, same /api/system/status source).
    vi.mocked(apiClient.fetchSystemStatus).mockResolvedValue({
      composer_available: true,
      composer_model: "anthropic/claude-sonnet-4.6",
      composer_provider: "openrouter",
      composer_reason: null,
      composer_missing_keys: [],
    });
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(<ChatPanel />);

    await waitFor(() => {
      expect(
        screen.getByLabelText("Composer model: anthropic/claude-sonnet-4.6"),
      ).toBeInTheDocument();
    });
    // The chip lives in the guided header's actions chrome.
    const header = container.querySelector(".chat-panel-header");
    expect(header?.querySelector(".chat-model-chip")).not.toBeNull();
  });

  it("non-tutorial guided: no 'always start in freeform mode' opt-out checkbox", () => {
    // Load preferences so the (now-removed) InlineOptOutCheckbox would render if
    // it were still wired — it returns null until prefs load, so without this
    // the assertion would pass vacuously.
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    expect(
      screen.queryByText(/always start new sessions in freeform mode/i),
    ).toBeNull();
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
    for (const label of ["Source", "Output", "Transforms", "Wire", "Ready"]) {
      expect(workflow).toHaveTextContent(label);
    }
    expect(screen.getByRole("listitem", { current: "step" })).toHaveTextContent(
      "Source",
    );
  });

  it("renders the chat box at step_3 with NO proposal (per-stage transforms entry, not a panel-less fall-through)", () => {
    // STEP_3 begins with no server turn: the per-stage transforms prompt drives
    // the build via /guided/chat (cold-start). The guided surface — crucially the
    // chat box — MUST render so the operator can describe the transforms; a
    // missing turn must NOT fall through to the freeform body / loading flash.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: { ...activeGuidedSession(), step: "step_3_transforms" },
      guidedNextTurn: null,
    });

    const { container } = render(<ChatPanel />);

    const chatMain = container.querySelector("#chat-main");
    expect(chatMain?.classList.contains("chat-panel--guided")).toBe(true);
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    // Transforms is the current stepper step (not the vestigial Recipe).
    expect(screen.getByRole("listitem", { current: "step" })).toHaveTextContent(
      "Transforms",
    );
  });

  it("marks STEP_4_WIRE as the current guided workflow step", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: { ...activeGuidedSession(), step: "step_4_wire" },
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    const workflow = screen.getByRole("list", { name: /guided workflow/i });
    expect(workflow).toHaveTextContent("Wire");
    expect(screen.getByRole("listitem", { current: "step" })).toHaveTextContent(
      "Wire",
    );
  });

  it("lays the guided workflow stepper out one column per step, with a mobile breakpoint", () => {
    const css = readFileSync(
      join(process.cwd(), "src/components/chat/guided/guided.css"),
      "utf8",
    );
    // 5 GUIDED_WORKFLOW_STEPS -> 5 columns. The prior repeat(6,...) left an
    // empty trailing column.
    expect(css).toContain("grid-template-columns: repeat(5, minmax(0, 1fr));");
    // Narrow viewports drop to 2 columns so single-word labels (e.g.
    // "Transforms") stay whole instead of shattering one character per line.
    expect(css).toMatch(
      /@media \(max-width: 640px\)[\s\S]*?grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/,
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

    // The decision now leads with the dynamic rationale (or, when no assistant
    // turn exists for the step, the step-purpose fallback) AS the heading;
    // "Current decision" is a decorative, aria-hidden eyebrow.
    expect(
      screen.getByRole("heading", {
        level: 2,
        name: /choose the input and confirm what elspeth can read/i,
      }),
    ).toBeInTheDocument();
  });

  it("leads the decision with the assistant rationale heading when chat_history has one", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        chat_history: [
          {
            role: "assistant",
            content: "Source created as a 3-row CSV.",
            seq: 1,
            step: "step_1_source",
            ts_iso: "t",
          },
        ],
      },
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    // With an assistant rationale for the active step, it leads AS the heading
    // (instead of the static step-purpose fallback).
    expect(
      screen.getByRole("heading", {
        level: 2,
        name: /source created as a 3-row csv/i,
      }),
    ).toBeInTheDocument();
  });

  // ── Slice C: the guided verification panel ────────────────────────────────
  //
  // The panel (gloss + plain validation summary) leads the guided column for
  // BOTH surfaces; the graph THUMBNAIL is tutorial-only (live-guided already
  // renders GraphMiniView in the SideRail, so the column would otherwise
  // duplicate it). No second GraphModal is mounted here — both surfaces expand
  // into the App-root GraphModal.
  function sourceLlmCsvComposition() {
    return makeComposition(1, {
      sources: { source: { plugin: "text", options: {} } },
      nodes: [
        {
          id: "rater",
          node_type: "transform",
          plugin: "llm",
          input: "source",
          on_success: null,
          on_error: null,
          options: {},
        },
      ],
      outputs: [{ name: "out", plugin: "csv", options: {} }],
    });
  }

  it("mounts the guided verification panel (gloss + validation + graph thumbnail) in the workspace rail for live guided", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      compositionState: sourceLlmCsvComposition(),
    });

    const { container } = render(<ChatPanel />);

    // The verification panel anchors the artifact rail.
    const rail = container.querySelector("aside.guided-workspace-rail");
    expect(rail).not.toBeNull();
    const panel = screen.getByRole("region", { name: "Pipeline so far" });
    expect(rail!.contains(panel)).toBe(true);
    // Gloss renders a plain-language sentence from the composition.
    expect(screen.getByTestId("pipeline-gloss")).toHaveTextContent(
      /this pipeline will read your data, rate each row, and write a csv\./i,
    );
    // Validation summary root is always present (neutral here — no result yet).
    expect(
      screen.getByTestId("pipeline-validation-summary"),
    ).toBeInTheDocument();
    // The graph thumbnail rides in the rail for EVERY guided session — the
    // App suppresses the freeform SideRail (its old home) while the guided
    // workspace is on screen.
    expect(
      screen.getByRole("button", {
        name: "Pipeline graph (click to expand)",
      }),
    ).toBeInTheDocument();
    // No second GraphModal is mounted in the panel (App-root one serves all).
    expect(screen.queryByTestId("graph-modal-backdrop")).toBeNull();
    expect(container.querySelector(".graph-modal")).toBeNull();
  });

  it("renders the graph thumbnail in the tutorial rail (which has no SideRail)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      compositionState: sourceLlmCsvComposition(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // Panel + gloss + summary present in the tutorial too — anchored in the
    // artifact rail (the workspace's right column), not the conversation.
    const rail = container.querySelector("aside.guided-workspace-rail");
    expect(rail).not.toBeNull();
    const panel = screen.getByRole("region", { name: "Pipeline so far" });
    expect(rail!.contains(panel)).toBe(true);
    expect(screen.getByTestId("pipeline-gloss")).toBeInTheDocument();
    expect(
      screen.getByTestId("pipeline-validation-summary"),
    ).toBeInTheDocument();
    // The tutorial gets the rail thumbnail (populated → the expand button).
    expect(
      screen.getByRole("button", {
        name: "Pipeline graph (click to expand)",
      }),
    ).toBeInTheDocument();
  });

  it("retains the per-step rationale prose below the verification panel (demoted, not deleted)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        chat_history: [
          {
            role: "assistant",
            content: "Source created as a 3-row CSV.",
            seq: 1,
            step: "step_1_source",
            ts_iso: "t",
          },
        ],
      },
      guidedNextTurn: singleSelectTurn(),
      compositionState: sourceLlmCsvComposition(),
    });

    const { container } = render(<ChatPanel />);

    // The rationale prose is RETAINED as the decision heading, inside the
    // conversation column's scroll region (the action zone)...
    const rationale = screen.getByRole("heading", {
      level: 2,
      name: /source created as a 3-row csv/i,
    });
    expect(rationale).toBeInTheDocument();
    expect(
      container
        .querySelector(".guided-workspace-scroll")!
        .contains(rationale),
    ).toBe(true);
    // ...while the verification panel (the canonical "what I built") holds
    // its own ambient surface in the artifact rail.
    const panel = screen.getByRole("region", { name: "Pipeline so far" });
    expect(
      container.querySelector("aside.guided-workspace-rail")!.contains(panel),
    ).toBe(true);
  });

  // Shared validation-readiness stub (PipelineValidationSummary ignores it).
  const READINESS = {
    authoring_valid: true,
    execution_ready: true,
    completion_ready: true,
    blockers: [],
  };

  it("D1: renders the guided verification panel from the store only — no source-DATA fetch", () => {
    // Scope the spies to SOURCE-DATA endpoints (blob content / upload). The
    // mode-agnostic auto-validate fires api.validatePipeline, which is
    // metadata-only and D1-safe — it is intentionally NOT spied (a blanket
    // api.* spy would false-fail). getBlobMetadata + previewBlobContent are
    // already vi.fn() stubs from the module mock above.
    const uploadSpy = vi.spyOn(apiClient, "uploadBlob");
    const previewSnippetSpy = vi.spyOn(apiClient, "previewBlobContentSnippet");
    const downloadSpy = vi.spyOn(apiClient, "downloadBlobContent");

    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      compositionState: sourceLlmCsvComposition(),
    });
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [],
        readiness: READINESS,
      },
    } as never);

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // The panel is built purely from compositionState + validationResult,
    // and lives in the artifact rail.
    expect(
      container.querySelector(
        'aside.guided-workspace-rail [data-testid="pipeline-gloss"]',
      ),
    ).not.toBeNull();
    expect(screen.getByTestId("pipeline-gloss")).toBeInTheDocument();
    expect(
      screen.getByTestId("pipeline-validation-summary"),
    ).toHaveTextContent(/looks good/i);
    expect(
      screen.getByRole("button", {
        name: "Pipeline graph (click to expand)",
      }),
    ).toBeInTheDocument();

    // Zero source-DATA reads — D1 (consumable source, zero rows).
    expect(uploadSpy).not.toHaveBeenCalled();
    expect(previewSnippetSpy).not.toHaveBeenCalled();
    expect(downloadSpy).not.toHaveBeenCalled();
    expect(apiClient.previewBlobContent).not.toHaveBeenCalled();
    expect(apiClient.getBlobMetadata).not.toHaveBeenCalled();
  });

  it("tutorial parity: the in-column validation summary reflects validationResult, and the thumbnail expands the App-root modal", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      compositionState: sourceLlmCsvComposition(),
    });
    // validationResult populates in the tutorial too (the auto-validate
    // subscription is version-keyed / mode-agnostic), so the in-column signal
    // shows there — pin it.
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [
          {
            component_id: "rater",
            component_type: "transform",
            message: "Review the prompt wording",
            suggestion: null,
          },
        ],
        readiness: READINESS,
      },
    } as never);

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // (a) the summary reflects validationResult, with the PLAIN node name
    // mapped from the finding's component_id (not the raw id) — rendered in
    // the artifact rail.
    const summary = screen.getByTestId("pipeline-validation-summary");
    expect(
      container
        .querySelector("aside.guided-workspace-rail")
        ?.contains(summary),
    ).toBe(true);
    expect(summary).toHaveTextContent(/rate each row/);
    expect(summary).toHaveTextContent(/review the prompt wording/i);

    // (b) clicking the thumbnail dispatches OPEN_GRAPH_MODAL_EVENT, caught by
    // the App-root GraphModal — no second modal is mounted in the column. The
    // per-node MARKER assertion targets the modal GraphView (GraphView.test.tsx
    // marker coverage), NOT GraphMiniView (which has no markers).
    const openSpy = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, openSpy);
    fireEvent.click(
      screen.getByRole("button", {
        name: "Pipeline graph (click to expand)",
      }),
    );
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, openSpy);
    expect(openSpy).toHaveBeenCalledTimes(1);
  });

  it("scrolls the guided log into view when the active step advances", () => {
    const scrollSpy = vi.fn();
    Element.prototype.scrollIntoView = scrollSpy;
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { rerender } = render(<ChatPanel />);
    scrollSpy.mockClear();
    useSessionStore.setState({
      guidedSession: { ...activeGuidedSession(), step: "step_2_sink" },
      guidedNextTurn: {
        type: "single_select",
        step_index: 1,
        payload: {
          question: "Which output plugin should we use?",
          options: [{ id: "json", label: "JSON", hint: null }],
          allow_custom: false,
        },
      },
    });
    rerender(<ChatPanel />);

    expect(scrollSpy).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "nearest",
    });
  });

  it("visually separates the guided intent box as describe what you want", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    const chatRegion = screen.getByRole("region", {
      name: /describe what you want/i,
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

  // C-3 (composer first-principles review 2026-07-04): a step-2 sink
  // commit-failure 400 ({code: "guided_step2_sink_commit_failed", detail})
  // must surface its OWN detail text in the existing error affordance,
  // not a generic "Failed to submit..." line. respondGuided's catch already
  // forwards `apiErr.detail` verbatim (sessionStore.ts) — this pins that
  // ChatPanel renders whatever specific detail string it receives rather
  // than substituting a canned message.
  it("surfaces the specific detail text from a guided_step2_sink_commit_failed rejection", () => {
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error:
        "The output configuration could not be applied. Review the options entered for this output and try again.",
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
      "The output configuration could not be applied. Review the options entered for this output and try again.",
    );
    // Not the generic fallback line.
    expect(screen.queryByText(/Failed to submit guided response/i)).toBeNull();
  });

  // C-3 (composer first-principles review 2026-07-04): the turn_not_emitted
  // self-heal notice renders as a calm role="status" line, never as an alert.
  it("renders guidedSelfHealNotice as a role=status notice, not an alert", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedSelfHealNotice:
        "The wizard had fallen out of sync with the server. We've refreshed to the current step — please try again.",
    });

    render(<ChatPanel />);

    const notice = screen.getByText(
      "The wizard had fallen out of sync with the server. We've refreshed to the current step — please try again.",
    );
    expect(notice).toHaveAttribute("role", "status");
    expect(screen.queryByRole("alert")).toBeNull();
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

  // D12 / P3.6: a pending user_approved interpretation card surfaced into the
  // interpretationEventsStore must block guided advancement — the wizard turn's
  // submit control is disabled until the card is resolved, even when no guided
  // response is in flight (guidedResponsePending is false here).
  it("disables the guided turn while a pending user_approved interpretation card exists", () => {
    const pendingCard: InterpretationEvent = {
      id: "card-1",
      session_id: "session-guided",
      composition_state_id: "state-1",
      affected_node_id: "rate_node",
      tool_call_id: "backend_auto_surface:abc",
      user_term: "llm_model_choice:rate_node",
      kind: "llm_model_choice",
      llm_draft: "anthropic/claude-sonnet-4.6",
      accepted_value: null,
      choice: "pending",
      created_at: "2026-06-22T00:00:00Z",
      resolved_at: null,
      actor: "system:composer",
      interpretation_source: "user_approved",
      model_identifier: "anthropic/claude-opus-4-7",
      model_version: "anthropic/claude-opus-4-7",
      provider: "anthropic",
      composer_skill_hash: "0".repeat(64),
      arguments_hash: null,
      hash_domain_version: null,
      runtime_model_identifier_at_resolve: null,
      runtime_model_version_at_resolve: null,
      resolved_prompt_template_hash: null,
    };
    useInterpretationEventsStore.setState({
      pendingBySession: { "session-guided": { "card-1": pendingCard } },
    });
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedResponsePending: false,
    });

    render(<ChatPanel />);

    // Same query the pending-response test above uses: GuidedTurn's primary
    // option button. InterpretationReviewTurn renders no "CSV" button, so this
    // uniquely targets the wizard turn's submit control.
    expect(screen.getByRole("button", { name: "CSV" })).toBeDisabled();
  });

  // M-10 (composer first-principles review 2026-07-04): GuidedTurn's disabled
  // gate was missing guidedChatPending, letting a wizard widget stay
  // clickable while a /guided/chat request raced an in-flight step-respond.
  // "Explain this step" already gated on both flags; this pins GuidedTurn now
  // matches.
  it("disables guided turn buttons while a guided chat is pending (M-10)", async () => {
    const respondGuidedSpy = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedResponsePending: false,
      guidedChatPending: true,
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
      "Describe the source you have — e.g. a CSV, a store query, or pages to scrape…",
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
      "Describe the output you want — the shape and fields the pipeline should produce…",
    );
  });

  it("renders the per-step placeholder for STEP_4_WIRE", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: { ...activeGuidedSession(), step: "step_4_wire" },
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    expect(screen.getByTestId("chat-input").dataset.placeholder).toBe(
      "Resolve any pending acknowledgements, then press Confirm wiring on the current decision card.",
    );
  });

  it("renders the per-step placeholder for STEP_2_5_RECIPE_MATCH", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      // The placeholder keys on guidedSession.step (not the turn type), so
      // singleSelectTurn() is fine regardless of step (verified ChatPanel.tsx:1375).
      guidedSession: { ...activeGuidedSession(), step: "step_2_5_recipe_match" },
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    expect(screen.getByTestId("chat-input").dataset.placeholder).toBe(
      "Describe how this recipe should change, or accept it as proposed…",
    );
  });

  it("renders the per-step placeholder for STEP_3_TRANSFORMS", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: { ...activeGuidedSession(), step: "step_3_transforms" },
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    expect(screen.getByTestId("chat-input").dataset.placeholder).toBe(
      "Describe what each row should become, or how to fix the proposed transforms…",
    );
  });

  it("non-tutorial guided: docks the intent box BELOW the editable form (chat-window layout)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(<ChatPanel />);

    // Classnames are unchanged: the intent box keeps `.guided-step-chat`.
    const intent = container.querySelector(".guided-step-chat");
    const form = container.querySelector(".guided-current-decision");
    expect(intent).not.toBeNull();
    expect(form).not.toBeNull();
    // Document-order (non-tutorial): the intent box docks at the BOTTOM, AFTER
    // the decision form — mirroring the freeform body, where the ChatInput docks
    // below the message log. compareDocumentPosition returns
    // DOCUMENT_POSITION_FOLLOWING (4) when `intent` follows `form`.
    expect(
      form!.compareDocumentPosition(intent!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    // The intent box's recaptioned heading is present.
    expect(
      screen.getByRole("region", { name: "Describe what you want" }),
    ).toBeInTheDocument();
  });

  // A pending user_approved interpretation card for the workspace-order tests:
  // seeds the AcknowledgementStack so the conversation column's full DOM-order
  // chain (transcript → acks → decision → composer) is assertable.
  function pendingAckCard(): InterpretationEvent {
    return {
      id: "card-workspace-1",
      session_id: "session-guided",
      composition_state_id: "state-1",
      affected_node_id: "rate_node",
      tool_call_id: "backend_auto_surface:workspace",
      user_term: "llm_model_choice:rate_node",
      kind: "llm_model_choice",
      llm_draft: "anthropic/claude-sonnet-4.6",
      accepted_value: null,
      choice: "pending",
      created_at: "2026-06-22T00:00:00Z",
      resolved_at: null,
      actor: "system:composer",
      interpretation_source: "user_approved",
      model_identifier: "anthropic/claude-opus-4-7",
      model_version: "anthropic/claude-opus-4-7",
      provider: "anthropic",
      composer_skill_hash: "0".repeat(64),
      arguments_hash: null,
      hash_domain_version: null,
      runtime_model_identifier_at_resolve: null,
      runtime_model_version_at_resolve: null,
      resolved_prompt_template_hash: null,
    };
  }

  it("tutorial guided: workspace — transcript, acks, decision and composer share the conversation column in order; the rail holds no decision", () => {
    useInterpretationEventsStore.setState({
      pendingBySession: { "session-guided": { "card-workspace-1": pendingAckCard() } },
    });
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        chat_history: [
          {
            role: "user",
            content: "create the source",
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-05-12T10:00:00Z",
          },
        ],
      },
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // The workspace: conversation column (stream) + artifact rail.
    const stream = container.querySelector(".guided-workspace-stream");
    const rail = container.querySelector("aside.guided-workspace-rail");
    expect(stream).not.toBeNull();
    expect(rail).not.toBeNull();
    // The rail is ambient pipeline state — labelled for what it now holds
    // ("Pipeline summary", distinct from the inner "Pipeline so far" section),
    // with NO decision card in it.
    expect(rail!.getAttribute("aria-label")).toBe("Pipeline summary");
    expect(rail!.querySelector(".guided-current-decision")).toBeNull();
    expect(rail!.querySelector(".guided-step-chat")).toBeNull();

    // The decision lives INSIDE the conversation column's internal scroll
    // region, and the composer docks after it in the stream.
    const scroll = container.querySelector(".guided-workspace-scroll");
    expect(scroll).not.toBeNull();
    expect(scroll!.querySelector(".guided-current-decision")).not.toBeNull();
    const composer = stream!.querySelector(".guided-step-chat");
    expect(composer).not.toBeNull();
    expect(scroll!.contains(composer)).toBe(false);

    // DOM-order chain inside the column: transcript → AcknowledgementStack →
    // decision → composer (DOCUMENT_POSITION_FOLLOWING = the argument follows
    // the receiver).
    const transcript = stream!.querySelector(".guided-chat-bubbles");
    const acks = stream!.querySelector('[data-testid="acknowledgement-stack"]');
    const decision = stream!.querySelector(".guided-current-decision");
    expect(transcript).not.toBeNull();
    expect(acks).not.toBeNull();
    expect(decision).not.toBeNull();
    const follows = (earlier: Element, later: Element) =>
      earlier.compareDocumentPosition(later) & Node.DOCUMENT_POSITION_FOLLOWING;
    expect(follows(transcript!, acks!)).toBeTruthy();
    expect(follows(acks!, decision!)).toBeTruthy();
    expect(follows(decision!, composer!)).toBeTruthy();

    // Tutorial suppresses the exit affordance, so there is no header/exit.
    expect(
      screen.queryByRole("button", { name: "Exit to freeform" }),
    ).toBeNull();
  });

  it("tutorial: the conversation column is a named, keyboard-focusable scroll region (not a live region)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    const scroll = container.querySelector<HTMLElement>(
      ".guided-workspace-scroll",
    );
    expect(scroll).not.toBeNull();
    // role="group" is required for the accessible name to be exposed
    // (aria-label on a role-less div is AT-invisible, elspeth-37293a3b7c);
    // tabIndex=0 makes it arrow-scrollable (elspeth-5e43a0c8b2).
    expect(scroll!.getAttribute("role")).toBe("group");
    expect(scroll!.getAttribute("aria-label")).toBe("Conversation");
    expect(scroll!.getAttribute("tabindex")).toBe("0");
    // NOT a live region — the transcript log and the wizard log live inside
    // it; nesting them in an outer live region would double-announce.
    expect(scroll!.getAttribute("aria-live")).toBeNull();
    expect(scroll!.getAttribute("role")).not.toBe("log");
  });

  it("tutorial: the artifact rail is a keyboard-scrollable tab stop", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // The rail scrolls (overflow-y:auto; the ≤900px strip caps at 30vh while
    // hiding its only focusable furniture) — without a tab stop its overflow
    // is keyboard-unreachable (WCAG 2.1.1). The complementary role already
    // carries the accessible name.
    const rail = container.querySelector("aside.guided-workspace-rail");
    expect(rail).not.toBeNull();
    expect(rail!.getAttribute("tabindex")).toBe("0");
  });

  it("presents the respond-rejection alert by scrolling it into view when it lands", () => {
    // A failed respond (e.g. a wire-confirm 409) mutates ONLY
    // error/errorDetails/guidedResponsePending — nothing the auto-scroll or
    // step-advance effects watch — and the alert renders as the LAST content
    // of the decision card at the bottom of the scroll region. The dedicated
    // presenter effect must scroll the alert itself into view or the
    // rejection is visually silent in the pinned-at-bottom state
    // (elspeth-3b35abf148 variant 3, reintroduced by geometry).
    const receivers: Element[] = [];
    Element.prototype.scrollIntoView = vi.fn(function (this: Element) {
      receivers.push(this);
    });
    // error/errorDetails reach ChatPanel through useComposer (mocked here) —
    // seeding the session store alone never reaches the panel.
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: "That wiring can't be confirmed yet.",
      errorDetails: ["Node llm-1 has no downstream consumer."],
    });
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    const rejection = container.querySelector(".guided-respond-rejection");
    expect(rejection).not.toBeNull();
    expect(receivers).toContain(rejection);
  });

  it("tutorial: pins the decision INSIDE the conversation column's scroll region and suppresses the rival source chips", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // The decision is the action zone at the END of the internal scroll region
    // — between the reply and the composer, never docked with the composer
    // (a tall schema/wire widget in a fixed dock crushes the transcript).
    expect(
      container.querySelector(".guided-workspace-scroll .guided-current-decision"),
    ).not.toBeNull();
    // .guided-scroll is the LIVE guided arrangement — the tutorial never
    // renders it (the old assertion here passed vacuously against it).
    expect(container.querySelector(".guided-scroll")).toBeNull();

    // The live, submit-on-click source chips are suppressed in the tutorial — a
    // passive learner's only action is Send. (singleSelectTurn() asks
    // "Which source plugin should we use?" with CSV / API options; clicking one
    // would submit an off-script source and derail the scripted build.)
    expect(
      screen.queryByText("Which source plugin should we use?"),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: "CSV" })).toBeNull();

    // The "press Send" coaching note still leads the decision.
    expect(
      container.querySelector(".guided-current-decision-tutorial-note"),
    ).not.toBeNull();
  });

  it("tutorial rail: pipeline summary + decisions so far, and nothing actionable (no decision/submit/composer affordances)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        step: "step_2_sink",
        // A PAST summarised step so GuidedHistory ("Decisions so far") renders
        // — it returns null until a step the learner moved past has a summary.
        history: [
          {
            step: "step_1_source",
            turn_type: "single_select",
            payload_hash: "h1",
            response_hash: "r1",
            summary: "Source selected: csv",
            emitter: "server",
          },
        ],
      },
      guidedNextTurn: singleSelectTurn(),
      compositionState: sourceLlmCsvComposition(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_2_sink: "create the sink" }}
      />,
    );

    const rail = container.querySelector("aside.guided-workspace-rail");
    expect(rail).not.toBeNull();
    // Summary card: gloss + validation + graph thumbnail, inside the rail.
    expect(rail!.querySelector('[data-testid="pipeline-gloss"]')).not.toBeNull();
    expect(
      rail!.querySelector('[data-testid="pipeline-validation-summary"]'),
    ).not.toBeNull();
    const expandButton = screen.getByRole("button", {
      name: "Pipeline graph (click to expand)",
    });
    expect(rail!.contains(expandButton)).toBe(true);
    // "Decisions so far" folded into the rail (single mount — it left the
    // conversation column).
    const decisionsHeading = screen.getByRole("heading", {
      name: "Decisions so far",
    });
    expect(rail!.contains(decisionsHeading)).toBe(true);
    expect(screen.getAllByRole("heading", { name: "Decisions so far" })).toHaveLength(1);

    // Nothing actionable: no composer, no decision widget, no submit controls.
    // (GraphMiniView's expand button above is the accepted exception — it
    // matches live's SideRail; "actionable" = decision/submit/composer
    // affordances.)
    expect(rail!.querySelector('[data-testid="chat-input"]')).toBeNull();
    expect(rail!.querySelector("fieldset")).toBeNull();
    expect(rail!.querySelector("textarea")).toBeNull();
    expect(rail!.querySelector(".guided-current-decision")).toBeNull();
    expect(rail!.querySelector(".guided-step-chat")).toBeNull();
  });

  it("tutorial: bare composer — locked read-only prompt in the named region, no visible card heading", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // The region + accessible name survive de-carding (a11y landmark + the
    // staging e2e locator both depend on the name)...
    const region = screen.getByRole("region", {
      name: "Describe what you want",
    });
    expect(region).toContainElement(screen.getByTestId("chat-input"));
    // ...and the composer is a bare docked strip — no visible card heading
    // (the dashed card + heading died with the pre-workspace flat layout).
    expect(container.querySelector(".guided-step-chat-heading")).toBeNull();
    expect(region.classList.contains("guided-step-chat")).toBe(true);
    // The locked-prompt contract: prefilled with the CURRENT stage's prompt,
    // read-only (the learner Sends, never types).
    const input = screen.getByTestId("chat-input");
    expect(input.dataset.value).toBe("create the source");
    expect(input.dataset.readOnly).toBe("true");
  });

  it("live guided: layout net — the SAME workspace as the tutorial, with an EDITABLE composer (no locked prompt)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(<ChatPanel />);

    // The workspace is THE guided layout (promoted from the tutorial,
    // operator directive 2026-07-03) — the old .guided-scroll flat
    // arrangement is gone for good.
    expect(container.querySelector(".guided-workspace")).not.toBeNull();
    expect(container.querySelector(".guided-scroll")).toBeNull();
    // Decision = the action zone at the end of the conversation column.
    expect(
      container.querySelector(".guided-workspace-scroll .guided-current-decision"),
    ).not.toBeNull();
    // Artifact rail present (the App suppresses the freeform SideRail while
    // this surface renders — the rail here replaces it).
    expect(container.querySelector("aside.guided-workspace-rail")).not.toBeNull();
    // Bare docked composer: no card heading in ANY mode...
    expect(container.querySelector(".guided-step-chat-heading")).toBeNull();
    // ...but live input stays EDITABLE — the tutorial's locked prompt is the
    // one delta the promotion kept tutorial-only.
    const input = screen.getByTestId("chat-input");
    expect(input.dataset.readOnly).toBe("false");
    // The mock coalesces the value prop (undefined → ""): live passes NO
    // controlled value, so the box starts empty and editable.
    expect(input.dataset.value).toBe("");
    // Live guided keeps its interactive decision widget (the tutorial
    // suppresses the rival chips; live does not).
    expect(
      screen.getByText("Which source plugin should we use?"),
    ).toBeInTheDocument();
  });

  it("tutorial: the workflow stepper renders above the workspace", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    const stepper = screen.getByRole("list", { name: /guided workflow/i });
    const workspace = container.querySelector(".guided-workspace");
    expect(workspace).not.toBeNull();
    expect(
      stepper.compareDocumentPosition(workspace!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("tutorial: the pending strip's status region mounts OUTSIDE every role=log region (elspeth-76a0cc485e parity)", () => {
    // chat_history is EMPTY while the chat is in flight — the user turn is
    // server-emitted on response. (A populated history plus a forward
    // affordance means tutorialStepBuilt, which renders the "Sent" line
    // instead of the composer/strip slot.)
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedChatPending: true,
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    const status = container.querySelector(".guided-pending-strip-status");
    expect(status).not.toBeNull();
    expect(status!.getAttribute("role")).toBe("status");
    // Never nested inside a role=log container (transcript log or wizard log)
    // where both live regions could announce the same change.
    for (const log of container.querySelectorAll('[role="log"]')) {
      expect(log.contains(status)).toBe(false);
    }
  });

  it("tutorial: the acknowledgement count announcer mounts once, in the stream, OUTSIDE the scroll wrapper", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // AcknowledgementLiveRegion: the persistent visually-hidden role=status
    // counter. It must sit OUTSIDE the scroll wrapper's churning subtree so
    // its 0→1 announce contract (content mutation inside a pre-existing node)
    // survives the relayout, and there must be exactly one.
    const announcers = container.querySelectorAll(
      '.guided-workspace-stream [role="status"].visually-hidden',
    );
    expect(announcers).toHaveLength(1);
    const scroll = container.querySelector(".guided-workspace-scroll");
    expect(scroll).not.toBeNull();
    expect(scroll!.contains(announcers[0])).toBe(false);
  });

  it("tutorial auto-scroll: an appended chat turn scrolls the conversation column when the user is at the bottom", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        chat_history: [
          {
            role: "user",
            content: "create the source",
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-05-12T10:00:00Z",
          },
        ],
      },
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    const scroll = container.querySelector<HTMLElement>(
      ".guided-workspace-scroll",
    );
    expect(scroll).not.toBeNull();
    // Stub the scroll geometry (jsdom has no layout): content overflows the
    // 400px viewport; track scrollTop assignments through a setter.
    let scrollTop = 0;
    Object.defineProperty(scroll!, "scrollHeight", {
      configurable: true,
      get: () => 1000,
    });
    Object.defineProperty(scroll!, "clientHeight", {
      configurable: true,
      get: () => 400,
    });
    Object.defineProperty(scroll!, "scrollTop", {
      configurable: true,
      get: () => scrollTop,
      set: (v: number) => {
        scrollTop = v;
      },
    });

    // No scroll event has fired — the at-bottom default (a fresh column starts
    // pinned) holds. Append the assistant reply.
    act(() => {
      const session = useSessionStore.getState().guidedSession!;
      useSessionStore.setState({
        guidedSession: {
          ...session,
          chat_history: [
            ...session.chat_history,
            {
              role: "assistant",
              content: "Source created.",
              seq: 2,
              step: "step_1_source",
              ts_iso: "2026-05-12T10:00:01Z",
            },
          ],
        },
      });
    });

    // The effect pinned the column to the bottom (scrollTop = scrollHeight).
    expect(scrollTop).toBe(1000);
  });

  it("tutorial auto-scroll: an appended chat turn does NOT yank the column when the user has scrolled up", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        chat_history: [
          {
            role: "user",
            content: "create the source",
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-05-12T10:00:00Z",
          },
        ],
      },
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    const scroll = container.querySelector<HTMLElement>(
      ".guided-workspace-scroll",
    );
    expect(scroll).not.toBeNull();
    let scrollTop = 100; // 1000 - 100 - 400 = 500px from the bottom (> 40px).
    Object.defineProperty(scroll!, "scrollHeight", {
      configurable: true,
      get: () => 1000,
    });
    Object.defineProperty(scroll!, "clientHeight", {
      configurable: true,
      get: () => 400,
    });
    Object.defineProperty(scroll!, "scrollTop", {
      configurable: true,
      get: () => scrollTop,
      set: (v: number) => {
        scrollTop = v;
      },
    });
    // The user scrolls up into the transcript — the handler records the
    // pre-append position.
    fireEvent.scroll(scroll!);

    act(() => {
      const session = useSessionStore.getState().guidedSession!;
      useSessionStore.setState({
        guidedSession: {
          ...session,
          chat_history: [
            ...session.chat_history,
            {
              role: "assistant",
              content: "Source created.",
              seq: 2,
              step: "step_1_source",
              ts_iso: "2026-05-12T10:00:01Z",
            },
          ],
        },
      });
    });

    // A reader reviewing earlier turns is not yanked to the bottom.
    expect(scrollTop).toBe(100);
  });

  it("tutorial completed: the completion surface renders under the guided shell with the stepper and the --completed frame escape hook", () => {
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
        step: "step_4_wire",
        history: [],
        terminal,
        chat_history: [],
        chat_turn_seq: 0,
        profile: null,
      },
      guidedTerminal: terminal,
    });

    const { container } = render(<ChatPanel isTutorial />);

    // The completed branch has NO internal scroll region — tutorial.css keys
    // its overflow escape (.tutorial-shell--guided .chat-panel--completed)
    // on this modifier class; without it the guided shell's overflow:hidden
    // frame would strand the completion content off-screen.
    const chatMain = container.querySelector("#chat-main");
    expect(chatMain?.classList.contains("chat-panel--completed")).toBe(true);
    // Stepper (all steps done → "Ready") + the completion summary render.
    expect(
      screen.getByRole("list", { name: /guided workflow/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Review YAML" }),
    ).toBeInTheDocument();
    // Tutorial completion suppresses the freeform handoff (concern B).
    expect(
      screen.queryByRole("button", { name: "Open freeform editor" }),
    ).toBeNull();
  });

  it("workspace CSS: internal scroll region focus ring is inset, and the 900px collapse bounds the stream row with the rail strip first", () => {
    // jsdom runs with css:false — media queries and computed styles are
    // invisible, so the responsive contract is pinned as CSS text (same idiom
    // as the stepper-grid test above).
    const css = readFileSync(
      join(process.cwd(), "src/components/chat/guided/guided.css"),
      "utf8",
    );
    // The keyboard-focusable scroll region draws its ring INSET — the
    // workspace frame clips overflow, so the default +2px offset ring would
    // be invisible on all four sides.
    expect(css).toMatch(
      /\.guided-workspace-scroll:focus-visible\s*\{[^}]*outline-offset: -2px/,
    );
    // ≤900px: single column; the rail collapses to a strip ABOVE the
    // conversation (order:-1) and the stream row stays bounded
    // (minmax(0, 1fr)) so the internal scroll still engages.
    const media900 = css.match(/@media \(max-width: 900px\)[\s\S]*?\n\}/);
    expect(media900).not.toBeNull();
    expect(media900![0]).toContain("grid-template-rows: auto minmax(0, 1fr);");
    expect(media900![0]).toContain("order: -1;");
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
      // The guided send path now threads an AbortSignal (Stop button +
      // client-timeout bound — elspeth-fb4464cdf0).
      expect(chatGuidedSpy).toHaveBeenCalledWith(
        "test-chat-message",
        expect.any(AbortSignal),
      );
    });
  });

  it("decision card offers Explain — one click sends the canned question down the chat path", async () => {
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

    await act(async () => {
      screen.getByRole("button", { name: "Explain this step" }).click();
    });

    await waitFor(() => {
      expect(chatGuidedSpy).toHaveBeenCalledWith(
        GUIDED_EXPLAIN_MESSAGE,
        expect.any(AbortSignal),
      );
    });
  });

  it("Explain is disabled while a chat or respond is in flight (same 409 guard as the composer)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedResponsePending: true,
    });

    render(<ChatPanel />);

    expect(
      (screen.getByRole("button", { name: "Explain this step" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("no Explain without a current decision (nothing on screen to explain)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: null,
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("button", { name: "Explain this step" }),
    ).toBeNull();
  });

  // C-2iii (composer first-principles review 2026-07-04): Retry on a
  // synthetic-failure turn.
  describe("synthetic-failure Retry", () => {
    function guidedSessionWithSyntheticFailure(): GuidedSession {
      return {
        step: "step_1_source",
        history: [],
        terminal: null,
        chat_history: [
          { role: "user", content: "scrape this page", seq: 1, step: "step_1_source", ts_iso: "t" },
          {
            role: "assistant",
            content: "I'm unavailable right now; you can still use the wizard controls.",
            seq: 2,
            step: "step_1_source",
            ts_iso: "t",
            assistant_message_kind: "synthetic_failure",
          },
        ],
        chat_turn_seq: 2,
        profile: null,
      };
    }

    it("renders the synthetic-failure turn as a distinct error turn, not an assistant bubble", () => {
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: guidedSessionWithSyntheticFailure(),
        guidedNextTurn: singleSelectTurn(),
      });

      render(<ChatPanel />);

      expect(screen.queryByText("ELSPETH said:", { exact: false })).toBeNull();
      expect(
        screen.getByText(
          "I'm unavailable right now; you can still use the wizard controls.",
        ),
      ).toBeInTheDocument();
    });

    it("the Current-decision heading falls back to the static step purpose instead of the synthetic message", () => {
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: guidedSessionWithSyntheticFailure(),
        guidedNextTurn: singleSelectTurn(),
      });

      render(<ChatPanel />);

      // Static step-purpose fallback (GUIDED_STEP_PURPOSES.step_1_source) —
      // the synthetic-failure turn is the only chat_history entry for this
      // step, so it must never win the headline.
      expect(
        screen.getByRole("heading", {
          level: 2,
          name: "Choose the input and confirm what ELSPETH can read.",
        }),
      ).toBeInTheDocument();
    });

    it("Retry resends the preceding user message via the normal chat path", async () => {
      const chatGuidedSpy = vi.fn().mockResolvedValue(undefined);
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: guidedSessionWithSyntheticFailure(),
        guidedNextTurn: singleSelectTurn(),
        chatGuided: chatGuidedSpy,
      });

      render(<ChatPanel />);

      await act(async () => {
        screen.getByRole("button", { name: "Retry" }).click();
      });

      await waitFor(() => {
        expect(chatGuidedSpy).toHaveBeenCalledWith(
          "scrape this page",
          expect.any(AbortSignal),
        );
      });
    });

    it("Retry falls back to refetching guided state when there is no preceding user turn to resend", async () => {
      const startGuidedSpy = vi.fn().mockResolvedValue(undefined);
      const noUserTurn: GuidedSession = {
        step: "step_1_source",
        history: [],
        terminal: null,
        chat_history: [
          {
            role: "assistant",
            content: "I'm unavailable right now; you can still use the wizard controls.",
            seq: 0,
            step: "step_1_source",
            ts_iso: "t",
            assistant_message_kind: "synthetic_failure",
          },
        ],
        chat_turn_seq: 0,
        profile: null,
      };
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: noUserTurn,
        guidedNextTurn: singleSelectTurn(),
        startGuided: startGuidedSpy,
      });

      render(<ChatPanel />);

      await act(async () => {
        screen.getByRole("button", { name: "Retry" }).click();
      });

      await waitFor(() => {
        expect(startGuidedSpy).toHaveBeenCalledWith("session-guided");
      });
    });

    it("Retry is disabled while a chat or respond is already in flight", () => {
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: guidedSessionWithSyntheticFailure(),
        guidedNextTurn: singleSelectTurn(),
        guidedChatPending: true,
      });

      render(<ChatPanel />);

      expect(screen.getByRole("button", { name: "Retry" })).toBeDisabled();
    });
  });

  it("tutorial: an Explain send does NOT count as the step's prompt (locked box must not flip to 'Sent')", () => {
    // On confirm-only steps the Explain user-turn shares the current step;
    // without the exact-content filter it would satisfy
    // tutorialPromptSentForStep and prematurely swap the locked box for the
    // "Sent" line, stranding the learner.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        chat_history: [
          {
            role: "user",
            content: GUIDED_EXPLAIN_MESSAGE,
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-07-03T00:00:00Z",
          },
        ],
      },
      guidedNextTurn: singleSelectTurn(),
    });

    render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    // The locked box (mocked ChatInput) is still offered — NOT the Sent line.
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    expect(
      screen.queryByText(/your request is in the transcript above/i),
    ).toBeNull();
  });

  it("swaps the ChatInput for the pending strip with Stop while a guided chat is in flight", () => {
    // Pending swap (elspeth-6a9673ecd3): the input is UNMOUNTED — not
    // disabled — while /guided/chat is in flight; the working strip carries
    // the abort affordance. The landmark section must survive the swap (AT
    // navigation + staging e2e locators find the composer by region name).
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedChatPending: true,
    });

    const { container } = render(<ChatPanel />);

    expect(screen.queryByTestId("chat-input")).toBeNull();
    expect(container.querySelector(".guided-pending-strip")).not.toBeNull();
    expect(
      screen.getByRole("button", { name: "Stop composing" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Describe what you want" }),
    ).toBeInTheDocument();
  });

  it("does NOT swap (and offers no Stop) while only a turn-respond is pending", () => {
    // Deliberate asymmetry: a respond in flight has its own adjacent
    // "Saving decision..." status and nothing abortable — and a live-guided
    // user can have a typed draft in the textarea when they submit a
    // decision card; unmounting the input would destroy the draft.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedChatPending: false,
      guidedResponsePending: true,
    });

    const { container } = render(<ChatPanel />);

    expect(container.querySelector(".guided-pending-strip")).toBeNull();
    const input = screen.getByTestId("chat-input");
    expect(input.dataset.disabled).toBe("true");
    expect(input.dataset.hasCancel).toBe("false");
  });

  it("tutorial: keeps the retry chat box (not the 'Sent' line) when a Send-driven step was sent but produced no forward affordance", () => {
    // Regression: a transient chain-solve failure at step_3 appends a user turn
    // but returns next_turn=null. The 'Sent' line must NOT replace the box, or
    // the passive learner is stranded with no widget and no exit.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        step: "step_3_transforms",
        chat_history: [
          {
            role: "user",
            content: "do the transforms",
            seq: 1,
            step: "step_3_transforms",
            ts_iso: "2026-05-12T10:00:00Z",
          },
        ],
      },
      guidedNextTurn: null,
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_3_transforms: "do the transforms" }}
      />,
    );

    // Retry box present; the stranding "Sent" line absent; the "press Send"
    // guidance still shown so the learner knows what to do.
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    expect(
      screen.queryByText(/your request is in the transcript above/i),
    ).toBeNull();
    expect(
      container.querySelector(".guided-current-decision-tutorial-note"),
    ).not.toBeNull();
  });

  it("tutorial: shows the 'Sent' line and drops the 'press Send' note once the step was sent AND a forward turn exists", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: {
        ...activeGuidedSession(),
        step: "step_1_source",
        chat_history: [
          {
            role: "user",
            content: "create the source",
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-05-12T10:00:00Z",
          },
        ],
      },
      guidedNextTurn: singleSelectTurn(),
    });

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{ step_1_source: "create the source" }}
      />,
    );

    expect(
      screen.getByText(/your request is in the transcript above/i),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("chat-input")).toBeNull();
    // The "press Send below" coaching note is gone once Sent.
    expect(
      container.querySelector(".guided-current-decision-tutorial-note"),
    ).toBeNull();
  });

  it("passes the backend guided-chat message limit to the guided ChatInput", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel />);

    expect(screen.getByTestId("chat-input").dataset.maxLength).toBe("4096");
  });

  it("unmounts the guided ChatInput while guidedChatPending=true (409-race pin)", () => {
    // Regression pin (sits next to the guided-resend 409 fix): with the
    // pending swap, no input is mounted while the build is in flight, so a
    // second send cannot race it AT ALL — stronger than the old disabled
    // gate, whose textarea never actually received `disabled`.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedChatPending: true,
    });

    const { container } = render(<ChatPanel />);

    expect(screen.queryByTestId("chat-input")).toBeNull();
    // Silent-compute affordance: the lean working strip is the busy signal
    // (the old detached ComposingIndicator card is gone from guided).
    expect(container.querySelector(".guided-pending-strip")).not.toBeNull();
    expect(container.querySelector(".composing-indicator")).toBeNull();
  });

  it("does not show the pending strip on the guided surface when guidedChatPending=false", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedChatPending: false,
    });

    const { container } = render(<ChatPanel />);

    expect(container.querySelector(".guided-pending-strip")).toBeNull();
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
  });

  it("pending-swap focus contract: strip wrapper takes focus in, composer section takes it back out", () => {
    // Unmounting the focused composer content would strand focus at <body>
    // (WCAG 2.4.3). Into pending: focus moves to the strip's tabIndex=-1
    // wrapper — never the Stop button (double-Enter would abort the request
    // just sent). Out of pending: focus returns into the composer (under the
    // ChatInput mock inputRef never attaches, so the section is the landing).
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedChatPending: false,
    });

    const { container } = render(<ChatPanel />);

    // Focus inside the composer (the mocked ChatInput button), as after Send.
    act(() => {
      screen.getByTestId("chat-input").focus();
    });
    act(() => {
      useSessionStore.setState({ guidedChatPending: true });
    });
    const strip = container.querySelector(".guided-pending-strip");
    expect(strip).not.toBeNull();
    expect(document.activeElement).toBe(strip);

    act(() => {
      useSessionStore.setState({ guidedChatPending: false });
    });
    const section = screen.getByRole("region", {
      name: "Describe what you want",
    });
    expect(section.contains(document.activeElement)).toBe(true);
  });

  it("pending-swap focus contract: leaves focus alone when the user had moved away from the composer", () => {
    // A user re-reading the transcript during a long compose must not be
    // yanked back into the composer when the request resolves.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
      guidedChatPending: false,
    });

    render(<ChatPanel />);

    // The mount-time step-advance effect focuses the decision widget's first
    // control — an element OUTSIDE the composer section. That is exactly the
    // "user is working elsewhere" state; both flips must leave it alone.
    const before = document.activeElement;
    expect(
      screen
        .getByRole("region", { name: "Describe what you want" })
        .contains(before),
    ).toBe(false);

    act(() => {
      useSessionStore.setState({ guidedChatPending: true });
    });
    expect(document.activeElement).toBe(before);

    act(() => {
      useSessionStore.setState({ guidedChatPending: false });
    });
    expect(document.activeElement).toBe(before);
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
        profile: null,
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
    expect(screen.getByTestId("inline-run-results")).toBeInTheDocument();
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
        profile: null,
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
    // Default-freeform contract: freeform body surfaces a "Switch to guided"
    // affordance in the header so the user can opt into guided mode.
    expect(
      screen.getByRole("button", { name: "Switch to guided" }),
    ).toBeInTheDocument();
  });

  it("'Switch to guided' button calls enterGuided() when clicked from the freeform body", async () => {
    const enterGuidedSpy = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: null,
      guidedNextTurn: null,
      enterGuided: enterGuidedSpy,
    });

    render(<ChatPanel />);

    const button = screen.getByRole("button", { name: "Switch to guided" });
    await act(async () => {
      button.click();
    });

    expect(enterGuidedSpy).toHaveBeenCalledTimes(1);
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
        profile: null,
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

  // C-4b (composer first-principles review 2026-07-04): "Switch to guided"
  // must not silently no-op on a permanently-terminal guided session.
  it("disables 'Switch to guided' with an explanation when guided ended via solver_exhausted", () => {
    const terminal: TerminalState = {
      kind: "exited_to_freeform",
      reason: "solver_exhausted",
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
        profile: null,
      },
      guidedNextTurn: null,
      guidedTerminal: terminal,
    });

    render(<ChatPanel />);

    const button = screen.getByRole("button", { name: "Switch to guided" });
    expect(button).toBeDisabled();
    expect(
      screen.getByText(
        "Guided ended for this session — start a new session to use guided.",
      ),
    ).toBeInTheDocument();
  });

  it("disables 'Switch to guided' with an explanation when guided ended via protocol_violation", () => {
    const terminal: TerminalState = {
      kind: "exited_to_freeform",
      reason: "protocol_violation",
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
        profile: null,
      },
      guidedNextTurn: null,
      guidedTerminal: terminal,
    });

    render(<ChatPanel />);

    expect(screen.getByRole("button", { name: "Switch to guided" })).toBeDisabled();
  });

  it("keeps 'Switch to guided' enabled (reenterable) when the terminal reason is user_pressed_exit", () => {
    // Reversible operator exit — POST /guided/reenter still honours it
    // (routes/composer/guided.py post_guided_reenter). Disabling here would
    // be false: the switch genuinely works via reenterGuided.
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
        profile: null,
      },
      guidedNextTurn: null,
      guidedTerminal: terminal,
    });

    render(<ChatPanel />);

    expect(
      screen.getByRole("button", { name: "Switch to guided" }),
    ).not.toBeDisabled();
  });

  it("clicking 'Switch to guided' on a user_pressed_exit terminal session actually resumes guided (C-4b — the old silent no-op)", async () => {
    // The old bug: guidedSession was null until the user clicked (no C-4a
    // restore-on-load), so the FIRST click always mis-routed through
    // enterGuided()'s startGuided/GET branch instead of reenterGuided,
    // observing the same terminal and landing back in freeform with zero
    // feedback. With guidedSession already populated (as selectSession now
    // does on load — sessionStore's C-4a fix), enterGuided() sees
    // terminal.kind === "exited_to_freeform" up front and correctly calls
    // reenterGuided() instead. This test pins the click reaching
    // enterGuided at all; enterGuided's internal branch to reenterGuided is
    // covered in sessionStore.guided.test.ts.
    const terminal: TerminalState = {
      kind: "exited_to_freeform",
      reason: "user_pressed_exit",
      pipeline_yaml: null,
    };
    const enterGuidedSpy = vi.fn().mockResolvedValue(undefined);

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
        profile: null,
      },
      guidedNextTurn: null,
      guidedTerminal: terminal,
      enterGuided: enterGuidedSpy,
    });

    render(<ChatPanel />);

    const button = screen.getByRole("button", { name: "Switch to guided" });
    expect(button).not.toBeDisabled();
    await act(async () => {
      button.click();
    });

    expect(enterGuidedSpy).toHaveBeenCalledTimes(1);
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
        profile: null,
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

  it("suppresses ExitToFreeformButton when isTutorial (concern B)", () => {
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(),
    });

    render(<ChatPanel isTutorial />);

    // The rival source-pick widget is SUPPRESSED in the tutorial — a passive
    // learner advances via Send, not by picking from a live submit-on-click menu
    // (whose options don't even include the scripted source). So the chip group
    // is gone...
    expect(
      screen.queryByRole("group", { name: "Which source plugin should we use?" }),
    ).toBeNull();
    // ...as is the freeform exit affordance — a tutorial must never expose a
    // switch-to-freeform path (spec §"Frontend" concern B).
    expect(
      screen.queryByRole("button", { name: "Exit to freeform" }),
    ).toBeNull();
  });

  it("renders a guided placeholder (not the freeform body) when isTutorial and the session is still loading (concern B startup flash)", () => {
    // TutorialGuidedShell clears guidedSession/guidedNextTurn to null before
    // the async start resolves (TutorialGuidedShell.tsx:61-81). Without the
    // tutorial guard, ChatPanel would fall through to the panel-less freeform
    // body during that window.
    useSessionStore.setState({
      activeSessionId: "session-guided",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: null,
      guidedNextTurn: null,
    });

    render(<ChatPanel isTutorial />);

    // Guided placeholder present...
    expect(
      screen.getByTestId("tutorial-guided-loading"),
    ).toBeInTheDocument();
    // ...and the freeform composer input is NOT rendered.
    expect(screen.queryByTestId("chat-input")).toBeNull();
  });

  it("renders a guided placeholder (not the freeform body) when isTutorial and terminal is exited_to_freeform (concern B defensive)", () => {
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
        step: "step_3_transforms",
        history: [],
        terminal,
        chat_history: [],
        chat_turn_seq: 0,
        profile: null,
      },
      guidedNextTurn: null,
      guidedTerminal: terminal,
    });

    render(<ChatPanel isTutorial />);

    expect(screen.getByTestId("tutorial-guided-loading")).toBeInTheDocument();
    expect(screen.queryByTestId("chat-input")).toBeNull();
    // The placeholder rail reflects the ACTUAL session step, not a hardcoded
    // step_1 (GuidedWorkflowStepper marks the current step with
    // aria-current="step", ChatPanel.tsx:1759). The transform-step rail item
    // must be the current one.
    const current = screen
      .getByTestId("tutorial-guided-loading")
      .querySelector('[aria-current="step"]');
    expect(current).not.toBeNull();
    expect(current).toHaveTextContent(/transform/i);
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
    return { step: "step_1_source", history: [], terminal: null, chat_history: [], chat_turn_seq: 0, profile: null };
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

  it("moves focus to the new decision when a same-step build changes the turn type", async () => {
    // The composer now docks at the BOTTOM for every session (tutorial
    // included). A same-step `/guided/chat` Send replaces the turn with a
    // different TYPE without advancing the step (single_select → schema_form at
    // step 1; null → propose_chain at step 3). The just-built decision lands
    // ABOVE the box the user Sent from, so the focus effect must re-fire on the
    // type change and bring it into view — otherwise the passive learner is left
    // looking at the docked box with the decision off-screen above it. This is
    // the type-change counterpart to the same-type "does NOT re-focus" guard
    // above; both share the [step_index, type] dependency.
    useSessionStore.setState({
      activeSessionId: "session-focus",
      sessions: [guidedSessionFixture],
      messages: [],
      guidedSession: activeGuidedSession(),
      guidedNextTurn: singleSelectTurn(0),
    });

    render(<ChatPanel />);

    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "CSV" }),
      );
    });

    // Same step_index (0), different turn TYPE — the build result.
    act(() => {
      useSessionStore.setState({
        guidedNextTurn: {
          type: "schema_form",
          step_index: 0,
          payload: {
            mode: "plugin_options",
            plugin: "csv",
            knobs: { fields: [] },
            prefilled: {},
          },
        },
      });
    });

    // Focus lands on the schema form's first control (the Edit toggle), proving
    // the effect re-fired on the type change rather than stranding the user on
    // the prior turn's "CSV" button.
    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Edit" }),
      );
    });
  });
});

// ── Inline-source projection (Phase 5a Task 3) ────────────────────────────────
//
// These tests cover the wiring that derives an InlineSourceSummary from
// `compositionState.sources[*].options["blob_ref"]` and the corresponding session
// blob's metadata + preview, then surfaces the InlineSourceCreatedTurn widget
// in the message stream.
//
// The widget itself is tested in InlineSourceCreatedTurn.test.tsx; here we
// only assert the predicate ("widget renders iff inline source is bound to
// the active session") and the absence case ("no inline source → no
// widget"). Detailed widget rendering, edit-button visibility per
// provenance, and audit-info disclosure all live in the widget test.
describe("ChatPanel inline-source projection", () => {
  const twoRowInlineSourceText = "url\nhttps://a.gov.au\nhttps://b.gov.au";
  const twoRowInlineSourceHash =
    "9b8d3393ad3be052da5f25595789f926a161a4f8c0090c61f10a9cbab69a473c";
  const oneRowInlineSourceText = "url\nhttps://a.gov.au";
  const differentInlineSourceHash =
    "e14713c61f9a7d0119925f46e9957e6d42a1604a5d62932853c46b03681af30b";

  const sessionFixture: Session = {
    id: "session-inline",
    title: "Inline session",
    created_at: "2026-05-18T10:00:00Z",
    updated_at: "2026-05-18T10:00:00Z",
  };

  function makeBlobMetadata(overrides: Partial<BlobMetadata> = {}): BlobMetadata {
    return {
      id: "blob-inline-1",
      session_id: "session-inline",
      filename: "chat.csv",
      mime_type: "text/csv",
      size_bytes: 42,
      content_hash: twoRowInlineSourceHash,
      created_at: "2026-05-18T10:00:01Z",
      created_by: "assistant",
      source_description: null,
      status: "ready",
      creation_modality: "llm_generated",
      created_from_message_id: "msg-1",
      creating_model_identifier: "claude-opus-4-7",
      creating_model_version: "20260101",
      creating_provider: "anthropic",
      creating_composer_skill_hash: "skill-hash",
      creating_arguments_hash: "args-hash",
      ...overrides,
    };
  }

  beforeEach(() => {
    vi.resetAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    resetStore(useInlineSourceStore);
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: null,
    });
  });

  it("renders the widget when a composition source blob_ref resolves to a session blob", async () => {
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeBlobMetadata(),
    );
    (
      apiClient.previewBlobContent as ReturnType<typeof vi.fn>
    ).mockResolvedValue(twoRowInlineSourceText);

    const composition = makeComposition(1, {
      sources: {
        source: {
          plugin: "inline_blob",
          options: { blob_ref: "blob-inline-1" },
        },
      },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    await waitFor(() => {
      expect(
        screen.getByRole("region", { name: /source created/i }),
      ).toBeInTheDocument();
    });

    // Provenance-derived: llm_generated → llm-generated (display form) →
    // Edit affordance present (F-4). Asserted here AS A WIRING TEST to
    // confirm the projection carries provenance end-to-end; the widget's
    // own test owns the per-provenance rendering matrix.
    expect(
      screen.getByRole("button", { name: /edit the list/i }),
    ).toBeInTheDocument();
  });

  it("renders the widget when a named inline source carries the blob_ref", async () => {
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeBlobMetadata(),
    );
    (
      apiClient.previewBlobContent as ReturnType<typeof vi.fn>
    ).mockResolvedValue(twoRowInlineSourceText);

    const composition = makeComposition(1, {
      sources: {
        created: {
          plugin: "inline_blob",
          options: { blob_ref: "blob-inline-1" },
        },
      },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    await waitFor(() => {
      expect(apiClient.getBlobMetadata).toHaveBeenCalledWith(
        "session-inline",
        "blob-inline-1",
      );
      expect(
        screen.getByRole("region", { name: /source created/i }),
      ).toBeInTheDocument();
    });
  });

  it("does NOT fetch content for an uploaded source blob_ref", async () => {
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeBlobMetadata({
        created_by: "user",
        source_description: "uploaded",
        creation_modality: "verbatim",
        created_from_message_id: null,
        creating_model_identifier: null,
        creating_model_version: null,
        creating_provider: null,
        creating_composer_skill_hash: null,
        creating_arguments_hash: null,
        size_bytes: 250_000_000,
      }),
    );

    const composition = makeComposition(1, {
      sources: {
        source: {
          plugin: "csv_file",
          options: { blob_ref: "blob-inline-1", path: "/data/upload.csv" },
        },
      },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    await waitFor(() => {
      expect(apiClient.getBlobMetadata).toHaveBeenCalledWith(
        "session-inline",
        "blob-inline-1",
      );
    });

    expect(apiClient.previewBlobContent).not.toHaveBeenCalled();
    expect(
      screen.queryByRole("region", { name: /source created/i }),
    ).toBeNull();
  });

  it("does NOT render the widget when compositionState has no inline source", () => {
    const composition = makeComposition(1, {
      sources: { source: { plugin: "csv_file", options: { path: "data.csv" } } },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /source created/i }),
    ).toBeNull();
    // No blob fetch attempted when no blob_ref is present.
    expect(apiClient.getBlobMetadata).not.toHaveBeenCalled();
    expect(apiClient.previewBlobContent).not.toHaveBeenCalled();
  });

  it("clears a stale inline-source summary when the active blob is not inline", async () => {
    useInlineSourceStore.getState().setSummary("session-inline", {
      blobId: "old-inline",
      filename: "old.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://old.gov.au",
      rowCount: 2,
      contentHash: twoRowInlineSourceHash,
      provenance: "llm-generated",
    });
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeBlobMetadata({
        id: "uploaded-blob",
        created_by: "user",
        created_from_message_id: null,
      }),
    );
    const composition = makeComposition(1, {
      sources: {
        source: {
          plugin: "csv_file",
          options: { blob_ref: "uploaded-blob" },
        },
      },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    await waitFor(() => {
      expect(useInlineSourceStore.getState().getSummary("session-inline")).toBeNull();
    });
    expect(apiClient.previewBlobContent).not.toHaveBeenCalled();
  });

  it("does NOT render the widget when compositionState has no sources", () => {
    const composition = makeComposition(1, { sources: {} });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /source created/i }),
    ).toBeNull();
  });

  // Tier-1 audit-trail invariant (see InlineSourceSummary.contentHash type
  // doc): a blob with a null content_hash is a wire-contract violation.
  // The projection effect throws on this case; the throw is caught and
  // logged; the inlineSourceStore is NEVER populated; the widget does
  // NOT render.  Substituting an empty string into the rendered audit
  // pane would assert a value the system never recorded — exactly the
  // fabrication CLAUDE.md forbids.
  it("does NOT render the widget when the blob's content_hash is null (audit-trail invariant)", async () => {
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...makeBlobMetadata(),
      content_hash: null,
    });
    (
      apiClient.previewBlobContent as ReturnType<typeof vi.fn>
    ).mockResolvedValue("url\nhttps://a.gov.au");

    // Suppress the expected console.error from the projection's catch
    // arm so the test output is clean.  The assertion below confirms
    // that the error WAS logged with the expected prefix — that's how
    // we know the invariant fired rather than the test silently
    // matching the negative case for an unrelated reason.
    const errorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    const composition = makeComposition(1, {
      sources: {
        source: {
          plugin: "inline_blob",
          options: { blob_ref: "blob-inline-1" },
        },
      },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    // Wait for the projection effect to resolve and throw.
    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringMatching(/\[inline-source\] projection failed:/),
        expect.any(Error),
      );
    });

    expect(
      screen.queryByRole("region", { name: /source created/i }),
    ).toBeNull();

    errorSpy.mockRestore();
  });

  it("does NOT render the widget when blob MIME metadata has malformed parameter syntax", async () => {
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeBlobMetadata({ mime_type: "text/csv; charset=" }),
    );
    (
      apiClient.previewBlobContent as ReturnType<typeof vi.fn>
    ).mockResolvedValue(twoRowInlineSourceText);
    const errorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    const composition = makeComposition(1, {
      sources: {
        source: {
          plugin: "inline_blob",
          options: { blob_ref: "blob-inline-1" },
        },
      },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringMatching(/\[inline-source\] projection failed:/),
        expect.objectContaining({
          message: expect.stringMatching(/invalid MIME metadata/i),
        }),
      );
    });

    expect(
      screen.queryByRole("region", { name: /source created/i }),
    ).toBeNull();
    expect(
      useInlineSourceStore.getState().getSummary("session-inline"),
    ).toBeNull();

    errorSpy.mockRestore();
  });

  it("does NOT render the widget when preview bytes disagree with blob metadata content_hash", async () => {
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeBlobMetadata({ content_hash: differentInlineSourceHash }),
    );
    (
      apiClient.previewBlobContent as ReturnType<typeof vi.fn>
    ).mockResolvedValue(oneRowInlineSourceText);
    const errorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    const composition = makeComposition(1, {
      sources: {
        source: {
          plugin: "inline_blob",
          options: { blob_ref: "blob-inline-1" },
        },
      },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringMatching(/\[inline-source\] projection failed:/),
        expect.objectContaining({
          message: expect.stringMatching(/content_hash mismatch/i),
        }),
      );
    });

    expect(
      screen.queryByRole("region", { name: /source created/i }),
    ).toBeNull();
    expect(
      useInlineSourceStore.getState().getSummary("session-inline"),
    ).toBeNull();

    errorSpy.mockRestore();
  });

  it("logs the bound projection error object when provenance translation throws", async () => {
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeBlobMetadata(),
    );
    (
      apiClient.previewBlobContent as ReturnType<typeof vi.fn>
    ).mockResolvedValue(twoRowInlineSourceText);
    const projectionError = new TypeError("projection dependency failed");
    vi.spyOn(apiClient, "toInlineSourceProvenance").mockImplementation(() => {
      throw projectionError;
    });
    const errorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    const composition = makeComposition(1, {
      sources: {
        source: {
          plugin: "inline_blob",
          options: { blob_ref: "blob-inline-1" },
        },
      },
    });

    useSessionStore.setState({
      activeSessionId: "session-inline",
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });

    render(<ChatPanel />);

    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringMatching(/\[inline-source\] projection failed:/),
        projectionError,
      );
    });
    expect(
      screen.queryByRole("region", { name: /source created/i }),
    ).toBeNull();

    errorSpy.mockRestore();
  });
});

// ── deriveRowCount unit tests (MIME parameter parsing) ────────────────────────
//
// Lives outside the rendering-test describe block because it's a pure
// helper with no React surface.  Imported from ChatPanel.tsx via the
// named export at the top of the file.
describe("deriveRowCount", () => {
  it("returns row count for vanilla 'text/csv'", () => {
    expect(deriveRowCount("text/csv", "url\nhttps://a\nhttps://b")).toBe(2);
  });

  it("returns row count for parameterised 'text/csv; charset=utf-8'", () => {
    // Real CSV uploads from a browser commonly carry a charset parameter.
    // A strict `===` comparison silently classified these as "unknown row
    // count"; the MIME-base normalisation in deriveRowCount fixes this.
    expect(
      deriveRowCount("text/csv; charset=utf-8", "url\nhttps://a\nhttps://b"),
    ).toBe(2);
  });

  it("is case-insensitive on the base MIME type", () => {
    expect(deriveRowCount("TEXT/CSV", "url\nhttps://a")).toBe(1);
  });

  it("returns null for non-CSV MIME types", () => {
    expect(deriveRowCount("application/json", "[1,2,3]")).toBeNull();
  });

  it("returns 0 for a header-only CSV (no data rows)", () => {
    expect(deriveRowCount("text/csv", "url")).toBe(0);
  });

  it("returns 0 for empty content", () => {
    expect(deriveRowCount("text/csv", "")).toBe(0);
  });
});

// ── isAmbiguousInlineProposal unit tests (Phase 5a Task 4) ────────────────────
//
// Phase 5a Task 4 ambiguity heuristic v1. False negatives are recoverable
// (proposal still routes through the standard banner); false positives are
// disruptive (a disambiguation widget appears where a banner would have
// sufficed). The canonical demo prompt MUST be a false negative — that's
// the load-bearing constraint.
describe("isAmbiguousInlineProposal", () => {
  function makeInlineProposal(
    summary: string,
    overrides: Partial<CompositionProposal> = {},
  ): CompositionProposal {
    return {
      id: "prop-1",
      session_id: "session-1",
      tool_call_id: "tc-1",
      tool_name: "set_pipeline",
      status: "pending",
      summary,
      rationale: "",
      affects: ["source"],
      arguments_redacted_json: {
        source: {
          plugin: "csv",
          on_success: "csv_rows",
          blob_id: null,
          options: "{}",
          on_validation_failure: null,
          inline_blob: {
            filename: "chat.csv",
            mime_type: "text/csv",
            content: "<inline-blob:42-bytes>",
            description: null,
          },
        },
        sources: null,
        nodes: [],
        edges: [],
        outputs: [],
        metadata: null,
      },
      base_state_id: null,
      committed_state_id: null,
      audit_event_id: null,
      created_at: "2026-05-18T10:00:00Z",
      updated_at: "2026-05-18T10:00:00Z",
      ...overrides,
    };
  }

  it("returns true when summary contains 'I read'", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("I read your message as 3 separate URLs."),
      ),
    ).toBe(true);
  });

  it("returns true when summary contains 'interpreted as'", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("Input interpreted as 3 rows for the inline source."),
      ),
    ).toBe(true);
  });

  it("is case-insensitive on the phrase match", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("I READ this as several rows."),
      ),
    ).toBe(true);
  });

  // The canonical demo proposal MUST be classified as non-ambiguous. The
  // composer narration for that case describes generation ("a list of 5
  // government web pages") not interpretation of user input.
  it("returns FALSE for the canonical demo proposal (no ambiguity phrases)", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal(
          "Created an inline source with 5 Australian government web pages " +
            "and wired an LLM transform to rate each on coolness.",
        ),
      ),
    ).toBe(false);
  });

  it("returns false when the tool_name is not set_pipeline", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("I read your input as 3 rows.", {
          tool_name: "patch_source_options",
        }),
      ),
    ).toBe(false);
  });

  it("returns false when arguments lack an inline_blob", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("I read your input as 3 rows.", {
          arguments_redacted_json: {
            source: {
              plugin: "csv_file",
              on_success: "csv_rows",
              blob_id: null,
              options: '{"path":"<redacted-option-value>"}',
              on_validation_failure: null,
            },
            sources: null,
            nodes: [],
            edges: [],
            outputs: [],
            metadata: null,
          },
        }),
      ),
    ).toBe(false);
  });

  it("returns false when set_pipeline includes hidden full-pipeline changes", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("I read your input as 3 rows.", {
          arguments_redacted_json: {
            source: {
              plugin: "csv",
              on_success: "csv_rows",
              blob_id: null,
              options: "{}",
              on_validation_failure: null,
              inline_blob: {
                filename: "chat.csv",
                mime_type: "text/csv",
                content: "<inline-blob:42-bytes>",
                description: null,
              },
            },
            sources: null,
            nodes: [],
            edges: [],
            outputs: [
              {
                sink_name: "exfil",
                plugin: "webhook",
                options: { url: "<redacted-option-value>" },
              },
            ],
            metadata: null,
          },
        }),
      ),
    ).toBe(false);
  });

  it("returns false when set_pipeline includes hidden transform nodes", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("I read your input as 3 rows.", {
          arguments_redacted_json: {
            source: {
              plugin: "csv",
              on_success: "csv_rows",
              blob_id: null,
              options: "{}",
              on_validation_failure: null,
              inline_blob: {
                filename: "chat.csv",
                mime_type: "text/csv",
                content: "<inline-blob:42-bytes>",
                description: null,
              },
            },
            sources: null,
            nodes: [
              {
                id: "unexpected_transform",
                node_type: "transform",
                plugin: "llm_filter",
                input: "source",
                options: { prompt: "<redacted-option-value>" },
              },
            ],
            edges: [],
            outputs: [],
            metadata: null,
          },
        }),
      ),
    ).toBe(false);
  });

  it("returns false when the source carries validation-failure routing", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("I read your input as 3 rows.", {
          arguments_redacted_json: {
            source: {
              plugin: "csv",
              on_success: "csv_rows",
              blob_id: null,
              options: "{}",
              on_validation_failure: "unexpected_route",
              inline_blob: {
                filename: "chat.csv",
                mime_type: "text/csv",
                content: "<inline-blob:42-bytes>",
                description: null,
              },
            },
            sources: null,
            nodes: [],
            edges: [],
            outputs: [],
            metadata: null,
          },
        }),
      ),
    ).toBe(false);
  });

  it("returns false when the inline source carries extra source options", () => {
    expect(
      isAmbiguousInlineProposal(
        makeInlineProposal("I read your input as 3 rows.", {
          arguments_redacted_json: {
            source: {
              plugin: "csv",
              on_success: "csv_rows",
              blob_id: null,
              options: '{"delimiter":"|"}',
              on_validation_failure: null,
              inline_blob: {
                filename: "chat.csv",
                mime_type: "text/csv",
                content: "<inline-blob:42-bytes>",
                description: null,
              },
            },
            sources: null,
            nodes: [],
            edges: [],
            outputs: [],
            metadata: null,
          },
        }),
      ),
    ).toBe(false);
  });

  it("returns false when set_pipeline includes non-default metadata", () => {
    const proposal = makeInlineProposal("I read your input as 3 rows.");
    expect(
      isAmbiguousInlineProposal({
        ...proposal,
        arguments_redacted_json: {
          ...proposal.arguments_redacted_json,
          metadata: { name: "Hidden pipeline name", description: null },
        },
      }),
    ).toBe(false);
  });

  it("returns false when set_pipeline includes alternate sources", () => {
    const proposal = makeInlineProposal("I read your input as 3 rows.");
    expect(
      isAmbiguousInlineProposal({
        ...proposal,
        arguments_redacted_json: {
          ...proposal.arguments_redacted_json,
          sources: {
            extra: {
              plugin: "csv",
              on_success: "csv_rows",
              options: "{}",
            },
          },
        },
      }),
    ).toBe(false);
  });

  it("returns false when the inline blob carries a non-default description", () => {
    const proposal = makeInlineProposal("I read your input as 3 rows.");
    const source = proposal.arguments_redacted_json[
      "source"
    ] as Record<string, unknown>;
    const inlineBlob = source["inline_blob"] as Record<string, unknown>;
    expect(
      isAmbiguousInlineProposal({
        ...proposal,
        arguments_redacted_json: {
          ...proposal.arguments_redacted_json,
          source: {
            ...source,
            inline_blob: {
              ...inlineBlob,
              description: "Hidden source description",
            },
          },
        },
      }),
    ).toBe(false);
  });
});

describe("hasExistingCompositionContent", () => {
  const proposal: CompositionProposal = {
    id: "prop-1",
    session_id: "session-1",
    tool_call_id: "tc-1",
    tool_name: "set_pipeline",
    status: "pending",
    summary: "I read your input as 3 rows.",
    rationale: "",
    affects: ["source"],
    arguments_redacted_json: {},
    base_state_id: null,
    committed_state_id: null,
    audit_event_id: null,
    created_at: "2026-05-18T10:00:00Z",
    updated_at: "2026-05-18T10:00:00Z",
  };

  it("returns false for null or empty composition state", () => {
    expect(hasExistingCompositionContent(null)).toBe(false);
    expect(
      hasExistingCompositionContent(
        makeComposition(1, {
          sources: {},
          nodes: [],
          edges: [],
          outputs: [],
          metadata: { name: null, description: null },
        }),
      ),
    ).toBe(false);
    expect(
      hasExistingCompositionContent(
        makeComposition(1, {
          sources: {},
          nodes: [],
          edges: [],
          outputs: [],
          metadata: { name: "Untitled Pipeline", description: "" },
        }),
      ),
    ).toBe(false);
  });

  it("returns true when any state content would be replaced by set_pipeline", () => {
    expect(hasExistingCompositionContent(makeComposition(1))).toBe(true);
    expect(
      hasExistingCompositionContent(
        makeComposition(1, {
          sources: {},
          nodes: [],
          edges: [
            {
              id: "edge-1",
              from_node: "a",
              to_node: "b",
              edge_type: "on_success",
              label: null,
            },
          ],
          outputs: [],
        }),
      ),
    ).toBe(true);
    expect(
      hasExistingCompositionContent(
        makeComposition(1, {
          sources: {},
          nodes: [],
          edges: [],
          outputs: [],
          metadata: { name: "demo", description: "" },
        }),
      ),
    ).toBe(true);
    expect(
      hasExistingCompositionContent(
        makeComposition(1, {
          sources: {},
          nodes: [],
          edges: [],
          outputs: [],
          metadata: { name: "Untitled Pipeline", description: "Custom notes" },
        }),
      ),
    ).toBe(true);
  });

  it("allows an unknown state only for proposals against a null base state", () => {
    expect(hasSafeInlineSourceDisambiguationBase(proposal, null)).toBe(true);
    expect(
      hasSafeInlineSourceDisambiguationBase(
        { ...proposal, base_state_id: "existing-state" },
        null,
      ),
    ).toBe(false);
  });

  it("denies any proposal when the known composition has replaceable content", () => {
    expect(
      hasSafeInlineSourceDisambiguationBase(proposal, makeComposition(1)),
    ).toBe(false);
  });
});

// ── findOriginatingMessageId unit tests ───────────────────────────────────────
describe("findOriginatingMessageId", () => {
  const userMessage: ChatMessage = {
    id: "user-1",
    session_id: "s",
    role: "user",
    content: "check these URLs: a, b, c",
    tool_calls: null,
    created_at: "2026-05-18T10:00:00Z",
  };
  const assistantMessage: ChatMessage = {
    id: "asst-1",
    session_id: "s",
    role: "assistant",
    content: "I'll add those.",
    tool_calls: [
      { id: "tc-1", type: "function", function: { name: "set_pipeline", arguments: "{}" } },
    ],
    created_at: "2026-05-18T10:00:01Z",
  };

  it("returns the immediately-preceding user message id", () => {
    expect(
      findOriginatingMessageId([userMessage, assistantMessage], "tc-1"),
    ).toBe("user-1");
  });

  it("returns null when no message carries that tool_call_id", () => {
    expect(
      findOriginatingMessageId([userMessage, assistantMessage], "tc-other"),
    ).toBeNull();
  });

  it("returns null when no user message precedes the assistant turn", () => {
    expect(findOriginatingMessageId([assistantMessage], "tc-1")).toBeNull();
  });
});

// ── parseProposedRowsFromUserInput unit tests ─────────────────────────────────
describe("parseProposedRowsFromUserInput", () => {
  it("splits a colon-led list on commas", () => {
    expect(
      parseProposedRowsFromUserInput("check these URLs: a.com, b.com, c.com"),
    ).toEqual(["a.com", "b.com", "c.com"]);
  });

  it("splits on newlines", () => {
    expect(parseProposedRowsFromUserInput("rows:\na\nb\nc")).toEqual([
      "a",
      "b",
      "c",
    ]);
  });

  it("returns the whole input as one row when no delimiter is present", () => {
    expect(parseProposedRowsFromUserInput("just one line")).toEqual([
      "just one line",
    ]);
  });
});

// ── ChatPanel disambiguation wiring tests (Phase 5a Task 4) ───────────────────
//
// Verifies the routing layer in ChatPanel that decides whether a pending
// inline-blob proposal surfaces via the disambiguation widget or via the
// standard PendingProposalsBanner. The widget itself is tested in
// InlineSourceDisambiguationTurn.test.tsx; here we only check the
// predicate-and-guard plumbing.
describe("ChatPanel inline-source disambiguation routing", () => {
  const sessionFixture: Session = {
    id: "session-disamb",
    title: "Disambiguation session",
    created_at: "2026-05-18T10:00:00Z",
    updated_at: "2026-05-18T10:00:00Z",
  };

  function makeAmbiguousProposalAndMessages(): {
    proposal: CompositionProposal;
    userMessage: ChatMessage;
    assistantMessage: ChatMessage;
  } {
    const userMessage: ChatMessage = {
      id: "user-disamb-1",
      session_id: sessionFixture.id,
      role: "user",
      content: "check these URLs: a.com, b.com, c.com",
      tool_calls: null,
      created_at: "2026-05-18T10:00:00Z",
    };
    const assistantMessage: ChatMessage = {
      id: "asst-disamb-1",
      session_id: sessionFixture.id,
      role: "assistant",
      content: "I'll add those.",
      tool_calls: [
        {
          id: "tc-disamb-1",
          type: "function",
          function: { name: "set_pipeline", arguments: "{}" },
        },
      ],
      created_at: "2026-05-18T10:00:01Z",
    };
    const proposal: CompositionProposal = {
      id: "prop-disamb-1",
      session_id: sessionFixture.id,
      tool_call_id: "tc-disamb-1",
      tool_name: "set_pipeline",
      status: "pending",
      summary: "I read your message as 3 separate URLs.",
      rationale: "",
      affects: ["source"],
      arguments_redacted_json: {
        source: {
          plugin: "csv",
          on_success: "csv_rows",
          blob_id: null,
          options: "{}",
          on_validation_failure: null,
          inline_blob: {
            filename: "chat.csv",
            mime_type: "text/csv",
            content: "<inline-blob:42-bytes>",
            description: null,
          },
        },
        sources: null,
        nodes: [],
        edges: [],
        outputs: [],
        metadata: null,
      },
      base_state_id: null,
      committed_state_id: null,
      audit_event_id: null,
      created_at: "2026-05-18T10:00:00Z",
      updated_at: "2026-05-18T10:00:00Z",
    };
    return { proposal, userMessage, assistantMessage };
  }

  beforeEach(() => {
    vi.resetAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    resetStore(useInlineSourceStore);
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: null,
    });
  });

  it("renders the disambiguation widget for an ambiguous inline-blob proposal", () => {
    const { proposal, userMessage, assistantMessage } =
      makeAmbiguousProposalAndMessages();
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [userMessage, assistantMessage],
      compositionProposals: [proposal],
    });

    render(<ChatPanel />);

    expect(
      screen.getByRole("region", { name: /row count/i }),
    ).toBeInTheDocument();
  });

  it("excludes the ambiguous proposal from the standard PendingProposalsBanner", () => {
    const { proposal, userMessage, assistantMessage } =
      makeAmbiguousProposalAndMessages();
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [userMessage, assistantMessage],
      compositionProposals: [proposal],
    });

    render(<ChatPanel />);

    // The PendingProposalsBanner uses aria-label="Pending changes (N)".
    // When the only pending proposal is claimed by the disambiguation
    // widget the banner has zero actionable items and returns null.
    expect(
      screen.queryByRole("region", { name: /pending changes/i }),
    ).toBeNull();
  });

  it("skips the widget when the originating message is in userRequestedSingleRowForMessageIds (F-11)", () => {
    const { proposal, userMessage, assistantMessage } =
      makeAmbiguousProposalAndMessages();
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [userMessage, assistantMessage],
      compositionProposals: [proposal],
    });
    // Seed the F-11 guard BEFORE render.
    act(() => {
      useInlineSourceStore
        .getState()
        .addUserRequestedSingleRow(userMessage.id);
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /row count/i }),
    ).toBeNull();
    // Falls back to the standard banner.
    expect(
      screen.getByRole("region", { name: /pending changes/i }),
    ).toBeInTheDocument();
  });

  it("skips the widget when the originating message is in nonSourceMessageIds (F-10)", () => {
    const { proposal, userMessage, assistantMessage } =
      makeAmbiguousProposalAndMessages();
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [userMessage, assistantMessage],
      compositionProposals: [proposal],
    });
    act(() => {
      useInlineSourceStore.getState().addNonSourceMessage(userMessage.id);
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /row count/i }),
    ).toBeNull();
    expect(
      screen.getByRole("region", { name: /pending changes/i }),
    ).toBeInTheDocument();
  });

  it("does NOT render the widget for a non-ambiguous proposal (canonical demo)", () => {
    const { proposal, userMessage, assistantMessage } =
      makeAmbiguousProposalAndMessages();
    // Rewrite the summary to the canonical demo narration; the
    // heuristic should classify this as non-ambiguous.
    const demoProposal: CompositionProposal = {
      ...proposal,
      summary:
        "Created an inline source with 5 Australian government web pages " +
          "and wired an LLM transform to rate each on coolness.",
    };
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [userMessage, assistantMessage],
      compositionProposals: [demoProposal],
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /row count/i }),
    ).toBeNull();
    // Falls back to the standard banner.
    expect(
      screen.getByRole("region", { name: /pending changes/i }),
    ).toBeInTheDocument();
  });

  it("routes inline_blob set_pipeline proposals with hidden outputs to the standard banner", () => {
    const { proposal, userMessage, assistantMessage } =
      makeAmbiguousProposalAndMessages();
    const fullPipelineProposal: CompositionProposal = {
      ...proposal,
      arguments_redacted_json: {
        ...proposal.arguments_redacted_json,
        outputs: [
          {
            sink_name: "unexpected",
            plugin: "jsonl",
            options: { path: "<redacted-option-value>" },
          },
        ],
      },
    };
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [userMessage, assistantMessage],
      compositionProposals: [fullPipelineProposal],
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /row count/i }),
    ).toBeNull();
    expect(
      screen.getByRole("region", { name: /pending changes/i }),
    ).toBeInTheDocument();
  });

  it("routes ambiguous inline_blob proposals to the banner when accepting would replace existing state", () => {
    const { proposal, userMessage, assistantMessage } =
      makeAmbiguousProposalAndMessages();
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [userMessage, assistantMessage],
      compositionProposals: [proposal],
      compositionState: makeComposition(1),
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /row count/i }),
    ).toBeNull();
    expect(
      screen.getByRole("region", { name: /pending changes/i }),
    ).toBeInTheDocument();
  });
});

// ── looksLikeData unit tests (Phase 5a Task 5) ────────────────────────────────
//
// Source-shaped-text detector. The fallback prompt fires only when the
// predicate is true; false positives produce a disruptive affordance on
// every conversational turn, so the predicate is intentionally narrow.
// CLOSED LIST tests pin the two recognised shapes (URL, comma-separated
// list 2..10 tokens). Spec clause 3 ("short typed phrase under 200 chars
// containing no ?") was OMITTED — see the comment on `looksLikeData` in
// ChatPanel.tsx for the rationale.
describe("looksLikeData", () => {
  it("returns true for a single http(s) URL", () => {
    expect(looksLikeData("https://example.com")).toBe(true);
    expect(looksLikeData("http://example.com/path")).toBe(true);
  });

  it("returns true for a URL embedded in prose", () => {
    expect(looksLikeData("check this https://example.com")).toBe(true);
  });

  it("returns true for a comma-separated list of 2..10 items", () => {
    expect(looksLikeData("alice, bob, carol")).toBe(true);
    expect(looksLikeData("a, b")).toBe(true);
    expect(looksLikeData("one, two, three, four, five, six, seven, eight, nine, ten")).toBe(true);
  });

  it("returns false for an empty string", () => {
    expect(looksLikeData("")).toBe(false);
    expect(looksLikeData("   ")).toBe(false);
  });

  it("returns false for prose with one or two embedded commas (anchored regex)", () => {
    // Anchored regex requires the entire trimmed content to consist of
    // comma-separated tokens. A sentence like "hello, world how are you"
    // contains a comma but is not a list.
    expect(looksLikeData("hello, world how are you doing today")).toBe(false);
  });

  it("returns false for a typical question turn", () => {
    expect(looksLikeData("what's the best way to do this?")).toBe(false);
    expect(looksLikeData("how do I add a transform?")).toBe(false);
  });

  it("returns false for a casual short message (no URL, no list)", () => {
    // Clause 3 from the spec ("short typed phrase under 200 chars
    // without ?") was deliberately omitted — see the looksLikeData
    // comment in ChatPanel.tsx. This test pins that omission so a
    // future "spec-tightening" pull request doesn't quietly re-add
    // the over-broad clause and dominate the predicate.
    expect(looksLikeData("ok")).toBe(false);
    expect(looksLikeData("Yes please go ahead")).toBe(false);
  });

  it("returns false for a comma-separated list with > 10 items", () => {
    // Eleven items — the {1,9} repeater caps at 10 total tokens.
    expect(
      looksLikeData("a, b, c, d, e, f, g, h, i, j, k"),
    ).toBe(false);
  });
});

// ── ChatPanel inline-source fallback wiring (Phase 5a Task 5) ─────────────────
//
// Integration of the LLM-skip safety-net prompt: ChatPanel computes the
// predicate (looksLikeData + source-not-bound + no inflight tool-call +
// not-dismissed) and renders <InlineSourceFallbackPrompt> above the chat
// input when all four hold. Predicate components are unit-tested directly
// above; here we check the wiring — predicate-true => render, each
// suppressor => no-render, accept => natural-language sendMessage,
// dismiss => markDismissed.
describe("ChatPanel inline-source fallback prompt", () => {
  const sessionFixture: Session = {
    id: "session-fallback",
    title: "Fallback session",
    created_at: "2026-05-18T10:00:00Z",
    updated_at: "2026-05-18T10:00:00Z",
  };

  function makeUserMessage(content: string, idSuffix = "1"): ChatMessage {
    return {
      id: `user-fallback-${idSuffix}`,
      session_id: sessionFixture.id,
      role: "user",
      content,
      tool_calls: null,
      created_at: "2026-05-18T10:00:00Z",
    };
  }

  function makeAssistantWithToolCall(name: string): ChatMessage {
    return {
      id: "asst-fallback-1",
      session_id: sessionFixture.id,
      role: "assistant",
      content: "Working on it.",
      tool_calls: [
        {
          id: "tc-fallback-1",
          type: "function",
          function: { name, arguments: "{}" },
        },
      ],
      created_at: "2026-05-18T10:00:02Z",
    };
  }

  beforeEach(() => {
    vi.resetAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    resetStore(useInlineSourceStore);
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: null,
    });
  });

  it("renders the fallback prompt when a recent user message looks like a URL and no source is bound", () => {
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [makeUserMessage("https://example.com")],
    });

    render(<ChatPanel />);

    expect(
      screen.getByRole("region", { name: /inline source fallback prompt/i }),
    ).toBeInTheDocument();
  });

  it("does NOT render while the composer is still responding to the user message", () => {
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: true,
      compositionState: null,
      error: null,
    });
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [makeUserMessage("https://example.com")],
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /inline source fallback prompt/i }),
    ).toBeNull();
  });

  it("does NOT render the prompt when there are no user messages", () => {
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [],
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /inline source fallback prompt/i }),
    ).toBeNull();
  });

  it("does NOT render when the latest user message does not look like data", () => {
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [makeUserMessage("how do I create a pipeline?")],
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /inline source fallback prompt/i }),
    ).toBeNull();
  });

  it("does NOT render when the composition already has a source", () => {
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [makeUserMessage("https://example.com")],
      // makeComposition's default source is csv_file (plugin !== ""),
      // which is exactly the source-bound state we want to suppress on.
      compositionState: makeComposition(1),
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /inline source fallback prompt/i }),
    ).toBeNull();
  });

  it("does NOT render when an in-flight source-related tool call is present", () => {
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [
        makeUserMessage("https://example.com"),
        makeAssistantWithToolCall("set_pipeline"),
      ],
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /inline source fallback prompt/i }),
    ).toBeNull();
  });

  it("does NOT render after the user dismisses (F-20 session-scoped)", () => {
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [makeUserMessage("https://example.com")],
    });
    act(() => {
      useInlineSourceStore.getState().markDismissed(sessionFixture.id);
    });

    render(<ChatPanel />);

    expect(
      screen.queryByRole("region", { name: /inline source fallback prompt/i }),
    ).toBeNull();
  });

  it("accept dispatches a natural-language chat turn (F-3: no API jargon)", () => {
    const sendMessage = vi.fn();
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage,
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: null,
    });
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [makeUserMessage("https://example.com")],
    });

    render(<ChatPanel />);

    fireEvent.click(
      screen.getByRole("button", { name: /create source/i }),
    );

    expect(sendMessage).toHaveBeenCalledTimes(1);
    const dispatched = sendMessage.mock.calls[0][0] as string;
    // F-3 — natural-language framing. Must NOT contain API tokens
    // ("set_pipeline", "inline_blob", "tool_call"); MUST embed the
    // candidate text verbatim.
    expect(dispatched).toMatch(/use this as my source data/i);
    expect(dispatched).toContain("https://example.com");
    expect(dispatched).not.toMatch(/set_pipeline|inline_blob|tool_call/i);
  });

  it("dismiss calls markDismissed on inlineSourceStore for the active session", () => {
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [makeUserMessage("https://example.com")],
    });

    render(<ChatPanel />);

    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));

    expect(
      useInlineSourceStore.getState().isDismissed(sessionFixture.id),
    ).toBe(true);
  });
});

// ── Interpretation review inline-message dispatch (Phase 5b Task 5) ──────────
//
// These tests cover ChatPanel's freeform-mode rendering of pending
// interpretation events via the InterpretationReviewInlineMessage widget.
// The widget itself is tested in InterpretationReviewInlineMessage.test.tsx;
// here we assert the dispatch predicate:
//
//   * Freeform mode + pending event in pendingBySession → render one
//     inline message per event, in created_at-ascending order.
//   * Guided mode → do NOT render the inline message (the guided turn
//     surface handles interpretation review via InterpretationReviewTurn).
//   * Opt-out clears pendingBySession locally → no inline messages even
//     if backend rows remain for audit.
//   * Negative-case routing predicate: an `inline_blob` proposal whose
//     summary does NOT contain "I read" / "interpreted as" routes to
//     the standard InlineSourceCreatedTurn, NOT to this widget.  This
//     pins the discriminator between the two surfaces so a future
//     change to either does not silently widen the interpretation-
//     review surface.

describe("ChatPanel interpretation-review inline-message dispatch", () => {
  const sessionFixture: Session = {
    id: "session-interp",
    title: "Interp session",
    created_at: "2026-05-18T10:00:00Z",
    updated_at: "2026-05-18T10:00:00Z",
  };

  function makeInterpretationEvent(
    overrides: Partial<InterpretationEvent> = {},
  ): InterpretationEvent {
    return {
      id: "evt-a",
      session_id: "session-interp",
      composition_state_id: "state-1",
      affected_node_id: "node-1",
      tool_call_id: "tool-1",
      user_term: "cool",
      kind: "vague_term",
      llm_draft: "trendy",
      accepted_value: null,
      choice: "pending",
      created_at: "2026-05-18T10:00:01Z",
      resolved_at: null,
      actor: "user:owner:u-1",
      interpretation_source: "user_approved",
      model_identifier: "anthropic/claude-opus-4-7",
      model_version: "20260518",
      provider: "anthropic",
      composer_skill_hash: "deadbeef",
      arguments_hash: null,
      hash_domain_version: null,
      runtime_model_identifier_at_resolve: null,
      runtime_model_version_at_resolve: null,
      resolved_prompt_template_hash: null,
      ...overrides,
    };
  }

  beforeEach(() => {
    vi.resetAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    resetStore(useInlineSourceStore);
    resetStore(useInterpretationEventsStore);
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      isComposing: false,
      compositionState: null,
      error: null,
    });
  });

  // Test 13: freeform mode + pending event → inline message rendered.
  it("renders an inline interpretation message in freeform mode when a pending event exists", () => {
    const event = makeInterpretationEvent();
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [],
    });
    act(() => {
      useInterpretationEventsStore
        .getState()
        .addPendingEvent(sessionFixture.id, event);
    });

    render(<ChatPanel />);

    expect(
      screen.getByTestId("acknowledgement-card"),
    ).toBeInTheDocument();
  });

  // Test 14: guided mode also unifies on the AcknowledgementStack — the same
  // card renders (one surface for both modes; they can no longer drift).
  it("renders the acknowledgement card in guided mode too (unified surface)", () => {
    const event = makeInterpretationEvent();
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [],
      guidedSession: {
        step: "step_1_source",
        history: [],
        terminal: null,
        chat_history: [],
        chat_turn_seq: 0,
        profile: null,
      },
      guidedNextTurn: {
        type: "single_select",
        step_index: 0,
        payload: {
          question: "Pick one",
          options: [{ id: "a", label: "A", hint: null }],
          allow_custom: false,
        },
      },
    });
    act(() => {
      useInterpretationEventsStore
        .getState()
        .addPendingEvent(sessionFixture.id, event);
    });

    render(<ChatPanel />);

    // The guided branch mounts the same AcknowledgementStack at the top of its
    // reply surface, so the card IS present in guided mode.
    expect(screen.getByTestId("acknowledgement-card")).toBeInTheDocument();
  });

  // Test 15: two pending events → two inline messages in created_at-ascending order.
  it("renders two inline messages in created_at-ascending order when two pending events exist", () => {
    // Seed in reverse-chronological order to ensure the component sorts
    // them (not just renders them in insertion order).
    const eventLater = makeInterpretationEvent({
      id: "evt-later",
      user_term: "later-term",
      created_at: "2026-05-18T11:00:00Z",
    });
    const eventEarlier = makeInterpretationEvent({
      id: "evt-earlier",
      user_term: "earlier-term",
      created_at: "2026-05-18T10:00:00Z",
    });
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [],
    });
    act(() => {
      useInterpretationEventsStore
        .getState()
        .addPendingEvent(sessionFixture.id, eventLater);
      useInterpretationEventsStore
        .getState()
        .addPendingEvent(sessionFixture.id, eventEarlier);
    });

    render(<ChatPanel />);

    const widgets = screen.getAllByTestId(
      "acknowledgement-card",
    );
    expect(widgets).toHaveLength(2);
    // The earlier-created event renders first (top-of-list).  Match by the
    // user_term text inside each widget so the assertion does not depend on
    // event-id ordering, which would be a fragile proxy.
    expect(widgets[0].textContent).toMatch(/earlier-term/);
    expect(widgets[1].textContent).toMatch(/later-term/);
  });

  // Test 16: after opt-out the pending map is cleared → no inline messages.
  it("renders no inline messages after opt-out clears the pending map locally", async () => {
    const event = makeInterpretationEvent();
    // Mock the opt-out API call so the store action completes
    // synchronously-as-far-as-the-store-is-concerned.
    const optOutSpy = vi
      .spyOn(apiClient, "optOutOfInterpretations")
      .mockResolvedValue({
        session_id: sessionFixture.id,
        interpretation_review_disabled: true,
        opted_out_at: "2026-05-18T12:00:00Z",
      });
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [],
    });
    act(() => {
      useInterpretationEventsStore
        .getState()
        .addPendingEvent(sessionFixture.id, event);
    });

    const { rerender } = render(<ChatPanel />);
    expect(
      screen.getByTestId("acknowledgement-card"),
    ).toBeInTheDocument();

    // Drive the opt-out via the store action — same surface the widget's
    // "Stop reviewing" confirm modal calls into.  The store clears
    // pendingBySession[sessionId] on success.
    await act(async () => {
      await useInterpretationEventsStore.getState().optOut(sessionFixture.id);
    });
    rerender(<ChatPanel />);

    expect(optOutSpy).toHaveBeenCalledWith(sessionFixture.id);
    expect(
      screen.queryByTestId("acknowledgement-card"),
    ).not.toBeInTheDocument();
  });

  // Test 17: negative-case routing predicate. An inline_blob proposal whose
  // summary contains neither "I read" nor "interpreted as" routes to the
  // standard InlineSourceCreatedTurn, NOT to this widget.  This pins the
  // discriminator between the two surfaces: the interpretation-review
  // widget keys off pendingInterpretationEvents (which is empty here),
  // and the InlineSourceCreatedTurn keys off inlineSourceSummary (which
  // we seed via the blob projection).
  it("an inline_blob proposal with no interpretation-context summary routes to InlineSourceCreatedTurn, not the interpretation-review inline message", async () => {
    (apiClient.getBlobMetadata as ReturnType<typeof vi.fn>).mockResolvedValue({
      id: "blob-routing-1",
      session_id: sessionFixture.id,
      filename: "rows.csv",
      mime_type: "text/csv",
      size_bytes: 32,
      content_hash:
        "bb34d52cc97aefb5ce4513edda086520863c513bd8f3bd9165404000347d1081",
      created_at: "2026-05-18T10:00:01Z",
      created_by: "assistant",
      source_description: null,
      status: "ready",
      // Provenance is llm_generated (not interpretation-related).  The
      // resulting summary in the InlineSourceCreatedTurn body reads
      // "Created a 5-row source from your input" — i.e., it does NOT
      // contain "I read" or "interpreted as".
      creation_modality: "llm_generated",
      created_from_message_id: "msg-1",
      creating_model_identifier: "claude-opus-4-7",
      creating_model_version: "20260101",
      creating_provider: "anthropic",
      creating_composer_skill_hash: "skill-hash",
      creating_arguments_hash: "args-hash",
    });
    (
      apiClient.previewBlobContent as ReturnType<typeof vi.fn>
    ).mockResolvedValue("a\nb\nc\nd\ne\nf");

    const composition = makeComposition(1, {
      sources: {
        source: {
          plugin: "inline_blob",
          options: { blob_ref: "blob-routing-1" },
        },
      },
    });
    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [],
      compositionState: composition,
    });
    // NO pending interpretation event seeded.  An inline_blob proposal
    // without interpretation context does NOT produce a pending
    // interpretation event on the wire, so pendingBySession is empty.

    render(<ChatPanel />);

    await waitFor(() => {
      expect(
        screen.getByTestId("inline-source-created-turn"),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByTestId("acknowledgement-card"),
    ).not.toBeInTheDocument();
  });

  // ── Phase 5b.18b.8 resolve-success confirmation copy ──────────────────────
  //
  // After the inline-review widget resolves (Use mine / Submit amend), the
  // chat shows a short assistant-styled confirmation line so the user has
  // a closure cue. The widget unmounts on resolve (pendingBySession clears
  // the event); the confirmation lives in ChatPanel-local state captured
  // via onResolved BEFORE unmount.
  //
  // Spec lines 768-774: "Got it — using your interpretation of *<user_term>*."
  // — pure UI nudge, NOT persisted to the audit trail (which already
  // recorded the resolved interpretation_event row).
  it("renders a resolve-success confirmation line after the user resolves an interpretation (Phase 5b.18b.8)", async () => {
    const event = makeInterpretationEvent({ user_term: "cool" });
    // Stub the resolve API so the store's resolveEvent action completes.
    const resolveSpy = vi
      .spyOn(apiClient, "resolveInterpretation")
      .mockResolvedValue({
        event: { ...event, choice: "accepted_as_drafted" },
        new_state: makeComposition(2),
      });

    useSessionStore.setState({
      activeSessionId: sessionFixture.id,
      sessions: [sessionFixture],
      messages: [],
    });
    act(() => {
      useInterpretationEventsStore
        .getState()
        .addPendingEvent(sessionFixture.id, event);
    });

    render(<ChatPanel />);

    // Initially the inline widget is mounted; no confirmation yet.
    expect(
      screen.getByTestId("acknowledgement-card"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("interpretation-review-confirmation"),
    ).not.toBeInTheDocument();

    // Click "Use my interpretation" — the widget calls the store's
    // resolveEvent, which calls api.resolveInterpretation, which our
    // spy resolves; on resolution the widget fires its onResolved
    // callback, ChatPanel records the confirmation, and the widget
    // unmounts (its event was removed from pendingBySession).
    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", {
          name: /Acknowledge the LLM's interpretation/i,
        }),
      );
    });

    // Confirmation copy is now visible; widget is gone.
    const confirmation = await screen.findByTestId(
      "interpretation-review-confirmation",
    );
    expect(confirmation.textContent).toMatch(
      /Got it — using your interpretation of/i,
    );
    expect(confirmation.textContent).toMatch(/cool/);
    expect(
      screen.queryByTestId("acknowledgement-card"),
    ).not.toBeInTheDocument();
    expect(resolveSpy).toHaveBeenCalledWith(
      sessionFixture.id,
      event.id,
      { choice: "accepted_as_drafted" },
    );
  });
});

describe("ChatPanel chat presentation (ux-review-2026-07-02)", () => {
  const session: Session = {
    id: "session-pres",
    title: "Presentation session",
    created_at: "2026-07-02T10:00:00Z",
    updated_at: "2026-07-02T10:00:00Z",
  };
  const userMessage: ChatMessage = {
    id: "message-pres-1",
    session_id: session.id,
    role: "user",
    content: "Build me a pipeline",
    tool_calls: null,
    created_at: "2026-07-02T10:00:01Z",
  };

  beforeEach(() => {
    vi.resetAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    (useComposer as ReturnType<typeof vi.fn>).mockReturnValue({
      sendMessage: vi.fn(),
      retryMessage: vi.fn(),
      cancelComposition: vi.fn(),
      isComposing: true,
      compositionState: null,
      error: null,
    });
    useSessionStore.setState({
      activeSessionId: session.id,
      sessions: [session],
      messages: [userMessage],
    });
  });

  it("makes the conversation scroll region keyboard-focusable with an accessible name (elspeth-5e43a0c8b2)", () => {
    const { container } = render(<ChatPanel />);

    const log = container.querySelector<HTMLElement>(".chat-panel-messages");
    expect(log).not.toBeNull();
    // Keyboard users must be able to focus the container to arrow-scroll it.
    expect(log?.getAttribute("tabindex")).toBe("0");
    expect(log?.getAttribute("aria-label")).toBe("Conversation");
    // The live-region semantics stay intact alongside focusability.
    expect(log?.getAttribute("role")).toBe("log");
    expect(log?.getAttribute("aria-live")).toBe("polite");
    expect(log?.getAttribute("aria-relevant")).toBe("additions");
  });

  it("mounts the composing indicator OUTSIDE the role=log live region (elspeth-76a0cc485e)", () => {
    // Default beforeEach useComposer mock has isComposing: true, so the
    // indicator is painted.
    const { container } = render(<ChatPanel />);

    const log = container.querySelector<HTMLElement>(".chat-panel-messages");
    const indicator = container.querySelector<HTMLElement>(".composing-indicator");
    expect(log).not.toBeNull();
    expect(indicator).not.toBeNull();
    // Structural fix for the nested-live-region finding: the indicator's
    // role="status" must be a SIBLING of the log container, never nested
    // inside it where both live regions could announce the same change.
    expect(log?.contains(indicator)).toBe(false);
    expect(indicator?.getAttribute("role")).toBe("status");
  });

  it("shows the composer model chip in the chat header (elspeth-e9f7678de8)", async () => {
    vi.mocked(apiClient.fetchSystemStatus).mockResolvedValue({
      composer_available: true,
      composer_model: "anthropic/claude-sonnet-4.6",
      composer_provider: "openrouter",
      composer_reason: null,
      composer_missing_keys: [],
    });

    const { container } = render(<ChatPanel />);

    await waitFor(() => {
      expect(
        screen.getByLabelText("Composer model: anthropic/claude-sonnet-4.6"),
      ).toBeInTheDocument();
    });
    // The chip lives in the header chrome, not in the message stream.
    const header = container.querySelector(".chat-panel-header");
    expect(
      header?.querySelector(".chat-model-chip"),
    ).not.toBeNull();
  });
});
