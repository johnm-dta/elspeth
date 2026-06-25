// ============================================================================
// TutorialGuidedShell.autodrive.test.tsx — p4 Task 8b
//
// Pins the tutorial PASSIVE AUTO-DRIVE wire contract: the learner specifies
// NOTHING and the shell drives the guided wizard to completion on its own.
//
// What this test pins (the WIRE CONTRACT — what the walker SENDS):
//   (a) the SOURCE phase chat seeds the FRONTEND scripted intent
//       (CANONICAL_TUTORIAL_PROMPT) + all 3 synthetic URLs from the 8a GET
//       surface — entry_seed never rides the wire;
//   (b) the SINK phase chat seeds the same canonical prompt (URLs only at the
//       source phase) — per-phase chat, mechanism (B);
//   (c) the scripted respondGuided walker advances the wizard through the form
//       confirms, the STEP_2.5 recipe accept (chosen=["accept"]), the STEP_3
//       chain accept, and the STEP_4 wire confirm — to terminal=completed;
//   (d) the D12 interpretation-review gate is resolved (accepted_as_drafted)
//       BEFORE the gated wire confirm, with no learner action.
//
// Mechanism: the guided store actions (startGuided/chatGuided/respondGuided)
// are replaced with SPIES THAT ADVANCE the real store (simulating the staged
// backend), so the REAL state-driven walker progresses and we assert the
// bodies + order it emits. Asserting the store-action body is equivalent to
// asserting the api-layer body — respondGuided/chatGuided pass the body through
// unchanged. The LLM's actual per-phase extraction is operator-verified at
// staging (P7 known-gap), NOT here.
//
// No fireEvent / userEvent is ever issued: "passive, no input affordance" holds
// because the wizard reaches completion with zero simulated learner events.
// ============================================================================

import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TutorialGuidedShell } from "./TutorialGuidedShell";
import { CANONICAL_TUTORIAL_PROMPT } from "./tutorialMachine";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import type {
  GuidedRespondRequest,
  GuidedStep,
  TurnPayload,
  TurnType,
} from "@/types/guided";
import type { InterpretationEvent } from "@/types/interpretation";

const SID = "tutorial-sess-1";
const SAMPLE_URLS = [
  "http://127.0.0.1:8000/tutorial-site/project-1.html",
  "http://127.0.0.1:8000/tutorial-site/project-2.html",
  "http://127.0.0.1:8000/tutorial-site/project-3.html",
];

const startGuidedSessionMock = vi.fn();
const getTutorialSampleMock = vi.fn();

vi.mock("@/api/client", () => ({
  startGuidedSession: (...args: unknown[]) => startGuidedSessionMock(...args),
  getTutorialSample: (...args: unknown[]) => getTutorialSampleMock(...args),
}));

// ChatPanel renders the live wizard passively; stub it so this test isolates
// the auto-drive orchestration (not ChatPanel's rendering).
vi.mock("@/components/chat/ChatPanel", () => ({
  ChatPanel: (props: { isTutorial?: boolean }) => (
    <div data-testid="chat-panel-stub" data-is-tutorial={String(props.isTutorial)} />
  ),
}));

// ── Canned staged-backend turn payloads ─────────────────────────────────────
function turn(type: TurnType, payload: unknown): TurnPayload {
  return { type, step_index: 0, payload };
}

const SOURCE_SINGLE_SELECT = turn("single_select", {
  question: "Pick a source",
  options: [],
  allow_custom: true,
});
const SOURCE_SCHEMA_FORM = turn("schema_form", {
  mode: "plugin_options",
  plugin: "json",
  knobs: { fields: [] },
  prefilled: { path: "blob://source" },
});
const SOURCE_INSPECT = turn("inspect_and_confirm", {
  observed: { columns: ["url"], samples: [], warnings: [] },
});
const SINK_SINGLE_SELECT = turn("single_select", {
  question: "Pick a sink",
  options: [],
  allow_custom: true,
});
const SINK_SCHEMA_FORM = turn("schema_form", {
  mode: "plugin_options",
  plugin: "json",
  knobs: { fields: [] },
  prefilled: { output_path: "outputs/rows.json" },
});
const SINK_MULTI_SELECT = turn("multi_select_with_custom", {
  question: "Which fields pass through?",
  options: [{ id: "url", label: "url", hint: null }],
  default_chosen: ["url"],
  escape_label: null,
});
const RECIPE_OFFER = turn("recipe_offer", {
  mode: "recipe_decision",
  knobs: { fields: [] },
  prefilled: { source_blob_id: "blob://source", output_path: "outputs/rows.json" },
  recipe_context: {
    recipe_name: "web-scrape-llm-rate-jsonl",
    description: "Scrape + LLM-extract + json",
    alternatives: [],
  },
});
const PROPOSE_CHAIN = turn("propose_chain", { steps: [], why: "", blockers: [] });
const CONFIRM_WIRING = turn("confirm_wiring", {
  topology: { sources: {}, nodes: [], outputs: [] },
  edge_contracts: [],
  semantic_contracts: [],
  warnings: [],
});

const SHIELD_EVENT_ID = "shield-evt-1";
function shieldReview(): InterpretationEvent {
  // The real always-on prompt-shield review (p3): kind pipeline_decision,
  // user_term prompt_injection_shield_recommendation — fires every run.
  return {
    id: SHIELD_EVENT_ID,
    session_id: SID,
    kind: "pipeline_decision",
    user_term: "prompt_injection_shield_recommendation",
    llm_draft: "Shield the LLM node against fetched-content prompt injection.",
    affected_node_id: "llm",
  } as unknown as InterpretationEvent;
}

function liveGuided(step: GuidedStep): unknown {
  return {
    step,
    history: [],
    terminal: null,
    chat_history: [],
    chat_turn_seq: 0,
    profile: {
      coaching: true,
      bookends: true,
      recipe_match: true,
      advisor_checkpoints: true,
    },
  };
}

function setTurn(step: GuidedStep, next: TurnPayload | null): void {
  useSessionStore.setState({
    guidedSession: liveGuided(step),
    guidedNextTurn: next,
  } as never);
}

function completeGuided(): void {
  useSessionStore.setState({
    guidedSession: {
      step: "step_4_wire",
      history: [],
      terminal: { kind: "completed", reason: null },
      chat_history: [],
      chat_turn_seq: 0,
      profile: null,
    },
    guidedNextTurn: null,
  } as never);
}

describe("TutorialGuidedShell auto-drive (passive worked example)", () => {
  let chatMessages: string[];
  let respondBodies: GuidedRespondRequest[];
  let resolveEventSpy: ReturnType<typeof vi.fn>;
  let chatGuidedSpy: ReturnType<typeof vi.fn>;
  let respondGuidedSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    chatMessages = [];
    respondBodies = [];

    startGuidedSessionMock.mockReset().mockResolvedValue(undefined);
    getTutorialSampleMock
      .mockReset()
      .mockResolvedValue({ sample_urls: SAMPLE_URLS, allowed_hosts: ["127.0.0.1/32", "::1/128"] });

    // startGuided spy: bind the active session + seed the initial STEP_1 turn.
    const startGuidedSpy = vi.fn(async () => {
      useSessionStore.setState({ activeSessionId: SID } as never);
      setTurn("step_1_source", SOURCE_SINGLE_SELECT);
    });

    // chatGuided spy: records the message + advances the SOURCE/SINK phase
    // exactly as p1's chat-apply does (authors result in place, step pointer
    // unchanged, re-renders the populated schema_form).
    chatGuidedSpy = vi.fn(async (message: string) => {
      chatMessages.push(message);
      const step = useSessionStore.getState().guidedSession?.step;
      if (step === "step_1_source") setTurn("step_1_source", SOURCE_SCHEMA_FORM);
      else if (step === "step_2_sink") setTurn("step_2_sink", SINK_SCHEMA_FORM);
    });

    // respondGuided spy: records the body + advances the staged wizard.
    respondGuidedSpy = vi.fn(async (body: GuidedRespondRequest) => {
      respondBodies.push(body);
      const step = useSessionStore.getState().guidedSession?.step;
      const turnType = useSessionStore.getState().guidedNextTurn?.type;
      if (step === "step_1_source" && turnType === "schema_form") {
        setTurn("step_1_source", SOURCE_INSPECT);
      } else if (step === "step_1_source" && turnType === "inspect_and_confirm") {
        setTurn("step_2_sink", SINK_SINGLE_SELECT);
      } else if (step === "step_2_sink" && turnType === "schema_form") {
        setTurn("step_2_sink", SINK_MULTI_SELECT);
      } else if (step === "step_2_sink" && turnType === "multi_select_with_custom") {
        setTurn("step_2_5_recipe_match", RECIPE_OFFER);
      } else if (step === "step_2_5_recipe_match" && turnType === "recipe_offer") {
        setTurn("step_3_transforms", PROPOSE_CHAIN);
      } else if (step === "step_3_transforms" && turnType === "propose_chain") {
        // The recipe inserted the llm node → the always-on shield review fires
        // and GATES the wire confirm (D12). Seed it pending now.
        useInterpretationEventsStore.setState({
          pendingBySession: { [SID]: { [SHIELD_EVENT_ID]: shieldReview() } },
        } as never);
        setTurn("step_4_wire", CONFIRM_WIRING);
      } else if (step === "step_4_wire" && turnType === "confirm_wiring") {
        completeGuided();
      }
    });

    resolveEventSpy = vi.fn(async (sessionId: string, eventId: string) => {
      // Clear the resolved event from the pending gate (mirrors the real store).
      const pending = { ...(useInterpretationEventsStore.getState().pendingBySession[sessionId] ?? {}) };
      delete pending[eventId];
      useInterpretationEventsStore.setState({
        pendingBySession: { [sessionId]: pending },
      } as never);
      return { new_state: null };
    });

    useSessionStore.setState({
      activeSessionId: null,
      messages: [],
      compositionState: null,
      compositionProposals: [],
      composerPreferences: null,
      staleProposalIds: [],
      proposalActionPendingIds: [],
      composerProgress: null,
      stateVersions: [],
      isComposing: false,
      error: null,
      selectedNodeId: null,
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
      guidedChatPending: false,
      guidedResponsePending: false,
      recoveryError: null,
      recoveryStartedCompositionVersion: null,
      startGuided: startGuidedSpy,
      chatGuided: chatGuidedSpy,
      respondGuided: respondGuidedSpy,
    } as never);

    useInterpretationEventsStore.setState({
      pendingBySession: {},
      resolveEvent: resolveEventSpy,
    } as never);
  });

  it("drives source+sink chat, the respond walker, and the D12 gate to completion with no learner input", async () => {
    const onCompleted = vi.fn();
    render(<TutorialGuidedShell sessionId={SID} onCompleted={onCompleted} />);

    // The walker drives all the way to terminal=completed on its own.
    await waitFor(() => expect(onCompleted).toHaveBeenCalledWith(SID), { timeout: 4000 });

    // (a) The shell consumed the 8a GET surface.
    expect(getTutorialSampleMock).toHaveBeenCalledWith(SID);

    // (b) Exactly two authoring chats: SOURCE (prompt + 3 URLs) then SINK (prompt).
    expect(chatGuidedSpy).toHaveBeenCalledTimes(2);
    const [sourceChat, sinkChat] = chatMessages;
    expect(sourceChat).toContain(CANONICAL_TUTORIAL_PROMPT);
    for (const url of SAMPLE_URLS) expect(sourceChat).toContain(url);
    // entry_seed never on the wire: the intent is the frontend constant only.
    expect(sourceChat.startsWith(CANONICAL_TUTORIAL_PROMPT)).toBe(true);
    // The sink chat seeds the canonical prompt alone (URLs only at the source phase).
    expect(sinkChat).toBe(CANONICAL_TUTORIAL_PROMPT);
    expect(sinkChat).not.toContain(SAMPLE_URLS[0]);

    // (c) The respondGuided walker fired the staged confirms incl. STEP_2.5 accept.
    const recipeAccept = respondBodies.find(
      (b) => Array.isArray(b.chosen) && b.chosen[0] === "accept" && b.edited_values?.recipe_name !== undefined,
    );
    expect(recipeAccept).toBeDefined();
    expect(recipeAccept?.chosen).toEqual(["accept"]);
    expect((recipeAccept?.edited_values as Record<string, unknown>).recipe_name).toBe(
      "web-scrape-llm-rate-jsonl",
    );
    expect((recipeAccept?.edited_values as Record<string, unknown>).slots).toBeDefined();

    const wireConfirm = respondBodies.find((b) => Array.isArray(b.chosen) && b.chosen[0] === "confirm");
    expect(wireConfirm).toBeDefined();

    // It is a MULTI-PHASE walker, not a single submit.
    expect(respondGuidedSpy.mock.calls.length).toBeGreaterThanOrEqual(5);

    // (d) The D12 shield gate was resolved accepted_as_drafted, BEFORE the wire confirm.
    expect(resolveEventSpy).toHaveBeenCalledWith(SID, SHIELD_EVENT_ID, {
      choice: "accepted_as_drafted",
    });
    const resolveOrder = resolveEventSpy.mock.invocationCallOrder[0];
    const wireConfirmCallIndex = respondGuidedSpy.mock.calls.findIndex(
      (call) => Array.isArray(call[0]?.chosen) && call[0].chosen[0] === "confirm",
    );
    const wireConfirmOrder = respondGuidedSpy.mock.invocationCallOrder[wireConfirmCallIndex];
    expect(resolveOrder).toBeLessThan(wireConfirmOrder);

    // Intent seeding precedes the confirms (chat before the first respond).
    expect(chatGuidedSpy.mock.invocationCallOrder[0]).toBeLessThan(
      respondGuidedSpy.mock.invocationCallOrder[0],
    );
  });
});
