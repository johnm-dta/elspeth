// ============================================================================
// ChatInput — listener-stabilisation regression coverage.
//
// Pins the contract that the PREFILL_CHAT_INPUT_EVENT listener must (a) be
// registered exactly once for the lifetime of the component (not re-registered
// on every parent re-render in controlled mode), and (b) always resolve to the
// latest setText / onChange handler — i.e. the ref-trampoline pattern at
// ChatInput.tsx:51-52 is load-bearing.  A behavioural test that only fired one
// event would still pass if a future refactor reverted to closing over setText
// directly; this test fires the event AFTER a parent re-render to catch that.
// ============================================================================

import { useRef, useState, type RefObject } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatInput } from "./ChatInput";
import { useSessionStore } from "@/stores/sessionStore";
import { useBlobStore } from "@/stores/blobStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import { PREFILL_CHAT_INPUT_EVENT } from "@/components/catalog/PluginCard";
import type { ChatMessage, CompositionState } from "@/types";
import type { InterpretationEvent } from "@/types/interpretation";

describe("ChatInput — controlled-mode prefill listener", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useBlobStore);
    resetStore(useInterpretationEventsStore);
  });

  function ControlledHarness() {
    const [value, setValue] = useState("");
    const [renderTick, setRenderTick] = useState(0);
    const inputRef = useRef<HTMLTextAreaElement>(
      null,
    ) as RefObject<HTMLTextAreaElement>;
    // CRITICAL: onChange closes over `renderTick`.  This is the discriminator
    // that turns the ref-trampoline test from a tautology into a real test.
    // - With the trampoline, setTextRef.current points at the LATEST onChange,
    //   which captures the LATEST renderTick.  After 3 rerenders, prefill
    //   writes "${detail}:3".
    // - Without the trampoline, the listener closes over the FIRST onChange,
    //   which captures renderTick=0.  Prefill writes "${detail}:0".
    // The suffix asymmetry is what proves the trampoline is load-bearing.
    return (
      <div>
        <button
          type="button"
          data-testid="force-rerender"
          onClick={() => setRenderTick((n) => n + 1)}
        >
          rerender {renderTick}
        </button>
        <ChatInput
          onSend={vi.fn()}
          disabled={false}
          inputRef={inputRef}
          value={value}
          onChange={(next) => setValue(`${next}:${renderTick}`)}
        />
      </div>
    );
  }

  it("populates the textarea when PREFILL_CHAT_INPUT_EVENT fires", async () => {
    render(<ControlledHarness />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.value).toBe("");

    act(() => {
      window.dispatchEvent(
        new CustomEvent(PREFILL_CHAT_INPUT_EVENT, {
          detail: "Add csv as the source",
        }),
      );
    });

    // renderTick=0 at this point — harness writes "${detail}:0".
    expect(textarea.value).toBe("Add csv as the source:0");
  });

  it("uses the LATEST setText after parent re-renders (ref-trampoline must be load-bearing)", async () => {
    // Regression: a previous bug closed over setText directly in the effect.
    // In controlled mode, setText identity changes on every parent render.
    // Without the ref trampoline (ChatInput.tsx:51-52), the listener would
    // hold a stale closure pointing to the FIRST onChange — which captures
    // renderTick=0.  After 3 rerenders, prefill should write "${detail}:3"
    // (latest tick) if the trampoline works.  If the listener has stale
    // closure, it writes "${detail}:0" and this test fails.
    const user = userEvent.setup();
    render(<ControlledHarness />);

    await user.click(screen.getByTestId("force-rerender"));
    await user.click(screen.getByTestId("force-rerender"));
    await user.click(screen.getByTestId("force-rerender"));

    act(() => {
      window.dispatchEvent(
        new CustomEvent(PREFILL_CHAT_INPUT_EVENT, { detail: "after rerenders" }),
      );
    });

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    // With the trampoline: latest onChange ran, captured renderTick=3.
    // Without the trampoline: stale onChange ran, captured renderTick=0.
    expect(textarea.value).toBe("after rerenders:3");
  });

  it("registers the listener exactly once across re-renders", async () => {
    // Verify the [] dep array on the prefill effect: addEventListener must
    // not fire on every render.  We spy on window.addEventListener and count
    // PREFILL_CHAT_INPUT_EVENT registrations.
    const addSpy = vi.spyOn(window, "addEventListener");
    const user = userEvent.setup();
    render(<ControlledHarness />);

    const initial = addSpy.mock.calls.filter(
      ([type]) => type === PREFILL_CHAT_INPUT_EVENT,
    ).length;
    expect(initial).toBe(1);

    await user.click(screen.getByTestId("force-rerender"));
    await user.click(screen.getByTestId("force-rerender"));

    const after = addSpy.mock.calls.filter(
      ([type]) => type === PREFILL_CHAT_INPUT_EVENT,
    ).length;
    expect(after).toBe(1);

    addSpy.mockRestore();
  });

  it("throws TypeError on non-string event detail (CLAUDE.md trust-tier: internal contract violations crash)", () => {
    // WHATWG DOM spec: event listener errors do NOT propagate through
    // dispatchEvent — the caller continues; the error is reported via
    // window.onerror / 'error' event.  Capture that report to prove the
    // crash actually fires and is loud (DevTools-visible) rather than
    // silently caught.
    const errorEvents: ErrorEvent[] = [];
    const errorListener = (e: ErrorEvent) => {
      errorEvents.push(e);
      e.preventDefault(); // suppress jsdom's "unhandled error" stderr noise
    };
    window.addEventListener("error", errorListener);

    render(<ControlledHarness />);

    // Dispatch the malformed event — listener throws synchronously.
    window.dispatchEvent(
      new CustomEvent(PREFILL_CHAT_INPUT_EVENT, {
        detail: { not: "a string" } as unknown as string,
      }),
    );

    // The TypeError must have been reported as an unhandled error.
    expect(errorEvents).toHaveLength(1);
    expect(errorEvents[0].error).toBeInstanceOf(TypeError);
    expect(errorEvents[0].error.message).toContain("PREFILL_CHAT_INPUT_EVENT");

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    // No silent state mutation: the bogus value must not have been written.
    expect(textarea.value).toBe("");

    window.removeEventListener("error", errorListener);
  });
});

// ============================================================================
// ChatInput — empty-state placeholder (Phase 5a Task 1).
//
// Primes the user to type data directly into the chat when the session is
// fresh (no messages, no composition state).  Reverts to the canonical
// "Describe the pipeline you want to build..." wording the moment either
// signal flips.  An explicit `placeholder` prop continues to win — Phase A
// slice 4 (guided-mode per-step nudge) depends on that override semantics.
// ============================================================================

describe("ChatInput empty-state placeholder", () => {
  const DATA_PRIMING =
    "Describe your pipeline, paste a URL, or type a few rows of data to start...";
  const STANDARD = "Describe the pipeline you want to build...";

  function StandaloneHarness(props: { placeholder?: string }) {
    const inputRef = useRef<HTMLTextAreaElement>(
      null,
    ) as RefObject<HTMLTextAreaElement>;
    return (
      <ChatInput
        onSend={vi.fn()}
        disabled={false}
        inputRef={inputRef}
        placeholder={props.placeholder}
      />
    );
  }

  function makeMessage(overrides: Partial<ChatMessage> = {}): ChatMessage {
    return {
      id: overrides.id ?? "m1",
      session_id: overrides.session_id ?? "s1",
      role: overrides.role ?? "user",
      content: overrides.content ?? "hello",
      tool_calls: overrides.tool_calls ?? null,
      created_at: overrides.created_at ?? "2026-05-18T00:00:00Z",
      ...overrides,
    };
  }

  function makeCompositionState(version: number): CompositionState {
    return {
      id: "comp-1",
      version,
      sources: {},
      nodes: [],
      edges: [],
      outputs: [],
      metadata: {} as CompositionState["metadata"],
    };
  }

  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useBlobStore);
    resetStore(useInterpretationEventsStore);
  });

  it("shows the data-priming placeholder when the session has no messages and no composition state", () => {
    // arrange: fresh store — messages=[], compositionState=null (version=0)
    render(<StandaloneHarness />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.placeholder).toBe(DATA_PRIMING);
  });

  it("reverts to the standard placeholder once the user has sent a message", () => {
    useSessionStore.setState({ messages: [makeMessage({ role: "user" })] });

    render(<StandaloneHarness />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.placeholder).toBe(STANDARD);
  });

  it("reverts to the standard placeholder once a composition state exists", () => {
    useSessionStore.setState({ compositionState: makeCompositionState(1) });

    render(<StandaloneHarness />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.placeholder).toBe(STANDARD);
  });

  it("respects an explicit `placeholder` prop override even in empty state", () => {
    // Empty state — store untouched — but the prop must still win.
    // This pins the Phase A slice 4 contract: guided-mode per-step nudges
    // override the empty-state default.
    render(<StandaloneHarness placeholder="custom" />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.placeholder).toBe("custom");
  });
});

describe("ChatInput composing cancel", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useBlobStore);
    resetStore(useInterpretationEventsStore);
  });

  it("shows a stop button while composing and calls onCancel", async () => {
    const user = userEvent.setup();
    const inputRef = { current: null } as RefObject<HTMLTextAreaElement>;
    const onCancel = vi.fn();

    render(
      <ChatInput
        onSend={vi.fn()}
        disabled={true}
        onCancel={onCancel}
        inputRef={inputRef}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Stop composing" }));

    expect(onCancel).toHaveBeenCalledOnce();
  });
});

describe("ChatInput max length", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useBlobStore);
    resetStore(useInterpretationEventsStore);
  });

  it("passes the configured maxLength to the textarea", () => {
    const inputRef = { current: null } as RefObject<HTMLTextAreaElement>;

    render(
      <ChatInput
        onSend={vi.fn()}
        disabled={false}
        inputRef={inputRef}
        maxLength={4096}
      />,
    );

    expect(screen.getByLabelText(/message input/i)).toHaveAttribute(
      "maxlength",
      "4096",
    );
  });
});

// ============================================================================
// ChatInput — pending-interpretation placeholder cue (Phase 5b Task 8).
//
// When an InterpretationReviewTurn widget is awaiting the user's decision and
// the underlying interpretation event has a non-null `user_term`, the
// chat-input placeholder briefly cues the user toward the widget above.
// Auto-baked rows (interpretation_source = auto_interpreted_*) have
// user_term=null and MUST NOT trigger the cue — they have no term to echo.
// The cue sits between the explicit `placeholder` prop (still wins) and
// the empty-state / standard placeholders (both lose to the cue when present).
// ============================================================================

describe("ChatInput pending-interpretation placeholder cue", () => {
  const EMPTY_STATE =
    "Describe your pipeline, paste a URL, or type a few rows of data to start...";
  const ACTIVE_SESSION_ID = "sess-1";

  function StandaloneHarness(props: { placeholder?: string }) {
    const inputRef = useRef<HTMLTextAreaElement>(
      null,
    ) as RefObject<HTMLTextAreaElement>;
    return (
      <ChatInput
        onSend={vi.fn()}
        disabled={false}
        inputRef={inputRef}
        placeholder={props.placeholder}
      />
    );
  }

  function makePendingEvent(
    overrides: Partial<InterpretationEvent> = {},
  ): InterpretationEvent {
    // Spread overrides last so explicit `null` (e.g. user_term=null for
    // auto-baked rows) overrides the defaults.  `??` short-circuits on
    // null/undefined and would silently drop intentional null overrides
    // — spread semantics keep them.
    const defaults: InterpretationEvent = {
      id: "evt-1",
      session_id: ACTIVE_SESSION_ID,
      composition_state_id: "state-1",
      affected_node_id: "node-1",
      tool_call_id: "tool-1",
      user_term: "cool",
      kind: "vague_term",
      llm_draft: "interesting and engaging",
      accepted_value: null,
      choice: "pending",
      created_at: "2026-05-18T00:00:00Z",
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
    };
    return { ...defaults, ...overrides };
  }

  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useBlobStore);
    resetStore(useInterpretationEventsStore);
  });

  it("shows the interpretation-review cue when a pending event with a user_term exists for the active session", () => {
    // arrange: active session + one pending event with user_term="cool"
    useSessionStore.setState({ activeSessionId: ACTIVE_SESSION_ID });
    const event = makePendingEvent({ user_term: "cool" });
    useInterpretationEventsStore.setState({
      pendingBySession: { [ACTIVE_SESSION_ID]: { [event.id]: event } },
    });

    render(<StandaloneHarness />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.placeholder).toBe(
      'Reviewing your interpretation of "cool" above — pick Use mine or Change it to continue.',
    );
  });

  it("does not show the cue when the pending event has user_term=null (auto-baked row)", () => {
    // Auto-baked rows (auto_interpreted_opt_out / no_surfaces) have no term
    // to echo.  Cue falls through to the empty-state placeholder.
    useSessionStore.setState({ activeSessionId: ACTIVE_SESSION_ID });
    const event = makePendingEvent({
      user_term: null,
      kind: null,
      interpretation_source: "auto_interpreted_opt_out",
      model_identifier: null,
      model_version: null,
      provider: null,
    });
    useInterpretationEventsStore.setState({
      pendingBySession: { [ACTIVE_SESSION_ID]: { [event.id]: event } },
    });

    render(<StandaloneHarness />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.placeholder).toBe(EMPTY_STATE);
  });

  it("does not show the cue when there is no active session", () => {
    // Pending events keyed under a different session must not leak through
    // when activeSessionId is null.
    const event = makePendingEvent({ user_term: "cool" });
    useInterpretationEventsStore.setState({
      pendingBySession: { [ACTIVE_SESSION_ID]: { [event.id]: event } },
    });

    render(<StandaloneHarness />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.placeholder).toBe(EMPTY_STATE);
  });

  it("respects an explicit `placeholder` prop override even when a pending cue would otherwise fire", () => {
    // Pins the precedence contract: explicit prop > pending-interpretation
    // cue > empty-state.  Guided-mode per-step nudges (Phase A slice 4)
    // remain authoritative even when an interpretation review is open.
    useSessionStore.setState({ activeSessionId: ACTIVE_SESSION_ID });
    const event = makePendingEvent({ user_term: "cool" });
    useInterpretationEventsStore.setState({
      pendingBySession: { [ACTIVE_SESSION_ID]: { [event.id]: event } },
    });

    render(<StandaloneHarness placeholder="step-specific nudge" />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.placeholder).toBe("step-specific nudge");
  });
});

// ============================================================================
// ChatInput — tutorial readOnly lock.
//
// The guided tutorial reuses the REAL guided flow; the only difference is the
// STEP_1 "Describe what you want" prompt is prepopulated AND locked, so the
// learner steps through the normal flow but types nothing. This pins that lock:
// the textarea shows the prepopulated value, is read-only, hides the
// source-composition affordances, and Send still submits the locked value.
// ============================================================================
describe("ChatInput — tutorial readOnly lock (prepopulated + locked prompt)", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useBlobStore);
    resetStore(useInterpretationEventsStore);
  });

  const LOCKED = "Scrape these three synthetic project-brief pages.";

  function LockedHarness({ onSend }: { onSend: (c: string) => void }) {
    const inputRef = useRef<HTMLTextAreaElement>(
      null,
    ) as RefObject<HTMLTextAreaElement>;
    return (
      <ChatInput
        onSend={onSend}
        disabled={false}
        inputRef={inputRef}
        value={LOCKED}
        onChange={() => undefined}
        readOnly
        onOpenSecrets={() => undefined}
        onToggleBlobManager={() => undefined}
      />
    );
  }

  it("prepopulates the textarea, locks it read-only, and hides source-composition affordances", () => {
    render(<LockedHarness onSend={() => undefined} />);
    const textarea = screen.getByLabelText(
      /message input/i,
    ) as HTMLTextAreaElement;
    expect(textarea.value).toBe(LOCKED);
    expect(textarea.readOnly).toBe(true);
    // The tutorial learner must not author a source by hand: the upload, file
    // manager, and secrets affordances are hidden in locked mode.
    expect(screen.queryByLabelText(/upload file/i)).toBeNull();
    expect(
      screen.queryByLabelText(/show file manager|hide file manager/i),
    ).toBeNull();
    expect(screen.queryByLabelText(/open secrets settings/i)).toBeNull();
  });

  it("Send submits the locked value (the learner presses Send, types nothing)", async () => {
    const sent: string[] = [];
    render(<LockedHarness onSend={(c) => sent.push(c)} />);
    await userEvent.click(screen.getByLabelText(/send message/i));
    expect(sent).toEqual([LOCKED]);
  });
});
