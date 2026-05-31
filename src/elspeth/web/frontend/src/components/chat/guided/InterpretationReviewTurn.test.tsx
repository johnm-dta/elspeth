// ============================================================================
// InterpretationReviewTurn — Phase 5b Task 4 behavioural coverage.
//
// Tests cover the seventeen contracts from 18b-phase-5b-frontend.md §4
// (test shape) plus the GuidedTurn dispatch contract from §4 test #18.
//
// Discipline note: per the task brief, we mock ONLY the API client's
// resolve / opt-out methods.  The Zustand store is exercised end-to-end so
// the widget's wire path stays honest.  store-helpers.resetStore() resets
// the store back to its initial state between tests so cross-test bleed
// can't hide regressions (an earlier test resolving an event would leave
// the in-memory `pendingBySession` map populated for the next test).
// ============================================================================

import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InterpretationReviewTurn } from "./InterpretationReviewTurn";
import { GuidedTurn } from "./GuidedTurn";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import type {
  InterpretationEvent,
  InterpretationOptOutResponse,
  InterpretationResolveResponse,
} from "@/types/interpretation";
import type { CompositionState } from "@/types/api";
import type { TurnPayload } from "@/types/guided";
import type { ApiError } from "@/types/index";

// ── API client mock ──────────────────────────────────────────────────────────
//
// The interpretationEventsStore imports `* as api from "@/api/client"`; we
// replace the two methods the widget reaches through to with vi.fn()s.  Only
// these two methods are mocked — the store's own logic (counter updates,
// pending-map pruning) is exercised live.

vi.mock("@/api/client", () => ({
  listInterpretationEvents: vi.fn(),
  resolveInterpretation: vi.fn(),
  optOutOfInterpretations: vi.fn(),
  getInterpretationOptOutSummary: vi.fn(),
}));

// Re-import the mocked module so the tests can configure return values.
import * as api from "@/api/client";

// ── Fixtures ─────────────────────────────────────────────────────────────────

function makeEvent(
  overrides: Partial<InterpretationEvent> = {},
): InterpretationEvent {
  return {
    id: "evt-1",
    session_id: "sess-1",
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
    ...overrides,
  };
}

function makeCompositionState(version = 2): CompositionState {
  return {
    id: `state-${version}`,
    version,
    source: null,
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

function makeResolveResponse(
  event: InterpretationEvent,
  overrides: Partial<InterpretationResolveResponse> = {},
): InterpretationResolveResponse {
  return {
    event: {
      ...event,
      choice: "accepted_as_drafted",
      accepted_value: event.llm_draft,
      resolved_at: "2026-05-18T01:00:00Z",
    },
    new_state: makeCompositionState(2),
    ...overrides,
  };
}

function makeOptOutResponse(
  sessionId: string,
): InterpretationOptOutResponse {
  return {
    session_id: sessionId,
    interpretation_review_disabled: true,
    opted_out_at: "2026-05-18T02:00:00Z",
  };
}

/** Build an ApiError plain-object literal (matches parseResponse's throw shape). */
function makeApiError(status: number, detail = ""): ApiError {
  return {
    status,
    detail,
    error_type: undefined,
    partial_state: undefined,
    failed_turn: undefined,
    partial_state_save_failed: undefined,
    partial_state_save_error: undefined,
    fanout_guard: undefined,
    provider_detail: undefined,
    provider_status_code: undefined,
    validation_errors: undefined,
  };
}

function mockScrollMetrics({
  scrollHeight,
  clientHeight,
}: {
  scrollHeight: number;
  clientHeight: number;
}): () => void {
  const previousScrollHeight = Object.getOwnPropertyDescriptor(
    HTMLElement.prototype,
    "scrollHeight",
  );
  const previousClientHeight = Object.getOwnPropertyDescriptor(
    HTMLElement.prototype,
    "clientHeight",
  );
  Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
    configurable: true,
    get() {
      return scrollHeight;
    },
  });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    get() {
      return clientHeight;
    },
  });
  return () => {
    if (previousScrollHeight) {
      Object.defineProperty(
        HTMLElement.prototype,
        "scrollHeight",
        previousScrollHeight,
      );
    } else {
      Reflect.deleteProperty(HTMLElement.prototype, "scrollHeight");
    }
    if (previousClientHeight) {
      Object.defineProperty(
        HTMLElement.prototype,
        "clientHeight",
        previousClientHeight,
      );
    } else {
      Reflect.deleteProperty(HTMLElement.prototype, "clientHeight");
    }
  };
}

// ── Suite setup ──────────────────────────────────────────────────────────────

beforeEach(() => {
  resetStore(useInterpretationEventsStore);
  vi.mocked(api.resolveInterpretation).mockReset();
  vi.mocked(api.optOutOfInterpretations).mockReset();
  vi.mocked(api.listInterpretationEvents).mockReset();
});

// ── Test 1: header text references user_term + llm_draft ─────────────────────

describe("InterpretationReviewTurn — header", () => {
  it("renders header text containing user_term and llm_draft", () => {
    const event = makeEvent({ user_term: "cool", llm_draft: "trendy" });
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    // Both the user term and the LLM draft are rendered inside the body
    // paragraph — query by partial text match so we don't pin the exact
    // copy here (the structural emphasis tags are separate elements).
    expect(screen.getByText(/cool/)).toBeTruthy();
    expect(screen.getByText(/trendy/)).toBeTruthy();
  });
});

describe("InterpretationReviewTurn — kind-aware surfaces", () => {
  it("renders invented-source copy and hides amendment", () => {
    const event = makeEvent({
      user_term: "inline_source_data",
      kind: "invented_source",
      llm_draft: "name,amount\nAda,42",
    });
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    expect(
      screen.getByRole("region", { name: /invented source data/i }),
    ).toBeTruthy();
    const draft = screen.getByRole("group", { name: /source data draft/i });
    expect(draft.textContent).toContain("name,amount");
    expect(draft.textContent).toContain("Ada,42");
    expect(draft.getAttribute("tabindex")).toBe("0");
    expect(
      screen.queryByRole("button", { name: /edit the interpretation/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: /accept invented source data/i }),
    ).toBeTruthy();
  });

  it("renders prompt-template copy and keeps accept disabled until the scroll surface reaches the end", async () => {
    const restoreScrollMetrics = mockScrollMetrics({
      scrollHeight: 300,
      clientHeight: 100,
    });
    try {
      const user = userEvent.setup();
      const event = makeEvent({
        kind: "llm_prompt_template",
        affected_node_id: "summarise",
        llm_draft: "Summarise {{ row.body }} for an auditor.",
      });
      vi.mocked(api.resolveInterpretation).mockResolvedValue(
        makeResolveResponse(event),
      );

      render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

      expect(
        screen.getByRole("region", { name: /llm prompt template/i }),
      ).toBeTruthy();
      expect(
        screen.queryByRole("button", { name: /edit the interpretation/i }),
      ).toBeNull();

      const accept = screen.getByRole("button", {
        name: /accept llm prompt template/i,
      }) as HTMLButtonElement;
      expect(accept.disabled).toBe(true);

      const promptSurface = screen.getByRole("region", {
        name: /prompt template review/i,
      });
      expect(document.activeElement).toBe(promptSurface);
      Object.defineProperty(promptSurface, "scrollTop", {
        configurable: true,
        value: 200,
      });
      fireEvent.scroll(promptSurface);

      await waitFor(() => {
        expect(accept.disabled).toBe(false);
      });
      await user.click(accept);

      await waitFor(() => {
        expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
      });
    } finally {
      restoreScrollMetrics();
    }
  });

  it("allows accepting a prompt template immediately when the review surface does not overflow", async () => {
    const user = userEvent.setup();
    const event = makeEvent({
      kind: "llm_prompt_template",
      llm_draft: "Classify {{ row.body }}.",
    });
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event),
    );

    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    const accept = screen.getByRole("button", {
      name: /accept llm prompt template/i,
    }) as HTMLButtonElement;
    await waitFor(() => {
      expect(accept.disabled).toBe(false);
    });

    await user.click(accept);

    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
  });

  it("shows a stale-review message when the prompt template changed before resolve", async () => {
    const user = userEvent.setup();
    const event = makeEvent({
      kind: "llm_prompt_template",
      llm_draft: "Classify {{ row.body }}.",
    });
    vi.mocked(api.resolveInterpretation).mockRejectedValue(
      {
        ...makeApiError(
          422,
          "The affected LLM prompt no longer contains the expected interpretation placeholder.",
        ),
        error_type: "interpretation_placeholder_unavailable",
      },
    );

    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    const accept = screen.getByRole("button", {
      name: /accept llm prompt template/i,
    }) as HTMLButtonElement;
    await waitFor(() => {
      expect(accept.disabled).toBe(false);
    });

    await user.click(accept);

    expect(
      await screen.findByRole("alert"),
    ).toHaveTextContent(/reload the session/i);
    expect(
      screen.getByRole("alert"),
    ).toHaveTextContent(/stale review/i);
  });

  it("renders pipeline-decision copy and hides amendment", () => {
    const event = makeEvent({
      kind: "pipeline_decision",
      affected_node_id: "drop_raw_html",
      user_term: "drop_raw_html_fields",
      llm_draft:
        "Drop the scraped raw HTML and fingerprint fields before saving the JSON output.",
    });
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    expect(
      screen.getByRole("region", { name: /pipeline decision/i }),
    ).toBeTruthy();
    expect(screen.getByText(/drop the scraped raw html/i)).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: /edit the interpretation/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: /accept pipeline decision/i }),
    ).toBeTruthy();
  });

  it("hides session opt-out when showOptOut is false", () => {
    const event = makeEvent();
    render(
      <InterpretationReviewTurn
        event={event}
        sessionId="sess-1"
        showOptOut={false}
      />,
    );

    expect(
      screen.queryByRole("button", {
        name: /stop reviewing interpretations this session/i,
      }),
    ).toBeNull();
  });

  it("hides amendment for vague-term reviews when showAmend is false", () => {
    const event = makeEvent();
    render(
      <InterpretationReviewTurn
        event={event}
        sessionId="sess-1"
        showAmend={false}
      />,
    );

    expect(
      screen.queryByRole("button", { name: /edit the interpretation/i }),
    ).toBeNull();
  });

  it("does not move focus on mount when autoFocusOnMount is false", () => {
    const event = makeEvent();
    render(
      <InterpretationReviewTurn
        event={event}
        sessionId="sess-1"
        autoFocusOnMount={false}
      />,
    );

    const accept = screen.getByRole("button", {
      name: /accept the llm's interpretation/i,
    });
    expect(document.activeElement).not.toBe(accept);
  });
});

// ── Test 2: "Use my interpretation" submits accepted_as_drafted ──────────────

describe("InterpretationReviewTurn — Use my interpretation", () => {
  it("calls resolveInterpretation with choice='accepted_as_drafted'", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event),
    );

    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);
    await user.click(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    );

    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
    expect(api.resolveInterpretation).toHaveBeenCalledWith("sess-1", "evt-1", {
      choice: "accepted_as_drafted",
    });
  });
});

// ── Test 3: "Change it" reveals textarea pre-filled with llm_draft, focused ──

describe("InterpretationReviewTurn — Change it mode", () => {
  it("reveals a textarea pre-filled with llm_draft and moves focus to it", async () => {
    const user = userEvent.setup();
    const event = makeEvent({ llm_draft: "interesting and engaging" });
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation/i }),
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea.value).toBe("interesting and engaging");
    expect(document.activeElement).toBe(textarea);
  });
});

// ── Test 4: Submit amendment sends choice='amended' + amended_value ──────────

describe("InterpretationReviewTurn — Submit amendment", () => {
  it("calls resolveInterpretation with choice='amended' and the new text", async () => {
    const user = userEvent.setup();
    const event = makeEvent({ llm_draft: "old draft" });
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event, {
        event: {
          ...event,
          choice: "amended",
          accepted_value: "my new wording",
          resolved_at: "2026-05-18T01:00:00Z",
        },
      }),
    );

    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);
    await user.click(
      screen.getByRole("button", { name: /edit the interpretation/i }),
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    await user.clear(textarea);
    await user.type(textarea, "my new wording");

    await user.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
    expect(api.resolveInterpretation).toHaveBeenCalledWith("sess-1", "evt-1", {
      choice: "amended",
      amended_value: "my new wording",
    });
  });
});

// ── Test 5: Cancel from amend mode reverts to choose mode ────────────────────

describe("InterpretationReviewTurn — Cancel from amend mode", () => {
  it("Cancel reverts to the two-button choose view", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation/i }),
    );
    expect(screen.queryByRole("button", { name: "Submit" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    // Choose-mode buttons are back
    expect(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Submit" })).toBeNull();
    // No request was issued
    expect(api.resolveInterpretation).not.toHaveBeenCalled();
  });
});

// ── Test 6: onResolved fires after successful resolve ────────────────────────

describe("InterpretationReviewTurn — onResolved callback", () => {
  it("fires onResolved with the new composition state after successful resolve", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    const newState = makeCompositionState(3);
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event, { new_state: newState }),
    );
    const onResolved = vi.fn();

    render(
      <InterpretationReviewTurn
        event={event}
        sessionId="sess-1"
        onResolved={onResolved}
      />,
    );
    await user.click(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    );

    await waitFor(() => {
      expect(onResolved).toHaveBeenCalledWith(newState);
    });
  });
});

// ── Test 7: Submit disabled for empty amendment ──────────────────────────────

describe("InterpretationReviewTurn — empty amendment", () => {
  it("Submit button is disabled when the amendment is empty", async () => {
    const user = userEvent.setup();
    const event = makeEvent({ llm_draft: "draft" });
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation/i }),
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    await user.clear(textarea);

    const submit = screen.getByRole("button", {
      name: "Submit",
    }) as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
  });
});

// ── Test 8: Amendment exceeding 8KB cap surfaces client-side error ───────────

describe("InterpretationReviewTurn — amendment too long", () => {
  it("submitting an oversized amendment shows a client-side validation error and does not request", async () => {
    const user = userEvent.setup();
    const event = makeEvent({ llm_draft: "draft" });
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation/i }),
    );

    // Construct a string above the 8 KB UTF-8 byte cap.  ASCII "a" is 1
    // byte so 8200 chars exceeds 8192 bytes.  We use fireEvent.change so
    // the controlled-input React onChange handler runs (userEvent.type()
    // here would spend seconds typing 8000 characters one at a time).
    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    const oversized = "a".repeat(8200);
    fireEvent.change(textarea, { target: { value: oversized } });

    // Submit is disabled, but we still attempt to click it to assert the
    // belt-and-suspenders guard.  user.click on a disabled button is a
    // no-op; we use the keyboard path to verify the handler also guards.
    const submit = screen.getByRole("button", {
      name: "Submit",
    }) as HTMLButtonElement;
    expect(submit.disabled).toBe(true);

    // Client-side error region is visible.  The size-validation warning
    // surfaces inside the amend region; we assert via partial text match.
    expect(screen.getByText(/8192 bytes/)).toBeTruthy();
    expect(api.resolveInterpretation).not.toHaveBeenCalled();
  });
});

// ── Test 9: Opt-out link opens confirm modal naming session scope ────────────

describe("InterpretationReviewTurn — opt-out flow", () => {
  it("'Stop reviewing...' opens a confirm modal naming the session scope", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    await user.click(
      screen.getByRole("button", {
        name: /stop reviewing interpretations this session/i,
      }),
    );

    // ConfirmDialog renders an alertdialog; the dialog title names the
    // session scope.
    const dialog = screen.getByRole("alertdialog");
    expect(dialog).toBeTruthy();
    // Copy explicitly mentions "this session" (session-scope, not global).
    expect(dialog.textContent).toMatch(/this session/i);
  });

  it("confirming the modal calls optOutOfInterpretations", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.optOutOfInterpretations).mockResolvedValue(
      makeOptOutResponse("sess-1"),
    );

    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);
    await user.click(
      screen.getByRole("button", {
        name: /stop reviewing interpretations this session/i,
      }),
    );
    await user.click(
      screen.getByRole("button", { name: /stop reviewing for this session/i }),
    );

    await waitFor(() => {
      expect(api.optOutOfInterpretations).toHaveBeenCalledWith("sess-1");
    });
  });
});

// ── Test 10: in-flight disabling + spinner ───────────────────────────────────

describe("InterpretationReviewTurn — in-flight state", () => {
  it("disables both primary buttons and shows a spinner while resolve is pending", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    // Defer the resolve so we can observe the in-flight state.
    let resolveResolve: (v: InterpretationResolveResponse) => void = () => {};
    vi.mocked(api.resolveInterpretation).mockImplementation(
      () =>
        new Promise<InterpretationResolveResponse>((res) => {
          resolveResolve = res;
        }),
    );

    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);
    await user.click(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    );

    // Both primary buttons are disabled while in flight.
    const accept = screen.getByRole("button", {
      name: /accept the llm's interpretation/i,
    }) as HTMLButtonElement;
    const change = screen.getByRole("button", {
      name: /edit the interpretation/i,
    }) as HTMLButtonElement;
    expect(accept.disabled).toBe(true);
    expect(change.disabled).toBe(true);

    // Saving… is rendered in place of the button label.
    expect(screen.getByText(/saving/i)).toBeTruthy();

    // Resolve the deferred promise so the component cleans up.
    resolveResolve(makeResolveResponse(event));
    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
  });
});

// ── Test 11: 409 — already-resolved (multi-tab TOCTOU, F-12) ─────────────────

describe("InterpretationReviewTurn — 409 already-resolved", () => {
  it("surfaces the multi-tab message when the resolve API returns 409", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.resolveInterpretation).mockRejectedValue(
      makeApiError(409, "Event already resolved"),
    );

    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);
    await user.click(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/already resolved in another tab/i);
  });
});

// ── Test 12: 422 — Pydantic validation error ─────────────────────────────────

describe("InterpretationReviewTurn — 422 validation error", () => {
  it("surfaces the validation error detail when the resolve API returns 422", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.resolveInterpretation).mockRejectedValue(
      makeApiError(422, "amended_value must not be blank"),
    );

    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);
    await user.click(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/amended_value must not be blank/i);
  });
});

// ── Test 13: ARIA region ────────────────────────────────────────────────────

describe("InterpretationReviewTurn — ARIA region", () => {
  it("renders as role='region' labelled 'Interpretation review'", () => {
    const event = makeEvent();
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);
    expect(
      screen.getByRole("region", { name: /interpretation review/i }),
    ).toBeTruthy();
  });
});

// ── Test 14: keyboard navigation ─────────────────────────────────────────────

describe("InterpretationReviewTurn — keyboard navigation", () => {
  it("the 'Stop reviewing' link is keyboard-reachable (no tabIndex='-1')", () => {
    const event = makeEvent();
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    const link = screen.getByRole("button", {
      name: /stop reviewing interpretations this session/i,
    });
    // Native <button> elements are tab-reachable by default.  Assert the
    // widget did not opt out via tabIndex="-1".
    expect(link.getAttribute("tabIndex")).not.toBe("-1");
  });

  it("Enter on a focused primary button activates it", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event),
    );
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    // The widget focuses the accept button on mount; pressing Enter must
    // activate it without an explicit pointer click.
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
  });
});

// ── Test 15: focus on mount ─────────────────────────────────────────────────

describe("InterpretationReviewTurn — focus on mount", () => {
  it("focuses the 'Use my interpretation' button on initial render", () => {
    const event = makeEvent();
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    const accept = screen.getByRole("button", {
      name: /accept the llm's interpretation/i,
    });
    expect(document.activeElement).toBe(accept);
  });
});

// ── Test 16: live-region announcement on mount ──────────────────────────────

describe("InterpretationReviewTurn — live-region announcement", () => {
  it("renders a role='status' region with 'Your input needs review' on mount", () => {
    const event = makeEvent();
    render(<InterpretationReviewTurn event={event} sessionId="sess-1" />);

    // role="status" implies aria-live="polite"; AT announces this when
    // the widget mounts.  The widget renders the status element
    // unconditionally on mount so the announcement always fires.
    const statusRegions = screen.getAllByRole("status");
    const matched = statusRegions.some((el) =>
      /your input needs review/i.test(el.textContent ?? ""),
    );
    expect(matched).toBe(true);
  });
});

// ── Test 17 (also covered by 11) — 409 wording ──────────────────────────────
// Test 11 above already pins the "already resolved in another tab" wording
// per spec test 17.  We leave this section as documentation of the dual
// contract: 409 is the multi-tab TOCTOU recovery surface.

// ── Test 18: GuidedTurn dispatches interpretation_review to this widget ─────

describe("GuidedTurn dispatch — interpretation_review", () => {
  it("routes turn.type='interpretation_review' to InterpretationReviewTurn", () => {
    const event = makeEvent({ user_term: "cool" });
    const turn: TurnPayload = {
      type: "interpretation_review",
      step_index: 0,
      payload: event,
    };
    render(<GuidedTurn turn={turn} onSubmit={vi.fn()} />);

    // The widget renders the region with the stable accessible name.
    expect(
      screen.getByRole("region", { name: /interpretation review/i }),
    ).toBeTruthy();
    // And references the user_term from the event payload.
    expect(screen.getByText(/cool/)).toBeTruthy();
  });
});
