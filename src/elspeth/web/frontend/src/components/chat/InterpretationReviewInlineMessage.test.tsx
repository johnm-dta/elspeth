// ============================================================================
// InterpretationReviewInlineMessage — Phase 5b Task 5 behavioural coverage.
//
// Mirrors the guided-mode widget's twelve behavioural tests (Task 4 #1-#12)
// against the freeform-mode inline-message variant.  Both widgets share the
// `useInterpretationResolver` hook (hooks/useInterpretationResolver.ts), so
// the contracts ARE identical at the wire and store level — these tests
// confirm the inline-message rendering does not break that contract.
//
// Spec deviations from the guided test suite:
//
//   * NO focus-on-mount test.  The inline message intentionally does NOT
//     focus on mount (it lives inside the chat log; auto-focus would yank
//     focus from a user typing in the chat input).  Tests 14 ("keyboard-
//     reachable opt-out button") and 15 ("Enter on a focused primary
//     button activates it") still hold — Tab-reachability is preserved via
//     real <button> elements.  We assert no mount-focus separately to pin
//     the contract.
//
// Discipline note: as in the guided test suite, ONLY the API client's
// resolve / opt-out methods are mocked.  The Zustand store is exercised
// end-to-end so the wire path stays honest.
// ============================================================================

import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InterpretationReviewInlineMessage } from "./InterpretationReviewInlineMessage";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import type {
  InterpretationEvent,
  InterpretationOptOutResponse,
  InterpretationResolveResponse,
} from "@/types/interpretation";
import type { CompositionState } from "@/types/api";
import type { ApiError } from "@/types/index";

vi.mock("@/api/client", () => ({
  listInterpretationEvents: vi.fn(),
  resolveInterpretation: vi.fn(),
  optOutOfInterpretations: vi.fn(),
  getInterpretationOptOutSummary: vi.fn(),
}));

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
    sources: {},
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

beforeEach(() => {
  resetStore(useInterpretationEventsStore);
  vi.mocked(api.resolveInterpretation).mockReset();
  vi.mocked(api.optOutOfInterpretations).mockReset();
  vi.mocked(api.listInterpretationEvents).mockReset();
});

// ── Test 1: header text references user_term + llm_draft ─────────────────────

describe("InterpretationReviewInlineMessage — header", () => {
  it("renders body text containing user_term and llm_draft", () => {
    const event = makeEvent({ user_term: "cool", llm_draft: "trendy" });
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

    expect(screen.getByText(/cool/)).toBeTruthy();
    expect(screen.getByText(/trendy/)).toBeTruthy();
  });
});

describe("InterpretationReviewInlineMessage — kind-aware surfaces", () => {
  it("renders invented-source copy and hides amendment", () => {
    const event = makeEvent({
      user_term: "inline_source_data",
      kind: "invented_source",
      llm_draft: "name,amount\nAda,42",
    });
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

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

      render(
        <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
      );

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
});

// ── Test 2: "Use my interpretation" submits accepted_as_drafted ──────────────

describe("InterpretationReviewInlineMessage — Use my interpretation", () => {
  it("calls resolveInterpretation with choice='accepted_as_drafted'", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event),
    );

    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );
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

describe("InterpretationReviewInlineMessage — Change it mode", () => {
  it("reveals a textarea pre-filled with llm_draft and moves focus to it", async () => {
    const user = userEvent.setup();
    const event = makeEvent({ llm_draft: "interesting and engaging" });
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation/i }),
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea.value).toBe("interesting and engaging");
    expect(document.activeElement).toBe(textarea);
  });
});

// ── Test 4: Submit amendment sends choice='amended' + amended_value ──────────

describe("InterpretationReviewInlineMessage — Submit amendment", () => {
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

    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );
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

describe("InterpretationReviewInlineMessage — Cancel from amend mode", () => {
  it("Cancel reverts to the two-button choose view", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation/i }),
    );
    expect(screen.queryByRole("button", { name: "Submit" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Submit" })).toBeNull();
    expect(api.resolveInterpretation).not.toHaveBeenCalled();
  });
});

// ── Test 6: onResolved fires after successful resolve ────────────────────────

describe("InterpretationReviewInlineMessage — onResolved callback", () => {
  it("fires onResolved with the new composition state after successful resolve", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    const newState = makeCompositionState(3);
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event, { new_state: newState }),
    );
    const onResolved = vi.fn();

    render(
      <InterpretationReviewInlineMessage
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

describe("InterpretationReviewInlineMessage — empty amendment", () => {
  it("Submit button is disabled when the amendment is empty", async () => {
    const user = userEvent.setup();
    const event = makeEvent({ llm_draft: "draft" });
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

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

describe("InterpretationReviewInlineMessage — amendment too long", () => {
  it("submitting an oversized amendment shows a client-side validation error and does not request", async () => {
    const user = userEvent.setup();
    const event = makeEvent({ llm_draft: "draft" });
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation/i }),
    );

    // 8200 ASCII bytes > 8192-byte cap.  fireEvent.change avoids the
    // multi-second user.type() typing cost.
    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    const oversized = "a".repeat(8200);
    fireEvent.change(textarea, { target: { value: oversized } });

    const submit = screen.getByRole("button", {
      name: "Submit",
    }) as HTMLButtonElement;
    expect(submit.disabled).toBe(true);

    expect(screen.getByText(/8192 bytes/)).toBeTruthy();
    expect(api.resolveInterpretation).not.toHaveBeenCalled();
  });
});

// ── Test 9: Opt-out link opens confirm modal naming session scope ────────────

describe("InterpretationReviewInlineMessage — opt-out flow", () => {
  it("'Stop reviewing...' opens a confirm modal naming the session scope", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

    await user.click(
      screen.getByRole("button", {
        name: /stop reviewing interpretations this session/i,
      }),
    );

    const dialog = screen.getByRole("alertdialog");
    expect(dialog).toBeTruthy();
    expect(dialog.textContent).toMatch(/this session/i);
  });

  it("confirming the modal calls optOutOfInterpretations", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.optOutOfInterpretations).mockResolvedValue(
      makeOptOutResponse("sess-1"),
    );

    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );
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

describe("InterpretationReviewInlineMessage — in-flight state", () => {
  it("disables both primary buttons and shows a spinner while resolve is pending", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    let resolveResolve: (v: InterpretationResolveResponse) => void = () => {};
    vi.mocked(api.resolveInterpretation).mockImplementation(
      () =>
        new Promise<InterpretationResolveResponse>((res) => {
          resolveResolve = res;
        }),
    );

    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );
    await user.click(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    );

    const accept = screen.getByRole("button", {
      name: /accept the llm's interpretation/i,
    }) as HTMLButtonElement;
    const change = screen.getByRole("button", {
      name: /edit the interpretation/i,
    }) as HTMLButtonElement;
    expect(accept.disabled).toBe(true);
    expect(change.disabled).toBe(true);

    expect(screen.getByText(/saving/i)).toBeTruthy();

    resolveResolve(makeResolveResponse(event));
    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
  });
});

// ── Test 11: 409 — already-resolved (multi-tab TOCTOU, F-12) ─────────────────

describe("InterpretationReviewInlineMessage — 409 already-resolved", () => {
  it("surfaces the multi-tab message when the resolve API returns 409", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.resolveInterpretation).mockRejectedValue(
      makeApiError(409, "Event already resolved"),
    );

    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );
    await user.click(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/already resolved in another tab/i);
  });
});

// ── Test 12: 422 — Pydantic validation error ─────────────────────────────────

describe("InterpretationReviewInlineMessage — 422 validation error", () => {
  it("surfaces the validation error detail when the resolve API returns 422", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.resolveInterpretation).mockRejectedValue(
      makeApiError(422, "amended_value must not be blank"),
    );

    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );
    await user.click(
      screen.getByRole("button", { name: /accept the llm's interpretation/i }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/amended_value must not be blank/i);
  });
});

// ── ARIA region ──────────────────────────────────────────────────────────────

describe("InterpretationReviewInlineMessage — ARIA region", () => {
  it("renders as role='region' labelled 'Interpretation review'", () => {
    const event = makeEvent();
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );
    expect(
      screen.getByRole("region", { name: /interpretation review/i }),
    ).toBeTruthy();
  });

  it("carries the discriminator data-testid for ChatPanel dispatch tests", () => {
    const event = makeEvent();
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );
    expect(
      screen.getByTestId("interpretation-review-inline-message"),
    ).toBeTruthy();
  });
});

// ── Keyboard navigation ──────────────────────────────────────────────────────

describe("InterpretationReviewInlineMessage — keyboard navigation", () => {
  it("the 'Stop reviewing' link is keyboard-reachable (no tabIndex='-1')", () => {
    const event = makeEvent();
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

    const link = screen.getByRole("button", {
      name: /stop reviewing interpretations this session/i,
    });
    expect(link.getAttribute("tabIndex")).not.toBe("-1");
  });
});

// ── Live-region announcement ────────────────────────────────────────────────

describe("InterpretationReviewInlineMessage — live-region announcement", () => {
  it("renders a role='status' region with 'Your input needs review' on mount", () => {
    const event = makeEvent();
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

    const statusRegions = screen.getAllByRole("status");
    const matched = statusRegions.some((el) =>
      /your input needs review/i.test(el.textContent ?? ""),
    );
    expect(matched).toBe(true);
  });
});

// ── No mount-focus contract ─────────────────────────────────────────────────
//
// Deliberately distinct from the guided turn: the inline-message variant
// must NOT focus the accept button on mount because it lives inside the
// chat log alongside a chat input the user may be typing into.  Pinning
// this here prevents an accidental copy-paste of the guided mount-focus
// effect into the inline component during a future refactor.

describe("InterpretationReviewInlineMessage — no mount focus", () => {
  it("does NOT move focus to the accept button on initial render", () => {
    const event = makeEvent();
    render(
      <InterpretationReviewInlineMessage event={event} sessionId="sess-1" />,
    );

    const accept = screen.getByRole("button", {
      name: /accept the llm's interpretation/i,
    });
    expect(document.activeElement).not.toBe(accept);
  });
});
