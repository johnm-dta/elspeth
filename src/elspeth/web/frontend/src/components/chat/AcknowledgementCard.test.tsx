// ============================================================================
// AcknowledgementCard — behavioural coverage ported from the retired
// InterpretationReviewTurn / InterpretationReviewInlineMessage suites.
//
// Discipline: mock ONLY the API client's resolve / opt-out methods; the
// Zustand store runs live so the wire path stays honest.  "Acknowledge" ==
// today's accept (`accepted_as_drafted`).
// ============================================================================

import { describe, it, expect, beforeEach, vi } from "vitest";
import type { ComponentProps } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AcknowledgementCard } from "./AcknowledgementCard";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import type {
  InterpretationEvent,
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
  const prevScroll = Object.getOwnPropertyDescriptor(
    HTMLElement.prototype,
    "scrollHeight",
  );
  const prevClient = Object.getOwnPropertyDescriptor(
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
    if (prevScroll) {
      Object.defineProperty(HTMLElement.prototype, "scrollHeight", prevScroll);
    } else {
      Reflect.deleteProperty(HTMLElement.prototype, "scrollHeight");
    }
    if (prevClient) {
      Object.defineProperty(HTMLElement.prototype, "clientHeight", prevClient);
    } else {
      Reflect.deleteProperty(HTMLElement.prototype, "clientHeight");
    }
  };
}

function renderCard(
  event: InterpretationEvent,
  props: Partial<ComponentProps<typeof AcknowledgementCard>> = {},
) {
  return render(
    <AcknowledgementCard
      event={event}
      sessionId="sess-1"
      stepLabel={props.stepLabel ?? "Summarise"}
      showAmend={props.showAmend ?? event.kind === "vague_term"}
      onResolved={props.onResolved}
    />,
  );
}

beforeEach(() => {
  resetStore(useInterpretationEventsStore);
  vi.mocked(api.resolveInterpretation).mockReset();
  vi.mocked(api.optOutOfInterpretations).mockReset();
  vi.mocked(api.listInterpretationEvents).mockReset();
});

// ── Copy & title ─────────────────────────────────────────────────────────────

describe("AcknowledgementCard — per-kind copy", () => {
  it("vague_term: title 'Interpretation' + user_term/llm_draft line", () => {
    renderCard(makeEvent({ user_term: "cool", llm_draft: "trendy" }));
    expect(screen.getByText("Interpretation")).toBeTruthy();
    expect(screen.getByText(/cool/)).toBeTruthy();
    expect(screen.getByText(/trendy/)).toBeTruthy();
  });

  it("llm_model_choice: '<step> step · model' title + 'The LLM picked' line", () => {
    renderCard(
      makeEvent({ kind: "llm_model_choice", llm_draft: "openai/gpt-4o-mini" }),
      { stepLabel: "Summarise", showAmend: false },
    );
    expect(screen.getByText("Summarise step · model")).toBeTruthy();
    expect(screen.getByText(/The LLM picked/)).toBeTruthy();
    expect(screen.getByText("openai/gpt-4o-mini")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: /acknowledge the llm model choice/i }),
    ).toBeTruthy();
  });

  it("pipeline_decision: '<step> step · decision' title + decision text inline", () => {
    renderCard(
      makeEvent({
        kind: "pipeline_decision",
        llm_draft: "Drop the scraped raw HTML before saving the JSON output.",
      }),
      { stepLabel: "Output", showAmend: false },
    );
    expect(screen.getByText("Output step · decision")).toBeTruthy();
    expect(screen.getByText(/drop the scraped raw html/i)).toBeTruthy();
    expect(
      screen.getByRole("button", { name: /acknowledge the pipeline decision/i }),
    ).toBeTruthy();
  });

  it("invented_source: title 'Source data' + review-before-fetching line", () => {
    renderCard(
      makeEvent({
        kind: "invented_source",
        llm_draft: '{"name":"Ada","amount":42}',
      }),
      { showAmend: false },
    );
    expect(screen.getByText("Source data")).toBeTruthy();
    expect(screen.getByText(/invented this source data/i)).toBeTruthy();
    expect(
      screen.getByRole("button", {
        name: /acknowledge the invented source data/i,
      }),
    ).toBeTruthy();
  });

  it("hides the amend affordance when showAmend is false", () => {
    renderCard(makeEvent(), { showAmend: false });
    expect(
      screen.queryByRole("button", { name: /edit the interpretation/i }),
    ).toBeNull();
  });
});

// ── Value rendering ──────────────────────────────────────────────────────────

describe("AcknowledgementCard — invented-source value", () => {
  it("pretty-prints a short JSON source value inline", () => {
    const { container } = renderCard(
      makeEvent({
        kind: "invented_source",
        llm_draft: '{"name":"Ada","amount":42}',
      }),
      { showAmend: false },
    );
    const pre = container.querySelector("pre");
    expect(pre!.getAttribute("data-codeblock-format")).toBe("json");
    // No View expander for a short value.
    expect(screen.queryByRole("button", { name: /^view$/i })).toBeNull();
  });

  it("falls back to plain monospace for a non-JSON source value", () => {
    const { container } = renderCard(
      makeEvent({ kind: "invented_source", llm_draft: "name,amount\nAda,42" }),
      { showAmend: false },
    );
    const pre = container.querySelector("pre");
    expect(pre!.getAttribute("data-codeblock-format")).toBe("plain");
  });

  it("puts a long source value behind a View expander", async () => {
    const user = userEvent.setup();
    const longJson = JSON.stringify(
      Array.from({ length: 40 }, (_, i) => ({ row: i, label: `item-${i}` })),
    );
    renderCard(
      makeEvent({ kind: "invented_source", llm_draft: longJson }),
      { showAmend: false },
    );
    const view = screen.getByRole("button", { name: /^view$/i });
    expect(view).toBeTruthy();
    await user.click(view);
    expect(screen.getByRole("button", { name: /^hide$/i })).toBeTruthy();
  });
});

// ── Acknowledge → accepted_as_drafted ────────────────────────────────────────

describe("AcknowledgementCard — Acknowledge", () => {
  it("resolves with choice='accepted_as_drafted'", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event),
    );

    renderCard(event);
    await user.click(
      screen.getByRole("button", {
        name: /acknowledge the llm's interpretation of cool/i,
      }),
    );

    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledWith("sess-1", "evt-1", {
        choice: "accepted_as_drafted",
      });
    });
  });

  it("fires onResolved with the new composition state", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    const newState = makeCompositionState(3);
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event, { new_state: newState }),
    );
    const onResolved = vi.fn();

    renderCard(event, { onResolved });
    await user.click(
      screen.getByRole("button", {
        name: /acknowledge the llm's interpretation of cool/i,
      }),
    );

    await waitFor(() => {
      expect(onResolved).toHaveBeenCalledWith(newState);
    });
  });

  it("disables both primary buttons and shows a spinner while in flight", async () => {
    const user = userEvent.setup();
    const event = makeEvent();
    let resolveResolve: (v: InterpretationResolveResponse) => void = () => {};
    vi.mocked(api.resolveInterpretation).mockImplementation(
      () =>
        new Promise<InterpretationResolveResponse>((res) => {
          resolveResolve = res;
        }),
    );

    renderCard(event);
    await user.click(
      screen.getByRole("button", {
        name: /acknowledge the llm's interpretation of cool/i,
      }),
    );

    const accept = screen.getByRole("button", {
      name: /acknowledge the llm's interpretation of cool/i,
    }) as HTMLButtonElement;
    const change = screen.getByRole("button", {
      name: /edit the interpretation of cool/i,
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

// ── Amend flow ───────────────────────────────────────────────────────────────

describe("AcknowledgementCard — amend", () => {
  it("Change… reveals a textarea pre-filled with llm_draft and focuses it", async () => {
    const user = userEvent.setup();
    renderCard(makeEvent({ llm_draft: "interesting and engaging" }));

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation of cool/i }),
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea.value).toBe("interesting and engaging");
    expect(document.activeElement).toBe(textarea);
  });

  it("Submit resolves with choice='amended' and the new text", async () => {
    const user = userEvent.setup();
    const event = makeEvent({ llm_draft: "old draft" });
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event, {
        event: { ...event, choice: "amended", accepted_value: "my new wording" },
      }),
    );

    renderCard(event);
    await user.click(
      screen.getByRole("button", { name: /edit the interpretation of cool/i }),
    );
    const textarea = screen.getByRole("textbox");
    await user.clear(textarea);
    await user.type(textarea, "my new wording");
    await user.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledWith("sess-1", "evt-1", {
        choice: "amended",
        amended_value: "my new wording",
      });
    });
  });

  it("Cancel reverts to the choose view without a request", async () => {
    const user = userEvent.setup();
    renderCard(makeEvent());

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation of cool/i }),
    );
    expect(screen.queryByRole("button", { name: "Submit" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(
      screen.getByRole("button", {
        name: /acknowledge the llm's interpretation of cool/i,
      }),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Submit" })).toBeNull();
    expect(api.resolveInterpretation).not.toHaveBeenCalled();
  });

  it("Submit is disabled for an empty amendment", async () => {
    const user = userEvent.setup();
    renderCard(makeEvent({ llm_draft: "draft" }));
    await user.click(
      screen.getByRole("button", { name: /edit the interpretation of cool/i }),
    );
    await user.clear(screen.getByRole("textbox"));
    expect(
      (screen.getByRole("button", { name: "Submit" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("surfaces the 8 KB cap and blocks the request for an oversized amendment", async () => {
    const user = userEvent.setup();
    renderCard(makeEvent({ llm_draft: "draft" }));
    await user.click(
      screen.getByRole("button", { name: /edit the interpretation of cool/i }),
    );
    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "a".repeat(8200) } });

    expect(
      (screen.getByRole("button", { name: "Submit" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(screen.getByText(/8192 bytes/)).toBeTruthy();
    expect(api.resolveInterpretation).not.toHaveBeenCalled();
  });
});

// ── Error mapping ────────────────────────────────────────────────────────────

describe("AcknowledgementCard — error mapping", () => {
  it("409 → already-resolved-in-another-tab message", async () => {
    const user = userEvent.setup();
    vi.mocked(api.resolveInterpretation).mockRejectedValue(makeApiError(409));
    renderCard(makeEvent());
    await user.click(
      screen.getByRole("button", {
        name: /acknowledge the llm's interpretation of cool/i,
      }),
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/already resolved in another tab/i);
  });

  it("other (500) → generic could-not-resolve message with detail", async () => {
    const user = userEvent.setup();
    vi.mocked(api.resolveInterpretation).mockRejectedValue(
      makeApiError(500, "upstream exploded"),
    );
    renderCard(makeEvent());
    await user.click(
      screen.getByRole("button", {
        name: /acknowledge the llm's interpretation of cool/i,
      }),
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/could not resolve interpretation/i);
    expect(alert.textContent).toMatch(/upstream exploded/i);
  });

  it("422 → validation detail", async () => {
    const user = userEvent.setup();
    vi.mocked(api.resolveInterpretation).mockRejectedValue(
      makeApiError(422, "amended_value must not be blank"),
    );
    renderCard(makeEvent());
    await user.click(
      screen.getByRole("button", {
        name: /acknowledge the llm's interpretation of cool/i,
      }),
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/amended_value must not be blank/i);
  });

  it("422 placeholder-unavailable on a prompt template → stale-review message", async () => {
    const restore = mockScrollMetrics({ scrollHeight: 100, clientHeight: 100 });
    try {
      const user = userEvent.setup();
      vi.mocked(api.resolveInterpretation).mockRejectedValue({
        ...makeApiError(422, "placeholder gone"),
        error_type: "interpretation_placeholder_unavailable",
      });
      renderCard(
        makeEvent({ kind: "llm_prompt_template", llm_draft: "Classify {{ x }}." }),
        { showAmend: false },
      );
      // Open the View expander so the scroll gate clears (no overflow → ready).
      await user.click(screen.getByRole("button", { name: /^view prompt/i }));
      const accept = screen.getByRole("button", {
        name: /acknowledge the llm prompt template/i,
      }) as HTMLButtonElement;
      await waitFor(() => expect(accept.disabled).toBe(false));
      await user.click(accept);
      const alert = await screen.findByRole("alert");
      expect(alert.textContent).toMatch(/stale review/i);
      expect(alert.textContent).toMatch(/reload the session/i);
    } finally {
      restore();
    }
  });
});

// ── Prompt-template scroll gate (inside the View expander) ────────────────────

describe("AcknowledgementCard — prompt-template scroll gate", () => {
  it("keeps Acknowledge disabled until the View expander is opened and scrolled to end", async () => {
    const restore = mockScrollMetrics({ scrollHeight: 300, clientHeight: 100 });
    try {
      const user = userEvent.setup();
      const event = makeEvent({
        kind: "llm_prompt_template",
        llm_draft: "Summarise {{ row.body }} for an auditor.",
      });
      vi.mocked(api.resolveInterpretation).mockResolvedValue(
        makeResolveResponse(event),
      );
      renderCard(event, { showAmend: false });

      const accept = screen.getByRole("button", {
        name: /acknowledge the llm prompt template/i,
      }) as HTMLButtonElement;
      // Collapsed: gate closed.
      expect(accept.disabled).toBe(true);

      await user.click(screen.getByRole("button", { name: /^view prompt/i }));
      // Open but not scrolled: still gated (content overflows).
      expect(accept.disabled).toBe(true);

      const surface = screen.getByRole("region", {
        name: /prompt template review/i,
      });
      Object.defineProperty(surface, "scrollTop", {
        configurable: true,
        value: 200,
      });
      fireEvent.scroll(surface);

      await waitFor(() => expect(accept.disabled).toBe(false));
      await user.click(accept);
      await waitFor(() =>
        expect(api.resolveInterpretation).toHaveBeenCalledTimes(1),
      );
    } finally {
      restore();
    }
  });

  it("enables Acknowledge once the View expander is opened when the prompt does not overflow", async () => {
    const restore = mockScrollMetrics({ scrollHeight: 100, clientHeight: 100 });
    try {
      const user = userEvent.setup();
      renderCard(
        makeEvent({ kind: "llm_prompt_template", llm_draft: "Classify {{ x }}." }),
        { showAmend: false },
      );
      const accept = screen.getByRole("button", {
        name: /acknowledge the llm prompt template/i,
      }) as HTMLButtonElement;
      expect(accept.disabled).toBe(true);
      await user.click(screen.getByRole("button", { name: /^view prompt/i }));
      await waitFor(() => expect(accept.disabled).toBe(false));
    } finally {
      restore();
    }
  });

  // elspeth-3b35abf148 variant 2: the disabled Acknowledge says WHY as visible
  // text (not title-only), and the View button carries its required intent.
  it("explains the closed gate in visible text wired to the disabled button", async () => {
    const restore = mockScrollMetrics({ scrollHeight: 300, clientHeight: 100 });
    try {
      const user = userEvent.setup();
      renderCard(
        makeEvent({
          kind: "llm_prompt_template",
          llm_draft: "Summarise {{ row.body }}.",
        }),
        { showAmend: false },
      );

      // Visible explanation (must NOT be visually-hidden) + intent-carrying label.
      const note = screen.getByText(/read it to the end to enable/i);
      expect(note.classList.contains("visually-hidden")).toBe(false);
      const view = screen.getByRole("button", { name: "View prompt (required)" });
      expect(view).toBeTruthy();
      // aria-describedby on the disabled button points at the SAME element.
      const accept = screen.getByRole("button", {
        name: /acknowledge the llm prompt template/i,
      });
      expect(accept.getAttribute("aria-describedby")).toBe(note.id);

      // Once the gate clears (open + no overflow scenario is covered above),
      // scrolling to the end removes the note.
      await user.click(view);
      const surface = screen.getByRole("region", {
        name: /prompt template review/i,
      });
      Object.defineProperty(surface, "scrollTop", {
        configurable: true,
        value: 200,
      });
      fireEvent.scroll(surface);
      await waitFor(() =>
        expect(screen.queryByText(/read it to the end to enable/i)).toBeNull(),
      );
    } finally {
      restore();
    }
  });

  it("has a stable DOM id the wire-stage blocker links can target", () => {
    const event = makeEvent({ kind: "llm_prompt_template" });
    renderCard(event, { showAmend: false });
    const section = document.getElementById(`ack-card-${event.id}`);
    expect(section).not.toBeNull();
    expect(section?.getAttribute("data-testid")).toBe("acknowledgement-card");
  });
});

// ── Accessibility / focus ────────────────────────────────────────────────────

describe("AcknowledgementCard — a11y", () => {
  it("does NOT move focus on mount (announce-don't-steal)", () => {
    renderCard(makeEvent());
    const accept = screen.getByRole("button", {
      name: /acknowledge the llm's interpretation of cool/i,
    });
    expect(document.activeElement).not.toBe(accept);
  });

  it("is a region labelled by the title and uses real <button>s", () => {
    renderCard(makeEvent());
    expect(screen.getByRole("region", { name: /interpretation/i })).toBeTruthy();
    const accept = screen.getByRole("button", {
      name: /acknowledge the llm's interpretation of cool/i,
    });
    expect(accept.tagName).toBe("BUTTON");
  });
});
