import { act, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  useInterpretationResolver,
  type UseInterpretationResolverResult,
} from "./useInterpretationResolver";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import type { CompositionState } from "@/types/api";
import type {
  InterpretationEvent,
  InterpretationResolveResponse,
} from "@/types/interpretation";

vi.mock("@/api/client", () => ({
  listInterpretationEvents: vi.fn(),
  resolveInterpretation: vi.fn(),
  optOutOfInterpretations: vi.fn(),
  getInterpretationOptOutSummary: vi.fn(),
}));

import * as api from "@/api/client";

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason?: unknown) => void = () => undefined;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function makeEvent(
  id: string,
  overrides: Partial<InterpretationEvent> = {},
): InterpretationEvent {
  return {
    id,
    session_id: "sess-1",
    composition_state_id: "state-1",
    affected_node_id: `node-${id}`,
    tool_call_id: `tool-${id}`,
    user_term: `term-${id}`,
    kind: "vague_term",
    llm_draft: `draft-${id}`,
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

function makeCompositionState(id: string, version: number): CompositionState {
  return {
    id,
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
  choice: "accepted_as_drafted" | "amended",
  acceptedValue: string,
  newState: CompositionState,
): InterpretationResolveResponse {
  return {
    event: {
      ...event,
      choice,
      accepted_value: acceptedValue,
      resolved_at: "2026-05-18T01:00:00Z",
    },
    new_state: newState,
  };
}

describe("useInterpretationResolver", () => {
  beforeEach(() => {
    resetStore(useInterpretationEventsStore);
    vi.mocked(api.resolveInterpretation).mockReset();
    vi.mocked(api.optOutOfInterpretations).mockReset();
  });

  it("preserves event B state when event A resolves late", async () => {
    const eventA = makeEvent("evt-a");
    const eventB = makeEvent("evt-b");
    const stateA = makeCompositionState("state-a-resolved-late", 2);
    const stateB = makeCompositionState("state-b-resolved-first", 3);
    const deferredA = deferred<InterpretationResolveResponse>();
    const deferredB = deferred<InterpretationResolveResponse>();
    const resolvedStateIds: string[] = [];
    let hooks!: {
      a: UseInterpretationResolverResult;
      b: UseInterpretationResolverResult;
    };

    vi.mocked(api.resolveInterpretation).mockImplementation(
      (_sessionId, eventId) => {
        if (eventId === eventA.id) return deferredA.promise;
        if (eventId === eventB.id) return deferredB.promise;
        throw new Error(`Unexpected event id ${eventId}`);
      },
    );

    useInterpretationEventsStore.getState().addPendingEvent("sess-1", eventA);
    useInterpretationEventsStore.getState().addPendingEvent("sess-1", eventB);

    function Harness() {
      hooks = {
        a: useInterpretationResolver({
          event: eventA,
          sessionId: "sess-1",
          onResolved: (newState) => {
            if (newState !== null) resolvedStateIds.push(newState.id);
          },
        }),
        b: useInterpretationResolver({
          event: eventB,
          sessionId: "sess-1",
          onResolved: (newState) => {
            if (newState !== null) resolvedStateIds.push(newState.id);
          },
        }),
      };
      return null;
    }

    render(<Harness />);

    let resolveA!: Promise<void>;
    act(() => {
      resolveA = hooks.a.handleUseMine();
    });
    await waitFor(() => expect(hooks.a.resolveInFlight).toBe(true));

    act(() => {
      hooks.b.handleOpenAmend();
    });
    await waitFor(() => expect(hooks.b.mode).toBe("amend"));
    act(() => {
      hooks.b.setAmendText("operator amendment for B");
    });
    await waitFor(() =>
      expect(hooks.b.amendText).toBe("operator amendment for B"),
    );

    let resolveB!: Promise<void>;
    act(() => {
      resolveB = hooks.b.handleSubmitAmend();
    });
    await waitFor(() => expect(hooks.b.resolveInFlight).toBe(true));

    await act(async () => {
      deferredB.resolve(
        makeResolveResponse(
          eventB,
          "amended",
          "operator amendment for B",
          stateB,
        ),
      );
      await resolveB;
    });

    expect(resolvedStateIds).toEqual(["state-b-resolved-first"]);
    expect(
      Object.keys(
        useInterpretationEventsStore.getState().pendingBySession["sess-1"],
      ),
    ).toEqual(["evt-a"]);
    expect(
      useInterpretationEventsStore.getState().resolvedCountBySession["sess-1"],
    ).toMatchObject({ amended: 1, accepted_as_drafted: 0 });

    await act(async () => {
      deferredA.resolve(
        makeResolveResponse(eventA, "accepted_as_drafted", eventA.llm_draft ?? "", stateA),
      );
      await resolveA;
    });

    expect(resolvedStateIds).toEqual([
      "state-b-resolved-first",
      "state-a-resolved-late",
    ]);
    expect(
      useInterpretationEventsStore.getState().pendingBySession["sess-1"],
    ).toEqual({});
    expect(
      useInterpretationEventsStore.getState().resolvedCountBySession["sess-1"],
    ).toMatchObject({ amended: 1, accepted_as_drafted: 1 });
  });
});
