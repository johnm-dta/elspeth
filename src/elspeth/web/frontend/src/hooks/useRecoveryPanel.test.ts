import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useRecoveryPanel } from "./useRecoveryPanel";
import type { ApiError, CompositionState, FailedTurn } from "@/types/api";

function makePartialState(version = 7): CompositionState {
  return {
    id: `state-${version}`,
    version,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: "Recovered", description: null },
  };
}

function makeFailedTurn(): FailedTurn {
  return {
    assistant_message_id: "assistant-1",
    tool_calls_attempted: 2,
    tool_responses_persisted: 1,
    transcript_url: null,
  };
}

function makeApiError(fields: Partial<ApiError>): ApiError {
  return {
    status: 500,
    detail: "compose failed",
    ...fields,
  };
}

describe("useRecoveryPanel", () => {
  it("opens only when partial_state and failed_turn are both present", () => {
    const onApplyState = vi.fn();
    const { result } = renderHook(() =>
      useRecoveryPanel({
        currentCompositionVersion: 7,
        recoveryStartedCompositionVersion: 7,
        onApplyState,
      }),
    );

    const cases: Array<[string, ApiError, boolean]> = [
      [
        "both present",
        makeApiError({
          partial_state: makePartialState(),
          failed_turn: makeFailedTurn(),
        }),
        true,
      ],
      [
        "partial only",
        makeApiError({ partial_state: makePartialState(), failed_turn: null }),
        false,
      ],
      [
        "failed turn only",
        makeApiError({ partial_state: null, failed_turn: makeFailedTurn() }),
        false,
      ],
      ["neither", makeApiError({}), false],
    ];

    for (const [, error, expectedOpened] of cases) {
      act(() => {
        result.current.discard();
      });

      let opened = false;
      act(() => {
        opened = result.current.openFromError(error);
      });

      expect(opened).toBe(expectedOpened);
      expect(result.current.isOpen).toBe(expectedOpened);
    }
  });

  it("requires confirmation when the current composition version changed", () => {
    const onApplyState = vi.fn();
    let currentVersion: number | null = 7;
    const partialState = makePartialState(8);
    const { result, rerender } = renderHook(() =>
      useRecoveryPanel({
        currentCompositionVersion: currentVersion,
        recoveryStartedCompositionVersion: 7,
        onApplyState,
      }),
    );

    act(() => {
      result.current.openFromError(
        makeApiError({
          partial_state: partialState,
          failed_turn: makeFailedTurn(),
        }),
      );
    });
    currentVersion = 9;
    rerender();

    let applyResult: ReturnType<typeof result.current.requestApply> | undefined;
    act(() => {
      applyResult = result.current.requestApply();
    });

    expect(applyResult).toEqual({ applied: false, needsConfirmation: true });
    expect(result.current.needsApplyConfirmation).toBe(true);
    expect(onApplyState).not.toHaveBeenCalled();
  });

  it("confirmed Apply calls onApplyState with the partial composition state", () => {
    const onApplyState = vi.fn();
    const partialState = makePartialState(8);
    const { result } = renderHook(() =>
      useRecoveryPanel({
        currentCompositionVersion: 9,
        recoveryStartedCompositionVersion: 7,
        onApplyState,
      }),
    );

    act(() => {
      result.current.openFromError(
        makeApiError({
          partial_state: partialState,
          failed_turn: makeFailedTurn(),
        }),
      );
      result.current.requestApply();
    });

    let applied = false;
    act(() => {
      applied = result.current.confirmApply();
    });

    expect(applied).toBe(true);
    expect(onApplyState).toHaveBeenCalledTimes(1);
    expect(onApplyState).toHaveBeenCalledWith(partialState);
    expect(result.current.isOpen).toBe(false);
    expect(result.current.needsApplyConfirmation).toBe(false);
  });

  it("applies immediately when the composition version matches the compose-start snapshot", () => {
    const onApplyState = vi.fn();
    const partialState = makePartialState(8);
    const { result } = renderHook(() =>
      useRecoveryPanel({
        currentCompositionVersion: 7,
        recoveryStartedCompositionVersion: 7,
        onApplyState,
      }),
    );

    act(() => {
      result.current.openFromError(
        makeApiError({
          partial_state: partialState,
          failed_turn: makeFailedTurn(),
        }),
      );
    });

    let applyResult: ReturnType<typeof result.current.requestApply> | undefined;
    act(() => {
      applyResult = result.current.requestApply();
    });

    expect(applyResult).toEqual({ applied: true, needsConfirmation: false });
    expect(onApplyState).toHaveBeenCalledWith(partialState);
    expect(result.current.isOpen).toBe(false);
  });

  it("Discard closes local state without applying or calling fetch", () => {
    const onApplyState = vi.fn();
    const onDiscard = vi.fn();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const { result } = renderHook(() =>
      useRecoveryPanel({
        currentCompositionVersion: 7,
        recoveryStartedCompositionVersion: 7,
        onApplyState,
        onDiscard,
      }),
    );

    act(() => {
      result.current.openFromError(
        makeApiError({
          partial_state: makePartialState(),
          failed_turn: makeFailedTurn(),
        }),
      );
      result.current.discard();
    });

    expect(result.current.isOpen).toBe(false);
    expect(result.current.recoveryError).toBeNull();
    expect(result.current.needsApplyConfirmation).toBe(false);
    expect(onApplyState).not.toHaveBeenCalled();
    expect(onDiscard).toHaveBeenCalledTimes(1);
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });
});
