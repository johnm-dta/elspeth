import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ChatPanel } from "./ChatPanel";
import { useSessionStore } from "@/stores/sessionStore";
import { useBlobStore } from "@/stores/blobStore";
import { useExecutionStore } from "@/stores/executionStore";
import { useInlineSourceStore } from "@/stores/inlineSourceStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";
import type { GetGuidedResponse, GuidedSession } from "@/types/guided";

const apiMocks = vi.hoisted(() => ({
  startGuidedSession: vi.fn(),
  reconcileGuidedStartOperation: vi.fn(),
  getGuided: vi.fn(),
  listInterpretationEvents: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/client")>();
  return { ...actual, ...apiMocks };
});

vi.mock("@/hooks/useComposer", () => ({
  useComposer: () => ({
    sendMessage: vi.fn(),
    retryMessage: vi.fn(),
    cancelComposition: vi.fn(),
    isComposing: false,
    error: null,
    errorDetails: null,
  }),
}));

vi.mock("./AcknowledgementStack", () => ({
  AcknowledgementLiveRegion: () => null,
  AcknowledgementStack: () => null,
  useHasPendingGuidedInterpretations: () => false,
  usePendingAcknowledgements: () => [],
}));
vi.mock("./guided/GuidedChatHistory", () => ({ GuidedChatHistory: () => null }));
vi.mock("./guided/GuidedHistory", () => ({ GuidedHistory: () => null }));
vi.mock("./guided/GuidedTurn", () => ({ GuidedTurn: () => null }));
vi.mock("./guided/GuidedPendingStrip", () => ({
  GuidedPendingStrip: ({ onStop }: { onStop: () => void }) => (
    <div role="status">
      Guided setup running
      <button type="button" onClick={onStop}>Stop</button>
    </div>
  ),
}));
vi.mock("./guided/PipelineGloss", () => ({ PipelineGloss: () => null }));
vi.mock("./guided/PipelineValidationSummary", () => ({ PipelineValidationSummary: () => null }));
vi.mock("./guided/ModeSwitchButton", () => ({ ModeSwitchButton: () => null }));
vi.mock("./ModelChip", () => ({ ModelChip: () => null }));
vi.mock("@/components/sidebar/GraphMiniView", () => ({ GraphMiniView: () => null }));
vi.mock("@/components/execution/InlineRunResults", () => ({ InlineRunResults: () => null }));

const SESSION_ID = "00000000-0000-4000-8000-000000000901";
const guidedSession: GuidedSession = {
  step: "step_3_transforms",
  history: [],
  terminal: null,
  chat_history: [],
  chat_turn_seq: 0,
  profile: null,
};
const completedResponse: GetGuidedResponse = {
  guided_session: guidedSession,
  next_turn: null,
  terminal: null,
  composition_state: {
    id: "00000000-0000-4000-8000-000000000902",
    session_id: SESSION_ID,
    version: 1,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
    is_valid: false,
    validation_errors: null,
    validation_warnings: null,
    validation_suggestions: null,
    derived_from_state_id: null,
    created_at: "2026-07-20T00:00:00Z",
    composer_meta: null,
    plugin_policy_findings: [],
  },
};

describe("ChatPanel cold guided-start recovery with the real ChatInput", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    window.sessionStorage.clear();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    resetStore(useBlobStore);
    resetStore(useExecutionStore);
    resetStore(useInlineSourceStore);
    resetStore(useInterpretationEventsStore);
    resetStore(usePreferencesStore);
    useInterpretationEventsStore.setState({
      refreshAll: vi.fn().mockResolvedValue(undefined),
    } as never);
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      sessions: [{
        id: SESSION_ID,
        title: "Cold guided start",
        created_at: "2026-07-20T00:00:00Z",
        updated_at: "2026-07-20T00:00:00Z",
      }],
      guidedSession,
      guidedNextTurn: null,
      compositionState: null,
      composeTimeoutReady: true,
    });
  });

  it("loses the submitted text immediately, then permits revised text after authoritative cancellation", async () => {
    const user = userEvent.setup();
    apiMocks.startGuidedSession
      .mockImplementationOnce(
        (_sessionId: string, _command: unknown, signal?: AbortSignal) =>
          new Promise((_resolve, reject) => {
            signal?.addEventListener("abort", () => reject(signal.reason));
          }),
      )
      .mockResolvedValueOnce(completedResponse);
    apiMocks.reconcileGuidedStartOperation.mockResolvedValueOnce({
      status: "failed",
      failure_code: "request_cancelled",
    });

    render(<ChatPanel />);
    const textarea = screen.getByLabelText("Message input");
    await user.type(textarea, "Original prompt that is not persisted");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(screen.queryByLabelText("Message input")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Guided setup running");
    expect(window.sessionStorage.getItem("elspeth_guided_operation_retries_v2")).not.toContain(
      "Original prompt that is not persisted",
    );

    await user.click(screen.getByRole("button", { name: "Stop" }));
    const recoveredTextarea = await screen.findByLabelText("Message input");
    expect(recoveredTextarea).toHaveValue("");
    expect(useSessionStore.getState().error).toMatch(/revise your request and send it again/i);

    await user.type(recoveredTextarea, "Revised prompt");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => expect(apiMocks.startGuidedSession).toHaveBeenCalledTimes(2));
    const firstOperationId = apiMocks.startGuidedSession.mock.calls[0]?.[1].operationId;
    const secondOperationId = apiMocks.startGuidedSession.mock.calls[1]?.[1].operationId;
    expect(secondOperationId).not.toBe(firstOperationId);
    expect(apiMocks.startGuidedSession.mock.calls[1]?.[1]).toEqual(
      expect.objectContaining({ intent: "Revised prompt" }),
    );
    expect(window.sessionStorage.getItem("elspeth_guided_operation_retries_v2") ?? "").not.toContain(
      "Revised prompt",
    );
  });
});
