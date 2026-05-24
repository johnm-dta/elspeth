import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api/client";
import { resetStore } from "@/test/store-helpers";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import type {
  ChatMessage,
  ComposerProgressSnapshot,
  CompositionState,
  Session,
} from "@/types/index";
import { buildTutorialDraft, TutorialTurn2Describe } from "./TutorialTurn2Describe";

vi.mock("@/api/client", () => ({
  acceptCompositionProposal: vi.fn(),
  createSession: vi.fn(),
  fetchComposerProgress: vi.fn(),
  fetchComposerPreferences: vi.fn(),
  fetchCompositionProposals: vi.fn(),
  fetchCompositionState: vi.fn(),
  fetchMessages: vi.fn(),
  listInterpretationEvents: vi.fn(),
  optOutOfInterpretations: vi.fn(),
  renameSession: vi.fn(),
  resolveInterpretation: vi.fn(),
  sendMessage: vi.fn(),
}));

const tutorialSession: Session = {
  id: "session-1",
  title: "New session",
  created_at: "2026-05-19T12:00:00Z",
  updated_at: "2026-05-19T12:00:00Z",
};

const assistantMessage: ChatMessage = {
  id: "message-1",
  session_id: "session-1",
  role: "assistant",
  content: "Drafted a pipeline.",
  tool_calls: null,
  created_at: "2026-05-19T12:00:01Z",
};

const compositionState: CompositionState = {
  id: "state-1",
  version: 1,
  sources: {
    source: {
      plugin: "inline_blob",
      options: { rows: [{ url: "dta.gov.au" }] },
    },
  },
  nodes: [
    {
      id: "rate",
      node_type: "transform",
      plugin: "llm_rate",
      input: "source",
      on_success: "sink",
      on_error: null,
      options: {},
    },
  ],
  edges: [],
  outputs: [{ name: "ratings", plugin: "jsonl", options: {} }],
  metadata: { name: null, description: null },
};

const composerProgress: ComposerProgressSnapshot = {
  session_id: tutorialSession.id,
  request_id: "message-1",
  phase: "using_tools",
  headline: "The model is choosing pipeline components.",
  evidence: ["Checking source, transform, and output schemas."],
  likely_next: "ELSPETH will draft the pipeline and validate the result.",
  reason: null,
  updated_at: "2026-05-19T12:00:03Z",
};

const composerPreferences = {
  session_id: tutorialSession.id,
  trust_mode: "explicit_approve",
  density_default: "medium",
  interpretation_review_disabled: false,
  updated_at: "2026-05-19T12:00:00Z",
} as const;

function primeHappyPath(): void {
  vi.mocked(api.createSession).mockResolvedValue(tutorialSession);
  vi.mocked(api.renameSession).mockResolvedValue({
    ...tutorialSession,
    title: "hello-world (pending)",
  });
  vi.mocked(api.optOutOfInterpretations).mockResolvedValue({
    session_id: tutorialSession.id,
    interpretation_review_disabled: true,
    opted_out_at: "2026-05-19T12:00:02Z",
  });
  vi.mocked(api.sendMessage).mockResolvedValue({
    message: assistantMessage,
    state: compositionState,
    proposals: [],
  });
  vi.mocked(api.fetchCompositionState).mockResolvedValue(compositionState);
  vi.mocked(api.listInterpretationEvents).mockResolvedValue([]);
  vi.mocked(api.fetchMessages).mockResolvedValue([assistantMessage]);
  vi.mocked(api.fetchCompositionProposals).mockResolvedValue([]);
  vi.mocked(api.fetchComposerPreferences).mockResolvedValue(composerPreferences);
  vi.mocked(api.fetchComposerProgress).mockResolvedValue({
    ...composerProgress,
    phase: "idle",
    headline: "",
    evidence: [],
    likely_next: null,
  });
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
} {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("buildTutorialDraft — surface backend errors, never silent-fallback", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useInterpretationEventsStore);
    vi.clearAllMocks();
  });

  it("happy path resolves with build result and publishes to session store", async () => {
    primeHappyPath();
    const result = await buildTutorialDraft("rate cool gov pages");
    expect(result.sessionId).toBe(tutorialSession.id);
    expect(result.prompt).toBe("rate cool gov pages");
    // sanity check: store was populated, not left with silent defaults
    const stored = useSessionStore.getState();
    expect(stored.messages).toEqual([assistantMessage]);
    expect(stored.compositionState).toEqual(compositionState);
  });

  it("tags the session with hello-world prefix before any other call so orphan cleanup catches abandoned sessions", async () => {
    primeHappyPath();
    await buildTutorialDraft("rate cool gov pages");

    // P2-4: orphan cleanup filters by the "hello-world (" title prefix. The
    // backend createSession defaults the title to "New session"; any abandon
    // between Turn 2 (create) and Turn 6 (final rename) would otherwise leave
    // an untagged session that cleanup misses.
    expect(api.renameSession).toHaveBeenCalledWith(
      tutorialSession.id,
      "hello-world (pending)",
    );
    // Rename must happen before sendMessage — if the user closes the tab
    // between createSession and rename, the session still goes out titled
    // "New session". Pin the ordering.
    const renameOrder = vi.mocked(api.renameSession).mock.invocationCallOrder[0];
    const sendOrder = vi.mocked(api.sendMessage).mock.invocationCallOrder[0];
    expect(renameOrder).toBeLessThan(sendOrder);
  });

  it("keeps tutorial interpretation review enabled and does not auto-resolve pending interpretation events", async () => {
    primeHappyPath();
    await buildTutorialDraft("rate cool gov pages");

    expect(api.optOutOfInterpretations).not.toHaveBeenCalled();
    expect(api.resolveInterpretation).not.toHaveBeenCalled();
  });

  it("rejects when composer preferences show interpretation review disabled", async () => {
    primeHappyPath();
    const disabledComposerPreferences = {
      ...composerPreferences,
      interpretation_review_disabled: true,
    } as const;
    vi.mocked(api.fetchComposerPreferences).mockResolvedValueOnce(
      disabledComposerPreferences,
    );

    await expect(buildTutorialDraft("rate cool gov pages")).rejects.toThrow(
      "tutorial sessions must not have interpretation review disabled",
    );
  });

  it("rejects when composer preferences omit the interpretation-review contract field", async () => {
    primeHappyPath();
    const { interpretation_review_disabled: _removed, ...driftedPreferences } =
      composerPreferences;
    vi.mocked(api.fetchComposerPreferences).mockResolvedValueOnce(
      driftedPreferences as Awaited<ReturnType<typeof api.fetchComposerPreferences>>,
    );

    await expect(buildTutorialDraft("rate cool gov pages")).rejects.toThrow(
      "composer preferences response missing interpretation_review_disabled",
    );
  });

  it("rejects when api.fetchMessages fails — does not silently substitute the inline assistantMessage", async () => {
    primeHappyPath();
    vi.mocked(api.fetchMessages).mockRejectedValueOnce(
      new Error("HTTP 401: session expired"),
    );
    await expect(buildTutorialDraft("rate cool gov pages")).rejects.toThrow(
      /401|session expired/,
    );
  });

  it("rejects when api.fetchCompositionProposals fails — does not silently substitute response.proposals", async () => {
    primeHappyPath();
    vi.mocked(api.fetchCompositionProposals).mockRejectedValueOnce(
      new Error("HTTP 500: upstream service unavailable"),
    );
    await expect(buildTutorialDraft("rate cool gov pages")).rejects.toThrow(
      /500|upstream service unavailable/,
    );
  });

  it("rejects when api.fetchComposerPreferences fails — does not silently substitute null", async () => {
    primeHappyPath();
    vi.mocked(api.fetchComposerPreferences).mockRejectedValueOnce(
      new Error("HTTP 503: degraded mode"),
    );
    await expect(buildTutorialDraft("rate cool gov pages")).rejects.toThrow(
      /503|degraded mode/,
    );
  });

  it("rejects when interpretationEventsStore.refreshAll fails — does not silently desync the store", async () => {
    // refreshAll calls api.listInterpretationEvents(sessionId, "all").
    // Mock that call to fail so the tutorial does not publish a session
    // whose interpretation-event store is out of sync with the backend.
    primeHappyPath();
    vi.mocked(api.listInterpretationEvents).mockImplementation(
      async (_sessionId: string, status?: "pending" | "all") => {
        if (status === "all") {
          throw new Error("HTTP 502: bad gateway");
        }
        return [];
      },
    );
    await expect(buildTutorialDraft("rate cool gov pages")).rejects.toThrow(
      /502|bad gateway/,
    );
  });
});

describe("TutorialTurn2Describe pending progress", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useInterpretationEventsStore);
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it("renders the tutorial shimmer while the LLM draft is still pending", async () => {
    primeHappyPath();
    vi.mocked(api.sendMessage).mockReturnValue(
      deferred<Awaited<ReturnType<typeof api.sendMessage>>>().promise,
    );

    render(
      <TutorialTurn2Describe
        initialPrompt="rate cool gov pages"
        onBuilt={vi.fn()}
        onBack={vi.fn()}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Build it" }));

    expect(await screen.findByRole("status")).toHaveTextContent(
      "Asking the composer to draft the pipeline.",
    );
    expect(document.querySelector(".tutorial-progress-bar")).toBeInTheDocument();
  });

  it("polls tutorial composer progress and renders provider-safe backend activity", async () => {
    primeHappyPath();
    const send = deferred<Awaited<ReturnType<typeof api.sendMessage>>>();
    vi.mocked(api.sendMessage).mockReturnValue(send.promise);
    vi.mocked(api.fetchComposerProgress).mockResolvedValue(composerProgress);

    render(
      <TutorialTurn2Describe
        initialPrompt="rate cool gov pages"
        onBuilt={vi.fn()}
        onBack={vi.fn()}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Build it" }));
    await waitFor(() =>
      expect(api.fetchComposerProgress).toHaveBeenCalledWith(tutorialSession.id),
    );

    expect(await screen.findByText(composerProgress.headline)).toBeInTheDocument();
    expect(screen.getByText(composerProgress.evidence[0])).toBeInTheDocument();
    expect(screen.getByText(composerProgress.likely_next ?? "")).toBeInTheDocument();
  });
});
