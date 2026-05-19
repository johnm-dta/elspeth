import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api/client";
import { resetStore } from "@/test/store-helpers";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { ChatMessage, CompositionState, Session } from "@/types/index";
import { buildTutorialDraft } from "./TutorialTurn2Describe";

vi.mock("@/api/client", () => ({
  acceptCompositionProposal: vi.fn(),
  createSession: vi.fn(),
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
  source: {
    plugin: "inline_blob",
    options: { rows: [{ url: "dta.gov.au" }] },
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
  vi.mocked(api.fetchComposerPreferences).mockResolvedValue({
    session_id: tutorialSession.id,
    trust_mode: "explicit_approve",
    density_default: "medium",
    updated_at: "2026-05-19T12:00:00Z",
  });
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
    // Rename must happen before optOut and sendMessage — if the user closes
    // the tab between createSession and rename, the session still goes out
    // titled "New session". Pin the ordering.
    const renameOrder = vi.mocked(api.renameSession).mock.invocationCallOrder[0];
    const optOutOrder = vi.mocked(api.optOutOfInterpretations).mock.invocationCallOrder[0];
    const sendOrder = vi.mocked(api.sendMessage).mock.invocationCallOrder[0];
    expect(renameOrder).toBeLessThan(optOutOrder);
    expect(renameOrder).toBeLessThan(sendOrder);
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
    // refreshAll calls api.listInterpretationEvents(sessionId, "all"); the
    // first call (resolveTutorialInterpretations, with filter "pending")
    // must succeed. Mock the "all" call to fail.
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
