import { StrictMode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { resetStore } from "@/test/store-helpers";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { ChatMessage, CompositionState, Session } from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import * as api from "@/api/client";
import { HelloWorldTutorial } from "./HelloWorldTutorial";
import { TutorialTurn4Run } from "./TutorialTurn4Run";

vi.mock("@/api/client", () => ({
  acceptCompositionProposal: vi.fn(),
  createSession: vi.fn(),
  deleteTutorialOrphans: vi.fn().mockResolvedValue({ deleted_count: 0 }),
  fetchComposerPreferences: vi.fn(),
  fetchCompositionProposals: vi.fn(),
  fetchCompositionState: vi.fn(),
  fetchMessages: vi.fn(),
  fetchUserComposerPreferences: vi.fn(),
  getRunAuditSummary: vi.fn(),
  listInterpretationEvents: vi.fn(),
  optOutOfInterpretations: vi.fn(),
  renameSession: vi.fn(),
  resolveInterpretation: vi.fn(),
  runTutorialPipeline: vi.fn(),
  sendMessage: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));

const tutorialSession: Session = {
  id: "session-1",
  title: "New session",
  created_at: "2026-05-19T12:00:00Z",
  updated_at: "2026-05-19T12:00:00Z",
};

const emptySession: Session = {
  id: "session-empty",
  title: "New session",
  created_at: "2026-05-19T12:30:00Z",
  updated_at: "2026-05-19T12:30:00Z",
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
      options: { rows: [{ url: "dta.gov.au" }, { url: "data.gov.au" }] },
    },
  },
  nodes: [
    {
      id: "scrape",
      node_type: "transform",
      plugin: "web_scrape",
      input: "source",
      on_success: "rate",
      on_error: null,
      options: {},
    },
    {
      id: "rate",
      node_type: "transform",
      plugin: "llm_rate",
      input: "scrape",
      on_success: "sink",
      on_error: null,
      options: {},
    },
  ],
  edges: [],
  outputs: [{ name: "ratings", plugin: "jsonl", options: {} }],
  metadata: { name: null, description: null },
};

const pendingInterpretation: InterpretationEvent = {
  id: "event-1",
  session_id: "session-1",
  composition_state_id: "state-1",
  affected_node_id: "rate",
  tool_call_id: "call-1",
  user_term: "cool",
  kind: "vague_term",
  llm_draft: "modern, useful, and interesting",
  accepted_value: null,
  choice: "pending",
  created_at: "2026-05-19T12:00:02Z",
  resolved_at: null,
  actor: "composer-llm",
  interpretation_source: "user_approved",
  model_identifier: "openrouter/openai/gpt-5.4-mini",
  model_version: "openai/gpt-5.4-mini-20260317",
  provider: "openrouter",
  composer_skill_hash: "a".repeat(64),
  arguments_hash: null,
  hash_domain_version: null,
  runtime_model_identifier_at_resolve: null,
  runtime_model_version_at_resolve: null,
  resolved_prompt_template_hash: null,
};

describe("HelloWorldTutorial", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    resetStore(useSessionStore);
    vi.clearAllMocks();
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      tutorialCompletedAt: null,
      tutorialCompleted: false,
    });
    vi.mocked(api.createSession)
      .mockResolvedValueOnce(tutorialSession)
      .mockResolvedValueOnce(emptySession);
    vi.mocked(api.sendMessage).mockResolvedValue({
      message: assistantMessage,
      state: compositionState,
      proposals: [],
    });
    vi.mocked(api.fetchMessages).mockResolvedValue([assistantMessage]);
    vi.mocked(api.fetchCompositionProposals).mockResolvedValue([]);
    vi.mocked(api.fetchComposerPreferences).mockResolvedValue({
      session_id: "session-1",
      trust_mode: "explicit_approve",
      density_default: "medium",
      interpretation_review_disabled: false,
      updated_at: "2026-05-19T12:00:00Z",
    });
    vi.mocked(api.listInterpretationEvents).mockResolvedValue([]);
    vi.mocked(api.optOutOfInterpretations).mockResolvedValue({
      session_id: "session-1",
      interpretation_review_disabled: true,
      opted_out_at: "2026-05-19T12:00:02Z",
    });
    vi.mocked(api.resolveInterpretation).mockResolvedValue({
      event: {
        ...pendingInterpretation,
        choice: "accepted_as_drafted",
        accepted_value: pendingInterpretation.llm_draft,
        resolved_at: "2026-05-19T12:00:03Z",
        arguments_hash: "b".repeat(64),
        hash_domain_version: "v1",
        resolved_prompt_template_hash: "c".repeat(64),
      },
      new_state: compositionState,
    });
    vi.mocked(api.runTutorialPipeline).mockResolvedValue({
      run_id: "run-1",
      output: {
        source_data_hash: "a7f3e2fullhash",
        rows: [
          { url: "dta.gov.au", score: 9, rationale: "bold" },
          { url: "data.gov.au", score: 8, rationale: "useful" },
        ],
      },
      seeded_from_cache: false,
      cache_key: null,
    });
    vi.mocked(api.getRunAuditSummary).mockResolvedValue({
      run_id: "run-1",
      session_id: "session-1",
      llm_call_count: 5,
      output_file_hash: "cafe1234567890",
      started_at: "2026-05-19T12:05:00Z",
      plugin_versions: { web_scrape: "1.0.0", llm_rate: "1.0.0" },
      seeded_from_cache: false,
      cache_key: null,
    });
    vi.mocked(api.renameSession).mockResolvedValue({
      ...tutorialSession,
      title: "hello-world (cool government pages)",
    });
    vi.mocked(api.updateUserComposerPreferences).mockImplementation(async (body) => {
      const updatedAt = "2026-05-19T12:10:00Z";
      return {
        default_mode: body.default_mode ?? usePreferencesStore.getState().defaultMode ?? "guided",
        banner_dismissed_at: null,
        tutorial_completed_at:
          body.tutorial_completed_at === undefined
            ? null
            : body.tutorial_completed_at,
        updated_at: updatedAt,
      };
    });
  });

  it("walks the tutorial through mode selection and graduation", async () => {
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);

    expect(api.deleteTutorialOrphans).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(screen.getByRole("button", { name: "Build it" }));

    expect(await screen.findByText(/Here is what the composer drafted/i)).toBeInTheDocument();
    expect(screen.getByText("dta.gov.au")).toBeInTheDocument();
    expect(api.optOutOfInterpretations).not.toHaveBeenCalled();
    expect(api.resolveInterpretation).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Looks good" }));
    await user.click(screen.getByRole("button", { name: "Looks good, run it" }));

    expect(await screen.findByText("bold")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByText(/This is the audit story/i)).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await user.click(screen.getByRole("radio", { name: /Freeform/i }));
    await user.click(screen.getByRole("button", { name: "Save and go" }));

    await waitFor(() => {
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith(
        { default_mode: "freeform" },
      );
    });
    expect(vi.mocked(api.updateUserComposerPreferences).mock.calls[0]?.[0]).toEqual({
      default_mode: "freeform",
    });
    expect(api.renameSession).toHaveBeenCalledWith(
      "session-1",
      "hello-world (cool government pages)",
    );
    expect(usePreferencesStore.getState().tutorialCompleted).toBe(false);
    expect(useSessionStore.getState().activeSessionId).toBe("session-1");

    expect(
      await screen.findByRole("heading", {
        name: "You're ready to use the composer.",
      }),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Take me to the composer" }),
    );

    await waitFor(() => {
      expect(api.updateUserComposerPreferences).toHaveBeenCalledTimes(2);
    });
    expect(vi.mocked(api.updateUserComposerPreferences).mock.calls[1]?.[0]).toEqual({
      tutorial_completed_at: expect.any(String),
    });
    expect(usePreferencesStore.getState().tutorialCompleted).toBe(true);
    expect(useSessionStore.getState().activeSessionId).toBe("session-empty");
  });

  it("surfaces pending interpretation drafts for explicit review before continuing", async () => {
    const user = userEvent.setup();
    vi.mocked(api.listInterpretationEvents).mockResolvedValue([
      pendingInterpretation,
    ]);

    render(<HelloWorldTutorial />);

    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(screen.getByRole("button", { name: "Build it" }));

    expect(await screen.findByText(/Here is what the composer drafted/i)).toBeInTheDocument();
    expect(screen.getByText("1 assumption to review")).toBeInTheDocument();
    expect(screen.getByText("cool")).toBeInTheDocument();
    expect(api.listInterpretationEvents).toHaveBeenCalledWith(
      "session-1",
      "all",
    );
    expect(api.optOutOfInterpretations).not.toHaveBeenCalled();
    expect(api.resolveInterpretation).not.toHaveBeenCalled();
    expect(useSessionStore.getState().compositionState?.id).toBe("state-1");
  });

  it("settles the run turn under React StrictMode without duplicating the run request", async () => {
    render(
      <StrictMode>
        <TutorialTurn4Run
          sessionId="strict-session"
          prompt="strict prompt"
          onCompleted={() => undefined}
          onCancelled={() => undefined}
          onBack={() => undefined}
        />
      </StrictMode>,
    );

    expect(await screen.findByText("bold")).toBeInTheDocument();
    expect(api.runTutorialPipeline).toHaveBeenCalledTimes(1);
    // Body shape unchanged; the second argument is the AbortSignal owned
    // by the per-(session, prompt) cache entry and is asserted by type
    // rather than by identity (a fresh signal is created per cache miss).
    const [body, signal] = vi.mocked(api.runTutorialPipeline).mock.calls[0];
    expect(body).toEqual({
      session_id: "strict-session",
      prompt: "strict prompt",
    });
    expect(signal).toBeInstanceOf(AbortSignal);
  });
});
