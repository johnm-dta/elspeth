// E2E spec: the first-run tutorial as the staged guided flow.
//
// P7.4/P7.5/P7.6 rewired the tutorial from the old big-bang
// describe→showBuilt→graph→mode turns to: welcome bookend →
// TutorialGuidedShell (the real ChatPanel guided surface, started with the
// "tutorial" profile) → run → audit → graduation. This spec drives that flow
// with the whole API surface route-mocked (no live backend), so it owns the
// guided protocol responses end to end.
//
// The happy path mocks:
//   POST /api/sessions                         → {id} (tutorial then graduation session)
//   POST /api/sessions/{id}/guided/start       → 200, idempotent (profile seed)
//   GET  /api/sessions/{id}/guided             → step_1_source turn
//   POST /api/sessions/{id}/guided/respond     → walks source → sink →
//                                                recipe-apply → step_4_wire,
//                                                then wire-confirm → completed
//   POST /api/tutorial/run                     → the canonical run result
//   GET  .../runs/{id}/audit-story             → the audit story
//
// Wire-stage assertions (M1 + D11/B4 prompt-shield):
//   - the topology + edge-contract overlay renders a `from`/`to` edge cell
//     (a "{from} to {to}" listitem), never from_id/to_id;
//   - the live prompt-injection shield advisory is visible for the canonical
//     web_scrape → llm shape, AND no azure_prompt_shield node appears.
// On terminal=completed the TutorialGuidedShell hands off to the run turn (no
// 409 dead-end), then the run/audit/graduation tail completes.

import { expect, test, type Page, type Route } from "@playwright/test";

const tutorialSession = {
  id: "session-tutorial",
  title: "New session",
  created_at: "2026-05-19T12:00:00Z",
  updated_at: "2026-05-19T12:00:00Z",
};

const graduationSession = {
  id: "session-graduation",
  title: "New session",
  created_at: "2026-05-19T12:10:00Z",
  updated_at: "2026-05-19T12:10:00Z",
};

// Composition state the guided respond/state endpoints return. The canonical
// tutorial pipeline is web_scrape → llm (rate) → jsonl, the shape that triggers
// the prompt-injection shield advisory at the wire stage.
const compositionState = {
  id: "state-1",
  version: 1,
  sources: {
    source: {
      plugin: "inline_blob",
      options: {
        rows: [{ url: "dta.gov.au" }, { url: "data.gov.au" }],
      },
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
      on_success: "ratings",
      on_error: null,
      options: {},
    },
  ],
  edges: [],
  outputs: [{ name: "ratings", plugin: "jsonl", options: {} }],
  metadata: { name: null, description: null },
};

interface GuidedFixtureState {
  sessionPostCount: number;
  guidedRespondCount: number;
  requestLog: string[];
}

// A guided session at a given step with no terminal yet.
function guidedSession(step: string): Record<string, unknown> {
  return {
    step,
    history: [],
    terminal: null,
    chat_history: [],
    chat_turn_seq: 0,
    // null profile == empty/live; the tutorial seeds via /guided/start. The
    // shell reads profile.bookends but tolerates a null profile (defaults true).
    profile: {
      coaching: true,
      bookends: true,
      recipe_match: true,
      advisor_checkpoints: true,
    },
  };
}

function singleSelectTurn(question: string, options: Array<[string, string]>): Record<string, unknown> {
  return {
    type: "single_select",
    step_index: 0,
    payload: {
      question,
      options: options.map(([id, label]) => ({ id, label, hint: null })),
      allow_custom: false,
    },
  };
}

// The step_4_wire turn payload. The canonical web_scrape → llm shape surfaces
// the prompt-injection shield advisory (warnings[].message) and renders the
// source → scrape → rate → output edges with from/to naming.
//
// IMPORTANT: the warning message contains "prompt-injection shield" (the
// advisory) but deliberately NOT the literal "azure_prompt_shield" — the
// count-0 assertion is about the absence of an azure_prompt_shield NODE, and a
// page-wide text match would otherwise trip on the advisory prose.
const wireTurn: Record<string, unknown> = {
  type: "confirm_wiring",
  step_index: 3,
  payload: {
    topology: {
      sources: {
        source: {
          id: "source",
          plugin: "inline_blob",
          on_success: "scrape",
          on_validation_failure: "discard",
        },
      },
      nodes: [
        {
          id: "scrape",
          node_type: "transform",
          plugin: "web_scrape",
          input: "scrape",
          on_success: "rate",
          on_error: "discard",
          routes: null,
          fork_to: null,
          branches: null,
        },
        {
          id: "rate",
          node_type: "transform",
          plugin: "llm_rate",
          input: "rate",
          on_success: "ratings",
          on_error: "discard",
          routes: null,
          fork_to: null,
          branches: null,
        },
      ],
      outputs: [
        {
          id: "output:ratings",
          sink_name: "ratings",
          plugin: "jsonl",
          on_write_failure: "discard",
        },
      ],
    },
    edge_contracts: [
      {
        from: "source",
        to: "scrape",
        producer_guarantees: ["url"],
        consumer_requires: ["url"],
        missing_fields: [],
        satisfied: true,
      },
      {
        from: "scrape",
        to: "rate",
        producer_guarantees: ["url", "html"],
        consumer_requires: ["html"],
        missing_fields: [],
        satisfied: true,
      },
    ],
    semantic_contracts: [],
    warnings: [
      {
        component: "rate",
        severity: "medium",
        // Advisory copy — contains "prompt-injection shield" but not the
        // azure_prompt_shield node literal (see note above).
        message:
          "LLM node 'rate' consumes externally-fetched content from a web_scrape upstream without an authorized prompt-injection shield between them. Continuing without it is allowed.",
      },
    ],
  },
};

function completedSession(): Record<string, unknown> {
  return {
    ...guidedSession("step_4_wire"),
    terminal: {
      kind: "completed",
      reason: null,
      pipeline_yaml: "sources:\n  source:\n    plugin: inline_blob\n",
    },
  };
}

async function installTutorialRoutes(
  page: Page,
  state: GuidedFixtureState,
): Promise<void> {
  await page.route("**/api/**", async (route: Route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    if (path === "/api/system/status" && method === "GET") {
      await route.fulfill({
        json: {
          composer_available: true,
          composer_model: "gpt-5.5",
          composer_provider: "test",
          composer_reason: null,
          composer_missing_keys: [],
          composer_timeout_seconds: 180,
          tutorial_ready: true,
          tutorial_reason: null,
          plugin_policy_readiness: {
            tutorial_ready: true,
            rows: [
              "policy_compilation",
              "required_core",
              "local_capability_configuration",
              "live_health",
              "tutorial_profile",
              "tutorial_required_control_coverage",
            ].map((id) => ({
              id,
              label: id,
              status: "ok",
              summary: "Ready for the tutorial fixture.",
              detail: null,
            })),
          },
        },
      });
      return;
    }

    if (path === "/api/composer-preferences" && method === "GET") {
      await route.fulfill({
        json: {
          default_mode: "guided",
          banner_dismissed_at: null,
          tutorial_completed_at: null,
          updated_at: null,
        },
      });
      return;
    }

    if (path === "/api/composer-preferences" && method === "PATCH") {
      const body = request.postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        json: {
          default_mode: body.default_mode ?? "guided",
          banner_dismissed_at: null,
          tutorial_completed_at:
            typeof body.tutorial_completed_at === "string"
              ? body.tutorial_completed_at
              : null,
          updated_at: "2026-05-19T12:11:00Z",
        },
      });
      return;
    }

    if (path === "/api/sessions" && method === "GET") {
      await route.fulfill({
        json: [
          {
            ...tutorialSession,
            title: "First-run tutorial",
            updated_at: "2026-05-19T12:11:00Z",
          },
        ],
      });
      return;
    }

    if (path === "/api/sessions" && method === "POST") {
      state.sessionPostCount += 1;
      state.requestLog.push(`session-post:${state.sessionPostCount}`);
      await route.fulfill({
        json: state.sessionPostCount === 1 ? tutorialSession : graduationSession,
      });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}` && method === "PATCH") {
      const body = request.postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        json: {
          ...tutorialSession,
          title: body.title,
          updated_at: "2026-05-19T12:11:00Z",
        },
      });
      return;
    }

    // ── Guided protocol ─────────────────────────────────────────────────────
    if (
      path === `/api/sessions/${tutorialSession.id}/guided/start` &&
      method === "POST"
    ) {
      state.requestLog.push("guided-start");
      await route.fulfill({
        json: {
          guided_session: guidedSession("step_1_source"),
          next_turn: singleSelectTurn("Which data source would you like to use?", [
            ["inline_blob", "inline_blob"],
            ["csv", "csv"],
          ]),
          terminal: null,
          composition_state: null,
        },
      });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/guided/tutorial-sample` &&
      method === "GET"
    ) {
      await route.fulfill({
        json: {
          sample_urls: [
            "https://johnm-dta.github.io/elspeth/tutorial-site/project-1.html",
            "https://johnm-dta.github.io/elspeth/tutorial-site/project-2.html",
            "https://johnm-dta.github.io/elspeth/tutorial-site/project-3.html",
          ],
        },
      });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/guided` &&
      method === "GET"
    ) {
      await route.fulfill({
        json: {
          guided_session: guidedSession("step_1_source"),
          next_turn: singleSelectTurn("Which data source would you like to use?", [
            ["inline_blob", "inline_blob"],
            ["csv", "csv"],
          ]),
          terminal: null,
          composition_state: null,
        },
      });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/guided/chat` &&
      method === "POST"
    ) {
      state.guidedRespondCount += 1;
      const n = state.guidedRespondCount;
      state.requestLog.push(`guided-chat:${n}`);
      let next: Record<string, unknown> | null;
      let session = guidedSession("step_2_sink");
      if (n === 1) {
        next = singleSelectTurn("What format should the output be in?", [
          ["jsonl", "jsonl"],
          ["json", "json"],
        ]);
      } else {
        next = {
          type: "recipe_offer",
          step_index: 2,
          payload: {
            mode: "recipe_decision",
            knobs: { fields: [] },
            prefilled: {},
            recipe_context: {
              recipe_name: "web-scrape-llm-rate-jsonl",
              description: "Scrape each URL and rate the page.",
              alternatives: [],
            },
          },
        };
        session = guidedSession("step_2_5_recipe_match");
      }
      await route.fulfill({
        json: {
          assistant_message:
            n === 1
              ? "I set this up as an inline source."
              : "I set up a JSONL output.",
          assistant_message_kind: "assistant",
          guided_session: session,
          next_turn: next,
          terminal: null,
          composition_state: compositionState,
        },
      });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/guided/respond` &&
      method === "POST"
    ) {
      state.guidedRespondCount += 1;
      const n = state.guidedRespondCount;
      state.requestLog.push(`guided-respond:${n}`);
      // Drive a deterministic walk: source → sink → recipe-apply → wire →
      // completed. The mock ignores the request body and advances by count.
      let next: Record<string, unknown> | null;
      let session = guidedSession("step_2_sink");
      if (n === 1) {
        // after source pick → sink pick turn
        next = singleSelectTurn("What format should the output be in?", [
          ["jsonl", "jsonl"],
          ["json", "json"],
        ]);
        session = guidedSession("step_2_sink");
      } else if (n === 2) {
        // after sink pick → recipe offer
        next = {
          type: "recipe_offer",
          step_index: 2,
          payload: {
            mode: "recipe_decision",
            knobs: { fields: [] },
            prefilled: {},
            recipe_context: {
              recipe_name: "web-scrape-llm-rate-jsonl",
              description: "Scrape each URL and rate the page.",
              alternatives: [],
            },
          },
        };
        session = guidedSession("step_2_5_recipe_match");
      } else if (n === 3) {
        // after apply recipe → wire turn
        next = wireTurn;
        session = guidedSession("step_4_wire");
      } else {
        // wire confirm → completed
        next = null;
        session = completedSession();
      }
      const terminal =
        next === null
          ? (session.terminal as Record<string, unknown>)
          : null;
      await route.fulfill({
        json: {
          guided_session: session,
          next_turn: next,
          terminal,
          composition_state: compositionState,
        },
      });
      return;
    }

    // Interpretation events: the canonical mock has none pending (the wire
    // confirm is not blocked by D12 in this fixture).
    if (
      path === `/api/sessions/${tutorialSession.id}/interpretations` &&
      method === "GET"
    ) {
      await route.fulfill({ json: { events: [] } });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/composer/preferences` &&
      method === "GET"
    ) {
      await route.fulfill({
        json: {
          session_id: tutorialSession.id,
          trust_mode: "explicit_approve",
          density_default: "medium",
          interpretation_review_disabled: false,
          updated_at: "2026-05-19T12:00:00Z",
        },
      });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/composer-progress` &&
      method === "GET"
    ) {
      await route.fulfill({
        json: {
          session_id: tutorialSession.id,
          request_id: null,
          phase: "idle",
          headline: "Idle.",
          evidence: [],
          likely_next: null,
          reason: "composer_idle",
          updated_at: "2026-05-19T12:00:00Z",
        },
      });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/state` && method === "GET") {
      await route.fulfill({ json: compositionState });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/state/versions` &&
      method === "GET"
    ) {
      await route.fulfill({
        json: [
          {
            id: compositionState.id,
            version: compositionState.version,
            created_at: "2026-05-19T12:00:00Z",
            node_count: compositionState.nodes.length,
          },
        ],
      });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/messages` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/proposals` &&
      method === "GET"
    ) {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/validate` && method === "POST") {
      await route.fulfill({
        json: {
          is_valid: true,
          summary: "Tutorial pipeline is valid.",
          checks: [],
          errors: [],
          warnings: [],
          semantic_contracts: [],
        },
      });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/audit-readiness` &&
      method === "GET"
    ) {
      await route.fulfill({
        json: {
          session_id: tutorialSession.id,
          composition_version: 1,
          checked_at: "2026-05-19T12:11:00Z",
          rows: [],
          validation_result: {
            is_valid: true,
            summary: "Tutorial pipeline is valid.",
            checks: [],
            errors: [],
            warnings: [],
            semantic_contracts: [],
          },
        },
      });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/runs` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/blobs` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === `/api/sessions/${graduationSession.id}/runs` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === "/api/tutorial/orphans" && method === "DELETE") {
      await route.fulfill({ json: { deleted_count: 0 } });
      return;
    }

    if (path === "/api/tutorial/run" && method === "POST") {
      await route.fulfill({
        json: {
          run_id: "run-1",
          output: {
            source_data_hash: "a7f3e2fullhash",
            rows: [
              { url: "dta.gov.au", score: 9, rationale: "bold" },
              { url: "data.gov.au", score: 8, rationale: "useful" },
            ],
            discarded_row_count: 0,
          },
          seeded_from_cache: false,
          cache_key: null,
        },
      });
      return;
    }

    if (
      path === `/api/sessions/${tutorialSession.id}/runs/run-1/audit-story` &&
      method === "GET"
    ) {
      await route.fulfill({
        json: {
          run_id: "run-1",
          session_id: tutorialSession.id,
          llm_call_count: 5,
          source_data_hash: "a7f3e2fullhash",
          started_at: "2026-05-19T12:05:00Z",
          plugin_versions: { web_scrape: "1.0.0", llm_rate: "1.0.0" },
          seeded_from_cache: false,
          cache_key: null,
        },
      });
      return;
    }

    await route.continue();
  });
}

test.describe("first-run tutorial (staged guided flow)", () => {
  test("welcome → guided (source/sink/recipe/wire) → run → audit → graduation", async ({
    page,
  }) => {
    const state: GuidedFixtureState = {
      sessionPostCount: 0,
      guidedRespondCount: 0,
      requestLog: [],
    };
    await installTutorialRoutes(page, state);

    // ── Welcome bookend ──────────────────────────────────────────────────────
    await page.goto("/");
    await expect(
      page.getByRole("main", { name: /first-run tutorial/i }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: /Welcome to ELSPETH/i }),
    ).toBeVisible();

    // Start mounts the guided surface (chat-panel--guided).
    await page.getByRole("button", { name: "Let's go" }).click();
    await expect(page.getByLabel(/guided composer/i)).toBeVisible();

    // ── Step 1 source ────────────────────────────────────────────────────────
    await page.getByRole("button", { name: "Send message", exact: true }).click();
    // ── Step 2 sink ──────────────────────────────────────────────────────────
    await expect(page.getByText(/Save the pipeline's results/i)).toBeVisible();
    await page.getByRole("button", { name: "Send message", exact: true }).click();
    // ── Step 2.5 recipe-apply ────────────────────────────────────────────────
    await expect(
      page.getByRole("button", { name: "Apply recipe", exact: true }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Apply recipe", exact: true }).click();

    // ── Step 4 wire stage: topology + edge-contract overlay (M1 from/to) ─────
    await expect(page.getByRole("heading", { name: "Review wiring" })).toBeVisible();
    await expect(
      page.getByRole("listitem", { name: "Source to Fetch step" }),
    ).toBeVisible();
    await expect(
      page.getByRole("listitem", { name: "Fetch step to Llm Rate step" }),
    ).toBeVisible();
    await expect(page.getByText(/Source\s+→\s+Fetch step\s+—\s+connected/)).toBeVisible();
    await expect(page.getByText(/Fetch step\s+→\s+Llm Rate step\s+—\s+connected/)).toBeVisible();
    // M1 guard: post-M1 naming, never from_id/to_id.
    await expect(
      page.getByRole("listitem", { name: /from_id|to_id/ }),
    ).toHaveCount(0);

    // The wire validation payload surfaces the live prompt-shield advisory for
    // the canonical web_scrape → llm shape (D11/B4 rev-4), and must NOT contain
    // an azure_prompt_shield node. The mock seeds the advisory (without the
    // azure_prompt_shield literal) and no such node in the topology.
    await expect(page.getByText(/prompt-injection shield/i)).toBeVisible();
    await expect(page.locator("text=azure_prompt_shield")).toHaveCount(0);

    // ── Wire confirm → completed → run turn (no 409 dead-end) ────────────────
    await page.getByRole("button", { name: "Confirm wiring", exact: true }).click();
    // TutorialGuidedShell handed off to the run turn on terminal=completed.
    await expect(
      page.getByRole("heading", { name: /Running your pipeline/i }),
    ).toBeVisible();
    await expect(page.getByText("bold")).toBeVisible();

    // ── Audit story ──────────────────────────────────────────────────────────
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByText(/This is the audit story/i)).toBeVisible();
    await expect(
      page
        .locator(".tutorial-audit-list div", { hasText: "LLM calls" })
        .getByText("5", { exact: true }),
    ).toBeVisible();

    // ── Graduation ───────────────────────────────────────────────────────────
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(
      page.getByRole("heading", { name: "You're ready to use the composer." }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Take me to the composer" }).click();

    // Graduation renamed the tutorial session, saved guided default, and
    // landed on the built pipeline instead of creating a fresh empty session.
    await expect.poll(() => state.sessionPostCount).toBe(1);
    await expect(
      page.getByRole("button", { name: /Session switcher: First-run tutorial/i }),
    ).toBeVisible();
    expect(state.requestLog).toContain("guided-start");
  });

  test("skip from welcome lands directly on graduation", async ({ page }) => {
    const state: GuidedFixtureState = {
      sessionPostCount: 0,
      guidedRespondCount: 0,
      requestLog: [],
    };
    await installTutorialRoutes(page, state);

    await page.goto("/");
    await page.getByRole("button", { name: "Skip the tutorial" }).click();
    await expect(
      page.getByRole("heading", { name: "You're ready to use the composer." }),
    ).toBeVisible();
  });

  test("welcome surfaces the privacy preamble before starting", async ({
    page,
  }) => {
    const state: GuidedFixtureState = {
      sessionPostCount: 0,
      guidedRespondCount: 0,
      requestLog: [],
    };
    await installTutorialRoutes(page, state);

    await page.goto("/");
    await expect(
      page.getByText(
        /This step calls the configured LLM and fetches pages over the network/i,
      ),
    ).toBeVisible();
  });
});
