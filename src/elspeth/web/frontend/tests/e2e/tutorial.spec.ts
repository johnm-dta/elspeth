import { expect, test, type Page, type Route } from "@playwright/test";

interface RouteState {
  sessionPostCount: number;
  completionPatchSeen: boolean;
  renamedSession: boolean;
}

const tutorialSession = {
  id: "session-tutorial",
  title: "New session",
  created_at: "2026-05-19T12:00:00Z",
  updated_at: "2026-05-19T12:00:00Z",
};

const emptySession = {
  id: "session-empty",
  title: "New session",
  created_at: "2026-05-19T12:10:00Z",
  updated_at: "2026-05-19T12:10:00Z",
};

const compositionState = {
  id: "state-1",
  version: 1,
  source: {
    plugin: "inline_blob",
    options: {
      rows: [
        { url: "dta.gov.au" },
        { url: "data.gov.au" },
        { url: "ato.gov.au" },
        { url: "finance.gov.au" },
        { url: "australia.gov.au" },
      ],
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

async function installTutorialRoutes(page: Page, state: RouteState): Promise<void> {
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
      state.completionPatchSeen =
        body.default_mode === "freeform" &&
        typeof body.tutorial_completed_at === "string";
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
      await route.fulfill({ json: [] });
      return;
    }

    if (path === "/api/sessions" && method === "POST") {
      state.sessionPostCount += 1;
      await route.fulfill({
        json: state.sessionPostCount === 1 ? tutorialSession : emptySession,
      });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}` && method === "PATCH") {
      const body = request.postDataJSON() as Record<string, unknown>;
      state.renamedSession =
        body.title === "hello-world (cool government pages)";
      await route.fulfill({
        json: {
          ...tutorialSession,
          title: body.title,
          updated_at: "2026-05-19T12:11:00Z",
        },
      });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/messages`) {
      if (method === "POST") {
        await route.fulfill({
          json: {
            message: {
              id: "message-1",
              session_id: tutorialSession.id,
              role: "assistant",
              content: "Drafted a tutorial pipeline.",
              tool_calls: null,
              created_at: "2026-05-19T12:00:01Z",
            },
            state: compositionState,
            proposals: [],
          },
        });
        return;
      }
      if (method === "GET") {
        await route.fulfill({
          json: [
            {
              id: "message-1",
              session_id: tutorialSession.id,
              role: "assistant",
              content: "Drafted a tutorial pipeline.",
              tool_calls: null,
              created_at: "2026-05-19T12:00:01Z",
            },
          ],
        });
        return;
      }
    }

    if (path === `/api/sessions/${tutorialSession.id}/proposals` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/composer/preferences` && method === "GET") {
      await route.fulfill({
        json: {
          session_id: tutorialSession.id,
          trust_mode: "explicit_approve",
          density_default: "medium",
          updated_at: "2026-05-19T12:00:00Z",
        },
      });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/composer-progress` && method === "GET") {
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

    if (path === `/api/sessions/${tutorialSession.id}/state/versions` && method === "GET") {
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

    if (path === `/api/sessions/${tutorialSession.id}/interpretations` && method === "GET") {
      await route.fulfill({ json: { events: [] } });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/interpretations/opt_out` && method === "POST") {
      await route.fulfill({
        json: {
          session_id: tutorialSession.id,
          interpretation_review_disabled: true,
          opted_out_at: "2026-05-19T12:00:02Z",
        },
      });
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

    if (path === `/api/sessions/${tutorialSession.id}/runs` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/audit-readiness` && method === "GET") {
      await route.fulfill({
        json: {
          session_id: tutorialSession.id,
          composition_version: compositionState.version,
          checked_at: "2026-05-19T12:00:00Z",
          rows: [
            {
              id: "validation",
              label: "Validation",
              status: "ok",
              summary: "Tutorial pipeline validates.",
              detail: null,
              component_ids: [],
            },
            {
              id: "plugin_trust",
              label: "Plugin trust",
              status: "ok",
              summary: "Tutorial plugins are trusted.",
              detail: null,
              component_ids: [],
            },
            {
              id: "provenance",
              label: "Provenance",
              status: "ok",
              summary: "Tutorial provenance is present.",
              detail: null,
              component_ids: [],
            },
            {
              id: "retention",
              label: "Retention",
              status: "ok",
              summary: "Tutorial retention is configured.",
              detail: null,
              component_ids: [],
            },
            {
              id: "llm_interpretations",
              label: "LLM interpretations",
              status: "not_applicable",
              summary: "No unresolved interpretations.",
              detail: null,
              component_ids: [],
            },
            {
              id: "secrets",
              label: "Secrets",
              status: "not_applicable",
              summary: "No tutorial secrets required.",
              detail: null,
              component_ids: [],
            },
          ],
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

    if (path === `/api/sessions/${tutorialSession.id}/blobs` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === `/api/sessions/${emptySession.id}/runs` && method === "GET") {
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
          },
          seeded_from_cache: false,
          cache_key: null,
        },
      });
      return;
    }

    if (path === `/api/sessions/${tutorialSession.id}/runs/run-1/audit-story` && method === "GET") {
      await route.fulfill({
        json: {
          run_id: "run-1",
          session_id: tutorialSession.id,
          llm_call_count: 5,
          output_file_hash: "cafe1234567890",
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

async function installTutorialRoutesWithSlowRun(
  page: Page,
  state: RouteState,
): Promise<void> {
  // Same routes as installTutorialRoutes but holds the /api/tutorial/run
  // response open for ~7s so the 5s cancel-button timer can fire and the
  // user can click cancel before the run resolves.
  await installTutorialRoutes(page, state);
  await page.route("**/api/tutorial/run", async (route) => {
    await new Promise<void>((resolve) => setTimeout(resolve, 7_000));
    await route.fulfill({
      json: {
        run_id: "run-slow",
        output: {
          source_data_hash: "neverreachedhash",
          rows: [],
        },
        seeded_from_cache: false,
        cache_key: null,
      },
    });
  });
}

test.describe("first-run tutorial", () => {
  test("walks the tutorial flow through final preference selection", async ({
    page,
  }) => {
    const state: RouteState = {
      sessionPostCount: 0,
      completionPatchSeen: false,
      renamedSession: false,
    };
    await installTutorialRoutes(page, state);

    await page.goto("/");
    await expect(page.getByRole("main", { name: /first-run tutorial/i })).toBeVisible();

    await page.getByRole("button", { name: "Let's go" }).click();
    await page.getByRole("button", { name: "Build it" }).click();
    await expect(page.getByText(/Here is what the composer drafted/i)).toBeVisible();
    await expect(page.getByText("dta.gov.au")).toBeVisible();

    await page.getByRole("button", { name: "Show me the graph" }).click();
    await page.getByRole("button", { name: "Looks good, run it" }).click();
    await expect(page.getByText("bold")).toBeVisible();

    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByText(/This is the audit story/i)).toBeVisible();
    await expect(
      page
        .locator(".tutorial-audit-list div", { hasText: "LLM calls" })
        .getByText("5", { exact: true }),
    ).toBeVisible();

    await page.getByRole("button", { name: "Continue" }).click();
    await page.getByRole("radio", { name: /Freeform/i }).click();
    await page.getByRole("button", { name: "Save and go" }).click();

    await expect.poll(() => state.completionPatchSeen).toBe(true);
    expect(state.renamedSession).toBe(true);
    expect(state.sessionPostCount).toBe(2);
  });

  test("Edit prompt button on Turn 2b returns to Describe with prompt preserved", async ({
    page,
  }) => {
    const state: RouteState = {
      sessionPostCount: 0,
      completionPatchSeen: false,
      renamedSession: false,
    };
    await installTutorialRoutes(page, state);

    await page.goto("/");
    await page.getByRole("button", { name: "Let's go" }).click();

    // Edit the prompt before building so we can confirm it survives the
    // back trip.
    const promptInput = page.getByLabel("Pipeline description");
    await promptInput.fill("a custom prompt the user typed");

    await page.getByRole("button", { name: "Build it" }).click();
    await expect(page.getByText(/Here is what the composer drafted/i)).toBeVisible();

    // The Edit-prompt button is the back-to-describe affordance on Turn 2b.
    await page.getByRole("button", { name: "Edit prompt" }).click();

    // We're back on Describe with the user's prompt intact.
    await expect(
      page.getByRole("heading", { name: /Describe your pipeline/i }),
    ).toBeVisible();
    await expect(promptInput).toHaveValue("a custom prompt the user typed");
  });

  test("Turn 5 hash copy button copies the full hash, not a truncation", async ({
    page,
  }) => {
    const state: RouteState = {
      sessionPostCount: 0,
      completionPatchSeen: false,
      renamedSession: false,
    };
    await installTutorialRoutes(page, state);
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);

    await page.goto("/");
    await page.getByRole("button", { name: "Let's go" }).click();
    await page.getByRole("button", { name: "Build it" }).click();
    await page.getByRole("button", { name: "Show me the graph" }).click();
    await page.getByRole("button", { name: "Looks good, run it" }).click();
    await expect(page.getByText("bold")).toBeVisible();
    await page.getByRole("button", { name: "Continue" }).click();

    // Audit story turn — full hash visible (not truncated to 12 chars + "...")
    await expect(page.getByText("a7f3e2fullhash")).toBeVisible();

    // Click the copy button for the source-data hash and verify clipboard
    // contents match the full hash, not a "..."-truncated prefix.
    await page
      .getByRole("button", { name: /Copy full source data hash/i })
      .click();
    const clipboardText = await page.evaluate(() =>
      navigator.clipboard.readText(),
    );
    expect(clipboardText).toBe("a7f3e2fullhash");

    // Button feedback flips to "Copied".
    await expect(
      page.getByRole("button", { name: /Copy full source data hash/i }),
    ).toHaveText("Copied");
  });

  test("Turn 4 cancel skips the audit story and acknowledges on Turn 6", async ({
    page,
  }) => {
    const state: RouteState = {
      sessionPostCount: 0,
      completionPatchSeen: false,
      renamedSession: false,
    };
    await installTutorialRoutesWithSlowRun(page, state);

    await page.goto("/");
    await page.getByRole("button", { name: "Let's go" }).click();
    await page.getByRole("button", { name: "Build it" }).click();
    await page.getByRole("button", { name: "Show me the graph" }).click();
    await page.getByRole("button", { name: "Looks good, run it" }).click();

    // The cancel button appears 5 seconds after the run starts (per the
    // SHOW_CANCEL_DELAY_MS constant in TutorialTurn4Run).
    const cancelButton = page.getByRole("button", { name: "Cancel run" });
    await expect(cancelButton).toBeVisible({ timeout: 10_000 });
    await cancelButton.click();

    // Should land directly on Turn 6 (skipping Turn 5 audit) with the
    // cancelled-run acknowledgement.
    await expect(
      page.getByRole("heading", { name: /Choose your default composer mode/i }),
    ).toBeVisible();
    await expect(
      page.getByText(/Your run was cancelled/i),
    ).toBeVisible();
  });

  test("Turn 1 surfaces the privacy preamble before the run starts", async ({
    page,
  }) => {
    const state: RouteState = {
      sessionPostCount: 0,
      completionPatchSeen: false,
      renamedSession: false,
    };
    await installTutorialRoutes(page, state);

    await page.goto("/");

    // Preamble appears on Turn 1 (welcome) and again on Turn 4 (run).
    // Verifying Turn 1 here; Turn 4 is exercised in the happy-path spec.
    await expect(
      page.getByText(
        /This is calling the configured LLM and fetching the URLs/i,
      ),
    ).toBeVisible();
  });
});
