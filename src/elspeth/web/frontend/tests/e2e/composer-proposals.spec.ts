import { expect, type Page, test } from "@playwright/test";
import { ComposerPage } from "./page-objects/composer-page";

const sessionId = "proposal-session-1";

const baseState = {
  id: "state-1",
  version: 1,
  sources: {
    source: {
      plugin: "csv",
      options: { path: "input.csv" },
      on_success: "classify_in",
    },
  },
  nodes: [
    {
      id: "classify",
      node_type: "transform",
      plugin: "llm_transform",
      input: "classify_in",
      on_success: "json_out",
      on_error: null,
      options: {},
    },
  ],
  edges: [],
  outputs: [
    {
      name: "json_out",
      plugin: "json",
      options: { path: "output.jsonl" },
    },
  ],
  metadata: { name: "Proposal workflow", description: null },
};

const committedState = {
  ...baseState,
  id: "state-2",
  version: 2,
  nodes: [
    {
      ...baseState.nodes[0],
      id: "normalize",
      plugin: "passthrough",
    },
  ],
};

const pendingProposal = {
  id: "proposal-1",
  session_id: sessionId,
  tool_call_id: "call-1",
  tool_name: "set_pipeline",
  status: "pending",
  summary: "Replace the pipeline with csv input, 1 transform, and 1 output.",
  rationale: "Requested by the current composer turn.",
  affects: ["graph", "validation", "yaml"],
  arguments_redacted_json: { source: { plugin: "csv" } },
  base_state_id: "state-1",
  committed_state_id: null,
  audit_event_id: "event-created-1",
  created_at: "2026-05-14T00:00:00Z",
  updated_at: "2026-05-14T00:00:00Z",
};

const committedProposal = {
  ...pendingProposal,
  status: "committed",
  committed_state_id: "state-2",
  audit_event_id: "event-accepted-1",
  updated_at: "2026-05-14T00:00:02Z",
};

async function installDeterministicComposerRoutes(page: Page): Promise<void> {
  let accepted = false;
  const sessions: unknown[] = [];

  await page.route("**/api/system/status", async (route) => {
    await route.fulfill({
      json: {
        composer_available: true,
        composer_model: "deterministic-e2e",
        composer_provider: "playwright-route",
        composer_reason: null,
        composer_missing_keys: [],
      },
    });
  });

  await page.route("**/api/composer-preferences", async (route) => {
    await route.fulfill({
      json: {
        default_mode: "freeform",
        banner_dismissed_at: null,
      },
    });
  });

  await page.route("**/api/sessions**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    if (path === "/api/sessions" && method === "GET") {
      await route.fulfill({ json: sessions });
      return;
    }

    if (path === "/api/sessions" && method === "POST") {
      const session = {
        id: sessionId,
        title: "Proposal workflow",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:00:00Z",
      };
      sessions.unshift(session);
      await route.fulfill({ json: session });
      return;
    }

    if (path === `/api/sessions/${sessionId}/guided` && method === "GET") {
      await route.fulfill({
        json: {
          guided_session: null,
          next_turn: null,
          terminal: null,
          composition_state: null,
        },
      });
      return;
    }

    if (path === `/api/sessions/${sessionId}/composer-progress` && method === "GET") {
      await route.fulfill({
        json: {
          session_id: sessionId,
          request_id: null,
          phase: "idle",
          headline: "Idle.",
          evidence: [],
          likely_next: null,
          reason: "composer_idle",
          updated_at: "2026-05-14T00:00:00Z",
        },
      });
      return;
    }

    if (path === `/api/sessions/${sessionId}/messages` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === `/api/sessions/${sessionId}/messages` && method === "POST") {
      await route.fulfill({
        json: {
          message: {
            id: "assistant-1",
            session_id: sessionId,
            role: "assistant",
            content: "I found one pipeline change that needs approval.",
            tool_calls: [
              {
                id: "call-1",
                type: "function",
                function: {
                  name: "set_pipeline",
                  arguments: "{\"source\":{\"plugin\":\"csv\"}}",
                },
              },
            ],
            created_at: "2026-05-14T00:00:01Z",
            composition_state_id: null,
            tool_call_id: null,
            parent_assistant_id: null,
            sequence_no: 1,
          },
          state: baseState,
          proposals: [pendingProposal],
        },
      });
      return;
    }

    if (path === `/api/sessions/${sessionId}/state` && method === "GET") {
      await route.fulfill({ json: accepted ? committedState : baseState });
      return;
    }

    if (path === `/api/sessions/${sessionId}/state/versions` && method === "GET") {
      await route.fulfill({
        json: [
          {
            id: accepted ? "state-2" : "state-1",
            version: accepted ? 2 : 1,
            created_at: "2026-05-14T00:00:00Z",
            node_count: 1,
          },
        ],
      });
      return;
    }

    if (path === `/api/sessions/${sessionId}/runs` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    if (path === `/api/sessions/${sessionId}/proposals` && method === "GET") {
      await route.fulfill({ json: accepted ? [] : [pendingProposal] });
      return;
    }

    if (path === `/api/sessions/${sessionId}/proposals/proposal-1/accept` && method === "POST") {
      accepted = true;
      await route.fulfill({ json: committedProposal });
      return;
    }

    if (path === `/api/sessions/${sessionId}/blobs` && method === "GET") {
      await route.fulfill({ json: [] });
      return;
    }

    await route.continue();
  });
}

test("explicit approve tool call is visible before commit", async ({ page }) => {
  await installDeterministicComposerRoutes(page);

  const composer = new ComposerPage(page);
  await composer.goto();
  await composer.createSession("Proposal workflow");
  await composer.sendMessage("Build a simple csv to json pipeline");

  await expect(page.getByText(/Proposed: set_pipeline/)).toBeVisible();
  const pendingChanges = page.getByRole("region", { name: /Pending changes/ });
  await expect(
    pendingChanges.getByRole("button", { name: /Accept proposal:/ }),
  ).toBeVisible();
  await expect(
    pendingChanges.getByRole("button", { name: /Reject proposal:/ }),
  ).toBeVisible();

  await pendingChanges.getByRole("button", { name: /Accept proposal:/ }).click();

  await expect(page.getByText(/Applied: set_pipeline/)).toBeVisible();
  await expect(
    // Accessible name updated with the keyboard-focusable conversation region
    // (elspeth-5e43a0c8b2).
    page.getByRole("log", { name: "Conversation" }).getByText("audit event-ac"),
  ).toBeVisible();
});
