// ============================================================================
// composer-capability-parity.staging.spec.ts — guided_staged live acceptance
//
// Plan 05 Task 5 authored / Task 6 executed. This spec drives the deployed
// guided-staged authoring surface through the two-LLM colour hybrid pipeline
// against a REAL provider, exports sanitized evidence for exactly one deployed
// revision, and hands that evidence to the same Python oracle the freeform and
// guided_full surfaces use (evals/composer-parity/live_acceptance.py).
//
// It is authored here but NOT run in the Task 5 workflow: it needs a live
// staging deployment AND a real provider key. Like tests/.../test_bedrock_live_smoke.py
// it is key-gated — with no ELSPETH_EVAL_API_KEY the whole describe is SKIPPED,
// so the rest of the composer-capability-parity work stays deterministic and
// offline. Task 6 supplies STAGING_* credentials (via the staging global-setup)
// and ELSPETH_EVAL_API_KEY, deploys the integrated revision, and runs:
//
//   STAGING_BASE_URL=https://elspeth.foundryside.dev \
//   PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
//   STAGING_USERNAME=... STAGING_PASSWORD=... \
//   ELSPETH_EVAL_API_KEY=... ELSPETH_EVAL_REVISION="$(git rev-parse HEAD)" \
//   npx playwright test --config=playwright.staging.config.ts \
//     tests/e2e/composer-capability-parity.staging.spec.ts --retries=0
//
// ── TASK 6 RECONCILIATION NOTES (do not silently paper over) ────────────────
//   1. Guided-staged stage sequence. This spec models the colour pipeline as a
//      chat-driven guided planner: enter guided, send the outcome-only request,
//      send an early reminder, then review the whole-graph ProposePipelineTurn.
//      If the deployed guided-staged flow interposes source/sink SINGLE_SELECT
//      stages before the planner chat (as composer-guided.spec.ts shows for the
//      step wizard), Task 6 threads those with `advanceDeterministicStages`
//      below — it is a documented seam, not a hidden assumption.
//   2. Evidence collector endpoint. The oracle consumes six sanitized JSON
//      documents (manifest/graph/run_llm_calls/run_accounting/business_output/
//      input_identities). There is no single backend endpoint that emits them
//      in the oracle's shape today; server-side assembly + redaction is where
//      `redact_evidence` should live (live_acceptance.py). `collectStagedParityEvidence`
//      calls the purpose-built export endpoint the Task-6 collector must expose
//      and FAILS LOUDLY, naming the contract, if it is absent — it never
//      fabricates a passing document (that would defeat the oracle's whole point).
// ============================================================================

import { spawnSync } from "node:child_process";
import { chmodSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  tokenFromStorageState,
  uploadBlob,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

// ── Paths ───────────────────────────────────────────────────────────────────
// This spec lives at src/elspeth/web/frontend/tests/e2e/; six `..` reach the
// repo root where the eval corpus and oracle live.
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../../../../../..");
const ORACLE = resolve(REPO_ROOT, "evals", "composer-parity", "live_acceptance.py");
const FIXTURE_CSV = resolve(REPO_ROOT, "evals", "composer-parity", "fixtures", "two_llm_colour.csv");
const REQUEST_TXT = resolve(REPO_ROOT, "evals", "composer-parity", "fixtures", "two_llm_colour_request.txt");
const EVIDENCE_ROOT = resolve(REPO_ROOT, "output", "playwright", "composer-parity");

const SURFACE = "guided_staged";
const BLOB_FILENAME = "two_llm_colour.csv";

// The six sanitized documents the oracle loads (live_acceptance.EVIDENCE_FILES).
const EVIDENCE_FILES = [
  "manifest.json",
  "graph.json",
  "run_llm_calls.json",
  "run_accounting.json",
  "business_output.json",
  "input_identities.json",
] as const;
type EvidenceBundle = Record<(typeof EVIDENCE_FILES)[number], unknown>;

// Key-gate: absent a real provider key this journey cannot prove a live run, so
// it is skipped rather than run against a fake. This keeps the Task 5 workflow
// deterministic/offline; Task 6 supplies the key on staging.
const LIVE_KEY = process.env.ELSPETH_EVAL_API_KEY;

test.describe("composer capability parity — guided_staged live acceptance (staging)", () => {
  test.skip(
    !LIVE_KEY,
    "requires a live staging deployment and ELSPETH_EVAL_API_KEY (real provider run)",
  );

  // A single real colour run can take minutes (ten rows × two LLM branches plus
  // planner completions), so give the journey generous headroom.
  test.setTimeout(10 * 60_000);

  test("derives, accepts, runs, exports, and passes the oracle on one revision", async ({
    page,
  }) => {
    const revision = resolveRevision();
    const csv = readFixture(FIXTURE_CSV);
    const request = readFixture(REQUEST_TXT).trim();

    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);

    let sessionId: string | undefined;
    try {
      // ── Clean session + bound colour source ────────────────────────────────
      const session = await createSession(ctx, `composer-parity-${SURFACE}`);
      sessionId = session.id;
      await uploadBlob(ctx, sessionId, BLOB_FILENAME, csv);

      const composer = new ComposerPage(page);
      await composer.goto(sessionId);
      await composer.waitForChatReady();

      // ── /guided/start ──────────────────────────────────────────────────────
      // "Switch to guided" issues POST /api/sessions/{id}/guided/start and mounts
      // the guided composer surface.
      await page.getByRole("button", { name: "Switch to guided" }).click();
      await expect(page.getByLabel(/guided composer/i)).toBeVisible();

      // Task-6 seam: interpose any deterministic source/sink stages the deployed
      // guided-staged flow requires before the planner chat.
      await advanceDeterministicStages(page);

      // ── Outcome-only request + early two-LLM reminder ──────────────────────
      const chat = page.getByRole("textbox", { name: "Message input" });
      await expect(chat).toBeEnabled();
      await sendGuided(page, request);

      // The "wait" behavior: while the planner works, the pending strip owns the
      // composer slot (GuidedPendingStrip — status region + Stop).
      await expect(page.locator(".guided-pending-strip")).toBeVisible();

      // Early reminder — sent before the proposal lands — to retain the two
      // INDEPENDENT LLM assessments and wait for both branches before merging.
      // This is the "retain the two independent LLM assessments" reminder the
      // reload must prove is persisted and consumed, not dropped.
      const REMINDER =
        "Reminder before you finish: keep the blue and red assessments as two " +
        "separate, independent LLM calls, and wait for BOTH branches on every " +
        "row before merging into the single hybrid row.";
      await waitForGuidedIdle(page);
      await sendGuided(page, REMINDER);

      // ── Reload → persist + consume ─────────────────────────────────────────
      // A mid-composition reload must not lose the request or the reminder
      // (persist), and the guided session must resume and consume the pending
      // work (consume) rather than restarting from an empty transcript.
      await page.reload();
      await composer.waitForChatReady();
      await expect(page.getByLabel(/guided composer/i)).toBeVisible();

      const transcript = page.getByRole("log", { name: /chat history/i });
      await expect(transcript).toContainText("two independent LLMs");
      await expect(transcript).toContainText("independent LLM calls");

      // ── Review the complete graph (ProposePipelineTurn) ────────────────────
      const proposal = page.getByRole("article", { name: "Review pipeline proposal" });
      await expect(proposal.getByRole("heading", { name: "Review pipeline proposal" })).toBeVisible({
        timeout: 5 * 60_000,
      });
      // The whole-DAG canvas and the two independent LLM assessment nodes. The
      // planner names the two branches for blue and red; both must surface in
      // the reviewed proposal (Components section / graph).
      await expect(proposal.getByRole("img", { name: /pipeline proposal graph/i })).toBeVisible();
      await expect(proposal.getByText(/blue/i).first()).toBeVisible();
      await expect(proposal.getByText(/red/i).first()).toBeVisible();
      // A require-all coalesce shows a fan-in join over both branches.
      await expect(proposal.getByText(/join/i).first()).toBeVisible();

      // ── Accept: Review wiring → Confirm wiring ─────────────────────────────
      const reviewWiring = proposal.getByRole("button", { name: "Review wiring" });
      await expect(reviewWiring).toBeEnabled();
      await reviewWiring.click();
      const confirmWiring = page.getByRole("button", { name: "Confirm wiring" });
      await expect(confirmWiring).toBeEnabled();
      await confirmWiring.click();

      // ── Run against the real provider ──────────────────────────────────────
      const runButton = composer.executeButton();
      await expect(runButton).toBeEnabled();
      await runButton.click();

      const results = page.getByRole("region", { name: "Pipeline run results" });
      await expect(results).toBeVisible({ timeout: 8 * 60_000 });
      // Ten successful hybrid rows and zero failures is the business contract;
      // the oracle re-proves it from evidence, this is the fast UI signal.
      await expect(results).toContainText(/10 .*succe/i);
      await expect(results).not.toContainText(/error|failed/i);

      // ── Export sanitized evidence + run the same oracle ────────────────────
      const runId = await readActiveRunId(page);
      const evidence = await collectStagedParityEvidence(ctx, {
        sessionId,
        runId,
        revision,
      });
      const revisionDir = writeEvidenceDir(EVIDENCE_ROOT, revision, evidence);

      const verdict = spawnSync(
        "uv",
        [
          "run",
          "python",
          ORACLE,
          "verify",
          "--evidence-dir",
          EVIDENCE_ROOT,
          "--revision",
          revision,
          "--surface",
          SURFACE,
        ],
        { cwd: REPO_ROOT, encoding: "utf-8" },
      );
      const oracleOut = `${verdict.stdout ?? ""}${verdict.stderr ?? ""}`;
      expect(
        verdict.status,
        `oracle rejected exported evidence in ${revisionDir}:\n${oracleOut}`,
      ).toBe(0);
      expect(oracleOut).toContain("ACCEPTED");
    } finally {
      if (sessionId !== undefined) {
        await deleteSession(ctx, sessionId);
      }
      await ctx.dispose();
    }
  });
});

// ── Journey helpers ──────────────────────────────────────────────────────────

/** Send a guided chat message through the shared composer input. */
async function sendGuided(page: Page, content: string): Promise<void> {
  const chat = page.getByRole("textbox", { name: "Message input" });
  await chat.fill(content);
  await page.getByRole("button", { name: "Send message" }).click();
}

/** Wait for the guided planner to return control (pending strip gone). */
async function waitForGuidedIdle(page: Page): Promise<void> {
  await expect(page.locator(".guided-pending-strip")).toHaveCount(0, {
    timeout: 5 * 60_000,
  });
}

/**
 * Task-6 seam: advance any deterministic source/sink stages the deployed
 * guided-staged flow requires before the planner chat. With the colour CSV
 * already uploaded, a source SINGLE_SELECT binds it via the upload fast-path;
 * a sink stage may default. Left intentionally permissive: it clicks a stage's
 * Continue only when one is actually presented, and is a no-op for a purely
 * chat-driven planner. Task 6 tightens this against the real staging sequence
 * rather than this spec guessing it blind.
 */
async function advanceDeterministicStages(page: Page): Promise<void> {
  const continueButton = page.getByRole("button", { name: "Continue", exact: true });
  // Bounded: never loop forever if the flow is chat-first.
  for (let stage = 0; stage < 4; stage += 1) {
    if ((await continueButton.count()) === 0) return;
    if (!(await continueButton.first().isVisible().catch(() => false))) return;
    if (!(await continueButton.first().isEnabled().catch(() => false))) return;
    await continueButton.first().click();
    await page.waitForTimeout(500);
  }
}

/**
 * Read the just-executed run id. The execution store exposes the active run id
 * to the DOM; Task 6 confirms the exact attribute on staging if this drifts.
 */
async function readActiveRunId(page: Page): Promise<string> {
  const region = page.getByRole("region", { name: "Pipeline run results" });
  const runId = await region.getAttribute("data-run-id");
  if (runId === null || runId.trim() === "") {
    throw new Error(
      "could not read the active run id from the run-results region " +
        "(expected a data-run-id attribute) — Task 6: wire the run-id selector to staging",
    );
  }
  return runId;
}

// ── Evidence collection (Task-6 collector seam) ──────────────────────────────

interface EvidenceRef {
  sessionId: string;
  runId: string;
  revision: string;
}

/**
 * Fetch the six sanitized oracle documents for the accepted proposal + run.
 *
 * The oracle (live_acceptance.py) intentionally REFUSES to fabricate provider
 * evidence: its default `LiveEvidenceCollector` raises `LiveCollectionUnavailable`
 * until a real collector is wired. This is the frontend counterpart. Sanitizing
 * and assembling the six documents belongs server-side — that is where
 * `redact_evidence` runs before anything is retained — so this helper reads a
 * purpose-built export endpoint rather than scraping and re-shaping raw audit in
 * the browser. If that endpoint is absent, it FAILS LOUDLY with the exact
 * contract Task 6 must satisfy; it never returns a hand-built passing bundle.
 */
async function collectStagedParityEvidence(
  ctx: APIRequestContext,
  ref: EvidenceRef,
): Promise<EvidenceBundle> {
  const path = `/api/sessions/${ref.sessionId}/runs/${ref.runId}/composer-parity-evidence?revision=${encodeURIComponent(ref.revision)}`;
  const resp = await ctx.get(path);
  if (resp.status() === 404) {
    throw new Error(
      "composer-parity evidence export endpoint is not wired on this deployment.\n" +
        `  Task 6 must expose GET ${path.split("?")[0]} returning the six sanitized\n` +
        "  documents the oracle loads, already passed through live_acceptance.redact_evidence:\n" +
        `    ${EVIDENCE_FILES.join(", ")}\n` +
        "  It must strip every credential/cookie/authorization header/resolved secret/raw\n" +
        "  provider response, keep the intrinsic live-provider proof (model_returned,\n" +
        "  provider_request_id, positive prompt/completion token usage), and be keyed to\n" +
        "  exactly this proposal/commit/run. Do NOT fabricate these documents client-side.",
    );
  }
  if (!resp.ok()) {
    throw new Error(
      `evidence export failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
    );
  }
  const bundle = (await resp.json()) as Partial<EvidenceBundle>;
  for (const name of EVIDENCE_FILES) {
    if (bundle[name] === undefined) {
      throw new Error(`evidence bundle is missing required document ${name}`);
    }
  }
  return bundle as EvidenceBundle;
}

/**
 * Write the six documents under <root>/<revision>/ with the restrictive
 * permissions the oracle's hygiene gate enforces (0o700 dir, 0o600 files),
 * mirroring live_acceptance._write_evidence_dir so `assert_no_sensitive` and the
 * permission checks pass on load. The server-side export already redacted; this
 * only persists.
 */
function writeEvidenceDir(root: string, revision: string, bundle: EvidenceBundle): string {
  if (revision.includes("/") || revision.includes("\\")) {
    throw new Error(`revision ${revision} must be a plain directory name`);
  }
  const revisionDir = resolve(root, revision);
  // Start clean so a stale prior export never bleeds into this revision.
  rmSync(revisionDir, { recursive: true, force: true });
  mkdirSync(revisionDir, { recursive: true, mode: 0o700 });
  chmodSync(revisionDir, 0o700);
  for (const name of EVIDENCE_FILES) {
    const filePath = resolve(revisionDir, name);
    writeFileSync(filePath, `${JSON.stringify(bundle[name], null, 2)}\n`, { mode: 0o600 });
    chmodSync(filePath, 0o600);
  }
  return revisionDir;
}

// ── Small utilities ──────────────────────────────────────────────────────────

function readFixture(path: string): string {
  // Read via node so the fixture bytes are the corpus's pinned form.
  return readFileSync(path, "utf-8");
}

/** The deployed revision under acceptance — env-provided, else git HEAD. */
function resolveRevision(): string {
  const fromEnv = process.env.ELSPETH_EVAL_REVISION ?? process.env.GITHUB_SHA;
  if (fromEnv && fromEnv.trim() !== "") return fromEnv.trim();
  const rev = spawnSync("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT, encoding: "utf-8" });
  const head = (rev.stdout ?? "").trim();
  if (head === "") {
    throw new Error("could not resolve the deployed revision (set ELSPETH_EVAL_REVISION)");
  }
  return head;
}
