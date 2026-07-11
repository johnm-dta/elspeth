#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { chromium } from "@playwright/test";
import {
  driveStagedGuidedTutorial,
  isComposeRequest,
  isRunRequest,
} from "./staging-tutorial-driver.mjs";

const TOKEN_KEY = "auth_token";
const DEFAULT_BASE_URL = "https://elspeth.foundryside.dev";
const DEFAULT_RUNS = 20;
const DEFAULT_BUILD_TIMEOUT_MS = 420_000;
// The tutorial seeds an uploaded sample; the guided source binds it via the
// upload fast-path (blob:<ref>, server_storage_bound) rather than authoring an
// LLM-invented source, so it stages NO invented_source review on the source
// node (parity with the normal guided seeded-upload flow — see
// _source_from_latest_uploaded_blob_for_step_1_chat and chat_solver
// server_storage_bound). The invented_source expectation therefore no longer
// holds; the bound-source invariant is asserted positively below
// (elspeth-52f42d09a2).
const BASE_EXPECTED_TUTORIAL_KINDS = ["llm_prompt_template", "pipeline_decision"];
const RAW_HTML_CLEANUP_TERM = "drop_raw_html_fields";
const PROMPT_SHIELD_RECOMMENDATION_TERM = "prompt_injection_shield_recommendation";

function looksLikeRawHtmlField(fieldName) {
  const normalized = String(fieldName).trim().toLowerCase().replaceAll("-", "_");
  return normalized === "content" || normalized.includes("html") || normalized.includes("fingerprint");
}

function outputName(output) {
  return typeof output?.name === "string" ? output.name : typeof output?.sink_name === "string" ? output.sink_name : null;
}

function producerForStream(nodes, streamName) {
  return nodes.find((node) => node?.on_success === streamName) ?? null;
}

function upstreamNodesForInput(nodes, inputStream) {
  const upstream = [];
  const visitedStreams = new Set();
  let stream = inputStream;
  while (typeof stream === "string" && stream.length > 0 && !visitedStreams.has(stream)) {
    visitedStreams.add(stream);
    const producer = producerForStream(nodes, stream);
    if (!producer) {
      break;
    }
    upstream.push(producer);
    stream = producer.input;
  }
  return upstream;
}

function parseMessageJson(message) {
  if (typeof message?.content !== "string" || message.content.trim() === "") {
    return null;
  }
  try {
    return JSON.parse(message.content);
  } catch {
    return null;
  }
}

function isPendingReviewDiagnostic(entry) {
  const message = typeof entry?.message === "string" ? entry.message : "";
  return message.includes("review pending");
}

function latestBlockingToolValidationDiagnostics(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }
  let latestDiagnostics = [];
  for (const message of messages) {
    if (message?.role !== "tool") {
      continue;
    }
    const payload = parseMessageJson(message);
    const validation = payload?.validation;
    if (!validation || typeof validation !== "object") {
      continue;
    }
    const errors = Array.isArray(validation.errors) ? validation.errors : [];
    const warnings = Array.isArray(validation.warnings) ? validation.warnings : [];
    latestDiagnostics = [
      ...errors,
      ...warnings.filter((warning) => warning?.severity === "high"),
    ].filter((entry) => !isPendingReviewDiagnostic(entry));
  }
  return latestDiagnostics.map((entry) =>
    typeof entry?.message === "string"
      ? `${entry.component ?? "unknown"}: ${entry.message}`
      : JSON.stringify(entry),
  );
}

function rawHtmlCleanupEvidence(state, pendingEvents) {
  const cleanupEvent = pendingEvents.find(
    (event) =>
      event.kind === "pipeline_decision" &&
      event.affected_node_id !== "source" &&
      event.user_term === RAW_HTML_CLEANUP_TERM,
  );
  if (!cleanupEvent || state === null || typeof state !== "object" || "error" in state) {
    return { ok: false, node_id: cleanupEvent?.affected_node_id ?? null, mapping: null, preserved_fields: [] };
  }
  const nodes = Array.isArray(state.nodes) ? state.nodes : [];
  const outputs = Array.isArray(state.outputs) ? state.outputs : [];
  const node = nodes.find((candidate) => candidate?.id === cleanupEvent.affected_node_id);
  const options = node?.options && typeof node.options === "object" ? node.options : {};
  const mapping = options.mapping && typeof options.mapping === "object" ? options.mapping : null;
  const preservedFields =
    mapping === null
      ? []
      : [...Object.entries(mapping).flatMap(([source, target]) => [source, target])].filter(looksLikeRawHtmlField);
  const sinkNames = outputs.map(outputName).filter((name) => typeof name === "string" && name.length > 0);
  const finalPredecessors = sinkNames
    .map((sinkName) => producerForStream(nodes, sinkName))
    .filter((producer) => producer !== null);
  const cleanupImmediatelyBeforeSink =
    finalPredecessors.length > 0 && finalPredecessors.some((producer) => producer.id === cleanupEvent.affected_node_id);
  const upstreamNodes = node ? upstreamNodesForInput(nodes, node.input) : [];
  const upstreamLlmResponseFields = upstreamNodes
    .filter((producer) => producer?.plugin === "llm")
    .map((producer) => producer?.options?.response_field)
    .filter((field) => typeof field === "string" && field.length > 0);
  const mappedFields = mapping === null ? [] : [...Object.entries(mapping).flatMap(([source, target]) => [source, target])];
  const preservesLlmOutputs =
    upstreamLlmResponseFields.length === 0 ||
    upstreamLlmResponseFields.some((field) => mappedFields.includes(field));
  return {
    ok:
      node?.plugin === "field_mapper" &&
      options.select_only === true &&
      mapping !== null &&
      preservedFields.length === 0 &&
      cleanupImmediatelyBeforeSink &&
      upstreamLlmResponseFields.length > 0 &&
      preservesLlmOutputs,
    node_id: cleanupEvent.affected_node_id,
    mapping,
    preserved_fields: preservedFields,
    final_predecessors: finalPredecessors.map((producer) => producer.id),
    upstream_llm_response_fields: upstreamLlmResponseFields,
  };
}

function promptShieldRecommendationEvidence(state, pendingEvents) {
  const nodes = Array.isArray(state?.nodes) ? state.nodes : [];
  const lowerPluginNames = nodes.map((node) => String(node?.plugin ?? "").toLowerCase());
  const promptShieldInserted = lowerPluginNames.some((name) => name.includes("prompt_shield"));
  const contentSafetyInserted = lowerPluginNames.some((name) => name.includes("content_safety"));
  const recommendationEvent = pendingEvents.find(
    (event) =>
      event.kind === "pipeline_decision" &&
      event.user_term === PROMPT_SHIELD_RECOMMENDATION_TERM &&
      nodes.some((node) => node?.id === event.affected_node_id && node?.plugin === "llm"),
  );
  const draft = typeof recommendationEvent?.llm_draft === "string" ? recommendationEvent.llm_draft : "";
  const draftMentionsPromptShield =
    draft.includes("azure_prompt_shield") ||
    /prompt[- ]?injection|prompt shield/i.test(draft);

  return {
    ok: recommendationEvent !== undefined && draftMentionsPromptShield && !promptShieldInserted && !contentSafetyInserted,
    event_node_id: recommendationEvent?.affected_node_id ?? null,
    draft: draft || null,
    prompt_shield_inserted: promptShieldInserted,
    content_safety_inserted: contentSafetyInserted,
  };
}

function requiredEnv(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} must be set`);
  }
  return value;
}

function intEnv(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === "") {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

const baseURL = (process.env.STAGING_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
const origin = new URL(baseURL).origin;
const username = requiredEnv("STAGING_USERNAME");
const password = requiredEnv("STAGING_PASSWORD");
const runs = intEnv("TUTORIAL_RUNS", DEFAULT_RUNS);
const buildTimeoutMs = intEnv("TUTORIAL_BUILD_TIMEOUT_MS", DEFAULT_BUILD_TIMEOUT_MS);
const promptOverride = process.env.TUTORIAL_PROMPT_OVERRIDE ?? "";
const expectedVagueTerm = process.env.TUTORIAL_EXPECT_VAGUE_TERM ?? "";
const diagnosticQuestion = process.env.TUTORIAL_DIAGNOSTIC_QUESTION ?? "";
const artifactsDir = resolve("test-results", "staging-tutorial-harness");

async function readResponseBody(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function apiFetch(token, path, options = {}) {
  const headers = {
    Authorization: `Bearer ${token}`,
    ...(options.body === undefined ? {} : { "Content-Type": "application/json" }),
    ...(options.headers ?? {}),
  };
  const response = await fetch(new URL(path, `${baseURL}/`), {
    method: options.method ?? "GET",
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  const body = await readResponseBody(response);
  if (!response.ok) {
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? body.detail
        : body;
    throw new Error(`${options.method ?? "GET"} ${path} failed (${response.status}): ${String(detail).slice(0, 500)}`);
  }
  return body;
}

async function login() {
  const response = await fetch(`${baseURL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const body = await readResponseBody(response);
  if (!response.ok) {
    throw new Error(`login failed (${response.status}): ${JSON.stringify(body).slice(0, 500)}`);
  }
  if (typeof body !== "object" || body === null || typeof body.access_token !== "string") {
    throw new Error("login response missing access_token");
  }
  return body.access_token;
}

async function resetTutorial(token) {
  await apiFetch(token, "/api/composer-preferences", {
    method: "PATCH",
    body: { default_mode: "guided", tutorial_completed_at: null },
  });
  await apiFetch(token, "/api/tutorial/orphans", { method: "DELETE" });
}

async function askDiagnosticQuestion(token, sessionId, question) {
  const response = await apiFetch(token, `/api/sessions/${sessionId}/messages`, {
    method: "POST",
    body: { content: question },
  });
  return {
    question,
    assistant_excerpt:
      typeof response?.message?.content === "string"
        ? response.message.content.replace(/\s+/g, " ").slice(0, 1200)
        : null,
    state_version: response?.state?.version ?? null,
  };
}

async function fetchSessionEvidence(token, sessionId) {
  const [state, eventsEnvelope, messages] = await Promise.all([
    apiFetch(token, `/api/sessions/${sessionId}/state`).catch((error) => ({ error: error.message })),
    apiFetch(token, `/api/sessions/${sessionId}/interpretations`).catch((error) => ({ error: error.message })),
    apiFetch(token, `/api/sessions/${sessionId}/messages`).catch((error) => ({ error: error.message })),
  ]);
  const events = Array.isArray(eventsEnvelope?.events) ? eventsEnvelope.events : [];
  const pendingEvents = events.filter((event) => event.choice === "pending");
  const reviewEvents = events.filter((event) => event.choice !== "abandoned");
  const pendingKinds = pendingEvents.map((event) => event.kind).sort();
  const reviewKinds = reviewEvents.map((event) => event.kind).sort();
  const expectedKinds = [...BASE_EXPECTED_TUTORIAL_KINDS, ...(expectedVagueTerm ? ["vague_term"] : [])].sort();
  const cleanupEvidence = rawHtmlCleanupEvidence(state, reviewEvents);
  const promptShieldEvidence = promptShieldRecommendationEvidence(state, reviewEvents);
  const blockingToolValidationDiagnostics = latestBlockingToolValidationDiagnostics(messages);
  const hasExpectedTutorialAssumptions =
    expectedKinds.every((kind) => reviewKinds.includes(kind)) &&
    // Bound-source invariant (elspeth-52f42d09a2): the seeded upload binds the
    // source via the fast-path, so the tutorial must NOT stage an
    // invented_source review on the source node. Its presence would mean the
    // source regressed to the LLM-invented path (or the seeded blob failed to
    // bind), which the pre-9f425de3d contract wrongly required.
    !reviewEvents.some(
      (event) => event.kind === "invented_source" && event.affected_node_id === "source",
    ) &&
    (!expectedVagueTerm ||
      reviewEvents.some(
        (event) => event.kind === "vague_term" && event.user_term === expectedVagueTerm,
      )) &&
    reviewEvents.some(
      (event) =>
        event.kind === "llm_prompt_template" &&
        typeof event.user_term === "string" &&
        event.user_term.startsWith("llm_prompt_template:"),
    ) &&
    reviewEvents.some(
      (event) =>
        event.kind === "pipeline_decision" &&
        event.affected_node_id !== "source" &&
        event.user_term === RAW_HTML_CLEANUP_TERM,
    ) &&
    cleanupEvidence.ok &&
    promptShieldEvidence.ok &&
    blockingToolValidationDiagnostics.length === 0;
  const assistantMessages = Array.isArray(messages)
    ? messages.filter((message) => message.role === "assistant")
    : [];
  const lastAssistant = assistantMessages.at(-1)?.content ?? null;
  return {
    state_exists: state !== null && typeof state === "object" && !("error" in state),
    state_error: state?.error ?? null,
    state_version: state?.version ?? null,
    interpretation_count: events.length,
    interpretation_kinds: reviewKinds,
    interpretation_terms: reviewEvents
      .map((event) => `${event.kind}:${event.affected_node_id ?? ""}:${event.user_term ?? ""}:${event.choice ?? ""}`)
      .sort(),
    pending_interpretation_count: pendingEvents.length,
    pending_interpretation_kinds: pendingKinds,
    pending_interpretation_terms: pendingEvents
      .map((event) => `${event.kind}:${event.affected_node_id ?? ""}:${event.user_term ?? ""}`)
      .sort(),
    raw_html_cleanup_contract_ok: cleanupEvidence.ok,
    raw_html_cleanup_node_id: cleanupEvidence.node_id,
    raw_html_cleanup_mapping: cleanupEvidence.mapping,
    raw_html_cleanup_preserved_fields: cleanupEvidence.preserved_fields,
    raw_html_cleanup_final_predecessors: cleanupEvidence.final_predecessors ?? [],
    raw_html_cleanup_upstream_llm_response_fields: cleanupEvidence.upstream_llm_response_fields ?? [],
    prompt_shield_recommendation_ok: promptShieldEvidence.ok,
    prompt_shield_recommendation_node_id: promptShieldEvidence.event_node_id,
    prompt_shield_recommendation_draft: promptShieldEvidence.draft,
    prompt_shield_inserted: promptShieldEvidence.prompt_shield_inserted,
    content_safety_inserted: promptShieldEvidence.content_safety_inserted,
    blocking_tool_validation_diagnostics: blockingToolValidationDiagnostics,
    tutorial_assumption_contract_ok: hasExpectedTutorialAssumptions,
    last_assistant_excerpt:
      typeof lastAssistant === "string" ? lastAssistant.replace(/\s+/g, " ").slice(0, 500) : null,
  };
}

async function runOne(browser, token, index) {
  await resetTutorial(token);

  const apiFailures = [];
  const consoleErrors = [];
  let rejectBlockingFailure = null;
  let blockingFailureSettled = false;
  const blockingFailurePromise = new Promise((_, reject) => {
    rejectBlockingFailure = reject;
  });
  void blockingFailurePromise.catch(() => undefined);
  const makeStep = () => ({
    fired: false,
    responded: false,
    status: null,
    body: null,
    elapsed_ms: null,
  });
  const steps = { compose: makeStep(), run: makeStep() };
  const stepStartedAt = { compose: null, run: null };
  const context = await browser.newContext({
    baseURL,
    storageState: {
      cookies: [],
      origins: [
        {
          origin,
          localStorage: [{ name: TOKEN_KEY, value: token }],
        },
      ],
    },
  });
  const page = await context.newPage();
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  page.on("request", (request) => {
    const url = request.url();
    const method = request.method();
    if (isComposeRequest(url, method) && !steps.compose.fired) {
      steps.compose.fired = true;
      stepStartedAt.compose = Date.now();
    } else if (isRunRequest(url, method)) {
      steps.run.fired = true;
      stepStartedAt.run = Date.now();
    }
  });
  page.on("requestfailed", (request) => {
    const url = request.url();
    const method = request.method();
    const errorText = request.failure()?.errorText ?? "connection failed";
    if (isComposeRequest(url, method)) {
      steps.compose.body ??= errorText;
    } else if (isRunRequest(url, method)) {
      steps.run.body ??= errorText;
    }
  });
  page.on("response", (response) => {
    const method = response.request().method();
    const url = response.url();
    const isSessionCreate = method === "POST" && url === `${baseURL}/api/sessions`;
    const compose = isComposeRequest(url, method);
    const run = isRunRequest(url, method);
    if (compose || run) {
      const target = compose ? steps.compose : steps.run;
      const startedAt = compose ? stepStartedAt.compose : stepStartedAt.run;
      target.responded = true;
      target.status = response.status();
      target.elapsed_ms = startedAt === null ? null : Date.now() - startedAt;
    }
    if (response.status() >= 500) {
      apiFailures.push({
        status: response.status(),
        method,
        url,
      });
    }
    if (isSessionCreate && response.ok()) {
      void response
        .json()
        .then((session) => {
          if (typeof session?.id === "string") {
            sessionId = session.id;
          }
        })
        .catch(() => {});
    }
    const isBlockingPost = compose || run;
    if (isBlockingPost && response.status() >= 400 && !blockingFailureSettled) {
      blockingFailureSettled = true;
      void readResponseBody(response)
        .then((body) => {
          const target = compose ? steps.compose : steps.run;
          target.body = JSON.stringify(body).slice(0, 1000);
          apiFailures.push({
            status: response.status(),
            method,
            url,
            body,
          });
          rejectBlockingFailure?.(
            new Error(
              `tutorial ${compose ? "compose" : "run"} POST failed (${response.status()}): ${JSON.stringify(body).slice(0, 1000)}`,
            ),
          );
        })
        .catch((error) => {
          rejectBlockingFailure?.(
            new Error(
              `tutorial ${compose ? "compose" : "run"} POST failed (${response.status()}) and body could not be read: ${
                error instanceof Error ? error.message : String(error)
              }`,
            ),
          );
        });
    }
  });

  let sessionId = null;
  let screenshot = null;
  let graduated = false;

  try {
    await page.goto("/", { waitUntil: "networkidle", timeout: 45_000 });
    const sessionResponsePromise = page
      .waitForResponse(
        (response) =>
          response.url() === `${baseURL}/api/sessions` &&
          response.request().method() === "POST",
        { timeout: 30_000 },
      )
      .catch(() => null);
    await page.getByRole("button", { name: "Let's go" }).click({ timeout: 30_000 });
    const sessionResponse = await sessionResponsePromise;
    if (sessionResponse?.ok()) {
      const session = await sessionResponse.json().catch(() => null);
      if (typeof session?.id === "string") {
        sessionId = session.id;
      }
    }
    if (promptOverride.trim()) {
      const legacyPrompt = page.locator("#tutorial-prompt");
      if ((await legacyPrompt.count().catch(() => 0)) > 0) {
        await legacyPrompt.fill(promptOverride);
      } else {
        console.warn(
          "[tutorial-harness] TUTORIAL_PROMPT_OVERRIDE ignored; staged tutorial uses locked per-stage prompts",
        );
      }
    }

    await Promise.race([
      page.getByLabel(/guided composer/i).waitFor({ state: "visible", timeout: 60_000 }),
      blockingFailurePromise,
    ]);
    await Promise.race([
      driveStagedGuidedTutorial(page, {
        timeoutMs: buildTimeoutMs,
        decisionScreenshotPath: resolve(artifactsDir, `run-${index}-guided-decision-summary.png`),
      }),
      blockingFailurePromise,
    ]);
    await Promise.race([
      page
        .getByRole("button", { name: "Continue", exact: true })
        .waitFor({ state: "visible", timeout: buildTimeoutMs }),
      blockingFailurePromise,
    ]);
    await page.getByRole("button", { name: "Continue", exact: true }).click();
    await page.getByText(/This is the audit story/i).waitFor({ timeout: 60_000 });
    await page.getByRole("button", { name: "Continue", exact: true }).click();
    await page
      .getByRole("heading", { name: "You're ready to use the composer." })
      .waitFor({ timeout: 60_000 });
    graduated = true;
    await page
      .getByRole("button", { name: "Take me to the composer" })
      .click({ timeout: 30_000 })
      .catch(() => undefined);

    const bodyText = await page.locator("body").innerText();
    const landed = graduated;
    if (!landed) {
      await mkdir(artifactsDir, { recursive: true });
      screenshot = resolve(artifactsDir, `run-${index}-failure.png`);
      await page.screenshot({ path: screenshot, fullPage: true });
    }

    const evidence = sessionId === null ? null : await fetchSessionEvidence(token, sessionId);
    const ok =
      landed &&
      evidence?.state_exists === true &&
      evidence?.tutorial_assumption_contract_ok === true &&
      apiFailures.length === 0 &&
      consoleErrors.length === 0;
    const diagnostic =
      !ok && diagnosticQuestion.trim() && sessionId !== null
        ? await askDiagnosticQuestion(token, sessionId, diagnosticQuestion.trim()).catch((error) => ({
            question: diagnosticQuestion.trim(),
            error: error instanceof Error ? error.message : String(error),
          }))
        : null;
    return {
      index,
      ok,
      session_id: sessionId,
      landed,
      graduated,
      steps,
      api_failures: apiFailures,
      console_errors: consoleErrors,
      screenshot,
      evidence,
      diagnostic,
      body_excerpt: landed ? null : bodyText.replace(/\s+/g, " ").slice(0, 800),
    };
  } catch (error) {
    await mkdir(artifactsDir, { recursive: true });
    screenshot = resolve(artifactsDir, `run-${index}-exception.png`);
    await page.screenshot({ path: screenshot, fullPage: true }).catch(() => undefined);
    return {
      index,
      ok: false,
      session_id: sessionId,
      landed: false,
      graduated,
      steps,
      api_failures: apiFailures,
      console_errors: consoleErrors,
      screenshot,
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    await context.close();
  }
}

async function main() {
  await mkdir(artifactsDir, { recursive: true });
  const token = await login();
  const browser = await chromium.launch({ headless: process.env.PLAYWRIGHT_HEADLESS !== "0" });
  const results = [];

  try {
    for (let i = 1; i <= runs; i += 1) {
      console.log(`[tutorial-harness] run ${i}/${runs}`);
      const result = await runOne(browser, token, i);
      results.push(result);
      console.log(
        `[tutorial-harness] run ${i}/${runs} ${result.ok ? "ok" : "failed"} session=${result.session_id ?? "none"}`,
      );
    }
  } finally {
    await browser.close();
  }

  const summary = {
    base_url: baseURL,
    username,
    prompt_override: promptOverride || null,
    expected_vague_term: expectedVagueTerm || null,
    diagnostic_question: diagnosticQuestion || null,
    runs,
    passed: results.filter((result) => result.ok).length,
    failed: results.filter((result) => !result.ok).length,
    results,
  };
  const summaryPath = resolve(artifactsDir, "summary.json");
  await writeFile(summaryPath, JSON.stringify(summary, null, 2), "utf8");
  console.log(JSON.stringify(summary, null, 2));
  console.log(`[tutorial-harness] wrote ${summaryPath}`);
  if (summary.failed > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exit(1);
});
