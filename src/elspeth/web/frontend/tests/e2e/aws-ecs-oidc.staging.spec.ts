import { createHash } from "node:crypto";

import { test, type APIRequestContext, type APIResponse, type Locator, type Page } from "@playwright/test";

import {
  OIDC_EVIDENCE_PHASES,
  OidcEvidenceError,
  buildOidcEvidence,
  validateAccessToken,
  validateAuthConfig,
  writeOidcEvidence,
  type OidcAudienceClaim,
  type OidcEvidencePhase,
} from "./harness/oidc-evidence";

const MAX_API_RESPONSE_BYTES = 1024 * 1024;
const SESSION_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const PKCE_CHALLENGE = /^[A-Za-z0-9_-]{43}$/;
const PKCE_VERIFIER = /^[A-Za-z0-9._~-]{43,128}$/;

function requiredEnvironment(name: string): string {
  const value = process.env[name];
  if (value === undefined || value === "" || Buffer.byteLength(value, "utf8") > 16 * 1024) {
    throw new OidcEvidenceError("oidc_environment");
  }
  return value;
}

function requireOidc(condition: boolean, check: string): asserts condition {
  if (!condition) throw new OidcEvidenceError(check);
}

function exactStagingOrigin(raw: string): string {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new OidcEvidenceError("oidc_staging_origin");
  }
  requireOidc(
    parsed.protocol === "https:" &&
      parsed.username === "" &&
      parsed.password === "" &&
      parsed.pathname === "/" &&
      parsed.search === "" &&
      parsed.hash === "" &&
      parsed.origin === raw,
    "oidc_staging_origin",
  );
  return raw;
}

async function boundedJson(response: APIResponse, check: string): Promise<Record<string, unknown>> {
  const body = await response.body();
  requireOidc(body.length <= MAX_API_RESPONSE_BYTES, check);
  let payload: unknown;
  try {
    payload = JSON.parse(body.toString("utf8"));
  } catch {
    throw new OidcEvidenceError(check);
  }
  requireOidc(payload !== null && typeof payload === "object" && !Array.isArray(payload), check);
  return payload as Record<string, unknown>;
}

async function firstVisible(page: Page, selector: string, check: string): Promise<Locator> {
  const candidates = page.locator(selector);
  const count = await candidates.count();
  for (let index = 0; index < count; index += 1) {
    const candidate = candidates.nth(index);
    if (await candidate.isVisible()) return candidate;
  }
  throw new OidcEvidenceError(check);
}

async function apiCall(
  request: APIRequestContext,
  method: "get" | "post" | "delete",
  url: string,
  token: string,
  data?: object,
): Promise<APIResponse> {
  return request[method](url, {
    headers: { Authorization: `Bearer ${token}` },
    ...(data === undefined ? {} : { data }),
    failOnStatusCode: false,
    maxRedirects: 0,
    timeout: 15_000,
  });
}

test("completes public-client OIDC and session round trip", async ({ page, request }) => {
  const stagingOrigin = exactStagingOrigin(requiredEnvironment("STAGING_BASE_URL"));
  const username = requiredEnvironment("OIDC_TEST_USERNAME");
  const password = requiredEnvironment("OIDC_TEST_PASSWORD");
  const issuer = requiredEnvironment("OIDC_EXPECTED_ISSUER");
  const audience = requiredEnvironment("OIDC_EXPECTED_AUDIENCE");
  const authorizationOrigin = requiredEnvironment("OIDC_EXPECTED_AUTHORIZATION_ORIGIN");
  const audienceClaim = requiredEnvironment("OIDC_EXPECTED_AUDIENCE_CLAIM") as OidcAudienceClaim;
  const evidencePhase = requiredEnvironment("OIDC_EVIDENCE_PHASE") as OidcEvidencePhase;
  const evidenceFile = requiredEnvironment("OIDC_EVIDENCE_FILE");
  requireOidc(OIDC_EVIDENCE_PHASES.includes(evidencePhase), "oidc_evidence_phase");

  const configResponse = await request.get(`${stagingOrigin}/api/auth/config`, {
    failOnStatusCode: false,
    maxRedirects: 0,
    timeout: 15_000,
  });
  requireOidc(configResponse.status() === 200, "oidc_auth_config_status");
  const authConfig = validateAuthConfig(await boundedJson(configResponse, "oidc_auth_config_body"), {
    issuer,
    audience,
    authorizationOrigin,
  });

  let callbackObserved = false;
  let callbackHadCode = false;
  let callbackHadState = false;
  let callbackHadToken = false;
  page.on("framenavigated", (frame) => {
    if (frame !== page.mainFrame()) return;
    try {
      const navigated = new URL(frame.url());
      if (navigated.origin === stagingOrigin && (navigated.search !== "" || navigated.hash !== "")) {
        const fragment = new URLSearchParams(navigated.hash.replace(/^#/, ""));
        callbackObserved = true;
        callbackHadCode = navigated.searchParams.has("code");
        callbackHadState = navigated.searchParams.has("state");
        callbackHadToken =
          navigated.searchParams.has("token") ||
          navigated.searchParams.has("access_token") ||
          fragment.has("token") ||
          fragment.has("access_token");
      }
    } catch {
      // Non-URL intermediate browser states are not callback evidence.
    }
  });

  let expectedChallenge = "";
  let tokenExchangeChecked = false;
  let resolveExchange!: () => void;
  const exchangeObserved = new Promise<void>((resolve) => {
    resolveExchange = resolve;
  });
  await page.route(authConfig.tokenEndpoint, async (route) => {
    try {
      const tokenRequest = route.request();
      const headers = await tokenRequest.allHeaders();
      const form = new URLSearchParams(tokenRequest.postData() ?? "");
      const current = new URL(page.url());
      const transactionRemoved = await page.evaluate(() => sessionStorage.getItem("oidc_transaction") === null);
      requireOidc(tokenRequest.method() === "POST", "oidc_token_method");
      requireOidc(current.origin === stagingOrigin && current.search === "" && current.hash === "", "oidc_callback_scrub");
      requireOidc(transactionRemoved, "oidc_transaction_cleanup");
      requireOidc(!("authorization" in headers), "oidc_token_authorization");
      requireOidc(!form.has("client_secret"), "oidc_client_secret");
      requireOidc(form.get("grant_type") === "authorization_code", "oidc_token_grant");
      requireOidc(form.get("client_id") === audience, "oidc_token_client");
      requireOidc(form.get("redirect_uri") === `${stagingOrigin}/`, "oidc_token_redirect");
      const verifier = form.get("code_verifier") ?? "";
      requireOidc(PKCE_VERIFIER.test(verifier), "oidc_pkce_verifier");
      const derivedChallenge = createHash("sha256").update(verifier).digest("base64url");
      requireOidc(derivedChallenge === expectedChallenge, "oidc_pkce_binding");
      tokenExchangeChecked = true;
      await route.continue();
    } catch {
      await route.abort("blockedbyclient");
    } finally {
      resolveExchange();
    }
  });

  await page.goto(stagingOrigin, { waitUntil: "domcontentloaded" });
  const authorizationRequestPromise = page.waitForRequest(
    (candidate) => {
      try {
        return new URL(candidate.url()).origin === authorizationOrigin;
      } catch {
        return false;
      }
    },
    { timeout: 15_000 },
  );
  await page.getByRole("button", { name: "Sign in with single sign-on", exact: true }).click();
  const authorizationRequest = await authorizationRequestPromise;
  const authorizationUrl = new URL(authorizationRequest.url());
  requireOidc(authorizationUrl.origin === authorizationOrigin, "oidc_authorization_origin");
  requireOidc(authorizationUrl.searchParams.get("response_type") === "code", "oidc_authorization_flow");
  requireOidc(authorizationUrl.searchParams.get("client_id") === audience, "oidc_authorization_client");
  requireOidc(authorizationUrl.searchParams.get("code_challenge_method") === "S256", "oidc_pkce_method");
  expectedChallenge = authorizationUrl.searchParams.get("code_challenge") ?? "";
  requireOidc(PKCE_CHALLENGE.test(expectedChallenge), "oidc_pkce_challenge");
  requireOidc(!authorizationUrl.searchParams.has("code_verifier"), "oidc_authorization_verifier");
  requireOidc(!authorizationUrl.searchParams.has("client_secret"), "oidc_authorization_secret");
  requireOidc(authorizationUrl.searchParams.get("redirect_uri") === `${stagingOrigin}/`, "oidc_authorization_redirect");

  await (await firstVisible(page, 'input[name="username"], input#signInFormUsername', "oidc_username_control")).fill(username);
  await (await firstVisible(page, 'input[name="password"], input#signInFormPassword', "oidc_password_control")).fill(password);
  await (
    await firstVisible(
      page,
      'input[name="signInSubmitButton"], button:has-text("Sign in")',
      "oidc_submit_control",
    )
  ).click();

  await exchangeObserved;
  requireOidc(tokenExchangeChecked, "oidc_token_exchange");
  await page.waitForFunction(() => typeof localStorage.getItem("auth_token") === "string", undefined, { timeout: 30_000 });
  const accessToken = await page.evaluate(() => localStorage.getItem("auth_token"));
  requireOidc(typeof accessToken === "string" && accessToken.length > 0, "oidc_access_token");
  requireOidc(callbackObserved && callbackHadCode && callbackHadState && !callbackHadToken, "oidc_callback_observation");
  const claims = validateAccessToken(accessToken, { issuer, audience, authorizationOrigin, audienceClaim });

  const authMe = await apiCall(request, "get", `${stagingOrigin}/api/auth/me`, accessToken);
  const authMeStatus = authMe.status();
  requireOidc(authMeStatus === 200, "oidc_auth_me");

  let sessionId: string | null = null;
  let sessionCreateStatus = 0;
  let sessionReadStatus = 0;
  let sessionDeleteStatus = 0;
  let sessionFailure: unknown = null;
  try {
    const created = await apiCall(request, "post", `${stagingOrigin}/api/sessions`, accessToken, {});
    sessionCreateStatus = created.status();
    requireOidc(sessionCreateStatus === 201, "oidc_session_create");
    const createdPayload = await boundedJson(created, "oidc_session_create_body");
    requireOidc(typeof createdPayload.id === "string" && SESSION_ID.test(createdPayload.id), "oidc_session_identity");
    sessionId = createdPayload.id;
    const read = await apiCall(request, "get", `${stagingOrigin}/api/sessions/${sessionId}`, accessToken);
    sessionReadStatus = read.status();
    requireOidc(sessionReadStatus === 200, "oidc_session_read");
    const readPayload = await boundedJson(read, "oidc_session_read_body");
    requireOidc(readPayload.id === sessionId, "oidc_session_binding");
  } catch (error) {
    sessionFailure = error;
  } finally {
    if (sessionId !== null) {
      const deleted = await apiCall(request, "delete", `${stagingOrigin}/api/sessions/${sessionId}`, accessToken);
      sessionDeleteStatus = deleted.status();
    }
  }
  if (sessionFailure !== null) throw sessionFailure;
  requireOidc(sessionDeleteStatus === 204, "oidc_session_delete");

  const evidence = buildOidcEvidence({
    phase: evidencePhase,
    timestamp: new Date().toISOString(),
    issuer,
    authorizationOrigin,
    audienceClaim,
    audience,
    subjectSha256: claims.subjectSha256,
    authMeStatus,
    sessionCreateStatus,
    sessionReadStatus,
    sessionDeleteStatus,
  });
  writeOidcEvidence(evidenceFile, evidence);
});
