import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { request } from "@playwright/test";

// Placeholder credential consumed only by the local playwright webServer,
// which spins up a fresh ELSPETH_WEB__data_dir per test session. Not a
// real password; the hook's credential scanner flags any high-entropy
// quoted string here, so keep this short and obviously synthetic.
const TEST_USER = {
  username: "e2e-tester",
  password: "pw-placeholder", // secret-scan: allow-this-line
  display_name: "E2E Tester",
};

const FRONTEND_PORT = process.env.PLAYWRIGHT_FRONTEND_PORT ?? "5173";
const BACKEND_PORT = process.env.PLAYWRIGHT_BACKEND_PORT ?? "8451";
const FRONTEND_ORIGIN =
  process.env.PLAYWRIGHT_FRONTEND_BASE_URL ?? `http://127.0.0.1:${FRONTEND_PORT}`;
const BACKEND_BASE_URL =
  process.env.PLAYWRIGHT_BACKEND_BASE_URL ?? `http://127.0.0.1:${BACKEND_PORT}`;
const TOKEN_KEY = "auth_token";

async function obtainToken(apiBaseURL: string): Promise<string> {
  const ctx = await request.newContext({ baseURL: apiBaseURL });
  try {
    const registerResp = await ctx.post("/api/auth/register", {
      data: TEST_USER,
    });

    if (registerResp.ok()) {
      const body = (await registerResp.json()) as { access_token: string };
      return body.access_token;
    }

    if (registerResp.status() !== 409) {
      const detail = await registerResp.text();
      throw new Error(
        `register failed (${registerResp.status()}): ${detail.slice(0, 500)}`,
      );
    }

    const loginResp = await ctx.post("/api/auth/login", {
      data: { username: TEST_USER.username, password: TEST_USER.password },
    });
    if (!loginResp.ok()) {
      const detail = await loginResp.text();
      throw new Error(
        `login fallback failed (${loginResp.status()}): ${detail.slice(0, 500)}`,
      );
    }
    const body = (await loginResp.json()) as { access_token: string };
    return body.access_token;
  } finally {
    await ctx.dispose();
  }
}

async function markTutorialCompleted(apiBaseURL: string, token: string): Promise<void> {
  const ctx = await request.newContext({
    baseURL: apiBaseURL,
    extraHTTPHeaders: { Authorization: `Bearer ${token}` },
  });
  try {
    const resp = await ctx.patch("/api/composer-preferences", {
      data: {
        default_mode: "guided",
        tutorial_completed_at: new Date().toISOString(),
      },
    });
    if (!resp.ok()) {
      const detail = await resp.text();
      throw new Error(
        `mark tutorial completed failed (${resp.status()}): ${detail.slice(0, 500)}`,
      );
    }
  } finally {
    await ctx.dispose();
  }
}

function writeStorageState(path: string, token: string): void {
  const state = {
    cookies: [],
    origins: [
      {
        origin: FRONTEND_ORIGIN,
        localStorage: [{ name: TOKEN_KEY, value: token }],
      },
    ],
  };
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(state, null, 2), { encoding: "utf-8" });
}

// The storageState path is resolved relative to this file rather than
// FullConfig.rootDir or process.cwd() — both vary in confusing ways across
// Playwright versions. import.meta.url is unambiguous.
const HERE = dirname(fileURLToPath(import.meta.url));
const STORAGE_STATE_PATH = resolve(HERE, "..", ".auth", "user.json");

export default async function globalSetup(): Promise<void> {
  const token = await obtainToken(BACKEND_BASE_URL);
  // Most E2E specs predate the first-run tutorial and assert on the normal
  // composer shell. Keep the baseline tester out of tutorial mode; specs that
  // exercise the tutorial explicitly reset the preference in their own setup.
  await markTutorialCompleted(BACKEND_BASE_URL, token);
  writeStorageState(STORAGE_STATE_PATH, token);
  // Confirm to the operator that globalSetup completed — otherwise a
  // mis-pathed write looks identical to "globalSetup never ran" in the
  // failure output.
  console.log(`[globalSetup] wrote storageState to ${STORAGE_STATE_PATH}`);
}
