// Staging-targeted Playwright config. Runs the existing specs against an
// already-deployed ELSPETH (elspeth.foundryside.dev by default) instead
// of spawning a local backend + frontend.
//
// Differences from playwright.config.ts:
//   - No webServer block. The deployment is expected to be up.
//   - baseURL points at the staging frontend origin.
//   - storageState is written by the staging globalSetup (which logs in
//     against staging using STAGING_USERNAME / STAGING_PASSWORD) and is
//     a separate file from the local user.json.
//   - Specs that talk to the backend via helpers/api.ts read
//     PLAYWRIGHT_BACKEND_BASE_URL from the environment, set here to the
//     same staging origin.
//
// Invocation:
//   STAGING_BASE_URL=https://elspeth.foundryside.dev \
//   STAGING_USERNAME=dta_user STAGING_PASSWORD=dta_pass \
//   PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
//   npx playwright test --config=playwright.staging.config.ts composer-preferences

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, devices } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const STORAGE_STATE_PATH = resolve(
  HERE,
  "tests",
  "e2e",
  ".auth",
  "staging-user.json",
);

const STAGING_BASE_URL =
  process.env.STAGING_BASE_URL ?? "https://elspeth.foundryside.dev";

export default defineConfig({
  testDir: "./tests/e2e",
  // `**/*.test.ts` are vitest unit tests (e.g. harness/classify.test.ts) which
  // import `vitest` — that collides with Playwright's expect. Playwright owns
  // `*.spec.ts`; vitest owns `*.test.ts`.
  testIgnore: ["**/setup/**", "**/page-objects/**", "**/helpers/**", "**/*.test.ts"],
  // Sequential, single-worker for staging — we are talking to a shared
  // service and don't want parallel test runs colliding on the same
  // dta_user account's preferences row.
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  expect: {
    timeout: 10_000,
  },
  globalSetup: "./tests/e2e/setup/staging-global-setup.ts",
  use: {
    baseURL: STAGING_BASE_URL,
    trace: "retain-on-failure",
    video: "retain-on-failure",
    screenshot: "only-on-failure",
    storageState: STORAGE_STATE_PATH,
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    // Staging uses a real TLS cert; this is here only as a safety net
    // for ad-hoc runs against self-signed deployments. Default false.
    ignoreHTTPSErrors: process.env.STAGING_IGNORE_HTTPS === "1",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
