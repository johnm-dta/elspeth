import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, devices } from "@playwright/test";

const FRONTEND_PORT = 5173;
const BACKEND_PORT = 8451;
const FRONTEND_URL = `http://localhost:${FRONTEND_PORT}`;
const BACKEND_HEALTH_URL = `http://127.0.0.1:${BACKEND_PORT}/api/health`;

const REPO_ROOT_FROM_FRONTEND = "../../../..";

const E2E_DATA_DIR = "./.e2e-data";

// Anchor path resolution to this config file rather than process.cwd() —
// Playwright's runtime cwd varies (config-load vs test-run vs globalSetup)
// and a relative storageState path produced subtly different files in each
// phase. Keep it absolute here and share the same anchor with globalSetup.
const HERE = dirname(fileURLToPath(import.meta.url));
const STORAGE_STATE_PATH = resolve(HERE, "tests", "e2e", ".auth", "user.json");

const isCI = !!process.env.CI;

const composerSettingsEnv: Record<string, string> = {
  ELSPETH_WEB__data_dir: E2E_DATA_DIR,
  ELSPETH_WEB__registration_mode: "open",
  ELSPETH_WEB__auth_provider: "local",
  ELSPETH_WEB__composer_max_composition_turns: "15",
  ELSPETH_WEB__composer_max_discovery_turns: "10",
  ELSPETH_WEB__composer_timeout_seconds: "180.0",
  ELSPETH_WEB__composer_rate_limit_per_minute: "60",
  ELSPETH_WEB__auth_rate_limit_per_minute: "120",
  // Placeholder JWT signing key for the local webServer instance. The
  // backend refuses startup if cors_origins contains a non-loopback host
  // and secret_key is left at its default; here we set it explicitly so
  // the ELSPETH_WEB__SECRET_KEY guard in src/elspeth/web/config.py is
  // satisfied even if a non-loopback CORS origin is configured later.
  ELSPETH_WEB__secret_key: "e2e-jwt-placeholder", // secret-scan: allow-this-line

  ELSPETH_WEB__cors_origins: JSON.stringify([FRONTEND_URL]),
};

export default defineConfig({
  testDir: "./tests/e2e",
  testIgnore: ["**/setup/**", "**/page-objects/**", "**/helpers/**"],
  fullyParallel: true,
  forbidOnly: isCI,
  retries: isCI ? 2 : 0,
  workers: isCI ? 2 : undefined,
  reporter: isCI ? [["github"], ["html", { open: "never" }]] : [["list"], ["html", { open: "never" }]],
  expect: {
    timeout: 5_000,
  },
  globalSetup: "./tests/e2e/setup/global-setup.ts",
  use: {
    baseURL: FRONTEND_URL,
    trace: "retain-on-failure",
    video: "retain-on-failure",
    screenshot: "only-on-failure",
    storageState: STORAGE_STATE_PATH,
    actionTimeout: 10_000,
    navigationTimeout: 15_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command:
        "uv run uvicorn elspeth.web.app:create_app --factory " +
        `--host 127.0.0.1 --port ${BACKEND_PORT}`,
      cwd: REPO_ROOT_FROM_FRONTEND,
      url: BACKEND_HEALTH_URL,
      reuseExistingServer: !isCI,
      timeout: 60_000,
      stdout: "pipe",
      stderr: "pipe",
      env: composerSettingsEnv,
    },
    {
      command: "npm run dev",
      url: FRONTEND_URL,
      reuseExistingServer: !isCI,
      timeout: 30_000,
      stdout: "pipe",
      stderr: "pipe",
    },
  ],
});
