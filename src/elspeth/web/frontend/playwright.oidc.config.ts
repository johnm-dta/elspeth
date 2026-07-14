import { defineConfig, devices } from "@playwright/test";

const isListOnly = process.argv.includes("--list");
const stagingBaseUrl = isListOnly ? "https://oidc-list.invalid" : process.env.STAGING_BASE_URL;

if (stagingBaseUrl === undefined) {
  throw new Error("oidc_staging_base_url");
}

let parsed: URL;
try {
  parsed = new URL(stagingBaseUrl);
} catch {
  throw new Error("oidc_staging_base_url");
}
if (
  parsed.protocol !== "https:" ||
  parsed.username !== "" ||
  parsed.password !== "" ||
  parsed.pathname !== "/" ||
  parsed.search !== "" ||
  parsed.hash !== "" ||
  parsed.origin !== stagingBaseUrl
) {
  throw new Error("oidc_staging_base_url");
}

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: "aws-ecs-oidc.staging.spec.ts",
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 10_000 },
  reporter: [["./tests/e2e/harness/oidc-redacting-reporter.ts"]],
  use: {
    baseURL: stagingBaseUrl,
    trace: "off",
    video: "off",
    screenshot: "off",
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
