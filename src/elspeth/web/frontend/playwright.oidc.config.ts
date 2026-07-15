import { defineConfig, devices } from "@playwright/test";

const isListOnly = process.argv.includes("--list");
const stagingBaseUrl = isListOnly ? "https://oidc-list.invalid" : process.env.STAGING_BASE_URL;
const tlsSpkiSha256 = process.env.OIDC_TLS_SPKI_SHA256;

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
if (tlsSpkiSha256 !== undefined && !/^[A-Za-z0-9+/]{43}=$/.test(tlsSpkiSha256)) {
  throw new Error("oidc_tls_spki_sha256");
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
      use: {
        ...devices["Desktop Chrome"],
        launchOptions:
          tlsSpkiSha256 === undefined
            ? undefined
            : { args: [`--ignore-certificate-errors-spki-list=${tlsSpkiSha256}`] },
      },
    },
  ],
});
