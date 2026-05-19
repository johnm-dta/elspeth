import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

function backendWebServerBlock(): string {
  const source = readFileSync("playwright.config.ts", "utf8");
  const match = source.match(
    /command:\s*\n\s*"uv run --extra webui python -m uvicorn elspeth\.web\.app:create_app --factory "[\s\S]*?env: composerSettingsEnv,\s*}/,
  );
  if (match === null) {
    throw new Error("Could not locate the Playwright backend webServer block");
  }
  return match[0];
}

describe("playwright backend webServer isolation", () => {
  it("always starts the managed backend with the E2E environment", () => {
    const backend = backendWebServerBlock();

    expect(backend).toContain("uv run --extra webui python -m uvicorn");
    expect(backend).toContain("env: composerSettingsEnv");
    expect(backend).toContain("reuseExistingServer: false");
    expect(backend).not.toContain("reuseExistingServer: !isCI");
  });
});
