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

function playwrightConfigSource(): string {
  return readFileSync("playwright.config.ts", "utf8");
}

describe("playwright backend webServer isolation", () => {
  it("always starts the managed backend with the E2E environment", () => {
    const backend = backendWebServerBlock();

    expect(backend).toContain("uv run --extra webui python -m uvicorn");
    expect(backend).toContain("env: composerSettingsEnv");
    expect(backend).toContain("reuseExistingServer: false");
    expect(backend).not.toContain("reuseExistingServer: !isCI");
  });

  it("derives frontend and backend ports from environment variables", () => {
    const source = playwrightConfigSource();

    expect(source).toContain('portFromEnv("PLAYWRIGHT_FRONTEND_PORT", 5173)');
    expect(source).toContain('portFromEnv("PLAYWRIGHT_BACKEND_PORT", 8451)');
    expect(source).toContain("PLAYWRIGHT_FRONTEND_BASE_URL");
    expect(source).toContain("PLAYWRIGHT_BACKEND_BASE_URL");
    expect(source).toContain("npm run dev -- --host 127.0.0.1 --port");
  });
});

describe("vite proxy backend isolation", () => {
  it("uses the same dynamic backend port as Playwright", () => {
    const source = readFileSync("vite.config.ts", "utf8");

    expect(source).toContain("PLAYWRIGHT_FRONTEND_PORT");
    expect(source).toContain("PLAYWRIGHT_BACKEND_PORT");
    expect(source).toContain("http://127.0.0.1:${backendPort}");
    expect(source).toContain("ws://127.0.0.1:${backendPort}");
  });
});
