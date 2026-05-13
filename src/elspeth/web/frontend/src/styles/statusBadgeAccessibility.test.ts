import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const appCss = readFileSync("src/App.css", "utf8");

describe("status badge accessibility", () => {
  it("does not inject status symbols through CSS pseudo-content", () => {
    expect(appCss).not.toMatch(/\.status-badge[\w-]*::before\s*\{/);
    expect(appCss).not.toMatch(/\.status-badge[\w-]*::before[\s\S]*content:/);
  });
});
