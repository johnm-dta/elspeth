import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

// .status-badge* rules live in shared.css. We read shared.css (not the
// tokens or barrel file) so the negative assertion below has actual content
// to inspect — reading a file that doesn't define .status-badge* would let
// regressions sneak through.
const appCss = readFileSync("src/styles/shared.css", "utf8");

describe("status badge accessibility", () => {
  it("does not inject status symbols through CSS pseudo-content", () => {
    expect(appCss).not.toMatch(/\.status-badge[\w-]*::before\s*\{/);
    expect(appCss).not.toMatch(/\.status-badge[\w-]*::before[\s\S]*content:/);
  });
});
