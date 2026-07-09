import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

// The design-system merge (d56197fde) added the components/ui primitive library
// — including ui/Input, which emits .input / .input-mono / .field-label /
// .field-hint — plus the canonical rules in the standalone website/tokens/
// primitives.css. That standalone tree is NOT in the app's Vite bundle, so the
// app stylesheet defined none of those selectors and the migrated LoginPage
// exemplar rendered its username/password fields as default browser inputs.
//
// These tests follow App.tsx's import path (the styles/index.css @import barrel)
// exactly like colorContrast.test.ts, so a primitive rule that is missing from
// the app cascade (not merely from an ad-hoc file) fails here.
const stylesheetBarrel = readFileSync("src/styles/index.css", "utf8");
const appCss = [
  ...Array.from(stylesheetBarrel.matchAll(/@import\s+"(?<path>[^"]+)";/g)).map((match) => {
    const importPath = match.groups?.path;
    if (importPath === undefined) {
      throw new Error("styles/index.css import regex produced no path");
    }
    return readFileSync(fileURLToPath(new URL(importPath, import.meta.url)), "utf8");
  }),
].join("\n");

describe("ui/Input primitive classes are wired into the app stylesheet barrel", () => {
  it("defines the base .input control rule with a strong border", () => {
    // The canonical Inputs block is `.input, .textarea, .select { … border … }`.
    expect(appCss).toMatch(/\.input[\s,][\s\S]*?border:[^;]*var\(--color-border-strong\)/);
  });

  it("defines the .field-label class the Input component renders above the control", () => {
    expect(appCss).toContain(".field-label");
  });

  it("defines the .field-hint class the Input component renders below the control", () => {
    expect(appCss).toContain(".field-hint");
  });

  it("defines the .input-mono forensic-register variant", () => {
    expect(appCss).toContain(".input-mono");
  });

  it("gives .input a visible focus outline so keyboard focus is never invisible", () => {
    expect(appCss).toMatch(/\.input:focus-visible[\s\S]*?outline:/);
  });
});
