import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

import { describe, expect, it } from "vitest";

// Regression net for elspeth-400f95605c: header.css referenced
// var(--color-bg-hover, rgba(143, 200, 200, 0.08)) but --color-bg-hover was
// defined NOWHERE — so the hardcoded dark-tinted fallback rendered in both
// themes. This class of bug is silent by construction (the fallback masks the
// missing token), so we gate it structurally: every custom property referenced
// by var() in any stylesheet must be DEFINED in some stylesheet. A fallback
// value does not excuse an undefined token — a fallback is a per-call-site
// palette fork that cannot be theme-paired.
//
// cwd-relative path per the established test idiom (GraphView.test.tsx,
// colorContrast.test.ts read stylesheets the same way; vitest runs from the
// frontend root).
const srcRoot = "src";

function walkCssFiles(dir: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      found.push(...walkCssFiles(path));
    } else if (entry.endsWith(".css")) {
      found.push(path);
    }
  }
  return found;
}

describe("custom property references (elspeth-400f95605c)", () => {
  it("only references custom properties that are defined in a stylesheet", () => {
    const cssFiles = walkCssFiles(srcRoot);
    expect(cssFiles.length).toBeGreaterThan(0);

    const defined = new Set<string>();
    const references = new Map<string, Set<string>>();

    for (const file of cssFiles) {
      const css = readFileSync(file, "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
      for (const definition of css.matchAll(/(--[\w-]+)\s*:/g)) {
        defined.add(definition[1]);
      }
      for (const reference of css.matchAll(/var\(\s*(--[\w-]+)/g)) {
        const name = reference[1];
        if (!references.has(name)) {
          references.set(name, new Set());
        }
        references.get(name)!.add(relative(srcRoot, file));
      }
    }

    const undefinedReferences = [...references.entries()]
      .filter(([name]) => !defined.has(name))
      .map(([name, files]) => `${name} (referenced by ${[...files].join(", ")})`);

    expect(
      undefinedReferences,
      "var() references to custom properties that no stylesheet defines — " +
        "define the token per theme in styles/tokens.css or use an existing " +
        "theme-paired token; do not paper over it with a var() fallback",
    ).toEqual([]);
  });
});
