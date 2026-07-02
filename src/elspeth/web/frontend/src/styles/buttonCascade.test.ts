import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

// Regression net for elspeth-b293bbd3c0: the base .btn / .btn-compact hover
// rules once guarded with bare :not(:disabled):not([aria-disabled="true"])
// chains, which pushed them to specificity (0,4,0) — ABOVE the variant hovers
// (.btn-primary:hover / .btn-danger:hover at (0,3,0)). Every filled
// primary/danger button dropped to the translucent surface wash on hover
// (white-on-wash measured 1.12:1 live). The fix keeps the guards inside
// :where(), which contributes zero specificity. These tests parse the real
// selectors out of the stylesheet barrel and compute their specificity, so
// the cascade ordering cannot silently regress again.
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

const cssWithoutComments = appCss.replace(/\/\*[\s\S]*?\*\//g, "");

/** Find the selector of the rule that starts with `prefix` (e.g. ".btn:hover"). */
function findSelectorStartingWith(prefix: string): string {
  for (const match of cssWithoutComments.matchAll(/([^{}]+)\{[^{}]*\}/g)) {
    for (const selector of match[1].split(",")) {
      const trimmed = selector.trim();
      if (trimmed.startsWith(prefix)) {
        return trimmed;
      }
    }
  }
  throw new Error(`No rule found with a selector starting with ${prefix}`);
}

/**
 * Class/attribute/pseudo-class specificity component (the "B" of (A,B,C)) for
 * the simple flat selectors used by the button system. Handles the two cases
 * that matter here: `:where(...)` contributes zero, and `:not(<simple>)`
 * contributes its argument's specificity.
 */
function classLevelSpecificity(selector: string): number {
  // :where(...) — zero specificity, drop the whole construct (the argument
  // may itself contain one level of parentheses, e.g. :where(:not(:disabled))).
  let s = selector.replace(/:where\((?:[^()]|\([^()]*\))*\)/g, "");
  // :not(<arg>) — counts as its argument; unwrap so the argument is counted.
  s = s.replace(/:not\(([^)]*)\)/g, "$1");
  const classes = s.match(/\.[\w-]+/g) ?? [];
  const attributes = s.match(/\[[^\]]+\]/g) ?? [];
  // Pseudo-classes (single colon), not pseudo-elements (double colon).
  const pseudoClasses = s.match(/(?<!:):[\w-]+/g) ?? [];
  return classes.length + attributes.length + pseudoClasses.length;
}

describe("button hover cascade (elspeth-b293bbd3c0)", () => {
  it("keeps the base .btn hover below the variant hovers", () => {
    const baseHover = findSelectorStartingWith(".btn:hover");
    const primaryHover = findSelectorStartingWith(".btn-primary:hover");
    const dangerHover = findSelectorStartingWith(".btn-danger:hover");

    expect(
      classLevelSpecificity(baseHover),
      `base hover selector "${baseHover}" must stay below the variant hovers — keep its guards inside :where()`,
    ).toBeLessThan(classLevelSpecificity(primaryHover));
    expect(classLevelSpecificity(baseHover)).toBeLessThan(
      classLevelSpecificity(dangerHover),
    );
  });

  it("keeps the base .btn-compact hover below the variant hovers", () => {
    // .btn-compact composes with .btn-primary at real call sites (e.g. the
    // inline-source fallback Accept button), so its base hover is subject to
    // the same cascade constraint.
    const compactHover = findSelectorStartingWith(".btn-compact:hover");
    const primaryHover = findSelectorStartingWith(".btn-primary:hover");

    expect(classLevelSpecificity(compactHover)).toBeLessThan(
      classLevelSpecificity(primaryHover),
    );
  });

  it("keeps the ghost hover above the base hover so ghost buttons stay transparent", () => {
    const baseHover = findSelectorStartingWith(".btn:hover");
    const ghostHover = findSelectorStartingWith(".btn-ghost:hover");

    expect(classLevelSpecificity(ghostHover)).toBeGreaterThan(
      classLevelSpecificity(baseHover),
    );
  });

  it("keeps the variant hover fills as solid tokens", () => {
    // The variant hover rules are what the user sees mid-click; they must
    // keep the solid filled tokens (not a translucent wash).
    const primaryHoverRule = /\.btn-primary:hover[^{]*\{([^}]*)\}/.exec(cssWithoutComments);
    const dangerHoverRule = /\.btn-danger:hover[^{]*\{([^}]*)\}/.exec(cssWithoutComments);
    expect(primaryHoverRule).not.toBeNull();
    expect(dangerHoverRule).not.toBeNull();
    expect(primaryHoverRule![1]).toContain(
      "background-color: var(--color-btn-primary-bg-hover)",
    );
    expect(dangerHoverRule![1]).toContain(
      "background-color: var(--color-btn-danger-bg-hover)",
    );
  });

  it("retains the disabled/aria-disabled hover guards (zero-specificity form)", () => {
    // The guards must survive inside :where() — dropping them entirely would
    // re-enable hover feedback on disabled buttons.
    const baseHover = findSelectorStartingWith(".btn:hover");
    const compactHover = findSelectorStartingWith(".btn-compact:hover");
    for (const selector of [baseHover, compactHover]) {
      expect(selector).toContain(":not(:disabled)");
      expect(selector).toContain(':not([aria-disabled="true"])');
    }
  });
});
