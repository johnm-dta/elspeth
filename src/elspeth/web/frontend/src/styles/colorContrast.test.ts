import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

// Concatenate the runtime stylesheet barrel so coverage follows App.tsx's
// import path. Missing imports in styles/index.css must break these focused
// CSS tests instead of being masked by ad hoc direct-file reads.
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

it("loads inspected CSS through the runtime stylesheet barrel", () => {
  expect(stylesheetBarrel).toContain('@import "./tokens.css";');
  expect(stylesheetBarrel).toContain('@import "./shared.css";');
  expect(stylesheetBarrel).toContain('@import "./themes.css";');
  expect(appCss).toContain(".btn-primary");
});

function extractForcedColorsBlock(): string {
  const start = appCss.indexOf("@media (forced-colors: active)");
  if (start === -1) {
    return "";
  }

  const end = appCss.indexOf("\n}\n\n/*", start);
  return end === -1 ? appCss.slice(start) : appCss.slice(start, end + 3);
}

function extractRootToken(tokenName: string): string {
  const blockMatch = /^:root\s*\{([\s\S]*?)\n\}/m.exec(appCss);
  if (!blockMatch) {
    throw new Error("Could not find root token block in styles/tokens.css");
  }

  return extractTokenFromBlock(tokenName, blockMatch[1], "root");
}

function extractLightThemeToken(tokenName: string): string {
  const blockMatch = /\[data-theme="light"\]\s*\{([\s\S]*?)\n\}/.exec(appCss);
  if (!blockMatch) {
    throw new Error("Could not find light theme token block in styles/tokens.css");
  }

  return extractTokenFromBlock(tokenName, blockMatch[1], "light theme");
}

function extractTokenFromBlock(tokenName: string, block: string, blockName: string): string {
  const tokenMatch = new RegExp(`${tokenName}:\\s*(#[0-9a-fA-F]{6})\\s*;`).exec(block);
  if (!tokenMatch) {
    throw new Error(`Could not find ${tokenName} in ${blockName} token block`);
  }

  return tokenMatch[1];
}

function extractCssRule(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const ruleMatch = new RegExp(`${escaped}\\s*\\{([\\s\\S]*?)\\n\\}`).exec(appCss);
  if (!ruleMatch) {
    throw new Error(`Could not find CSS rule for ${selector}`);
  }
  return ruleMatch[1];
}

function channelToLinear(value: number): number {
  const normalized = value / 255;
  if (normalized <= 0.04045) {
    return normalized / 12.92;
  }
  return ((normalized + 0.055) / 1.055) ** 2.4;
}

function relativeLuminance(hex: string): number {
  const match = /^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/.exec(hex);
  if (!match) {
    throw new Error(`Expected six-digit hex colour, got ${hex}`);
  }

  const red = channelToLinear(Number.parseInt(match[1], 16));
  const green = channelToLinear(Number.parseInt(match[2], 16));
  const blue = channelToLinear(Number.parseInt(match[3], 16));
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(foreground: string, background: string): number {
  const foregroundLuminance = relativeLuminance(foreground);
  const backgroundLuminance = relativeLuminance(background);
  const lighter = Math.max(foregroundLuminance, backgroundLuminance);
  const darker = Math.min(foregroundLuminance, backgroundLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

describe("light theme colour contrast", () => {
  it("keeps muted body text at WCAG AA contrast against the light background", () => {
    const background = extractLightThemeToken("--color-bg");
    const mutedText = extractLightThemeToken("--color-text-muted");

    expect(contrastRatio(mutedText, background)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps the dark theme focus ring distinct from gate badges", () => {
    const background = extractRootToken("--color-bg");
    const gateBadge = extractRootToken("--color-badge-gate");
    const focusRing = extractRootToken("--color-focus-ring");

    expect(focusRing).not.toBe(gateBadge);
    expect(contrastRatio(focusRing, gateBadge)).toBeGreaterThanOrEqual(2);
    expect(contrastRatio(focusRing, background)).toBeGreaterThanOrEqual(3);
  });

  it("keeps the light theme focus ring distinct from gate badges", () => {
    const background = extractLightThemeToken("--color-bg");
    const gateBadge = extractLightThemeToken("--color-badge-gate");
    const focusRing = extractLightThemeToken("--color-focus-ring");

    expect(focusRing).not.toBe(gateBadge);
    expect(contrastRatio(focusRing, gateBadge)).toBeGreaterThanOrEqual(2);
    expect(contrastRatio(focusRing, background)).toBeGreaterThanOrEqual(3);
  });

  it("keeps primary button text at WCAG AA contrast in both themes", () => {
    const darkText = extractRootToken("--color-text-inverse");
    const darkBg = extractRootToken("--color-btn-primary-bg");
    const lightText = extractLightThemeToken("--color-text-inverse");
    const lightBg = extractLightThemeToken("--color-btn-primary-bg");

    expect(contrastRatio(darkText, darkBg)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightText, lightBg)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps danger button text at WCAG AA contrast in both themes", () => {
    const darkText = extractRootToken("--color-text-inverse");
    const darkBg = extractRootToken("--color-btn-danger-bg");
    const lightText = extractLightThemeToken("--color-text-inverse");
    const lightBg = extractLightThemeToken("--color-btn-danger-bg");

    expect(contrastRatio(darkText, darkBg)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightText, lightBg)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps info-coloured text at WCAG AA contrast in both themes", () => {
    // --color-info is used on the .tutorial-kicker (bold uppercase ~14px) and
    // on other "info" labels. The light-theme value was deepened from #2890b8
    // to #176d8a explicitly to pass this assertion; do not regress.
    const darkInfo = extractRootToken("--color-info");
    const darkBg = extractRootToken("--color-bg");
    const lightInfo = extractLightThemeToken("--color-info");
    const lightBg = extractLightThemeToken("--color-bg");

    expect(contrastRatio(darkInfo, darkBg)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightInfo, lightBg)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps link text at WCAG AA contrast in both themes", () => {
    // --color-link governs the tutorial skip-button and every inline link in
    // the app. Deepened in light theme to #176d8a so AA holds.
    const darkLink = extractRootToken("--color-link");
    const darkBg = extractRootToken("--color-bg");
    const lightLink = extractLightThemeToken("--color-link");
    const lightBg = extractLightThemeToken("--color-bg");

    expect(contrastRatio(darkLink, darkBg)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightLink, lightBg)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps state-positive text at WCAG AA contrast in both themes", () => {
    // --color-state-positive replaces the prior overloaded --color-success
    // usage on .tutorial-run-summary. Light value (#056e6c) chosen for AA.
    const darkPositive = extractRootToken("--color-state-positive");
    const darkBg = extractRootToken("--color-bg");
    const lightPositive = extractLightThemeToken("--color-state-positive");
    const lightBg = extractLightThemeToken("--color-bg");

    expect(contrastRatio(darkPositive, darkBg)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightPositive, lightBg)).toBeGreaterThanOrEqual(4.5);
  });
});

describe("disabled button contrast", () => {
  // The .btn:disabled / .btn[aria-disabled="true"] rule was migrated off
  // opacity:0.4 (which silently fails contrast against any non-white chrome)
  // to a tokenised background+text pair. We use --color-bg (page background)
  // rather than --color-surface-elevated because the latter is too close in
  // luminance to --color-text-muted in the dark theme (3.84:1, fails AA).
  // The chosen pair passes AA in both themes (dark 4.78:1, light 6.32:1).
  // These assertions are the design system's gate: if a future colour rebalance
  // pushes the pair below AA, this test fails before the regression ships.
  it("keeps --color-text-muted on --color-bg at AA in the dark theme", () => {
    const text = extractRootToken("--color-text-muted");
    const surface = extractRootToken("--color-bg");
    expect(contrastRatio(text, surface)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps --color-text-muted on --color-bg at AA in the light theme", () => {
    const text = extractLightThemeToken("--color-text-muted");
    const surface = extractLightThemeToken("--color-bg");
    expect(contrastRatio(text, surface)).toBeGreaterThanOrEqual(4.5);
  });

  it("verifies .btn:disabled uses the token pair rather than opacity dimming", () => {
    // Inspect the .btn rule body to confirm we did not regress to opacity:0.4
    // when the design system was simplified. The presence of color/background
    // declarations in the disabled selector is the load-bearing signal.
    const disabledMatch = /\.btn:disabled,\s*\n\s*\.btn\[aria-disabled="true"\]\s*\{([\s\S]*?)\n\}/.exec(appCss);
    expect(disabledMatch, ".btn:disabled rule must exist with token-based colours").not.toBeNull();
    const body = disabledMatch![1];
    expect(body).toContain("background-color: var(--color-bg)");
    expect(body).toContain("color: var(--color-text-muted)");
    expect(body).not.toMatch(/opacity:\s*0\.[0-9]+/);
  });
});

describe("base interaction tokens", () => {
  it("uses a 16px base font size token", () => {
    expect(appCss).toMatch(/--font-size-base:\s*16px;/);
  });

  it("sets the base button hit target to at least 44px via --size-control", () => {
    // .btn composes --size-control rather than redeclaring the literal 44px.
    // We verify both the composition AND the token's resolved value so a
    // future maintainer can't silently drop --size-control below the WCAG
    // 2.5.5 AAA floor.
    expect(extractCssRule(".btn")).toMatch(/min-height:\s*var\(--size-control\);/);
    expect(appCss).toMatch(/--size-control:\s*44px;/);
  });

  it("sets the compact button hit target via --size-control-compact (≥WCAG 2.5.8 AA)", () => {
    // .btn-compact is the chrome-row variant (header buttons, dense toolbars).
    // It MUST clear the WCAG 2.5.8 AA floor of 24×24 — we enforce ≥32px so the
    // hit target stays comfortable on touch screens.
    expect(extractCssRule(".btn-compact")).toMatch(
      /min-height:\s*var\(--size-control-compact\);/,
    );
    const compactMatch = /--size-control-compact:\s*(\d+)px;/.exec(appCss);
    expect(compactMatch, "--size-control-compact must be defined").not.toBeNull();
    expect(Number.parseInt(compactMatch![1], 10)).toBeGreaterThanOrEqual(32);
  });

  it("uses solid primary button tokens instead of low-alpha text-on-tint styling", () => {
    const primaryRule = extractCssRule(".btn-primary");

    expect(primaryRule).toContain("background-color: var(--color-btn-primary-bg);");
    expect(primaryRule).toContain("color: var(--color-text-inverse);");
    expect(primaryRule).not.toContain("rgba(");
  });

  it("uses solid danger button tokens instead of low-alpha text-on-tint styling", () => {
    const dangerRule = extractCssRule(".btn-danger");

    expect(dangerRule).toContain("background-color: var(--color-btn-danger-bg);");
    expect(dangerRule).toContain("color: var(--color-text-inverse);");
    expect(dangerRule).not.toContain("rgba(");
  });
});

describe("role-family surface contrast", () => {
  // The DTA/AGDS three-family role palette (navy chrome, teal workspace, warm
  // inspection paper) introduced --color-surface-nav, --color-surface-paper,
  // and re-bound --color-surface-inspector to a warm neutral. These assertions
  // gate that the new surfaces still host body text at WCAG AA (4.5:1) and
  // muted text at AA in both themes — the muted-on-warm pair is the tightest
  // and was the constraint that set the warm-neutral hex.

  it("keeps body text at AA on the navy navigation surface in both themes", () => {
    const darkText = extractRootToken("--color-text");
    const darkNav = extractRootToken("--color-surface-nav");
    const lightText = extractLightThemeToken("--color-text");
    const lightNav = extractLightThemeToken("--color-surface-nav");

    expect(contrastRatio(darkText, darkNav)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightText, lightNav)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps body text at AA on the warm inspector surface in both themes", () => {
    const darkText = extractRootToken("--color-text");
    const darkInspector = extractRootToken("--color-surface-inspector");
    const lightText = extractLightThemeToken("--color-text");
    const lightInspector = extractLightThemeToken("--color-surface-inspector");

    expect(contrastRatio(darkText, darkInspector)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightText, lightInspector)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps body text at AA on the paper modal surface in both themes", () => {
    const darkText = extractRootToken("--color-text");
    const darkPaper = extractRootToken("--color-surface-paper");
    const lightText = extractLightThemeToken("--color-text");
    const lightPaper = extractLightThemeToken("--color-surface-paper");

    expect(contrastRatio(darkText, darkPaper)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightText, lightPaper)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps muted text at AA on the warm inspector surface in both themes", () => {
    // This is the tightest pair in the role palette — muted text on the warm
    // neutral inspector. The warm-neutral hex (#2a2826 dark, #faf7f3 light)
    // was chosen at the bright end of "warm dark" / dark end of "warm light"
    // so muted text still passes 4.5:1. Tightening the warm-neutral toward
    // pure black/white would silently regress this; this assertion is the
    // gate.
    const darkMuted = extractRootToken("--color-text-muted");
    const darkInspector = extractRootToken("--color-surface-inspector");
    const lightMuted = extractLightThemeToken("--color-text-muted");
    const lightInspector = extractLightThemeToken("--color-surface-inspector");

    expect(contrastRatio(darkMuted, darkInspector)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightMuted, lightInspector)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps the coalesce badge hue distinct from --color-success in both themes", () => {
    // The original palette had --color-success and --color-badge-coalesce both
    // set to #14b0ae (byte-identical) — a "completed" badge next to a coalesce
    // node was indistinguishable. The remediation shifted coalesce cyan-ward.
    // Asserting strict inequality keeps the two tokens from re-converging in a
    // future palette rebalance.
    expect(extractRootToken("--color-badge-coalesce")).not.toBe(
      extractRootToken("--color-success"),
    );
    expect(extractLightThemeToken("--color-badge-coalesce")).not.toBe(
      extractLightThemeToken("--color-success"),
    );
  });
});

describe("forced-colors accessibility fallbacks", () => {
  it("defines system-color fallbacks for stateful high-contrast surfaces", () => {
    const forcedColorsBlock = extractForcedColorsBlock();

    expect(forcedColorsBlock).toContain("@media (forced-colors: active)");
    expect(forcedColorsBlock).toContain(".validation-banner-fail");
    expect(forcedColorsBlock).toContain(".alert-banner");
    expect(forcedColorsBlock).toContain(".type-badge-source");
    expect(forcedColorsBlock).toContain(".type-badge-transform");
    expect(forcedColorsBlock).toContain(".type-badge-gate");
    expect(forcedColorsBlock).toContain(".type-badge-sink");
    expect(forcedColorsBlock).toContain(".type-badge-aggregation");
    expect(forcedColorsBlock).toContain(".type-badge-coalesce");
    expect(forcedColorsBlock).toContain(".react-flow__edge-path");
    expect(forcedColorsBlock).toContain(".yaml-toolbar-btn[data-copied=\"true\"]");
    // First-run tutorial high-contrast fallbacks
    expect(forcedColorsBlock).toContain(".tutorial-graph-node");
    expect(forcedColorsBlock).toContain(".tutorial-progress-dot");
    expect(forcedColorsBlock).toContain(".tutorial-progress-dot--active");
    expect(forcedColorsBlock).toContain(".tutorial-progress-bar");
  });
});
