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

const BADGE_TOKEN_KINDS = [
  "source",
  "transform",
  "gate",
  "sink",
  "aggregation",
  "coalesce",
] as const;

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

function extractLightThemeRawToken(tokenName: string): string {
  const blockMatch = /\[data-theme="light"\]\s*\{([\s\S]*?)\n\}/.exec(appCss);
  if (!blockMatch) {
    throw new Error("Could not find light theme token block in styles/tokens.css");
  }

  return extractRawTokenFromBlock(tokenName, blockMatch[1], "light theme");
}

function extractTokenFromBlock(tokenName: string, block: string, blockName: string): string {
  const tokenMatch = new RegExp(`${tokenName}:\\s*(#[0-9a-fA-F]{6})\\s*;`).exec(block);
  if (!tokenMatch) {
    throw new Error(`Could not find ${tokenName} in ${blockName} token block`);
  }

  return tokenMatch[1];
}

function extractRawTokenFromBlock(tokenName: string, block: string, blockName: string): string {
  const tokenMatch = new RegExp(`${tokenName}:\\s*([^;]+);`).exec(block);
  if (!tokenMatch) {
    throw new Error(`Could not find ${tokenName} in ${blockName} token block`);
  }

  return tokenMatch[1].trim();
}

function parseRgba(value: string): { red: number; green: number; blue: number; alpha: number } {
  const match = /^rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\)$/.exec(value);
  if (!match) {
    throw new Error(`Expected rgba() colour, got ${value}`);
  }

  return {
    red: Number.parseInt(match[1], 10),
    green: Number.parseInt(match[2], 10),
    blue: Number.parseInt(match[3], 10),
    alpha: Number.parseFloat(match[4]),
  };
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
  // to a tokenised background+text pair on --color-bg (page background).
  // Historical note: --color-surface-elevated was rejected at the time
  // because the OLD dark muted value (#7a9a9a) was only 3.84:1 there; the
  // elspeth-dae08efdc9 remediation raised dark muted to #93b3b3, which now
  // clears every panel surface (see the muted-on-panel-surface composition
  // suite below). The chosen pair passes AA in both themes
  // (dark 6.45:1, light 6.32:1).
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

  // F4: side-rail buttons sit on the warm inspection surface, where the global
  // --color-bg disabled fill reads as a raised chip rather than "inactive". A
  // rail-scoped override re-merges disabled rail buttons into
  // --color-surface-inspector so they read as recessed/outlined. These gate
  // both that the override exists and that its muted text stays AA on the rail.
  it("re-merges disabled side-rail buttons into the inspector surface", () => {
    const railMatch =
      /\.layout-siderail \.btn:disabled,[\s\S]*?\{([\s\S]*?)\n\}/.exec(appCss);
    expect(
      railMatch,
      "rail-scoped disabled rule must exist in shared.css",
    ).not.toBeNull();
    expect(railMatch![1]).toContain(
      "background-color: var(--color-surface-inspector)",
    );
  });

  it("keeps disabled rail-button text at AA on the inspector surface in both themes", () => {
    const darkText = extractRootToken("--color-text-muted");
    const darkSurface = extractRootToken("--color-surface-inspector");
    const lightText = extractLightThemeToken("--color-text-muted");
    const lightSurface = extractLightThemeToken("--color-surface-inspector");
    expect(contrastRatio(darkText, darkSurface)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightText, lightSurface)).toBeGreaterThanOrEqual(4.5);
  });
});

describe("form input placeholder contrast", () => {
  // The .input/.textarea ::placeholder colour was migrated off
  // --color-text-muted (at the time #7a9a9a, only ~3.84:1 on
  // --color-surface-elevated, the input background — below AA) to
  // --color-text-secondary as part of the
  // acknowledge-card-stack restyle. These assertions gate both the rule
  // (placeholder uses the secondary token) and the underlying contrast so a
  // future palette rebalance cannot silently regress placeholder legibility.
  it("uses --color-text-secondary (not muted) for ::placeholder", () => {
    const placeholderMatch =
      /\.input::placeholder,\s*\n\s*\.textarea::placeholder\s*\{([\s\S]*?)\n\}/.exec(
        appCss,
      );
    expect(
      placeholderMatch,
      "::placeholder rule must exist for .input/.textarea",
    ).not.toBeNull();
    const body = placeholderMatch![1];
    expect(body).toContain("color: var(--color-text-secondary)");
    expect(body).not.toContain("var(--color-text-muted)");
  });

  it("keeps placeholder text at AA on the input surface in both themes", () => {
    const darkSecondary = extractRootToken("--color-text-secondary");
    const darkSurface = extractRootToken("--color-surface-elevated");
    const lightSecondary = extractLightThemeToken("--color-text-secondary");
    const lightSurface = extractLightThemeToken("--color-surface-elevated");

    expect(contrastRatio(darkSecondary, darkSurface)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lightSecondary, lightSurface)).toBeGreaterThanOrEqual(4.5);
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

  it("keeps light theme hover surfaces visibly darker than resting surfaces", () => {
    const hover = parseRgba(extractLightThemeRawToken("--color-surface-hover"));

    expect(hover).toMatchObject({ red: 15, green: 45, blue: 53 });
    expect(hover.alpha).toBeGreaterThanOrEqual(0.06);
    expect(hover.alpha).toBeLessThanOrEqual(0.07);
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

  it("keeps dark success/status visually green instead of collapsing into teal info chrome", () => {
    const success = hexToRgb(extractRootToken("--color-success"));
    const statePositive = hexToRgb(extractRootToken("--color-state-positive"));
    const info = hexToRgb(extractRootToken("--color-info"));

    expect(extractRootToken("--color-status-completed")).toBe(
      extractRootToken("--color-success"),
    );
    expect(extractRootToken("--color-success")).not.toBe(
      extractRootToken("--color-state-positive"),
    );
    expect(extractRootToken("--color-success")).not.toBe(
      extractRootToken("--color-info"),
    );
    // Success should read as green at a glance; info/state-positive retain
    // the cyan/teal family. The old dark success token differed from teal by
    // only two blue-channel points, making status and info hierarchy mushy.
    expect(success.g - success.b).toBeGreaterThanOrEqual(35);
    expect(Math.abs(statePositive.g - statePositive.b)).toBeLessThanOrEqual(40);
    expect(info.b).toBeGreaterThan(info.g);
  });

  it("uses opaque badge background tokens with non-text contrast in both themes", () => {
    for (const kind of BADGE_TOKEN_KINDS) {
      const foregroundToken = `--color-badge-${kind}`;
      const backgroundToken = `--color-badge-${kind}-bg`;

      const darkForeground = extractRootToken(foregroundToken);
      const darkBackground = extractRootToken(backgroundToken);
      const lightForeground = extractLightThemeToken(foregroundToken);
      const lightBackground = extractLightThemeToken(backgroundToken);

      expect(contrastRatio(darkForeground, darkBackground)).toBeGreaterThanOrEqual(3);
      expect(contrastRatio(lightForeground, lightBackground)).toBeGreaterThanOrEqual(3);
    }
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

// ---------------------------------------------------------------------------
// Design-review remediation (2026-06-29) — lock in the M02/M03/M04 contrast
// fixes so the deepened semantic/status tokens, the muted→secondary switch on
// elevated surfaces, and the new --color-input-border cannot silently regress.
// These pairings were previously UNGATED by this file (see the design-review
// epic elspeth-ea551cca69).
// ---------------------------------------------------------------------------

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const m = /^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/.exec(hex);
  if (!m) {
    throw new Error(`Expected six-digit hex, got ${hex}`);
  }
  return {
    r: Number.parseInt(m[1], 16),
    g: Number.parseInt(m[2], 16),
    b: Number.parseInt(m[3], 16),
  };
}

function rgbToHex({ r, g, b }: { r: number; g: number; b: number }): string {
  const c = (n: number) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`;
}

function extractRootRawToken(tokenName: string): string {
  const blockMatch = /^:root\s*\{([\s\S]*?)\n\}/m.exec(appCss);
  if (!blockMatch) {
    throw new Error("Could not find root token block");
  }
  return extractRawTokenFromBlock(tokenName, blockMatch[1], "root");
}

// Resolve a token to a flat hex for a theme, compositing rgba() tints over the
// theme's --color-bg so a tinted alert/banner background can be contrast-tested.
function resolveHex(theme: "dark" | "light", tokenName: string): string {
  const rawOf = theme === "dark" ? extractRootRawToken : extractLightThemeRawToken;
  const hexOf = theme === "dark" ? extractRootToken : extractLightThemeToken;
  const raw = rawOf(tokenName);
  if (raw.startsWith("#")) {
    return raw;
  }
  const rgba = parseRgba(raw);
  const base = hexToRgb(hexOf("--color-bg"));
  return rgbToHex({
    r: rgba.red * rgba.alpha + base.r * (1 - rgba.alpha),
    g: rgba.green * rgba.alpha + base.g * (1 - rgba.alpha),
    b: rgba.blue * rgba.alpha + base.b * (1 - rgba.alpha),
  });
}

// Composite an rgba() TINT token over a SPECIFIC surface (not always
// --color-bg). A tinted banner background is not self-contained — its real
// contrast depends on the surface rendered behind it (e.g. the YamlView
// validation banner sits on --color-surface-paper, not the page bg).
function resolveTintOver(
  theme: "dark" | "light",
  tintToken: string,
  surfaceToken: string,
): string {
  const rawOf = theme === "dark" ? extractRootRawToken : extractLightThemeRawToken;
  const hexOf = theme === "dark" ? extractRootToken : extractLightThemeToken;
  const rgba = parseRgba(rawOf(tintToken));
  const base = hexToRgb(hexOf(surfaceToken));
  return rgbToHex({
    r: rgba.red * rgba.alpha + base.r * (1 - rgba.alpha),
    g: rgba.green * rgba.alpha + base.g * (1 - rgba.alpha),
    b: rgba.blue * rgba.alpha + base.b * (1 - rgba.alpha),
  });
}

describe("design-review contrast remediation (2026-06-29)", () => {
  const themes: Array<"dark" | "light"> = ["dark", "light"];

  it("keeps semantic banner text at AA on its tint over EVERY surface a banner mounts on (M02)", () => {
    // .alert-banner (page / LoginPage), .validation-banner (side-rail, on
    // --color-surface-inspector), and YamlView's modal validation banners (on
    // --color-surface-paper) all render color:var(--color-X) on the TINT
    // var(--color-X-bg). The tint's real contrast depends on the surface BEHIND
    // it — so gate every such surface, not just --color-bg. (The paper modal
    // was the instance the first remediation pass missed; sign-off caught it.)
    const surfaces = [
      "--color-bg",
      "--color-surface",
      "--color-surface-inspector",
      "--color-surface-paper",
    ];
    for (const theme of themes) {
      for (const kind of ["error", "warning", "success"] as const) {
        const fg = resolveHex(theme, `--color-${kind}`);
        for (const surface of surfaces) {
          const bg = resolveTintOver(theme, `--color-${kind}-bg`, surface);
          expect(
            contrastRatio(fg, bg),
            `${kind} text on ${kind}-bg over ${surface} (${theme})`,
          ).toBeGreaterThanOrEqual(4.5);
        }
      }
    }
  });

  it("keeps semantic + status text at AA on the elevated surface (M02)", () => {
    const tokens = [
      "--color-error",
      "--color-warning",
      "--color-success",
      "--color-status-failed",
      "--color-status-completed",
      "--color-status-cancelled",
      "--color-status-running",
      "--color-status-empty",
    ];
    for (const theme of themes) {
      const elevated = resolveHex(theme, "--color-surface-elevated");
      for (const token of tokens) {
        expect(
          contrastRatio(resolveHex(theme, token), elevated),
          `${token} on elevated (${theme})`,
        ).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("keeps secondary text (not muted) at AA on elevated + paper surfaces (M03)", () => {
    // Components that previously used --color-text-muted on elevated/paper
    // (audit-icon 10px label, catalog-reference-meta) were switched to
    // --color-text-secondary because muted is only ~3.84:1 there.
    for (const theme of themes) {
      const secondary = resolveHex(theme, "--color-text-secondary");
      for (const surface of ["--color-surface-elevated", "--color-surface-paper"]) {
        expect(
          contrastRatio(secondary, resolveHex(theme, surface)),
          `secondary on ${surface} (${theme})`,
        ).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("keeps secondary text at AA on the page background in both themes (guided decision summary)", () => {
    // The guided read-only decision summary renders small secondary-coloured
    // text directly over the page/section background: the
    // .guided-current-decision-eyebrow ("Current decision" label), the
    // .guided-schema-summary-caveat (validation-failure note), and the
    // .guided-schema-summary-needs-edit banner. Gate --color-text-secondary on
    // --color-bg so a palette rebalance can't drop these below AA.
    for (const theme of themes) {
      // The eyebrow/caveat/needs-edit text actually renders on the decision
      // card (--color-surface); the section sits on --color-bg. Gate both so a
      // palette rebalance can't drop these below AA on either surface.
      for (const surface of ["--color-bg", "--color-surface"]) {
        expect(
          contrastRatio(resolveHex(theme, "--color-text-secondary"), resolveHex(theme, surface)),
          `secondary on ${surface} (${theme})`,
        ).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("gives form inputs a resting boundary clearing WCAG 1.4.11 (3:1) (M04)", () => {
    // --color-input-border replaces the 1px --color-border-strong (~1.7:1)
    // boundary on .input/.textarea/.select/.guided-schema-input.
    for (const theme of themes) {
      expect(
        contrastRatio(resolveHex(theme, "--color-input-border"), resolveHex(theme, "--color-surface-elevated")),
        `input-border on elevated (${theme})`,
      ).toBeGreaterThanOrEqual(3);
    }
  });
});

// ---------------------------------------------------------------------------
// Muted-on-panel-surface COMPOSITIONS (elspeth-dae08efdc9, 2026-07-02).
// The gap that let ~46 dark-theme axe hits ship: this file gated token PAIRS
// (muted-on-bg, muted-on-inspector) but never the full set of surfaces muted
// text actually sits on — composing-card labels on elevated (3.83:1 with the
// old #7a9a9a), graph-mini labels on paper-adjacent surfaces (4.26:1), the
// message-tools toggle on tinted washes. One token raise (#7a9a9a → #93b3b3
// dark) cleared them; this suite gates muted against EVERY surface-family
// token in BOTH themes, plus the hover-wash composites, so no single
// surface pairing can silently regress again.
// ---------------------------------------------------------------------------

describe("muted text on every panel surface (elspeth-dae08efdc9)", () => {
  const themes: Array<"dark" | "light"> = ["dark", "light"];
  const surfaceTokens = [
    "--color-bg",
    "--color-surface",
    "--color-surface-elevated",
    "--color-surface-raised",
    "--color-surface-inspector",
    "--color-surface-paper",
    "--color-surface-nav",
    "--color-surface-nav-raised",
    "--color-surface-input",
  ];

  it("keeps muted text at AA on every surface-family token in both themes", () => {
    for (const theme of themes) {
      const muted = resolveHex(theme, "--color-text-muted");
      for (const surface of surfaceTokens) {
        expect(
          contrastRatio(muted, resolveHex(theme, surface)),
          `muted on ${surface} (${theme})`,
        ).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("keeps muted text at AA on hover-washed surfaces in both themes", () => {
    // Live composition: .version-selector-item--focused paints
    // --color-surface-hover over the elevated dropdown while its meta/tag
    // spans stay muted. The wash lightens dark surfaces (and darkens light
    // ones), so it is the tightest composite muted text sits on.
    for (const theme of themes) {
      const muted = resolveHex(theme, "--color-text-muted");
      for (const surface of ["--color-surface", "--color-surface-elevated"]) {
        const washed = resolveTintOver(theme, "--color-surface-hover", surface);
        expect(
          contrastRatio(muted, washed),
          `muted on surface-hover over ${surface} (${theme})`,
        ).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("keeps muted text visually quieter than secondary text in both themes", () => {
    // The token raise must not collapse the muted/secondary hierarchy: muted
    // stays the quieter register (lower contrast against the page background
    // than secondary in each theme).
    for (const theme of themes) {
      const bg = resolveHex(theme, "--color-bg");
      expect(
        contrastRatio(resolveHex(theme, "--color-text-muted"), bg),
        `muted vs secondary hierarchy (${theme})`,
      ).toBeLessThan(
        contrastRatio(resolveHex(theme, "--color-text-secondary"), bg),
      );
    }
  });
});
