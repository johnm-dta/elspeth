#!/usr/bin/env node
// check-contrast.mjs — guard WCAG AA contrast for the website's terminal/code
// text in BOTH themes, against silent token/override drift. No dependencies.
//
//   node website/check-contrast.mjs   → exit 0 if every pairing clears AA, 1 if not.
//
// Why this exists: website/tokens/ is a hand-maintained MIRROR of the product
// design tokens and has no equivalent of the app's colorContrast.test.ts. The
// success colour and the code-accent colours dip below 4.5:1 as small text on
// the light code surface unless explicitly re-tinted (see the
// [data-theme="light"] overrides in site.css). This asserts they stay above the
// line, so the A-1/A-2 regression cannot come back unnoticed. Wire it into CI.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(join(here, p), "utf8");

// --- parse `--name: #hex;` pairs inside a named top-level CSS block ----------
function blockVars(css, selector) {
  const esc = selector.replace(/[[\]"^$.*+?()|{}\\]/g, "\\$&");
  const m = css.match(new RegExp(esc + "\\s*\\{([^}]*)\\}"));
  const vars = {};
  if (!m) return vars;
  for (const line of m[1].split(";")) {
    const mm = line.match(/(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{3,8})/);
    if (mm) vars[mm[1]] = mm[2];
  }
  return vars;
}

const colors = read("tokens/colors.css");
const site = read("site.css");

const dark = blockVars(colors, ":root");
const light = blockVars(colors, '[data-theme="light"]');

// The actual light-mode rendered colours for the re-tinted text uses, read
// straight from the site.css override rules so the check tracks reality.
const ov = (re) => (site.match(re) || [])[1] || null;
const ovOk = ov(/\.proof \.term-body \.hl\s*\{\s*color:\s*(#[0-9a-fA-F]{6})/);
const ovKey = ov(/pre\.code \.key\s*\{\s*color:\s*(#[0-9a-fA-F]{6})/);
const ovStr = ov(/pre\.code \.str\s*\{\s*color:\s*(#[0-9a-fA-F]{6})/);

// --- WCAG relative luminance + contrast ratio -------------------------------
function lum(hex) {
  const h = hex.replace("#", "");
  const v = [0, 2, 4]
    .map((i) => parseInt(h.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
}
const ratio = (fg, bg) =>
  (Math.max(lum(fg), lum(bg)) + 0.05) / (Math.min(lum(fg), lum(bg)) + 0.05);

const AA = 4.5; // normal-size body/mono text
const checks = [
  // dark theme — terminal/code text sits on surface-nav
  ["dark  success → terminal", dark["--color-success"], dark["--color-surface-nav"]],
  ["dark  code .key", dark["--color-badge-aggregation"], dark["--color-surface-nav"]],
  ["dark  code .str", dark["--color-badge-source"], dark["--color-surface-nav"]],
  ["dark  body → bg", dark["--color-text"], dark["--color-bg"]],
  ["dark  secondary → surface", dark["--color-text-secondary"], dark["--color-surface"]],
  // muted text (.eyebrow, .term-bar .t, pre.code .cm, card body copy) sits on
  // the page bg, cards (surface/elevated) and the terminal (surface-nav).
  // Mirrors the product-side muted-on-panel-surface gate (elspeth-dae08efdc9).
  ["dark  muted → bg", dark["--color-text-muted"], dark["--color-bg"]],
  ["dark  muted → elevated", dark["--color-text-muted"], dark["--color-surface-elevated"]],
  ["dark  muted → terminal", dark["--color-text-muted"], dark["--color-surface-nav"]],
  // light theme — the re-tinted text on the (worst-case) light code surface
  ["light success/hl override", ovOk, light["--color-surface-nav"]],
  ["light code .key override", ovKey, light["--color-surface-nav"]],
  ["light code .str override", ovStr, light["--color-surface-nav"]],
  ["light body → bg", light["--color-text"], light["--color-bg"]],
  ["light secondary → surface", light["--color-text-secondary"], light["--color-surface"]],
  ["light muted → bg", light["--color-text-muted"], light["--color-bg"]],
  ["light muted → elevated", light["--color-text-muted"], light["--color-surface-elevated"]],
  ["light muted → terminal", light["--color-text-muted"], light["--color-surface-nav"]],
];

// Component badges render each foreground token directly on its matching
// background token (see tokens/primitives.css). They are normal-size text, so
// every one must clear AA in both themes.
const badgeKinds = ["source", "transform", "gate", "sink", "aggregation", "coalesce"];
for (const [theme, vars] of [["dark", dark], ["light", light]]) {
  for (const kind of badgeKinds) {
    checks.push([
      `${theme} badge ${kind}`,
      vars[`--color-badge-${kind}`],
      vars[`--color-badge-${kind}-bg`],
    ]);
  }
}

let failed = 0;
for (const [label, fg, bg] of checks) {
  if (!fg || !bg) {
    console.log(`MISS  ${label}  (fg=${fg} bg=${bg})`);
    failed++;
    continue;
  }
  const r = ratio(fg, bg);
  const ok = r >= AA;
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label.padEnd(28)} ${fg} on ${bg} = ${r.toFixed(2)}:1`);
}
if (failed) {
  console.error(`\n${failed} contrast check(s) below WCAG AA (${AA}:1).`);
  process.exit(1);
}
console.log(`\nAll ${checks.length} pairings clear WCAG AA (${AA}:1).`);
