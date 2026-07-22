// Postbuild prune for the emptyOutDir:false retention policy (vite.config.ts).
//
// With emptyOutDir:false, dist/assets accumulates one generation of hashed
// assets per rebuild so stale open tabs keep lazy-loading their own chunks
// until the version beacon's refresh (f2d105691). This box rebuilds dozens
// of times a night, so growth must be bounded: delete assets older than
// MAX_AGE_DAYS that the CURRENT index.html does not reference.
//
// Safety argument (verified empirically): vite/rollup rewrites every output
// file on every build — an unchanged chunk keeps its hashed name but gets a
// fresh mtime — so at postbuild time the current generation is always
// younger than any age threshold; age alone can never select a live file.
// The index.html-reference check is a redundant belt for the entry/css.
// Runs as npm postbuild; a fresh checkout with no dist is a clean no-op.

import { readdirSync, readFileSync, statSync, unlinkSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const MAX_AGE_DAYS = 7;

const distDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "dist");
const assetsDir = path.join(distDir, "assets");

let referenced = new Set();
try {
  const indexHtml = readFileSync(path.join(distDir, "index.html"), "utf-8");
  referenced = new Set(
    [...indexHtml.matchAll(/\/assets\/([^"'?\s]+)/g)].map((match) => match[1]),
  );
} catch {
  // No dist/index.html (fresh checkout, dist-less CI): nothing to prune.
  process.exit(0);
}

let entries;
try {
  entries = readdirSync(assetsDir);
} catch {
  process.exit(0);
}

const cutoffMs = Date.now() - MAX_AGE_DAYS * 24 * 60 * 60 * 1000;
let pruned = 0;
for (const name of entries) {
  if (referenced.has(name)) continue;
  const filePath = path.join(assetsDir, name);
  let stats;
  try {
    stats = statSync(filePath);
  } catch {
    continue;
  }
  if (!stats.isFile() || stats.mtimeMs >= cutoffMs) continue;
  try {
    unlinkSync(filePath);
    pruned += 1;
  } catch {
    // Best-effort: a locked/undeletable file just survives to the next prune.
  }
}
if (pruned > 0) {
  console.log(`[prune-stale-assets] removed ${pruned} asset(s) older than ${MAX_AGE_DAYS} days`);
}
