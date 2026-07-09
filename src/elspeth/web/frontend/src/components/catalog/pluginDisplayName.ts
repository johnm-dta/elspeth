// ============================================================================
// pluginDisplayName — human-register labels for catalog plugin ids.
//
// The catalog API identifies plugins by their machine id (``azure_blob``,
// ``batch_top_k``). Those ids are the right register for YAML authors but
// machine-register for the catalog's browsing audience (ux review
// elspeth-5ee1f76e39). This module derives a display name for the card's
// PRIMARY label; the raw id stays visible as secondary mono metadata so
// operators can still copy the exact plugin name into a pipeline.
//
// Derivation, in order:
//   1. Curated override (Map below) — for ids whose plain title-casing is
//      wrong or unhelpful (``azure_blob`` is Azure Blob Storage, not
//      "Azure Blob").
//   2. Humanised fallback — underscores to spaces, Title Case, with a
//      closed acronym set upper-cased (``json_explode`` → "JSON Explode").
//
// Update discipline: new plugins get a sensible humanised name for free;
// add an override only when that name misleads.
// ============================================================================

/** Words rendered fully upper-case by the humanised fallback. */
const ACRONYMS: ReadonlySet<string> = new Set([
  "ai",
  "api",
  "csv",
  "db",
  "http",
  "https",
  "id",
  "io",
  "json",
  "llm",
  "rag",
  "sql",
  "url",
  "yaml",
]);

/**
 * Curated display names, keyed by plugin id. A Map (not a plain object)
 * so ids like "constructor" can never collide with Object.prototype.
 */
const DISPLAY_NAME_OVERRIDES: ReadonlyMap<string, string> = new Map([
  ["azure_blob", "Azure Blob Storage"],
  ["dataverse", "Microsoft Dataverse"],
  ["chroma_sink", "Chroma Vector Store"],
  ["batch_top_k", "Batch Top-K"],
  // The resume-only placeholder source. Its id is literally "null"; the
  // display name says what it is for instead of echoing a developer value
  // at end users (the card also carries the internal badge below).
  ["null", "Resume Placeholder"],
]);

/**
 * Plugin ids that exist for internal/resume machinery rather than for
 * end-user pipelines. The catalog keeps them visible (it is a reference,
 * not a picker) but badges them so first-run users do not reach for them.
 */
const INTERNAL_PLUGIN_IDS: ReadonlySet<string> = new Set(["null"]);

function titleCaseWord(word: string): string {
  if (ACRONYMS.has(word.toLowerCase())) return word.toUpperCase();
  return word.charAt(0).toUpperCase() + word.slice(1);
}

/** Human display name for a plugin id. Presentation only — never sent back
 *  to the backend; the raw id remains the wire identifier. */
export function pluginDisplayName(pluginId: string): string {
  const override = DISPLAY_NAME_OVERRIDES.get(pluginId);
  if (override !== undefined) return override;
  return pluginId
    .split(/[_\s]+/)
    .filter((word) => word.length > 0)
    .map(titleCaseWord)
    .join(" ");
}

/** True for plugins that are internal machinery (badged in the catalog). */
export function isInternalPlugin(pluginId: string): boolean {
  return INTERNAL_PLUGIN_IDS.has(pluginId);
}
