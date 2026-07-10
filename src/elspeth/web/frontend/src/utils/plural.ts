// plural.ts — shared pluralisation helper for "N thing(s)" labels.
//
// Not to be confused with `describeRowCount` in contentStructure.ts, which
// owns the row-count phrasing (including its `null` → "unknown row count"
// case) and is intentionally separate.

/**
 * Format a count with a singular/plural noun: `plural(1, "row")` → "1 row",
 * `plural(3, "row")` → "3 rows". Pass `pluralLabel` for nouns whose plural
 * isn't a simple "+s" suffix, e.g. `plural(2, "child", "children")`.
 */
export function plural(count: number, singular: string, pluralLabel = `${singular}s`): string {
  return count === 1 ? `${count} ${singular}` : `${count} ${pluralLabel}`;
}
