// contentStructure.ts — shared pure helpers for describing the shape of
// textual source content (row counts, CSV columns, JSON record keys).
//
// `describeRowCount` was extracted from InlineSourceCreatedTurn.tsx (Phase
// 5a Task 3) so BlobRow can reuse the exact same "1 row" / "N rows" /
// "unknown row count" phrasing rather than re-inventing it.
//
// `summarizeContentStructure` is new: static introspection of already-held
// text content (no network calls) so uploaded blobs can get the same kind
// of structural self-disclosure that LLM-authored inline sources already
// have (T-3, docs-archive/2026-07-04-composer-first-principles-review.md
// Part 6). It is defensive by design — see the HONESTY discipline below.
//
// HONESTY DISCIPLINE (Tier-1, non-negotiable): when the content can't be
// confidently parsed (ragged CSV, invalid/foreign-shaped JSON, a truncated
// preview, oversized input), the result carries a plain-language `caveat`
// and leaves `rowCount`/`fields` `null` rather than guessing or defaulting
// to 0. Never silently omit the failure — surface it.

/** Format a row count for display. `null` means "not confidently known". */
export function describeRowCount(rowCount: number | null): string {
  if (rowCount === null) return "unknown row count";
  if (rowCount === 1) return "1 row";
  return `${rowCount} rows`;
}

/**
 * Defensive cap on how much text `summarizeContentStructure` will attempt
 * to parse synchronously. Callers that already bound their fetch (e.g.
 * BlobRow's preview, capped at 5000 chars server-side) will never hit this;
 * it exists so the helper stays safe if reused against unbounded text.
 */
export const MAX_STRUCTURAL_SUMMARY_CHARS = 200_000;

export type BlobStructuralFormat = "csv" | "json-records" | "jsonlines" | "unsupported";

export interface BlobStructuralSummary {
  format: BlobStructuralFormat;
  /** Confirmed record count, excluding a CSV header row. Null when not confidently known. */
  rowCount: number | null;
  /** CSV column names, or JSON object keys (from the first record). Null when not confidently known. */
  fields: readonly string[] | null;
  /**
   * Plain-language reason `rowCount`/`fields` are null, or why a value that
   * IS present (e.g. truncated-but-known columns) is incomplete. Null on
   * the fully-confident happy path.
   */
  caveat: string | null;
}

function detectStructuralFormat(mimeType: string): BlobStructuralFormat {
  const base = mimeType.split(";")[0]?.trim().toLowerCase() ?? "";
  if (base === "text/csv") return "csv";
  if (base === "application/json") return "json-records";
  if (base === "application/x-jsonlines") return "jsonlines";
  return "unsupported";
}

/**
 * Minimal CSV tokenizer: splits text into rows of fields, honouring
 * double-quoted fields (with `""` as an escaped quote) so a comma or
 * newline inside a quoted field doesn't get mistaken for a delimiter —
 * which would otherwise produce false "ragged row" verdicts.
 */
interface CsvParseResult {
  rows: string[][];
  /** True when the text ended while still inside a quoted field — a hard sign of malformed or cut-off CSV. */
  endedInQuotes: boolean;
}

function parseCsvRows(text: string): CsvParseResult {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }
    if (ch === '"' && field === "") {
      inQuotes = true;
      continue;
    }
    if (ch === ",") {
      row.push(field);
      field = "";
      continue;
    }
    if (ch === "\r") {
      continue;
    }
    if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }
    field += ch;
  }

  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return { rows, endedInQuotes: inQuotes };
}

function summarizeCsv(text: string, truncated: boolean): BlobStructuralSummary {
  if (text.trim() === "") {
    return {
      format: "csv",
      rowCount: null,
      fields: null,
      caveat: "Content is empty — nothing to summarise.",
    };
  }

  const { rows, endedInQuotes } = parseCsvRows(text);
  if (rows.length === 0) {
    return {
      format: "csv",
      rowCount: null,
      fields: null,
      caveat: "Content is empty — nothing to summarise.",
    };
  }

  // An unterminated quoted field at end-of-text is expected when the
  // preview was truncated mid-row (handled below via the dropped last
  // row); outside truncation it means the content itself is malformed —
  // don't report a count derived from a parse that never closed.
  if (!truncated && endedInQuotes) {
    return {
      format: "csv",
      rowCount: null,
      fields: null,
      caveat: "Content has an unterminated quoted field — structure couldn't be read.",
    };
  }

  // If the preview was truncated, only trust the header as "seen" when a
  // later row proves it was terminated by a real newline rather than by
  // running out of preview budget mid-line.
  const headerConfirmed = !truncated || rows.length >= 2;
  if (!headerConfirmed) {
    return {
      format: "csv",
      rowCount: null,
      fields: null,
      caveat:
        "Preview is truncated before a complete row was captured — structure couldn't be confirmed.",
    };
  }

  const header = rows[0];
  const dataRowsAll = rows.slice(1);
  // The last row of a truncated preview is likely cut mid-row; don't count
  // or validate it — we simply don't know whether it's real.
  const dataRows = truncated ? dataRowsAll.slice(0, -1) : dataRowsAll;

  const ragged = dataRows.some((r) => r.length !== header.length);
  if (ragged) {
    return {
      format: "csv",
      rowCount: null,
      fields: null,
      caveat:
        "Rows don't all have the same number of columns — structure couldn't be read.",
    };
  }

  if (truncated) {
    return {
      format: "csv",
      rowCount: null,
      fields: header,
      caveat: "Preview is truncated — full row count couldn't be confirmed.",
    };
  }

  return { format: "csv", rowCount: dataRows.length, fields: header, caveat: null };
}

function summarizeJson(text: string, truncated: boolean): BlobStructuralSummary {
  if (text.trim() === "") {
    return {
      format: "json-records",
      rowCount: null,
      fields: null,
      caveat: "Content is empty — nothing to summarise.",
    };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return {
      format: "json-records",
      rowCount: null,
      fields: null,
      caveat: truncated
        ? "Preview is truncated — structure couldn't be confirmed."
        : "Content isn't valid JSON — structure couldn't be read.",
    };
  }

  if (!Array.isArray(parsed)) {
    return {
      format: "json-records",
      rowCount: null,
      fields: null,
      caveat: "JSON content isn't a list of records — structure couldn't be read.",
    };
  }

  if (parsed.length === 0) {
    return { format: "json-records", rowCount: 0, fields: [], caveat: null };
  }

  const allRecords = parsed.every(
    (item) => typeof item === "object" && item !== null && !Array.isArray(item),
  );
  if (!allRecords) {
    return {
      format: "json-records",
      rowCount: null,
      fields: null,
      caveat: "JSON list entries aren't records — structure couldn't be read.",
    };
  }

  const first = parsed[0] as Record<string, unknown>;
  return {
    format: "json-records",
    rowCount: parsed.length,
    fields: Object.keys(first),
    caveat: null,
  };
}

function summarizeJsonLines(text: string, truncated: boolean): BlobStructuralSummary {
  if (text.trim() === "") {
    return {
      format: "jsonlines",
      rowCount: null,
      fields: null,
      caveat: "Content is empty — nothing to summarise.",
    };
  }

  const rawLines = text.split("\n");
  // A trailing "" element means the text ended with a real newline; that's
  // not a data line, just the terminal-newline artifact of split().
  const lines =
    rawLines[rawLines.length - 1] === "" ? rawLines.slice(0, -1) : rawLines;
  if (lines.length === 0) {
    return {
      format: "jsonlines",
      rowCount: null,
      fields: null,
      caveat: "Content is empty — nothing to summarise.",
    };
  }

  // Same conservative proxy as CSV's header check: a line is only trusted
  // as "seen in full" once a later \n-confirmed line proves the preview
  // didn't just run out of budget mid-line.
  const firstLineConfirmed = !truncated || lines.length >= 2;
  if (!firstLineConfirmed) {
    return {
      format: "jsonlines",
      rowCount: null,
      fields: null,
      caveat:
        "Preview is truncated before a complete line was captured — structure couldn't be confirmed.",
    };
  }

  // The last line of a truncated preview is likely cut mid-line; don't
  // count or validate it.
  const consideredLines = truncated ? lines.slice(0, -1) : lines;

  const records: Array<Record<string, unknown>> = [];
  for (const rawLine of consideredLines) {
    if (rawLine.trim() === "") {
      return {
        format: "jsonlines",
        rowCount: null,
        fields: null,
        caveat: "Content has a blank line — structure couldn't be read.",
      };
    }
    let value: unknown;
    try {
      value = JSON.parse(rawLine);
    } catch {
      return {
        format: "jsonlines",
        rowCount: null,
        fields: null,
        caveat: "A line isn't valid JSON — structure couldn't be read.",
      };
    }
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return {
        format: "jsonlines",
        rowCount: null,
        fields: null,
        caveat: "A line isn't a JSON record — structure couldn't be read.",
      };
    }
    records.push(value as Record<string, unknown>);
  }

  const fields = Object.keys(records[0]);
  if (truncated) {
    return {
      format: "jsonlines",
      rowCount: null,
      fields,
      caveat: "Preview is truncated — full row count couldn't be confirmed.",
    };
  }

  return { format: "jsonlines", rowCount: records.length, fields, caveat: null };
}

/**
 * Static, synchronous introspection of already-held text content — no
 * network calls. `truncated` should reflect whatever the caller already
 * knows about whether `text` is a prefix of a larger file (e.g. a bounded
 * preview fetch): a truncated CSV/JSON body can't yield a trustworthy row
 * count, so the result says so rather than guessing.
 */
export function summarizeContentStructure(
  mimeType: string,
  text: string,
  options: { truncated?: boolean } = {},
): BlobStructuralSummary {
  const format = detectStructuralFormat(mimeType);
  if (format === "unsupported") {
    return { format, rowCount: null, fields: null, caveat: null };
  }

  if (text.length > MAX_STRUCTURAL_SUMMARY_CHARS) {
    return {
      format,
      rowCount: null,
      fields: null,
      caveat: "Content is too large to summarise structurally.",
    };
  }

  const truncated = options.truncated ?? false;
  if (format === "csv") return summarizeCsv(text, truncated);
  if (format === "json-records") return summarizeJson(text, truncated);
  return summarizeJsonLines(text, truncated);
}

/**
 * One-line human summary for display, e.g. "3 rows — columns: name, age"
 * or "1 row — keys: id, status". Returns null when there's no confident
 * row count or field list to show (callers should fall back to the
 * summary's `caveat`).
 */
export function describeStructuralSummary(summary: BlobStructuralSummary): string | null {
  const parts: string[] = [];
  if (summary.rowCount !== null) {
    parts.push(describeRowCount(summary.rowCount));
  }
  if (summary.fields !== null && summary.fields.length > 0) {
    const label = summary.format === "csv" ? "columns" : "keys";
    parts.push(`${label}: ${summary.fields.join(", ")}`);
  }
  return parts.length > 0 ? parts.join(" — ") : null;
}
