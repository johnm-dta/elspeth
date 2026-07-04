import { describe, expect, it } from "vitest";
import {
  MAX_STRUCTURAL_SUMMARY_CHARS,
  describeRowCount,
  describeStructuralSummary,
  summarizeContentStructure,
} from "./contentStructure";

describe("describeRowCount", () => {
  it("renders 'unknown row count' for null", () => {
    expect(describeRowCount(null)).toBe("unknown row count");
  });

  it("renders singular for 1", () => {
    expect(describeRowCount(1)).toBe("1 row");
  });

  it("renders plural for 0 and >1", () => {
    expect(describeRowCount(0)).toBe("0 rows");
    expect(describeRowCount(5)).toBe("5 rows");
  });
});

describe("summarizeContentStructure — CSV", () => {
  it("parses a well-formed CSV: row count excludes header, columns from header row", () => {
    const csv = "name,age\nAlice,30\nBob,40\n";
    const result = summarizeContentStructure("text/csv", csv);
    expect(result).toEqual({
      format: "csv",
      rowCount: 2,
      fields: ["name", "age"],
      caveat: null,
    });
  });

  it("handles a CSV with no trailing newline on the final row", () => {
    const csv = "name,age\nAlice,30";
    const result = summarizeContentStructure("text/csv", csv);
    expect(result.rowCount).toBe(1);
    expect(result.fields).toEqual(["name", "age"]);
  });

  it("respects quoted fields containing commas (not a false ragged-row positive)", () => {
    const csv = 'name,note\n"Doe, Jane","says ""hi"""\n';
    const result = summarizeContentStructure("text/csv", csv);
    expect(result.caveat).toBeNull();
    expect(result.rowCount).toBe(1);
    expect(result.fields).toEqual(["name", "note"]);
  });

  it("header-only CSV: confidently reports 0 rows, not a failure", () => {
    const csv = "name,age\n";
    const result = summarizeContentStructure("text/csv", csv);
    expect(result).toEqual({
      format: "csv",
      rowCount: 0,
      fields: ["name", "age"],
      caveat: null,
    });
  });

  it("ragged rows: honest failure, no guessed count, no partial fields", () => {
    const csv = "name,age\nAlice,30\nBob\n";
    const result = summarizeContentStructure("text/csv", csv);
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/same number of columns/i);
  });

  it("empty CSV content: honest 'nothing to summarise', not a guessed 0", () => {
    const result = summarizeContentStructure("text/csv", "");
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/empty/i);

    const whitespaceOnly = summarizeContentStructure("text/csv", "   \n  ");
    expect(whitespaceOnly.caveat).toMatch(/empty/i);
  });

  it("truncated CSV with a confirmed header: shows columns, refuses a row count", () => {
    const csv = "name,age\nAlice,30\nBob,4"; // last row likely cut off mid-value
    const result = summarizeContentStructure("text/csv", csv, { truncated: true });
    expect(result.rowCount).toBeNull();
    expect(result.fields).toEqual(["name", "age"]);
    expect(result.caveat).toMatch(/truncated/i);
  });

  it("truncated CSV cut off before any full row: refuses fields too", () => {
    const csv = "name,age,addr"; // no newline at all — header itself unconfirmed
    const result = summarizeContentStructure("text/csv", csv, { truncated: true });
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/truncated/i);
  });

  it("non-truncated CSV with an unterminated quoted field: honest failure, not a lucky-looking count", () => {
    const csv = 'name,note\nAlice,"unterminated';
    const result = summarizeContentStructure("text/csv", csv);
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/unterminated quoted field/i);
  });

  it("truncated CSV does not false-flag ragged: the dropped partial last row is excluded from the check", () => {
    const csv = "name,age\nAlice,30\nBob,4"; // "Bob,4" is a well-formed (if partial) 2-field row
    const result = summarizeContentStructure("text/csv", csv, { truncated: true });
    expect(result.caveat).toMatch(/truncated/i);
    expect(result.caveat).not.toMatch(/columns/i);
  });
});

describe("summarizeContentStructure — JSON", () => {
  it("parses an array of objects: row count + keys from the first record", () => {
    const json = JSON.stringify([
      { id: 1, status: "ok" },
      { id: 2, status: "ok" },
      { id: 3, status: "error" },
    ]);
    const result = summarizeContentStructure("application/json", json);
    expect(result).toEqual({
      format: "json-records",
      rowCount: 3,
      fields: ["id", "status"],
      caveat: null,
    });
  });

  it("empty JSON array: confidently 0, not a failure", () => {
    const result = summarizeContentStructure("application/json", "[]");
    expect(result).toEqual({
      format: "json-records",
      rowCount: 0,
      fields: [],
      caveat: null,
    });
  });

  it("empty content: honest 'nothing to summarise'", () => {
    const result = summarizeContentStructure("application/json", "");
    expect(result.rowCount).toBeNull();
    expect(result.caveat).toMatch(/empty/i);
  });

  it("invalid JSON: honest failure, not a guessed count", () => {
    const result = summarizeContentStructure("application/json", "{not valid json");
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/valid json/i);
  });

  it("truncated content that fails to parse: truncation-specific honest message", () => {
    const cut = JSON.stringify([{ id: 1 }, { id: 2 }]).slice(0, -3);
    const result = summarizeContentStructure("application/json", cut, { truncated: true });
    expect(result.rowCount).toBeNull();
    expect(result.caveat).toMatch(/truncated/i);
  });

  it("JSON root that isn't an array: honest failure", () => {
    const result = summarizeContentStructure("application/json", '{"id": 1}');
    expect(result.rowCount).toBeNull();
    expect(result.caveat).toMatch(/list of records/i);
  });

  it("JSON array of non-records: honest failure", () => {
    const result = summarizeContentStructure("application/json", "[1, 2, 3]");
    expect(result.rowCount).toBeNull();
    expect(result.caveat).toMatch(/records/i);
  });
});

describe("summarizeContentStructure — JSONL (application/x-jsonlines)", () => {
  it("happy path: line count + keys of the first record", () => {
    const jsonl = '{"id":1,"status":"ok"}\n{"id":2,"status":"error"}\n{"id":3,"status":"ok"}\n';
    const result = summarizeContentStructure("application/x-jsonlines", jsonl);
    expect(result).toEqual({
      format: "jsonlines",
      rowCount: 3,
      fields: ["id", "status"],
      caveat: null,
    });
  });

  it("happy path without a trailing newline on the final line", () => {
    const jsonl = '{"id":1}\n{"id":2}';
    const result = summarizeContentStructure("application/x-jsonlines", jsonl);
    expect(result.rowCount).toBe(2);
    expect(result.fields).toEqual(["id"]);
    expect(result.caveat).toBeNull();
  });

  it("empty content: honest 'nothing to summarise'", () => {
    const result = summarizeContentStructure("application/x-jsonlines", "");
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/empty/i);

    const whitespaceOnly = summarizeContentStructure("application/x-jsonlines", "  \n  ");
    expect(whitespaceOnly.caveat).toMatch(/empty/i);
  });

  it("malformed line (invalid JSON): honest failure, not a guessed count", () => {
    const jsonl = '{"id":1}\nnot json\n{"id":3}\n';
    const result = summarizeContentStructure("application/x-jsonlines", jsonl);
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/valid json/i);
  });

  it("malformed line (JSON but not a record, e.g. an array or scalar): honest failure", () => {
    const jsonl = '{"id":1}\n[1,2,3]\n';
    const result = summarizeContentStructure("application/x-jsonlines", jsonl);
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/record/i);
  });

  it("blank line: honest failure", () => {
    const jsonl = '{"id":1}\n\n{"id":3}\n';
    const result = summarizeContentStructure("application/x-jsonlines", jsonl);
    expect(result.rowCount).toBeNull();
    expect(result.caveat).toMatch(/blank line/i);
  });

  it("truncated: drops the likely-partial last line, shows keys, refuses a count", () => {
    const jsonl = '{"id":1,"status":"ok"}\n{"id":2,"status":"er'; // second line cut mid-value
    const result = summarizeContentStructure("application/x-jsonlines", jsonl, {
      truncated: true,
    });
    expect(result.rowCount).toBeNull();
    expect(result.fields).toEqual(["id", "status"]);
    expect(result.caveat).toMatch(/truncated/i);
  });

  it("truncated before any complete line: refuses fields too", () => {
    const jsonl = '{"id":1,"status":"o'; // no newline reached at all
    const result = summarizeContentStructure("application/x-jsonlines", jsonl, {
      truncated: true,
    });
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/truncated/i);
  });

  it("truncated does not false-flag the dropped partial line as malformed", () => {
    const jsonl = '{"id":1}\nnot even close to json'; // partial last line would fail to parse if checked
    const result = summarizeContentStructure("application/x-jsonlines", jsonl, {
      truncated: true,
    });
    expect(result.caveat).toMatch(/truncated/i);
    expect(result.caveat).not.toMatch(/valid json/i);
  });
});

describe("summarizeContentStructure — unsupported / oversized", () => {
  it("unsupported mime types are reported as such, with no caveat (not a failure)", () => {
    const result = summarizeContentStructure("text/plain", "just some text");
    expect(result).toEqual({
      format: "unsupported",
      rowCount: null,
      fields: null,
      caveat: null,
    });
  });

  it("oversized content is honestly capped, never parsed", () => {
    const huge = "a,b\n" + "1,2\n".repeat(MAX_STRUCTURAL_SUMMARY_CHARS);
    expect(huge.length).toBeGreaterThan(MAX_STRUCTURAL_SUMMARY_CHARS);
    const result = summarizeContentStructure("text/csv", huge);
    expect(result.rowCount).toBeNull();
    expect(result.fields).toBeNull();
    expect(result.caveat).toMatch(/too large/i);
  });
});

describe("describeStructuralSummary", () => {
  it("combines row count and CSV columns", () => {
    const line = describeStructuralSummary({
      format: "csv",
      rowCount: 3,
      fields: ["name", "age"],
      caveat: null,
    });
    expect(line).toBe("3 rows — columns: name, age");
  });

  it("combines row count and JSON keys", () => {
    const line = describeStructuralSummary({
      format: "json-records",
      rowCount: 1,
      fields: ["id"],
      caveat: null,
    });
    expect(line).toBe("1 row — keys: id");
  });

  it("returns null when nothing confident to show", () => {
    const line = describeStructuralSummary({
      format: "csv",
      rowCount: null,
      fields: null,
      caveat: "Rows don't all have the same number of columns — structure couldn't be read.",
    });
    expect(line).toBeNull();
  });

  it("shows fields alone when only the header survived truncation", () => {
    const line = describeStructuralSummary({
      format: "csv",
      rowCount: null,
      fields: ["name", "age"],
      caveat: "Preview is truncated — full row count couldn't be confirmed.",
    });
    expect(line).toBe("columns: name, age");
  });
});
