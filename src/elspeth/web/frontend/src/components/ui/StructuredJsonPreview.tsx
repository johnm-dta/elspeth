import { useMemo, useState } from "react";
import { CodeBlock, hasLossyNumberLiteral } from "@/components/chat/CodeBlock";
import { PreviewTable, type PreviewTableModel } from "./PreviewTable";

interface StructuredJsonPreviewProps {
  text: string;
  truncated?: boolean;
}

type TableModel = PreviewTableModel;

type PreviewMode = "json" | "table";

export function StructuredJsonPreview({
  text,
  truncated = false,
}: StructuredJsonPreviewProps) {
  // Cheap: parse + a lexical lossy-literal scan. Deliberately stops short of
  // building the TableModel (header union + formatCellValue per cell) —
  // that's the expensive part `table` below defers until table mode is
  // actually active.
  const parsed = useMemo(() => parseJsonPreviewValue(text), [text]);
  const [mode, setMode] = useState<PreviewMode>("json");
  const canShowTable = parsed.isValid && !parsed.isLossy;
  const activeMode = canShowTable ? mode : "json";

  const table = useMemo(() => {
    if (!canShowTable || activeMode !== "table") {
      return null;
    }
    return buildTableModel(parsed.value);
  }, [canShowTable, activeMode, parsed]);

  return (
    <div className="structured-preview">
      <div className="structured-preview-toolbar" aria-label="Preview format">
        <button
          type="button"
          className="structured-preview-toggle"
          aria-pressed={activeMode === "json"}
          onClick={() => setMode("json")}
        >
          JSON view
        </button>
        {canShowTable && (
          <button
            type="button"
            className="structured-preview-toggle"
            aria-pressed={activeMode === "table"}
            onClick={() => setMode("table")}
          >
            Table view
          </button>
        )}
      </div>

      {activeMode === "table" && table ? (
        <PreviewTable table={table} />
      ) : (
        <CodeBlock code={text} prettyJson ariaLabel="JSON preview" />
      )}

      {truncated && (
        <p className="structured-preview-note">
          Preview truncated before parsing; download the file for complete data.
        </p>
      )}
      {!parsed.isValid && (
        <p className="structured-preview-note">
          Could not parse as JSON, so the raw preview is shown.
        </p>
      )}
    </div>
  );
}

interface JsonPreviewParse {
  isValid: boolean;
  isLossy: boolean;
  value: unknown;
}

// Cheap membership check for "is this text safe to tabularise": parse +
// the shared lossy-literal scan from CodeBlock. Deliberately stops short of
// building the TableModel — see `table` in StructuredJsonPreview above,
// which defers that (header union + formatCellValue per cell) until table
// mode is actually active (elspeth-37dc3472de).
function parseJsonPreviewValue(text: string): JsonPreviewParse {
  try {
    const value = JSON.parse(text) as unknown;
    return { isValid: true, isLossy: hasLossyNumberLiteral(text), value };
  } catch {
    return { isValid: false, isLossy: false, value: undefined };
  }
}

function buildTableModel(value: unknown): TableModel | null {
  if (Array.isArray(value)) {
    return buildArrayTable(value);
  }
  if (isRecord(value)) {
    return {
      headers: ["key", "value"],
      rows: Object.entries(value).map(([key, entryValue]) => [
        key,
        formatCellValue(entryValue),
      ]),
    };
  }
  return {
    headers: ["value"],
    rows: [[formatCellValue(value)]],
  };
}

function buildArrayTable(values: unknown[]): TableModel | null {
  if (values.length === 0) {
    return {
      headers: ["value"],
      rows: [["[]"]],
    };
  }

  const allRowsAreRecords = values.every(isRecord);
  if (!allRowsAreRecords) {
    return {
      headers: ["index", "value"],
      rows: values.map((value, index) => [
        String(index),
        formatCellValue(value),
      ]),
    };
  }

  const headers: string[] = [];
  const seenHeaders = new Set<string>();
  for (const row of values) {
    for (const key of Object.keys(row)) {
      if (!seenHeaders.has(key)) {
        seenHeaders.add(key);
        headers.push(key);
      }
    }
  }
  if (headers.length === 0) {
    return {
      headers: ["value"],
      rows: values.map((row) => [formatCellValue(row)]),
    };
  }

  return {
    headers,
    rows: values.map((row) =>
      headers.map((header) =>
        Object.prototype.hasOwnProperty.call(row, header)
          ? formatCellValue(row[header])
          : "",
      ),
    ),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatCellValue(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}
