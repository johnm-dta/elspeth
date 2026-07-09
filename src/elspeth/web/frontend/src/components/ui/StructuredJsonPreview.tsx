import { useMemo, useState } from "react";

interface StructuredJsonPreviewProps {
  text: string;
  truncated?: boolean;
}

interface TableModel {
  headers: string[];
  rows: string[][];
}

type PreviewMode = "json" | "table";

export function StructuredJsonPreview({
  text,
  truncated = false,
}: StructuredJsonPreviewProps) {
  const parsed = useMemo(() => parseJsonPreview(text), [text]);
  const [mode, setMode] = useState<PreviewMode>("json");
  const canShowTable = parsed.table !== null;
  const activeMode = canShowTable ? mode : "json";

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

      {activeMode === "table" && parsed.table ? (
        <StructuredPreviewTable table={parsed.table} />
      ) : (
        <pre className="structured-preview-pre">{parsed.prettyText}</pre>
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

function StructuredPreviewTable({ table }: { table: TableModel }) {
  return (
    <div className="structured-preview-table-wrap">
      <table className="structured-preview-table">
        <thead>
          <tr>
            {table.headers.map((header) => (
              <th key={header} scope="col">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {table.headers.map((header, columnIndex) => (
                <td key={`${header}-${columnIndex}`}>
                  {row[columnIndex] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function parseJsonPreview(text: string): {
  isValid: boolean;
  prettyText: string;
  table: TableModel | null;
} {
  try {
    const value = JSON.parse(text) as unknown;
    return {
      isValid: true,
      prettyText: JSON.stringify(value, null, 2),
      table: buildTableModel(value),
    };
  } catch {
    return {
      isValid: false,
      prettyText: text,
      table: null,
    };
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
  for (const row of values) {
    for (const key of Object.keys(row)) {
      if (!headers.includes(key)) {
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
