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

const JSON_NUMBER_PATTERN =
  /^(-?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/;

interface DecimalValue {
  sign: "" | "-";
  digits: string;
  scale: number;
}

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
    if (hasLossyNumberLiteral(text)) {
      return {
        isValid: true,
        prettyText: text,
        table: null,
      };
    }

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

function hasLossyNumberLiteral(text: string): boolean {
  let inString = false;
  let escaped = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char !== "-" && !isDigit(char)) {
      continue;
    }

    const start = index;
    if (char === "-") {
      index += 1;
      if (index >= text.length || !isDigit(text[index])) {
        index = start;
        continue;
      }
    }

    if (text[index] === "0") {
      index += 1;
    } else {
      while (index < text.length && isDigit(text[index])) {
        index += 1;
      }
    }

    if (text[index] === ".") {
      index += 1;
      while (index < text.length && isDigit(text[index])) {
        index += 1;
      }
    }

    if (text[index] === "e" || text[index] === "E") {
      index += 1;
      if (text[index] === "+" || text[index] === "-") {
        index += 1;
      }
      while (index < text.length && isDigit(text[index])) {
        index += 1;
      }
    }

    if (isLossyJsonNumberLiteral(text.slice(start, index))) {
      return true;
    }
    index -= 1;
  }

  return false;
}

function isLossyJsonNumberLiteral(token: string): boolean {
  const numericValue = Number(token);
  if (!Number.isFinite(numericValue)) {
    return true;
  }

  const rendered = JSON.stringify(numericValue);
  if (typeof rendered !== "string" || rendered === "null") {
    return true;
  }

  const sourceValue = parseDecimalValue(token);
  const renderedValue = parseDecimalValue(rendered);
  if (sourceValue === null || renderedValue === null) {
    return false;
  }

  return (
    sourceValue.sign !== renderedValue.sign ||
    sourceValue.digits !== renderedValue.digits ||
    sourceValue.scale !== renderedValue.scale
  );
}

function parseDecimalValue(token: string): DecimalValue | null {
  const match = token.match(JSON_NUMBER_PATTERN);
  if (!match) {
    return null;
  }

  const [, rawSign, integerPart, fractionPart = "", exponentPart = "0"] = match;
  const exponent = Number(exponentPart);
  if (!Number.isSafeInteger(exponent)) {
    return null;
  }
  const sign = rawSign === "-" ? "-" : "";

  let digits = `${integerPart}${fractionPart}`.replace(/^0+/, "");
  if (digits.length === 0) {
    return { sign, digits: "0", scale: 0 };
  }

  let scale = fractionPart.length - exponent;
  if (scale < 0) {
    const zerosToAppend = -scale;
    if (zerosToAppend > 400) {
      return null;
    }
    digits += "0".repeat(zerosToAppend);
    scale = 0;
  }

  while (scale > 0 && digits.endsWith("0")) {
    digits = digits.slice(0, -1);
    scale -= 1;
  }

  if (digits.length === 0) {
    return { sign, digits: "0", scale: 0 };
  }

  return {
    sign,
    digits,
    scale,
  };
}

function isDigit(char: string): boolean {
  return char >= "0" && char <= "9";
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
