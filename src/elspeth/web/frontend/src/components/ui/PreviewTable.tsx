// ============================================================================
// PreviewTable
//
// Shared headers+rows table renderer for preview panes. Two divergent
// implementations used to render this: StructuredJsonPreview's JSON table
// view (proper thead/th scope="col") and RunOutputsPanel's TabularPreview
// for csv/jsonl (inline styles, a bold-td fake header). Both now build a
// PreviewTableModel and render it through this one component, so a11y and
// visual treatment can't drift between the two data sources
// (elspeth-611a05668e).
// ============================================================================

export interface PreviewTableModel {
  headers: string[];
  // Cells beyond headers.length are not rendered; builders must widen
  // headers to the widest row (short rows are padded with "").
  rows: string[][];
}

interface PreviewTableProps {
  table: PreviewTableModel;
}

export function PreviewTable({ table }: PreviewTableProps) {
  return (
    <div className="structured-preview-table-wrap">
      <table className="structured-preview-table">
        <thead>
          <tr>
            {table.headers.map((header, columnIndex) => (
              <th key={`${header}-${columnIndex}`} scope="col">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {table.headers.map((_, columnIndex) => (
                <td key={columnIndex}>{row[columnIndex] ?? ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
