/**
 * Format an ISO timestamp as a compact, sortable absolute time string
 * "YYYY-MM-DD HH:MM:SS" in the *local* timezone. Used in audit panels
 * (run-outputs inventory, etc.) where forensic precision matters more
 * than locale-friendliness — sortable lexically, unambiguous between
 * DD/MM and MM/DD conventions, no relative-time decay over a session.
 *
 * Pair with `title={dateStr}` on the rendered span so the unmodified
 * wire-format timestamp (including timezone marker) stays queryable
 * via hover for anyone diffing against the audit DB directly.
 */
export function absoluteTime(dateStr: string): string {
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  const pad = (n: number) => n.toString().padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

/** Format a date string as a relative time ("2 min ago", "yesterday"). */
export function relativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);

  if (diffSec < 60) return "just now";
  if (diffMin < 60) return `${diffMin} min ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDay === 1) return "yesterday";
  return `${diffDay}d ago`;
}
