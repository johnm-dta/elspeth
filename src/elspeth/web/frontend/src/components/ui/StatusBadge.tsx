import type { HTMLAttributes, ReactNode } from "react";

const GLYPH: Partial<Record<string, string>> = {
  completed_with_failures: "⚠",
  empty: "∅",
};

export interface StatusBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  status?:
    | "pending" | "running" | "completed" | "completed_with_failures"
    | "failed" | "empty" | "cancelled" | "cancelling";
  children?: ReactNode;
}

export function StatusBadge({ status = "pending", className = "", children, ...rest }: StatusBadgeProps) {
  const colorKey =
    status === "completed_with_failures" ? "completed"
    : status === "cancelling" ? "cancelled"
    : status;
  const cls = ["status-badge", `status-badge-${colorKey}`, className].filter(Boolean).join(" ");
  const glyph = GLYPH[status];
  return (
    <span className={cls} {...rest}>
      {glyph ? <span aria-hidden="true">{glyph}</span> : null}
      {children ?? status}
    </span>
  );
}
