import type { HTMLAttributes, ReactNode } from "react";

export interface TabItem {
  id: string;
  label: ReactNode;
  /** Optional count pill (used by the catalog tab strip). */
  count?: number;
}

export interface TabsProps extends Omit<HTMLAttributes<HTMLDivElement>, "onChange"> {
  tabs: TabItem[];
  /** id of the active tab. */
  value: string;
  /** Called with the id of the newly selected tab. */
  onChange?: (id: string) => void;
}

/**
 * Underline tab strip. Controlled via `value`/`onChange`. Each tab is
 * `{ id, label, count? }`; an optional count renders as a small pill.
 */
export function Tabs({ tabs = [], value, onChange, className = "", ...rest }: TabsProps) {
  const cls = ["tab-strip", className].filter(Boolean).join(" ");
  return (
    <div className={cls} role="tablist" {...rest}>
      {tabs.map((t) => {
        const active = t.id === value;
        return (
          <button
            key={t.id}
            role="tab"
            aria-selected={active}
            className={["tab-strip-tab", active ? "tab-strip-tab-active" : ""].filter(Boolean).join(" ")}
            onClick={() => onChange?.(t.id)}
          >
            {t.label}
            {typeof t.count === "number" ? (
              <span
                style={{
                  marginLeft: 6,
                  fontSize: "var(--font-size-3xs)",
                  padding: "1px 5px",
                  borderRadius: "var(--radius-lg)",
                  fontWeight: 600,
                  background: active ? "var(--color-accent)" : "var(--color-surface-elevated)",
                  color: active ? "var(--color-text-inverse)" : "var(--color-text-muted)",
                }}
              >
                {t.count}
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
