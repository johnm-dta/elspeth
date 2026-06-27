import type { HTMLAttributes, ReactNode } from "react";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Use the warm-neutral inspection (paper) surface. @default false */
  paper?: boolean;
  /** Apply default padding. Set false for full-bleed media. @default true */
  pad?: boolean;
}

/**
 * Surface card. `paper` switches to the warm-neutral inspection family used by
 * right-rail and modal panels. `pad` toggles default padding off for media.
 */
export function Card({ paper = false, pad = true, className = "", style, children, ...rest }: CardProps) {
  const cls = ["card", paper ? "card-paper" : "", className].filter(Boolean).join(" ");
  return (
    <div className={cls} style={{ ...(pad ? null : { padding: 0 }), ...style }} {...rest}>
      {children}
    </div>
  );
}

export interface CardHeaderProps {
  title: ReactNode;
  /** Right-aligned actions (buttons, menu). */
  actions?: ReactNode;
  /** Small uppercase label above the title. */
  eyebrow?: ReactNode;
}

/** Optional header row for a Card: title + right-aligned actions slot. */
export function CardHeader({ title, actions = null, eyebrow = null }: CardHeaderProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "space-between",
        gap: "var(--space-sm)",
        marginBottom: "var(--space-md)",
      }}
    >
      <div style={{ minWidth: 0 }}>
        {eyebrow ? (
          <div
            style={{
              fontSize: "var(--font-size-3xs)",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "var(--color-text-muted)",
              marginBottom: 2,
            }}
          >
            {eyebrow}
          </div>
        ) : null}
        <div style={{ fontSize: "var(--font-size-base)", fontWeight: 700, color: "var(--color-text)" }}>
          {title}
        </div>
      </div>
      {actions}
    </div>
  );
}
