import type { HTMLAttributes, ReactNode } from "react";

export interface AlertBannerProps extends Omit<HTMLAttributes<HTMLDivElement>, "role"> {
  /** Semantic tone. @default "error" */
  tone?: "error" | "warning" | "info" | "success";
  /** Right-aligned action (e.g. a Retry button). */
  action?: ReactNode;
  /** Override the inferred ARIA role. */
  role?: string;
}

/**
 * Inline alert / status banner. Tone maps to ELSPETH semantic colours. Error
 * banners get role="alert" (assertive); softer tones get role="status" (polite)
 * by default, overridable. `action` renders a right-aligned slot (e.g. Retry).
 */
export function AlertBanner({
  tone = "error",
  action = null,
  role,
  className = "",
  children,
  ...rest
}: AlertBannerProps) {
  const toneClass =
    tone === "info"
      ? "alert-banner--info"
      : tone === "warning"
        ? "alert-banner--warning"
        : tone === "success"
          ? "alert-banner--success"
          : "";
  const cls = ["alert-banner", toneClass, className].filter(Boolean).join(" ");
  const resolvedRole = role ?? (tone === "error" ? "alert" : "status");
  return (
    <div className={cls} role={resolvedRole} {...rest}>
      <span>{children}</span>
      {action ? <span style={{ flexShrink: 0 }}>{action}</span> : null}
    </div>
  );
}
