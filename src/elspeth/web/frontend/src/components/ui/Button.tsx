import type { ButtonHTMLAttributes, ReactNode } from "react";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual style. @default "secondary" */
  variant?: "primary" | "secondary" | "danger" | "ghost";
  /** Use the 36px chrome-row size instead of the 44px default. @default false */
  compact?: boolean;
  iconLeft?: ReactNode;
  iconRight?: ReactNode;
}

export function Button({
  variant = "secondary",
  compact = false,
  type = "button",
  iconLeft,
  iconRight,
  className = "",
  children,
  ...rest
}: ButtonProps) {
  const base = compact ? "btn-compact" : "btn";
  const variantClass =
    variant === "primary" ? "btn-primary"
    : variant === "danger" ? "btn-danger"
    : variant === "ghost" ? "btn-ghost"
    : "";
  const cls = [base, variantClass, className].filter(Boolean).join(" ");
  return (
    <button type={type} className={cls} {...rest}>
      {iconLeft}
      {children}
      {iconRight}
    </button>
  );
}
