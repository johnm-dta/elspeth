/**
 * Design tokens for use in JS contexts (e.g. React Flow inline node styles).
 *
 * These use CSS custom property references (`var(--color-*)`) so they
 * automatically adapt to light/dark theme without JavaScript intervention.
 *
 * Note: CSS variables in inline styles work in all modern browsers.
 */

// ── Component type badge colours ─────────────────────────────────────────────

export const BADGE_COLORS = {
  source: "var(--color-badge-source)",
  transform: "var(--color-badge-transform)",
  gate: "var(--color-badge-gate)",
  aggregation: "var(--color-badge-aggregation)",
  coalesce: "var(--color-badge-coalesce)",
  sink: "var(--color-badge-sink)",
} as const;

export const BADGE_BACKGROUNDS = {
  source: "var(--color-badge-source-bg)",
  transform: "var(--color-badge-transform-bg)",
  gate: "var(--color-badge-gate-bg)",
  aggregation: "var(--color-badge-aggregation-bg)",
  coalesce: "var(--color-badge-coalesce-bg)",
  sink: "var(--color-badge-sink-bg)",
} as const;

// ── Edge colours ─────────────────────────────────────────────────────────────

export const EDGE_COLORS = {
  normal: "var(--color-text-muted)",
  error: "var(--color-error)",
} as const;

export const EDGE_LABEL_COLOR = "var(--color-text-secondary)";

// ── Validation indicator colours ─────────────────────────────────────────────

export const VALIDATION_COLORS = {
  valid: "var(--color-success)",
  warning: "var(--color-warning)",
  invalid: "var(--color-error)",
  unchecked: "var(--color-status-pending)",
};

export const VALIDATION_BACKGROUNDS = {
  valid: "var(--color-success-bg)",
  warning: "var(--color-warning-bg)",
  invalid: "var(--color-error-bg)",
  unchecked: "rgba(122, 154, 154, 0.1)",
};
