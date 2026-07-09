// ============================================================================
// ui/ primitive library — adopt-as-touched policy (elspeth-e1c5ad0b53)
//
// When an edit touches a control that one of these primitives covers (a raw
// <button> the Button primitive matches, a hand-rolled status/type chip,
// a bare labelled <input>), swap in the primitive as part of that edit rather
// than patching the bespoke copy — that is how a11y fixes (e.g. StatusBadge's
// ⚠/∅ glyph map) propagate instead of being re-fixed one file at a time.
// No big-bang sweeps; no new primitive without a concrete first consumer.
// Unused primitives get retired: Card, Tabs, and Textarea were deleted in
// wave 3 of the 2026-07-02 UX remediation because they had no consumer and
// no mechanical adoption site (see epic elspeth-6cf0e1e188 notes).
// ============================================================================

export { Button } from "./Button";
export type { ButtonProps } from "./Button";

export { TypeBadge } from "./TypeBadge";
export type { TypeBadgeProps } from "./TypeBadge";

export { StatusBadge } from "./StatusBadge";
export type { StatusBadgeProps } from "./StatusBadge";

export { AlertBanner } from "./AlertBanner";
export type { AlertBannerProps } from "./AlertBanner";

export { Input } from "./Input";
export type { InputProps } from "./Input";

export { WordMark } from "./WordMark";
export type { WordMarkProps } from "./WordMark";

export { Icon } from "./Icon";
export type { IconName, IconProps } from "./Icon";

export { StructuredJsonPreview } from "./StructuredJsonPreview";
