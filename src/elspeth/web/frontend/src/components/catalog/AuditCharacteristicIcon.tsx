// ============================================================================
// AuditCharacteristicIcon
//
// Single-flag renderer used by the plugin card and the filter chip strip.
// Looks up the flag in the centralised metadata table; falls back to a
// "unknown" chip for forward compatibility with backend flag additions
// that predate the corresponding frontend metadata.
// ============================================================================

import { lookupAuditCharacteristic } from "./auditCharacteristics";

interface AuditCharacteristicIconProps {
  flag: string;
}

export function AuditCharacteristicIcon({ flag }: AuditCharacteristicIconProps) {
  const meta = lookupAuditCharacteristic(flag);
  if (meta === null) {
    return (
      <span
        className="audit-icon audit-icon-unknown"
        title={`Unknown audit characteristic: ${flag}`}
      >
        {flag}
      </span>
    );
  }
  return (
    <span
      className={`audit-icon audit-icon-${meta.tone}`}
      title={meta.tooltip}
    >
      <span className="audit-icon-glyph" aria-hidden="true">{meta.glyph}</span>
      <span className="audit-icon-label">{meta.label}</span>
    </span>
  );
}
