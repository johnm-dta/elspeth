// ============================================================================
// auditCharacteristics — centralised metadata for audit-characteristic flags
//
// Source of truth for how audit-characteristic flag strings render on the
// plugin card and in the filter chip strip. The backend (Phase 7A
// _derive_audit_characteristics) emits flag strings; this table maps each
// to its UI metadata.
//
// Unknown flags (no entry here) render as a small grey "unknown" chip with
// the raw flag string as label; this is the forward-compatibility path
// for flags added on the backend without a corresponding frontend update.
// ============================================================================

/** Visual tone for an audit-characteristic chip / icon.
 *  "positive"      — green checkmark-style chip (safe / good news)
 *  "attention"     — yellow warning-style chip (operator should be aware)
 *  "informational" — neutral blue-style chip (factual, neither good nor bad) */
export type AuditCharacteristicTone = "positive" | "attention" | "informational";

/** Display metadata for one audit-characteristic flag. */
export interface AuditCharacteristicMeta {
  /** Canonical flag string (matches the backend wire format). */
  flag: string;
  /** Short label shown next to the icon on the card. */
  label: string;
  /** Single-character or short glyph (Unicode) used as the icon. */
  glyph: string;
  /** Plain-language tooltip explaining the flag to a non-developer. */
  tooltip: string;
  /** Visual tone — "positive" renders as a green checkmark-style chip;
   *  "attention" renders as a yellow warning-style chip;
   *  "informational" renders as a neutral blue chip. */
  tone: AuditCharacteristicTone;
}

export const AUDIT_CHARACTERISTICS: AuditCharacteristicMeta[] = [
  // ── Determinism-derived ───────────────────────────────────────────────
  {
    flag: "deterministic",
    label: "deterministic",
    glyph: "≡",
    tooltip:
      "Plugin produces identical output on every run with the same input. " +
      "Safe to re-run; no replay machinery needed.",
    tone: "positive",
  },
  {
    flag: "seeded",
    label: "seeded",
    glyph: "🎲",
    tooltip:
      "Plugin uses pseudo-randomness controlled by a seed. Replay captures " +
      "the seed; re-running with the same seed reproduces the output.",
    tone: "positive",
  },
  {
    flag: "io_read",
    label: "reads I/O",
    glyph: "📥",
    tooltip:
      "Plugin reads from an external file, environment variable, or local " +
      "filesystem. Replay captures what was read.",
    tone: "positive",
  },
  {
    flag: "io_write",
    label: "writes I/O",
    glyph: "📤",
    tooltip:
      "Plugin writes to a file, environment, or local filesystem. Be " +
      "careful — replay re-applies the side effects.",
    tone: "informational",
  },
  {
    flag: "external_call",
    label: "Network call",
    glyph: "🌐",
    tooltip:
      "Plugin reaches an external system over the network (HTTP, API, " +
      "service call). Replay records the request and response.",
    tone: "attention",
  },
  {
    flag: "non_deterministic",
    label: "non-deterministic",
    glyph: "⁇",
    tooltip:
      "Plugin output is not reproducible from inputs alone. Replay must " +
      "record the full output verbatim.",
    tone: "attention",
  },

  // ── Source / sink behaviour ───────────────────────────────────────────
  {
    flag: "quarantine",
    label: "quarantines bad rows",
    glyph: "🛡",
    tooltip:
      "Source quarantines malformed rows to a designated sink instead of " +
      "crashing or silently discarding them. The audit trail records " +
      "which rows were quarantined and why.",
    tone: "positive",
  },
  {
    flag: "coerce",
    label: "coerces types",
    glyph: "↔",
    tooltip:
      "Plugin coerces external string-typed cells to typed columns at the " +
      "Tier-3 boundary (e.g., \"42\" → 42). Coercion is meaning-preserving; " +
      "fabrication is not. See CLAUDE.md \"Data Manifesto\" for the rule.",
    tone: "positive",
  },
  {
    flag: "retention",
    label: "retention-aware",
    glyph: "🗄",
    tooltip:
      "Plugin respects the configured retention policy — data emitted or " +
      "stored by this plugin will be purged according to the pipeline's " +
      "retention settings.",
    tone: "positive",
  },

  // ── Authored characteristics ──────────────────────────────────────────
  {
    flag: "provenance",
    label: "extra provenance",
    glyph: "🔍",
    tooltip:
      "Plugin emits additional provenance records beyond the standard " +
      "pipeline lineage (e.g., per-row hashes of source bytes).",
    tone: "positive",
  },
  {
    flag: "signed",
    label: "HMAC-signed",
    glyph: "🔏",
    tooltip:
      "Plugin output is HMAC-signed for tamper-evidence. The signing key " +
      "is part of the audit trail.",
    tone: "positive",
  },
  {
    flag: "credentials",
    label: "needs credentials",
    glyph: "🔑",
    tooltip:
      "Plugin requires user secrets (API keys, tokens) to operate. " +
      "Credentials are stored via the secret-handling pathway, not in " +
      "pipeline config.",
    tone: "attention",
  },
  {
    flag: "network",
    label: "Network call",
    glyph: "📡",
    tooltip:
      "Plugin reaches an external system over the network. This is the " +
      "authored variant of the determinism-derived external_call flag — " +
      "plugin authors set it explicitly when the call is not captured by " +
      "the determinism derivation rules.",
    tone: "attention",
  },
];

const _byFlag = new Map<string, AuditCharacteristicMeta>(
  AUDIT_CHARACTERISTICS.map((m) => [m.flag, m]),
);

export const KNOWN_AUDIT_FLAGS: string[] = AUDIT_CHARACTERISTICS.map((m) => m.flag);

/** Look up the display metadata for a flag string.
 *  Returns null for unknown flags (forward-compatible). */
export function lookupAuditCharacteristic(
  flag: string,
): AuditCharacteristicMeta | null {
  return _byFlag.get(flag) ?? null;
}
