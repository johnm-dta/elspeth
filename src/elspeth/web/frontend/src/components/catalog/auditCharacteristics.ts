// ============================================================================
// auditCharacteristics — centralised metadata for audit-characteristic flags
//
// Source of truth for how audit-characteristic flag strings render on the
// plugin card and in the filter chip strip. The backend (Phase 7A
// _derive_audit_characteristics in `src/elspeth/web/catalog/service.py`)
// emits flag strings drawn from the closed Python vocabulary
// `AuditCharacteristic` enum in `src/elspeth/contracts/enums.py`; this
// table maps each to its UI metadata.
//
// PARALLEL FILES (keep in sync — Python is the source of truth):
//   - Python enum:   src/elspeth/contracts/enums.py :: AuditCharacteristic
//   - This file:     AUDIT_CHARACTERISTICS (one entry per enum member)
//   - Parity test:   tests/unit/web/catalog/
//                    test_audit_characteristic_vocabulary_parity.py
//   - Wire shape:    src/elspeth/web/frontend/src/types/index.ts ::
//                    PluginSummary.audit_characteristics (literal-union
//                    type derived from this file's AuditCharacteristicFlag)
//
// Adding or removing a member in any one of those four sites without the
// matching change in the others fails CI (vocabulary parity test on
// adds/removes, TS compiler on type drift).
//
// Unknown flags (no entry here) render as a small grey "unknown" chip with
// the raw flag string as label; this is the forward-compatibility path
// for flags added on the backend without a corresponding frontend update.
// ============================================================================

/** Closed vocabulary of audit-characteristic flag strings.
 *
 * Mirrors the Python ``AuditCharacteristic`` StrEnum in
 * ``src/elspeth/contracts/enums.py``. Python is the source of truth; the
 * vocabulary parity test at
 * ``tests/unit/web/catalog/test_audit_characteristic_vocabulary_parity.py``
 * fails CI when either side drifts.
 *
 * Why a literal union and not ``string``:
 *
 *   - Internal call sites (test mocks, hand-constructed ``PluginSummary``
 *     objects) cannot accidentally introduce a typo'd flag — the TS
 *     compiler rejects ``"signe"`` where ``AuditCharacteristicFlag`` is
 *     expected.
 *   - Every ``AUDIT_CHARACTERISTICS`` metadata entry is type-checked
 *     against the union: a record whose ``flag`` is a string not in this
 *     union fails compilation, so the metadata table cannot silently
 *     reference a flag the closed vocabulary does not define.
 *
 * Forward compatibility for unknown-on-the-wire flags is preserved by
 * keeping ``lookupAuditCharacteristic(flag: string)`` accepting
 * ``string`` — an unknown wire value still resolves to ``null`` and
 * renders the grey "unknown" chip rather than crashing.
 */
export type AuditCharacteristicFlag =
  // Determinism-derived (composed from Determinism enum)
  | "io_read"
  | "io_write"
  | "external_call"
  | "deterministic"
  | "seeded"
  | "non_deterministic"
  // Author-declared (08-catalog-reshape.md vocabulary)
  | "provenance"
  | "retention"
  | "quarantine"
  | "coerce"
  | "signed"
  | "credentials";

/** Visual tone for an audit-characteristic chip / icon.
 *  "positive"      — green checkmark-style chip (safe / good news)
 *  "attention"     — yellow warning-style chip (operator should be aware)
 *  "informational" — neutral blue-style chip (factual, neither good nor bad) */
export type AuditCharacteristicTone = "positive" | "attention" | "informational";

/** Display metadata for one audit-characteristic flag. */
export interface AuditCharacteristicMeta {
  /** Canonical flag string (matches the backend wire format).
   *  Typed as the closed-vocabulary union so the metadata table cannot
   *  silently reference a flag outside ``AuditCharacteristic``. */
  flag: AuditCharacteristicFlag;
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
];

const _byFlag = new Map<string, AuditCharacteristicMeta>(
  AUDIT_CHARACTERISTICS.map((m) => [m.flag, m]),
);

export const KNOWN_AUDIT_FLAGS: AuditCharacteristicFlag[] = AUDIT_CHARACTERISTICS.map((m) => m.flag);

/** Look up the display metadata for a flag string.
 *  Returns null for unknown flags (forward-compatible). */
export function lookupAuditCharacteristic(
  flag: string,
): AuditCharacteristicMeta | null {
  return _byFlag.get(flag) ?? null;
}
