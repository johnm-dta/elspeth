import { describe, it, expect } from "vitest";
import {
  lookupAuditCharacteristic,
  KNOWN_AUDIT_FLAGS,
} from "./auditCharacteristics";

// Source of truth: src/elspeth/contracts/enums.py AuditCharacteristic
// StrEnum (the 13 string-valued members). When the backend vocabulary
// grows, update this expected set AND add metadata entries to
// AUDIT_CHARACTERISTICS for the new flags.
const EXPECTED_VOCABULARY = [
  "io_read", "io_write", "external_call",
  "deterministic", "seeded", "non_deterministic",
  "provenance", "retention", "quarantine", "coerce",
  "signed", "network", "credentials",
] as const;

describe("audit-characteristic vocabulary parity", () => {
  it("KNOWN_AUDIT_FLAGS covers every member of 16a's AuditCharacteristic enum", () => {
    const known = new Set(KNOWN_AUDIT_FLAGS);
    const missing = EXPECTED_VOCABULARY.filter((flag) => !known.has(flag));
    expect(missing).toEqual([]);
  });
});

describe("auditCharacteristics metadata", () => {
  it("exposes a metadata entry for io_read", () => {
    const meta = lookupAuditCharacteristic("io_read");
    expect(meta).not.toBeNull();
    expect(meta?.label).toMatch(/i\/?o.read|reads/i);
    expect(meta?.tone).toBe("positive");
  });

  it("exposes a metadata entry for external_call with attention tone", () => {
    const meta = lookupAuditCharacteristic("external_call");
    expect(meta).not.toBeNull();
    expect(meta?.tone).toBe("attention");
    expect(meta?.tooltip).toMatch(/external|network/i);
  });

  it("exposes provenance / retention / quarantine / coerce / signed", () => {
    expect(lookupAuditCharacteristic("provenance")).not.toBeNull();
    expect(lookupAuditCharacteristic("retention")).not.toBeNull();
    expect(lookupAuditCharacteristic("quarantine")).not.toBeNull();
    expect(lookupAuditCharacteristic("coerce")).not.toBeNull();
    expect(lookupAuditCharacteristic("signed")).not.toBeNull();
  });

  it("exposes network flag with attention tone", () => {
    const meta = lookupAuditCharacteristic("network");
    expect(meta).not.toBeNull();
    expect(meta?.tone).toBe("attention");
    expect(meta?.tooltip).toMatch(/network|external/i);
  });

  it("io_write has informational tone (not attention)", () => {
    const meta = lookupAuditCharacteristic("io_write");
    expect(meta).not.toBeNull();
    expect(meta?.tone).toBe("informational");
  });

  it("returns null for an unknown flag rather than crashing", () => {
    // Future flags added on the backend without a frontend metadata
    // entry should render as a small grey "unknown" chip, not crash.
    expect(lookupAuditCharacteristic("future_flag_2027")).toBeNull();
  });

  it("KNOWN_AUDIT_FLAGS lists every metadata key", () => {
    expect(KNOWN_AUDIT_FLAGS).toContain("io_read");
    expect(KNOWN_AUDIT_FLAGS).toContain("external_call");
    expect(KNOWN_AUDIT_FLAGS).toContain("quarantine");
  });

  it("AUDIT_CHARACTERISTICS table includes determinism-derived flags", () => {
    // The Phase-7A derivation rules turn Determinism enum values into
    // flag strings verbatim (io_read, io_write, external_call,
    // deterministic, seeded, non_deterministic). The frontend metadata
    // table must cover each so the inferred-flag case has a renderer.
    for (const flag of ["io_read", "io_write", "external_call", "deterministic", "seeded", "non_deterministic"]) {
      expect(lookupAuditCharacteristic(flag)).not.toBeNull();
    }
  });
});
