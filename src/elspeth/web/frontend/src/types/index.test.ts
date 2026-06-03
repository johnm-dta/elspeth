import { describe, it, expect, expectTypeOf } from "vitest";
import type { AuditCharacteristicFlag } from "../components/catalog/auditCharacteristics";
import type { PluginSummary } from "./index";

describe("PluginSummary type extension (Phase 7B)", () => {
  it("has the Phase-7A reference-content fields", () => {
    const summary: PluginSummary = {
      name: "csv",
      plugin_type: "source",
      description: "Read CSV files",
      config_fields: [],
      usage_when_to_use: "When you have a CSV file.",
      usage_when_not_to_use: "When the data is inline.",
      example_use: "source:\n  plugin: csv",
      capability_tags: ["csv", "file"],
      audit_characteristics: ["io_read", "quarantine", "coerce"],
    };
    expectTypeOf(summary.usage_when_to_use).toEqualTypeOf<string | null>();
    expectTypeOf(summary.capability_tags).toEqualTypeOf<string[]>();
    // audit_characteristics is the closed-vocabulary union, not bare
    // string[]: a typo'd flag in an internal call site (test mock,
    // hand-constructed PluginSummary) is rejected at compile time.
    // Forward compatibility for unknown wire values is preserved by the
    // lookup boundary at lookupAuditCharacteristic(flag: string).
    expectTypeOf(summary.audit_characteristics).toEqualTypeOf<AuditCharacteristicFlag[]>();
  });

  it("accepts null / empty defaults for unfilled plugins", () => {
    const summary: PluginSummary = {
      name: "azure_blob",
      plugin_type: "source",
      description: "Read Azure blobs",
      config_fields: [],
      usage_when_to_use: null,
      usage_when_not_to_use: null,
      example_use: null,
      capability_tags: [],
      audit_characteristics: ["io_read"],
    };
    expect(summary.usage_when_to_use).toBeNull();
  });
});
