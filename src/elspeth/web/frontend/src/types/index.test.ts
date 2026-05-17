import { describe, it, expect, expectTypeOf } from "vitest";
import type { PluginSummary, DataTrustTier } from "./index";

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
      data_trust_tier: 3,
    };
    expectTypeOf(summary.usage_when_to_use).toEqualTypeOf<string | null>();
    expectTypeOf(summary.capability_tags).toEqualTypeOf<string[]>();
    expectTypeOf(summary.audit_characteristics).toEqualTypeOf<string[]>();
    expectTypeOf(summary.data_trust_tier).toEqualTypeOf<DataTrustTier | null>();
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
      data_trust_tier: null,
    };
    expect(summary.usage_when_to_use).toBeNull();
  });

  it("DataTrustTier is the literal union 1 | 2 | 3", () => {
    expectTypeOf<DataTrustTier>().toEqualTypeOf<1 | 2 | 3>();
  });
});
