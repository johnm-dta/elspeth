/**
 * Audit-domain exemplars sourced from README.md lines 560-571; the empty-state chat consumes this.
 * Update discipline — if README.md's table changes, update this file and
 * the snapshot in `templates_data.test.ts` in the same PR.
 */

import { describe, expect, it } from "vitest";
import { TEMPLATES, type ExampleUseCase } from "./templates_data";

describe("templates_data — README Example Use Cases mapping", () => {
  it("contains exactly six audit-domain exemplars", () => {
    expect(TEMPLATES).toHaveLength(6);
  });

  it("every template has the four READMEs-table columns", () => {
    for (const t of TEMPLATES) {
      expect(t.domain).toBeTruthy();
      expect(t.sense).toBeTruthy();
      expect(t.decide).toBeTruthy();
      expect(t.act).toBeTruthy();
    }
  });

  it("every template has a seed_prompt suitable for chat dispatch", () => {
    for (const t of TEMPLATES) {
      expect(t.seed_prompt.length).toBeGreaterThan(40);
      expect(t.seed_prompt.length).toBeLessThan(400);
    }
  });

  it("every template has a recommended_starting_point", () => {
    for (const t of TEMPLATES) {
      expect(["dynamic_source_from_chat", "csv_upload", "api_source"]).toContain(
        t.recommended_starting_point,
      );
    }
  });

  it("ids are stable and unique", () => {
    const ids = TEMPLATES.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
    // Stability test: hard-coded snapshot so a future PR that
    // rearranges the array breaks here, not in a downstream test.
    expect(ids).toEqual([
      "tender-evaluation",
      "document-qa",
      "weather-monitoring",
      "satellite-operations",
      "financial-compliance",
      "content-moderation",
    ]);
  });

  it("the type is exported (used by TemplateCards.tsx)", () => {
    const sample: ExampleUseCase = TEMPLATES[0];
    expect(sample.id).toBeTruthy();
  });
});
