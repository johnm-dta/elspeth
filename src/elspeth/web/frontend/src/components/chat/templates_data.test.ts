/**
 * Tests for templates_data — pins the README.md §"Example Use Cases" mapping.
 *
 * The id-snapshot test here is a change-detector: any PR that reorders or
 * renames entries in templates_data.ts will break here, not silently in a
 * downstream consumer. That's intentional — see templates_data.ts for the
 * update-discipline comment.
 */

import { describe, expect, it } from "vitest";
import { TEMPLATES, type ExampleUseCase } from "./templates_data";

describe("templates_data — README Example Use Cases mapping", () => {
  it("contains twelve audit-domain exemplars", () => {
    expect(TEMPLATES).toHaveLength(12);
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
      "clinical-triage",
      "insurance-claims",
      "supply-chain-risk",
      "security-incident-triage",
      "research-review",
      "support-quality",
    ]);
  });

  it("the type is exported (used by TemplateCards.tsx)", () => {
    const sample: ExampleUseCase = TEMPLATES[0];
    expect(sample.id).toBeTruthy();
  });
});
