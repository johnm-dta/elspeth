import { describe, it, expect } from "vitest";
import { toInlineSourceProvenance } from "@/api/client";
import type {
  BlobCreationModalityWire,
  InlineSourceProvenance,
} from "@/types/index";

/**
 * Wire → display translation for the creation_modality field
 * (Phase 5a Task 2.5). The adapter is the single translation point;
 * these assertions pin the exhaustive mapping so a future server-side
 * enum extension or hyphenation change is caught at the test layer
 * before reaching any UI component.
 */
describe("toInlineSourceProvenance", () => {
  it("maps verbatim wire form to verbatim display form", () => {
    const wire: BlobCreationModalityWire = "verbatim";
    const display: InlineSourceProvenance = toInlineSourceProvenance(wire);
    expect(display).toBe("verbatim");
  });

  it("hyphenates llm_generated → llm-generated", () => {
    const wire: BlobCreationModalityWire = "llm_generated";
    const display: InlineSourceProvenance = toInlineSourceProvenance(wire);
    expect(display).toBe("llm-generated");
  });

  it("preserves disambiguated (no hyphenation needed)", () => {
    const wire: BlobCreationModalityWire = "disambiguated";
    const display: InlineSourceProvenance = toInlineSourceProvenance(wire);
    expect(display).toBe("disambiguated");
  });

  it("hyphenates llm_generated_then_amended → llm-generated-then-amended", () => {
    const wire: BlobCreationModalityWire = "llm_generated_then_amended";
    const display: InlineSourceProvenance = toInlineSourceProvenance(wire);
    expect(display).toBe("llm-generated-then-amended");
  });

  it("is exhaustive over the closed wire vocabulary", () => {
    // Exhaustive across the closed BlobCreationModalityWire union;
    // a future enum addition forces this table to be extended (and
    // the adapter itself to widen) — the typed `never` arm in
    // `toInlineSourceProvenance` does the rest at compile time.
    const allWireValues: BlobCreationModalityWire[] = [
      "verbatim",
      "llm_generated",
      "disambiguated",
      "llm_generated_then_amended",
    ];
    const allDisplayValues: InlineSourceProvenance[] = allWireValues.map(
      toInlineSourceProvenance,
    );
    expect(new Set(allDisplayValues).size).toBe(allWireValues.length);
  });
});
