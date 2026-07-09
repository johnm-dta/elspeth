// ============================================================================
// pluginDisplayName — display-name derivation for catalog plugin ids
// (elspeth-5ee1f76e39: display name primary, raw id demoted to metadata).
// ============================================================================

import { describe, expect, it } from "vitest";
import { isInternalPlugin, pluginDisplayName } from "./pluginDisplayName";

describe("pluginDisplayName", () => {
  it("uses curated overrides where plain title-casing would mislead", () => {
    expect(pluginDisplayName("azure_blob")).toBe("Azure Blob Storage");
    expect(pluginDisplayName("dataverse")).toBe("Microsoft Dataverse");
    expect(pluginDisplayName("chroma_sink")).toBe("Chroma Vector Store");
    expect(pluginDisplayName("batch_top_k")).toBe("Batch Top-K");
  });

  it("never surfaces the developer value 'null' as a display name", () => {
    expect(pluginDisplayName("null")).toBe("Resume Placeholder");
  });

  it("humanises underscore ids to Title Case", () => {
    expect(pluginDisplayName("azure_document_intelligence")).toBe(
      "Azure Document Intelligence",
    );
    expect(pluginDisplayName("field_mapper")).toBe("Field Mapper");
    expect(pluginDisplayName("web_scrape")).toBe("Web Scrape");
  });

  it("upper-cases known acronyms in the fallback path", () => {
    expect(pluginDisplayName("csv")).toBe("CSV");
    expect(pluginDisplayName("json_explode")).toBe("JSON Explode");
    expect(pluginDisplayName("llm")).toBe("LLM");
    expect(pluginDisplayName("rag_retrieval")).toBe("RAG Retrieval");
  });

  it("is not confused by Object.prototype key names", () => {
    // A plugin hypothetically named "constructor" must humanise, not
    // resolve to Object.prototype.constructor.
    expect(pluginDisplayName("constructor")).toBe("Constructor");
  });
});

describe("isInternalPlugin", () => {
  it("flags the resume-only null source", () => {
    expect(isInternalPlugin("null")).toBe(true);
  });

  it("does not flag ordinary plugins", () => {
    expect(isInternalPlugin("csv")).toBe(false);
    expect(isInternalPlugin("azure_blob")).toBe(false);
  });
});
