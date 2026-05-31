import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  projectInlineSourceSummary,
  useInlineSourceStore,
} from "./inlineSourceStore";
import { resetStore } from "@/test/store-helpers";
import type {
  BlobMetadata,
  BlobCreationModalityWire,
  InlineSourceProvenance,
} from "@/types/api";

const ONE_ROW_TEXT = "url\nhttps://a.gov.au";
const ONE_ROW_HASH =
  "c669ca6b4e11a7f2689ece48b8dc747463832d78833ea5d07c26be7e616d96b9";
const DIFFERENT_HASH =
  "9b8d3393ad3be052da5f25595789f926a161a4f8c0090c61f10a9cbab69a473c";

function makeBlobMetadata(overrides: Partial<BlobMetadata> = {}): BlobMetadata {
  return {
    id: "blob-inline-1",
    session_id: "session-1",
    filename: "inline.csv",
    mime_type: "text/csv",
    size_bytes: ONE_ROW_TEXT.length,
    content_hash: ONE_ROW_HASH,
    created_at: "2026-05-18T00:00:00Z",
    created_by: "assistant",
    source_description: "Inline source",
    status: "ready",
    creation_modality: "verbatim",
    created_from_message_id: "message-1",
    creating_model_identifier: null,
    creating_model_version: null,
    creating_provider: null,
    creating_composer_skill_hash: null,
    creating_arguments_hash: null,
    ...overrides,
  };
}

function toProvenance(wire: BlobCreationModalityWire): InlineSourceProvenance {
  if (wire === "verbatim") return "verbatim";
  if (wire === "llm_generated") return "llm-generated";
  if (wire === "disambiguated") return "disambiguated";
  if (wire === "llm_generated_then_amended")
    return "llm-generated-then-amended";
  const _exhaustive: never = wire;
  throw new Error(`Unhandled modality ${String(_exhaustive)}`);
}

describe("inlineSourceStore", () => {
  beforeEach(() => resetStore(useInlineSourceStore));

  it("returns null when no inline source is bound to the session", () => {
    expect(useInlineSourceStore.getState().getSummary("session-1")).toBeNull();
  });

  it("stores a verbatim summary and retrieves it by session", () => {
    useInlineSourceStore.getState().setSummary("session-1", {
      blobId: "blob-uuid",
      filename: "chat.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://example.com",
      rowCount: 1,
      contentHash: "abc123",
      provenance: "verbatim",
    });
    const summary = useInlineSourceStore.getState().getSummary("session-1");
    expect(summary?.provenance).toBe("verbatim");
    expect(summary?.rowCount).toBe(1);
  });

  it("clears the summary when the source is replaced or removed", () => {
    useInlineSourceStore.getState().setSummary("session-1", {
      blobId: "blob-clear-1",
      filename: "clear.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://example.com",
      rowCount: 1,
      contentHash: "deadbeef01",
      provenance: "verbatim",
    });
    useInlineSourceStore.getState().clearSummary("session-1");
    expect(useInlineSourceStore.getState().getSummary("session-1")).toBeNull();
  });

  it("namespaces summaries per session", () => {
    useInlineSourceStore.getState().setSummary("session-1", {
      blobId: "blob-ns-1",
      filename: "verbatim.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://gov.au",
      rowCount: 1,
      contentHash: "aabbcc0011",
      provenance: "verbatim",
    });
    useInlineSourceStore.getState().setSummary("session-2", {
      blobId: "blob-ns-2",
      filename: "llm-gen.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://a.gov\nhttps://b.gov",
      rowCount: 2,
      contentHash: "112233aabb",
      provenance: "llm-generated",
    });
    expect(useInlineSourceStore.getState().getSummary("session-1")?.provenance).toBe("verbatim");
    expect(useInlineSourceStore.getState().getSummary("session-2")?.provenance).toBe("llm-generated");
  });

  // --- Disambiguation re-fire guard tests (F-11) ---

  it("addUserRequestedSingleRow stores the message ID and prevents re-check", () => {
    useInlineSourceStore.getState().addUserRequestedSingleRow("msg-1");
    expect(
      useInlineSourceStore.getState().userRequestedSingleRowForMessageIds.has("msg-1"),
    ).toBe(true);
    expect(
      useInlineSourceStore.getState().userRequestedSingleRowForMessageIds.has("msg-2"),
    ).toBe(false);
  });

  // --- "Not source data" escape tests (F-10) ---

  it("addNonSourceMessage stores the message ID", () => {
    useInlineSourceStore.getState().addNonSourceMessage("msg-escape-1");
    expect(
      useInlineSourceStore.getState().nonSourceMessageIds.has("msg-escape-1"),
    ).toBe(true);
  });

  // --- Fallback-prompt dismiss persistence tests (F-20) ---

  it("markDismissed records a session-scoped dismissal timestamp", () => {
    const before = Date.now();
    useInlineSourceStore.getState().markDismissed("session-1");
    const ts = useInlineSourceStore.getState().dismissedAt.get("session-1");
    expect(ts).toBeGreaterThanOrEqual(before);
    expect(useInlineSourceStore.getState().isDismissed("session-1")).toBe(true);
  });

  it("isDismissed returns false for sessions that were never dismissed", () => {
    expect(useInlineSourceStore.getState().isDismissed("session-never")).toBe(false);
  });
});

describe("projectInlineSourceSummary", () => {
  it("refuses malformed MIME metadata instead of storing a confident projection", async () => {
    await expect(
      projectInlineSourceSummary({
        metadata: makeBlobMetadata({ mime_type: "text/csv; charset=" }),
        contentText: ONE_ROW_TEXT,
        toProvenance,
      }),
    ).rejects.toThrow(/invalid MIME metadata/i);
  });

  it("refuses hash drift between metadata and preview bytes", async () => {
    await expect(
      projectInlineSourceSummary({
        metadata: makeBlobMetadata({ content_hash: DIFFERENT_HASH }),
        contentText: ONE_ROW_TEXT,
        toProvenance,
      }),
    ).rejects.toThrow(/content_hash mismatch/i);
  });

  it("lets unrelated projection dependency exceptions bubble from the bound catch arm", async () => {
    const projectionError = new TypeError("projection dependency failed");
    const throwingMapper = vi.fn(() => {
      throw projectionError;
    });

    await expect(
      projectInlineSourceSummary({
        metadata: makeBlobMetadata(),
        contentText: ONE_ROW_TEXT,
        toProvenance: throwingMapper,
      }),
    ).rejects.toBe(projectionError);
    expect(throwingMapper).toHaveBeenCalledWith("verbatim");
  });
});
