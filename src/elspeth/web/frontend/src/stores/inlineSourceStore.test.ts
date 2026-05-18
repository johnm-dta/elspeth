import { describe, it, expect, beforeEach } from "vitest";
import { useInlineSourceStore } from "./inlineSourceStore";
import { resetStore } from "@/test/store-helpers";

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
