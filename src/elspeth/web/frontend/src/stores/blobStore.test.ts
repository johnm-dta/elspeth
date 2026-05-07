import { describe, it, expect, vi, beforeEach } from "vitest";
import { useBlobStore } from "./blobStore";
import type { BlobMetadata } from "@/types/api";

// Mock the API client
vi.mock("@/api/client", () => ({
  listBlobs: vi.fn(),
  uploadBlob: vi.fn(),
  deleteBlob: vi.fn(),
  downloadBlobContent: vi.fn(),
}));

function makeBlob(overrides: Partial<BlobMetadata> = {}): BlobMetadata {
  return {
    id: "blob-1",
    session_id: "session-1",
    filename: "data.csv",
    mime_type: "text/csv",
    size_bytes: 1024,
    content_hash: "abc123",
    created_at: "2026-04-01T00:00:00Z",
    created_by: "user",
    source_description: null,
    status: "ready",
    ...overrides,
  };
}

describe("blobStore", () => {
  beforeEach(() => {
    useBlobStore.getState().reset();
    vi.restoreAllMocks();
  });

  describe("loadBlobs", () => {
    it("fetches blobs and sets them in state", async () => {
      const blobs = [makeBlob(), makeBlob({ id: "blob-2", filename: "other.json" })];
      const { listBlobs } = await import("@/api/client");
      (listBlobs as ReturnType<typeof vi.fn>).mockResolvedValue(blobs);

      const promise = useBlobStore.getState().loadBlobs("session-1");

      // isLoading should be true while in-flight
      expect(useBlobStore.getState().isLoading).toBe(true);

      await promise;

      const state = useBlobStore.getState();
      expect(state.blobs).toEqual(blobs);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });

    it("sets error state when API call fails", async () => {
      const { listBlobs } = await import("@/api/client");
      (listBlobs as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error("Network error"),
      );

      await useBlobStore.getState().loadBlobs("session-1");

      const state = useBlobStore.getState();
      expect(state.error).toBe("Failed to load files.");
      expect(state.isLoading).toBe(false);
      expect(state.blobs).toEqual([]);
    });
  });

  describe("uploadBlob", () => {
    it("uploads and prepends blob to existing list", async () => {
      const existing = makeBlob({ id: "blob-old" });
      useBlobStore.setState({ blobs: [existing] });

      const newBlob = makeBlob({ id: "blob-new", filename: "upload.csv" });
      const { uploadBlob } = await import("@/api/client");
      (uploadBlob as ReturnType<typeof vi.fn>).mockResolvedValue(newBlob);

      const file = new File(["content"], "upload.csv", { type: "text/csv" });
      const result = await useBlobStore.getState().uploadBlob("session-1", file);

      expect(result).toEqual(newBlob);
      const state = useBlobStore.getState();
      expect(state.blobs[0]).toEqual(newBlob);
      expect(state.blobs[1]).toEqual(existing);
      expect(state.error).toBeNull();
    });

    it("sets quota-exceeded message for 413 error", async () => {
      const { uploadBlob } = await import("@/api/client");
      (uploadBlob as ReturnType<typeof vi.fn>).mockRejectedValue({ status: 413 });

      const file = new File(["x"], "big.csv", { type: "text/csv" });
      await expect(
        useBlobStore.getState().uploadBlob("session-1", file),
      ).rejects.toEqual({ status: 413 });

      expect(useBlobStore.getState().error).toBe(
        "File exceeds the maximum upload size.",
      );
    });

    it("sets unsupported-type message for 415 error", async () => {
      const { uploadBlob } = await import("@/api/client");
      (uploadBlob as ReturnType<typeof vi.fn>).mockRejectedValue({ status: 415 });

      const file = new File(["x"], "bad.exe", { type: "application/octet-stream" });
      await expect(
        useBlobStore.getState().uploadBlob("session-1", file),
      ).rejects.toEqual({ status: 415 });

      expect(useBlobStore.getState().error).toBe(
        "Unsupported file type. Please use CSV, JSON, JSONL, or plain text.",
      );
    });
  });

  describe("deleteBlob", () => {
    it("removes blob from state on success", async () => {
      const blob1 = makeBlob({ id: "blob-1" });
      const blob2 = makeBlob({ id: "blob-2" });
      useBlobStore.setState({ blobs: [blob1, blob2] });

      const { deleteBlob } = await import("@/api/client");
      (deleteBlob as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);

      await useBlobStore.getState().deleteBlob("session-1", "blob-1");

      const state = useBlobStore.getState();
      expect(state.blobs).toHaveLength(1);
      expect(state.blobs[0].id).toBe("blob-2");
    });

    it("sets active-run message for 409 error", async () => {
      useBlobStore.setState({ blobs: [makeBlob()] });

      const { deleteBlob } = await import("@/api/client");
      (deleteBlob as ReturnType<typeof vi.fn>).mockRejectedValue({ status: 409 });

      await useBlobStore.getState().deleteBlob("session-1", "blob-1");

      expect(useBlobStore.getState().error).toBe(
        "Cannot delete \u2014 file is linked to an active run.",
      );
    });
  });

  describe("downloadBlob", () => {
    it("triggers browser download via object URL", async () => {
      const fakeBlob = new Blob(["data"], { type: "text/csv" });
      const { downloadBlobContent } = await import("@/api/client");
      (downloadBlobContent as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: fakeBlob,
        filename: "result.csv",
      });

      const fakeUrl = "blob:http://localhost/fake-url";
      const createObjectURL = vi.fn().mockReturnValue(fakeUrl);
      const revokeObjectURL = vi.fn();
      globalThis.URL.createObjectURL = createObjectURL;
      globalThis.URL.revokeObjectURL = revokeObjectURL;

      const clickSpy = vi.fn();
      const fakeAnchor = {
        href: "",
        download: "",
        click: clickSpy,
      } as unknown as HTMLAnchorElement;
      const createElementSpy = vi
        .spyOn(document, "createElement")
        .mockReturnValue(fakeAnchor as unknown as HTMLElement);
      const appendChildSpy = vi
        .spyOn(document.body, "appendChild")
        .mockImplementation((node) => node);
      const removeChildSpy = vi
        .spyOn(document.body, "removeChild")
        .mockImplementation((node) => node);

      await useBlobStore.getState().downloadBlob("session-1", "blob-1");

      expect(createObjectURL).toHaveBeenCalledWith(fakeBlob);
      expect(createElementSpy).toHaveBeenCalledWith("a");
      expect(fakeAnchor.href).toBe(fakeUrl);
      expect(fakeAnchor.download).toBe("result.csv");
      expect(appendChildSpy).toHaveBeenCalledWith(fakeAnchor);
      expect(clickSpy).toHaveBeenCalled();
      expect(removeChildSpy).toHaveBeenCalledWith(fakeAnchor);
      expect(revokeObjectURL).toHaveBeenCalledWith(fakeUrl);
      expect(useBlobStore.getState().error).toBeNull();
    });
  });

  describe("clearBlobs", () => {
    it("clears blobs and error", () => {
      useBlobStore.setState({
        blobs: [makeBlob()],
        error: "some error",
      });

      useBlobStore.getState().clearBlobs();

      const state = useBlobStore.getState();
      expect(state.blobs).toEqual([]);
      expect(state.error).toBeNull();
    });
  });

  describe("reset", () => {
    it("restores full initial state", () => {
      useBlobStore.setState({
        blobs: [makeBlob()],
        isLoading: true,
        error: "something",
      });

      useBlobStore.getState().reset();

      const state = useBlobStore.getState();
      expect(state.blobs).toEqual([]);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });
  });
});
