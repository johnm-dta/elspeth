import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { previewBlobContentSnippet } from "./client";

describe("api/client blob preview", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("fetches the bounded preview endpoint instead of full blob content", async () => {
    fetchSpy.mockResolvedValue({
      ok: true,
      headers: {
        get: (name: string) => {
          if (name.toLowerCase() === "x-preview-truncated") return "true";
          if (name.toLowerCase() === "x-preview-limit") return "5000";
          return null;
        },
      },
      text: async () => "preview text",
    } as Response);

    const preview = await previewBlobContentSnippet("session-1", "blob-1", 5000);

    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/sessions/session-1/blobs/blob-1/preview?limit=5000",
      expect.objectContaining({
        headers: expect.any(Object),
      }),
    );
    expect(preview).toEqual({
      text: "preview text",
      truncated: true,
      limit: 5000,
    });
  });
});
