import { beforeEach, describe, expect, it, vi } from "vitest";

import { importCompositionYaml } from "./client";

describe("api/client YAML import errors", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
    localStorage.clear();
  });

  it("preserves plugin-policy rejection details from the import endpoint", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          detail: {
            error_code: "plugin_not_enabled",
            component_id: "main",
            plugin_id: "sink:database",
            snapshot_fingerprint: "snapshot-a",
          },
        }),
        {
          status: 422,
          statusText: "Unprocessable Entity",
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    await expect(
      importCompositionYaml("session-1", "sources: {}"),
    ).rejects.toMatchObject({
      status: 422,
      error_type: "plugin_not_enabled",
      component_id: "main",
      plugin_id: "sink:database",
      snapshot_fingerprint: "snapshot-a",
    });
  });
});
