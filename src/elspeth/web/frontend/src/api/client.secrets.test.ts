import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createSecret,
  deleteSecret,
  PLUGIN_CATALOG_INVALIDATED_EVENT,
} from "./client";

describe("api/client secret catalog invalidation", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
    localStorage.setItem("auth_token", "test-token");
  });

  it("emits catalog invalidation after a successful create or update", async () => {
    const invalidated = vi.fn();
    window.addEventListener(PLUGIN_CATALOG_INVALIDATED_EVENT, invalidated);
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify({ name: "LLM_KEY", scope: "user", available: true }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );

    await createSecret("LLM_KEY", "replacement-value");

    expect(invalidated).toHaveBeenCalledOnce();
    window.removeEventListener(PLUGIN_CATALOG_INVALIDATED_EVENT, invalidated);
  });

  it("emits catalog invalidation after a successful delete", async () => {
    const invalidated = vi.fn();
    window.addEventListener(PLUGIN_CATALOG_INVALIDATED_EVENT, invalidated);
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(null, { status: 204 }),
    );

    await deleteSecret("LLM_KEY");

    expect(invalidated).toHaveBeenCalledOnce();
    window.removeEventListener(PLUGIN_CATALOG_INVALIDATED_EVENT, invalidated);
  });

  it("does not invalidate the catalog when a secret mutation fails", async () => {
    const invalidated = vi.fn();
    window.addEventListener(PLUGIN_CATALOG_INVALIDATED_EVENT, invalidated);
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "rejected" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      }),
    );

    await expect(createSecret("LLM_KEY", "bad-value")).rejects.toMatchObject({
      status: 400,
    });

    expect(invalidated).not.toHaveBeenCalled();
    window.removeEventListener(PLUGIN_CATALOG_INVALIDATED_EVENT, invalidated);
  });
});
