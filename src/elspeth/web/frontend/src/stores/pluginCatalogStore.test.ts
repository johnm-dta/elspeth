import { beforeEach, describe, expect, it, vi } from "vitest";

import * as api from "@/api/client";
import {
  createPluginCatalogStore,
  PLUGIN_CATALOG_INVALIDATED_EVENT,
} from "./pluginCatalogStore";

vi.mock("@/api/client", () => ({
  fetchPluginPolicy: vi.fn(),
  listSources: vi.fn(),
  listTransforms: vi.fn(),
  listSinks: vi.fn(),
  getPluginSchema: vi.fn(),
}));

const SOURCE = {
  name: "csv",
  plugin_type: "source" as const,
  description: "CSV source",
  config_fields: [],
  usage_when_to_use: null,
  usage_when_not_to_use: null,
  example_use: null,
  capability_tags: [],
  audit_characteristics: [],
};

function policy(snapshotFingerprint: string) {
  return {
    snapshot_fingerprint: snapshotFingerprint,
    policy_hash: "policy-1",
    available_plugin_ids: ["source:csv"],
    capability_groups: [],
    selections: [],
    control_modes: [],
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe("pluginCatalogStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.listSources).mockResolvedValue([SOURCE]);
    vi.mocked(api.listTransforms).mockResolvedValue([]);
    vi.mocked(api.listSinks).mockResolvedValue([]);
    vi.mocked(api.getPluginSchema).mockResolvedValue({
      name: "csv",
      plugin_type: "source",
      description: "CSV source",
      json_schema: { type: "object" },
    });
  });

  it("never reuses a catalog across principal or snapshot fingerprints", async () => {
    vi.mocked(api.fetchPluginPolicy)
      .mockResolvedValueOnce(policy("a"))
      .mockResolvedValueOnce(policy("b"));
    const store = createPluginCatalogStore();

    await store.getState().load({ principal: "local:alice", fingerprint: "a" });
    await store.getState().load({ principal: "oidc:alice", fingerprint: "b" });

    expect(api.fetchPluginPolicy).toHaveBeenCalledTimes(2);
    expect(api.listSources).toHaveBeenCalledTimes(2);
    expect(store.getState().key).toBe("oidc:alice:b");
    store.getState().dispose();
  });

  it("discards list and schema caches when the fingerprint changes", async () => {
    vi.mocked(api.fetchPluginPolicy)
      .mockResolvedValueOnce(policy("a"))
      .mockResolvedValueOnce(policy("b"));
    const store = createPluginCatalogStore();
    await store.getState().load({ principal: "local:alice", fingerprint: "a" });
    await store.getState().loadSchema("source", "csv");
    expect(store.getState().schemas["source:csv"]).toBeDefined();

    await store.getState().load({ principal: "local:alice", fingerprint: "b" });

    expect(store.getState().key).toBe("local:alice:b");
    expect(store.getState().schemas).toEqual({});
    expect(api.listSources).toHaveBeenCalledTimes(2);
    store.getState().dispose();
  });

  it("clears stale aliases immediately and refetches policy after secret invalidation", async () => {
    const refreshed = deferred<ReturnType<typeof policy>>();
    vi.mocked(api.fetchPluginPolicy)
      .mockResolvedValueOnce(policy("a"))
      .mockReturnValueOnce(refreshed.promise);
    const store = createPluginCatalogStore();
    await store.getState().load({ principal: "local:alice", fingerprint: "a" });

    window.dispatchEvent(new Event(PLUGIN_CATALOG_INVALIDATED_EVENT));

    expect(store.getState().key).toBeNull();
    expect(store.getState().sources).toBeNull();
    expect(store.getState().isLoading).toBe(true);
    refreshed.resolve(policy("b"));
    await vi.waitFor(() => expect(store.getState().key).toBe("local:alice:b"));
    expect(api.fetchPluginPolicy).toHaveBeenCalledTimes(2);
    store.getState().dispose();
  });

  it("ignores a late response from the previous principal", async () => {
    const alicePolicy = deferred<ReturnType<typeof policy>>();
    vi.mocked(api.fetchPluginPolicy)
      .mockReturnValueOnce(alicePolicy.promise)
      .mockResolvedValueOnce(policy("bob"));
    const store = createPluginCatalogStore();

    const aliceLoad = store.getState().load({ principal: "local:alice" });
    const bobLoad = store.getState().load({ principal: "local:bob" });
    await bobLoad;
    alicePolicy.resolve(policy("alice"));
    await aliceLoad;

    expect(store.getState().key).toBe("local:bob:bob");
    store.getState().dispose();
  });
});
