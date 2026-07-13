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

function policy(snapshotFingerprint: string, principalScope = "local:alice") {
  return {
    principal_scope: principalScope,
    snapshot_fingerprint: snapshotFingerprint,
    policy_hash: "policy-1",
    available_plugin_ids: ["source:csv"],
    capability_groups: [],
    selections: [],
    control_modes: [],
  };
}

function snapshot<T>(data: T, fingerprint: string) {
  return { data, snapshotFingerprint: fingerprint };
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
    vi.mocked(api.listSources).mockResolvedValue(snapshot([SOURCE], "a"));
    vi.mocked(api.listTransforms).mockResolvedValue(snapshot([], "a"));
    vi.mocked(api.listSinks).mockResolvedValue(snapshot([], "a"));
    vi.mocked(api.getPluginSchema).mockResolvedValue(snapshot({
      name: "csv",
      plugin_type: "source",
      description: "CSV source",
      json_schema: { type: "object" },
    }, "a"));
  });

  it("never reuses a catalog across principal or snapshot fingerprints", async () => {
    vi.mocked(api.fetchPluginPolicy)
      .mockResolvedValueOnce(snapshot(policy("a", "local:alice"), "a"))
      .mockResolvedValueOnce(snapshot(policy("b", "oidc:alice"), "b"));
    vi.mocked(api.listSources)
      .mockResolvedValueOnce(snapshot([SOURCE], "a"))
      .mockResolvedValueOnce(snapshot([SOURCE], "b"));
    vi.mocked(api.listTransforms)
      .mockResolvedValueOnce(snapshot([], "a"))
      .mockResolvedValueOnce(snapshot([], "b"));
    vi.mocked(api.listSinks)
      .mockResolvedValueOnce(snapshot([], "a"))
      .mockResolvedValueOnce(snapshot([], "b"));
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
      .mockResolvedValueOnce(snapshot(policy("a"), "a"))
      .mockResolvedValueOnce(snapshot(policy("b"), "b"));
    vi.mocked(api.listSources)
      .mockResolvedValueOnce(snapshot([SOURCE], "a"))
      .mockResolvedValueOnce(snapshot([SOURCE], "b"));
    vi.mocked(api.listTransforms)
      .mockResolvedValueOnce(snapshot([], "a"))
      .mockResolvedValueOnce(snapshot([], "b"));
    vi.mocked(api.listSinks)
      .mockResolvedValueOnce(snapshot([], "a"))
      .mockResolvedValueOnce(snapshot([], "b"));
    vi.mocked(api.getPluginSchema).mockResolvedValueOnce(snapshot({
      name: "csv",
      plugin_type: "source",
      description: "CSV source",
      json_schema: { type: "object" },
    }, "a"));
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
    const refreshed = deferred<{ data: ReturnType<typeof policy>; snapshotFingerprint: string }>();
    vi.mocked(api.fetchPluginPolicy)
      .mockResolvedValueOnce(snapshot(policy("a"), "a"))
      .mockReturnValueOnce(refreshed.promise);
    const store = createPluginCatalogStore();
    await store.getState().load({ principal: "local:alice", fingerprint: "a" });

    window.dispatchEvent(new Event(PLUGIN_CATALOG_INVALIDATED_EVENT));

    expect(store.getState().key).toBeNull();
    expect(store.getState().sources).toBeNull();
    expect(store.getState().isLoading).toBe(true);
    vi.mocked(api.listSources).mockResolvedValueOnce(snapshot([SOURCE], "b"));
    vi.mocked(api.listTransforms).mockResolvedValueOnce(snapshot([], "b"));
    vi.mocked(api.listSinks).mockResolvedValueOnce(snapshot([], "b"));
    refreshed.resolve(snapshot(policy("b"), "b"));
    await vi.waitFor(() => expect(store.getState().key).toBe("local:alice:b"));
    expect(api.fetchPluginPolicy).toHaveBeenCalledTimes(2);
    store.getState().dispose();
  });

  it("ignores a late response from the previous principal", async () => {
    const alicePolicy = deferred<{ data: ReturnType<typeof policy>; snapshotFingerprint: string }>();
    vi.mocked(api.fetchPluginPolicy)
      .mockReturnValueOnce(alicePolicy.promise)
      .mockResolvedValueOnce(snapshot(policy("bob", "local:bob"), "bob"));
    vi.mocked(api.listSources).mockResolvedValueOnce(snapshot([SOURCE], "bob"));
    vi.mocked(api.listTransforms).mockResolvedValueOnce(snapshot([], "bob"));
    vi.mocked(api.listSinks).mockResolvedValueOnce(snapshot([], "bob"));
    const store = createPluginCatalogStore();

    const aliceLoad = store.getState().load({ principal: "local:alice" });
    const bobLoad = store.getState().load({ principal: "local:bob" });
    await bobLoad;
    alicePolicy.resolve(snapshot(policy("alice", "local:alice"), "alice"));
    await aliceLoad;

    expect(store.getState().key).toBe("local:bob:bob");
    store.getState().dispose();
  });

  it("restarts when policy A is followed by catalog list responses from snapshot B", async () => {
    vi.mocked(api.fetchPluginPolicy)
      .mockResolvedValueOnce(snapshot(policy("a"), "a") as never)
      .mockResolvedValueOnce(snapshot(policy("b"), "b") as never);
    vi.mocked(api.listSources)
      .mockResolvedValueOnce(snapshot([SOURCE], "b") as never)
      .mockResolvedValueOnce(snapshot([SOURCE], "b") as never);
    vi.mocked(api.listTransforms)
      .mockResolvedValueOnce(snapshot([], "b") as never)
      .mockResolvedValueOnce(snapshot([], "b") as never);
    vi.mocked(api.listSinks)
      .mockResolvedValueOnce(snapshot([], "b") as never)
      .mockResolvedValueOnce(snapshot([], "b") as never);
    const store = createPluginCatalogStore();

    await store.getState().load({ principal: "bootstrap:alice" });

    expect(api.fetchPluginPolicy).toHaveBeenCalledTimes(2);
    expect(store.getState().key).toBe("local:alice:b");
    expect(store.getState().sources).toEqual([SOURCE]);
    store.getState().dispose();
  });

  it("restarts and re-owns a schema response when snapshot changes during the request", async () => {
    vi.mocked(api.fetchPluginPolicy)
      .mockResolvedValueOnce(snapshot(policy("a"), "a") as never)
      .mockResolvedValueOnce(snapshot(policy("b"), "b") as never);
    vi.mocked(api.listSources)
      .mockResolvedValueOnce(snapshot([SOURCE], "a") as never)
      .mockResolvedValueOnce(snapshot([SOURCE], "b") as never);
    vi.mocked(api.listTransforms)
      .mockResolvedValueOnce(snapshot([], "a") as never)
      .mockResolvedValueOnce(snapshot([], "b") as never);
    vi.mocked(api.listSinks)
      .mockResolvedValueOnce(snapshot([], "a") as never)
      .mockResolvedValueOnce(snapshot([], "b") as never);
    vi.mocked(api.getPluginSchema).mockResolvedValueOnce(snapshot({
      name: "csv",
      plugin_type: "source",
      description: "CSV source",
      json_schema: { type: "object" },
    }, "b") as never);
    const store = createPluginCatalogStore();
    await store.getState().load({ principal: "bootstrap:alice" });

    await store.getState().loadSchema("source", "csv");

    expect(api.fetchPluginPolicy).toHaveBeenCalledTimes(2);
    expect(store.getState().key).toBe("local:alice:b");
    expect(store.getState().schemas["source:csv"]).toBeDefined();
    store.getState().dispose();
  });

  it("restarts when an A-enabled schema becomes disabled under snapshot B", async () => {
    vi.mocked(api.fetchPluginPolicy)
      .mockResolvedValueOnce(snapshot(policy("a"), "a"))
      .mockResolvedValueOnce(snapshot(policy("b"), "b"));
    vi.mocked(api.listSources)
      .mockResolvedValueOnce(snapshot([SOURCE], "a"))
      .mockResolvedValueOnce(snapshot([], "b"));
    vi.mocked(api.listTransforms)
      .mockResolvedValueOnce(snapshot([], "a"))
      .mockResolvedValueOnce(snapshot([], "b"));
    vi.mocked(api.listSinks)
      .mockResolvedValueOnce(snapshot([], "a"))
      .mockResolvedValueOnce(snapshot([], "b"));
    vi.mocked(api.getPluginSchema).mockRejectedValueOnce({
      status: 404,
      detail: "plugin_not_enabled",
      snapshot_fingerprint: "b",
    });
    const store = createPluginCatalogStore();
    await store.getState().load({ principal: "bootstrap:alice" });

    await store.getState().loadSchema("source", "csv");

    expect(api.fetchPluginPolicy).toHaveBeenCalledTimes(2);
    expect(store.getState().key).toBe("local:alice:b");
    expect(store.getState().schemas["source:csv"]).toBeUndefined();
    expect(store.getState().schemaErrors["source:csv"]).toBeUndefined();
    store.getState().dispose();
  });
});
