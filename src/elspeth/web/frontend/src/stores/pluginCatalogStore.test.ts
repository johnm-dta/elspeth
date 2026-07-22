import { beforeEach, describe, expect, it, vi } from "vitest";

import * as api from "@/api/client";
import type { PluginPolicyResponse, PluginSummary } from "@/types/index";
import pluginPolicyMatrixFixture from "./__fixtures__/pluginPolicyMatrix.json";
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

interface PluginPolicyMatrixCase extends PluginPolicyResponse {
  name: string;
  authorized_control_ids: string[];
  available_control_ids: string[];
}

interface PluginPolicyMatrixFixture {
  core_plugin_ids: string[];
  cases: PluginPolicyMatrixCase[];
}

const PLUGIN_POLICY_MATRIX = pluginPolicyMatrixFixture as PluginPolicyMatrixFixture;

const EXPECTED_CONTROL_IDS: Record<string, string[]> = {
  core_only: [],
  azure_only: [
    "transform:azure_prompt_shield",
    "transform:azure_content_safety",
  ],
  aws_only: [
    "transform:aws_bedrock_prompt_shield",
    "transform:aws_bedrock_content_safety",
  ],
  both_with_azure_preference: [
    "transform:azure_prompt_shield",
    "transform:azure_content_safety",
    "transform:aws_bedrock_prompt_shield",
    "transform:aws_bedrock_content_safety",
  ],
  neither_control_implementation: [],
};

const EXPECTED_CONTROL_SELECTIONS: Record<string, Array<string | null>> = {
  core_only: [null, null],
  azure_only: [
    "transform:azure_prompt_shield",
    "transform:azure_content_safety",
  ],
  aws_only: [
    "transform:aws_bedrock_prompt_shield",
    "transform:aws_bedrock_content_safety",
  ],
  both_with_azure_preference: [
    "transform:azure_prompt_shield",
    "transform:azure_content_safety",
  ],
  neither_control_implementation: [null, null],
};

function pluginSummary(pluginId: string): PluginSummary {
  const separator = pluginId.indexOf(":");
  const kind = pluginId.slice(0, separator);
  const name = pluginId.slice(separator + 1);
  if (
    separator < 1 ||
    (kind !== "source" && kind !== "transform" && kind !== "sink")
  ) {
    throw new Error(`Invalid plugin identity in matrix fixture: ${pluginId}`);
  }
  return {
    ...SOURCE,
    name,
    plugin_type: kind,
    description: `${name} ${kind}`,
  };
}

function policy(
  snapshotFingerprint: string,
  principalScope = "local:alice",
  availablePluginIds = ["source:csv"],
) {
  return {
    principal_scope: principalScope,
    snapshot_fingerprint: snapshotFingerprint,
    policy_hash: "policy-1",
    available_plugin_ids: availablePluginIds,
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

  it("defines the complete five-case UI policy contract", () => {
    expect(PLUGIN_POLICY_MATRIX.core_plugin_ids).toEqual([
      "source:csv",
      "source:json",
      "source:text",
      "transform:field_mapper",
      "transform:llm",
      "transform:web_scrape",
      "sink:csv",
      "sink:json",
      "sink:text",
    ]);
    expect(PLUGIN_POLICY_MATRIX.cases.map((matrixCase) => matrixCase.name)).toEqual([
      "core_only",
      "azure_only",
      "aws_only",
      "both_with_azure_preference",
      "neither_control_implementation",
    ]);
    for (const matrixCase of PLUGIN_POLICY_MATRIX.cases) {
      expect(matrixCase.principal_scope).toBe(`local:${matrixCase.name}`);
      expect(matrixCase.snapshot_fingerprint).toMatch(/^[0-9a-f]{64}$/);
      expect(matrixCase.policy_hash).toMatch(/^[0-9a-f]{64}$/);
      expect(matrixCase.available_control_ids).toEqual(
        EXPECTED_CONTROL_IDS[matrixCase.name],
      );
      expect(matrixCase.available_plugin_ids).toEqual([
        ...PLUGIN_POLICY_MATRIX.core_plugin_ids,
        ...matrixCase.available_control_ids,
      ]);
      expect(matrixCase.capability_groups.length).toBeGreaterThan(0);
      expect(
        matrixCase.capability_groups.every(
          (group) => group.available_plugin_ids.length > 0,
        ),
      ).toBe(true);
      expect(
        matrixCase.capability_groups
          .flatMap((group) => group.available_plugin_ids)
          .sort(),
      ).toEqual(
        ["transform:llm", ...matrixCase.available_control_ids].sort(),
      );
      expect(matrixCase.selections).toEqual([
        { capability: "llm", plugin_id: "transform:llm" },
        {
          capability: "prompt_shield",
          plugin_id: EXPECTED_CONTROL_SELECTIONS[matrixCase.name][0],
        },
        {
          capability: "content_safety",
          plugin_id: EXPECTED_CONTROL_SELECTIONS[matrixCase.name][1],
        },
      ]);
      expect(matrixCase.control_modes).toEqual([
        { capability: "prompt_shield", mode: "recommend" },
        { capability: "content_safety", mode: "recommend" },
      ]);
    }
  });

  it.each(PLUGIN_POLICY_MATRIX.cases)(
    "loads the real five-configuration policy/catalog projection for $name",
    async (matrixCase) => {
      const fingerprint = matrixCase.snapshot_fingerprint;
      const policyResponse: PluginPolicyResponse = {
        principal_scope: matrixCase.principal_scope,
        snapshot_fingerprint: fingerprint,
        policy_hash: matrixCase.policy_hash,
        available_plugin_ids: matrixCase.available_plugin_ids,
        capability_groups: matrixCase.capability_groups,
        selections: matrixCase.selections,
        control_modes: matrixCase.control_modes,
      };
      const summaries = matrixCase.available_plugin_ids.map(pluginSummary);
      vi.mocked(api.fetchPluginPolicy).mockResolvedValue(
        snapshot(policyResponse, fingerprint),
      );
      vi.mocked(api.listSources).mockResolvedValue(
        snapshot(
          summaries.filter((item) => item.plugin_type === "source"),
          fingerprint,
        ),
      );
      vi.mocked(api.listTransforms).mockResolvedValue(
        snapshot(
          summaries.filter((item) => item.plugin_type === "transform"),
          fingerprint,
        ),
      );
      vi.mocked(api.listSinks).mockResolvedValue(
        snapshot(
          summaries.filter((item) => item.plugin_type === "sink"),
          fingerprint,
        ),
      );
      const store = createPluginCatalogStore();

      await store.getState().load({
        principal: matrixCase.principal_scope,
        fingerprint,
      });

      const state = store.getState();
      const rendered = [
        ...(state.sources ?? []).map((item) => `source:${item.name}`),
        ...(state.transforms ?? []).map((item) => `transform:${item.name}`),
        ...(state.sinks ?? []).map((item) => `sink:${item.name}`),
      ].sort();
      expect(rendered).toEqual([...matrixCase.available_plugin_ids].sort());
      expect(state.principal).toBe(matrixCase.principal_scope);
      expect(state.fingerprint).toBe(fingerprint);
      expect(state.key).toBe(`${matrixCase.principal_scope}:${fingerprint}`);
      expect(state.policy).toEqual(policyResponse);
      expect(state.policy?.capability_groups).toEqual(
        matrixCase.capability_groups,
      );
      expect(state.policy?.selections).toEqual(matrixCase.selections);
      expect(state.policy?.control_modes).toEqual(matrixCase.control_modes);

      if (matrixCase.name === "neither_control_implementation") {
        expect(matrixCase.authorized_control_ids).toHaveLength(4);
        expect(matrixCase.available_control_ids).toEqual([]);
        expect(
          matrixCase.authorized_control_ids.every(
            (pluginId) => !state.policy?.available_plugin_ids.includes(pluginId),
          ),
        ).toBe(true);
        expect(state.policy?.selections).toEqual([
          { capability: "llm", plugin_id: "transform:llm" },
          { capability: "prompt_shield", plugin_id: null },
          { capability: "content_safety", plugin_id: null },
        ]);
      }
      store.getState().dispose();
    },
  );

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
