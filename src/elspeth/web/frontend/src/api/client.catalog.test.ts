import { beforeEach, describe, expect, it, vi } from "vitest";

import { fetchPluginPolicy, getPluginSchema, listSources } from "./client";

const SNAPSHOT_HEADER = "X-ELSPETH-Plugin-Snapshot";

describe("api/client plugin catalog snapshot metadata", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  it("returns the policy body with its response snapshot fingerprint", async () => {
    const body = {
      principal_scope: "local:alice",
      snapshot_fingerprint: "snapshot-a",
      policy_hash: "policy-a",
      available_plugin_ids: [],
      capability_groups: [],
      selections: [],
      control_modes: [],
    };
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json", [SNAPSHOT_HEADER]: "snapshot-a" },
    }));

    await expect(fetchPluginPolicy()).resolves.toEqual({
      data: body,
      snapshotFingerprint: "snapshot-a",
    });
  });

  it("returns list and schema bodies with their independent response fingerprints", async () => {
    vi.mocked(globalThis.fetch)
      .mockResolvedValueOnce(new Response("[]", {
        status: 200,
        headers: { "content-type": "application/json", [SNAPSHOT_HEADER]: "snapshot-b" },
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        name: "csv",
        plugin_type: "source",
        description: "CSV source",
        json_schema: { type: "object" },
      }), {
        status: 200,
        headers: { "content-type": "application/json", [SNAPSHOT_HEADER]: "snapshot-c" },
      }));

    expect(await listSources()).toMatchObject({ snapshotFingerprint: "snapshot-b" });
    expect(await getPluginSchema("source", "csv")).toMatchObject({ snapshotFingerprint: "snapshot-c" });
  });

  it("preserves the snapshot fingerprint on a policy-dependent schema error", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(new Response(JSON.stringify({ detail: "plugin_not_enabled" }), {
      status: 404,
      headers: { "content-type": "application/json", [SNAPSHOT_HEADER]: "snapshot-b" },
    }));

    await expect(getPluginSchema("source", "disabled")).rejects.toMatchObject({
      status: 404,
      snapshot_fingerprint: "snapshot-b",
    });
  });
});
