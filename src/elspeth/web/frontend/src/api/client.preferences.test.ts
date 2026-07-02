/**
 * Tests for the account-level user-composer-preferences API helpers
 * (Phase 1B Task 1).
 *
 * Convention: vi.spyOn(globalThis, "fetch") — matches client.guided.test.ts
 * and client.recovery.test.ts. Spying on the real fetch exercises the real
 * authHeaders() / parseResponse<T>() pipeline (including the 401-logout
 * interceptor and the FastAPI envelope decode); a module-level vi.mock
 * would stub those out and leave them uncovered.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchUserComposerPreferences,
  updateUserComposerPreferences,
} from "./client";
import type { UserComposerPreferencesPayload } from "@/types/api";

function makePayload(
  overrides: Partial<UserComposerPreferencesPayload> = {},
): UserComposerPreferencesPayload {
  return {
    default_mode: "guided",
    banner_dismissed_at: null,
    tutorial_completed_at: null,
    tutorial_stage: null,
    tutorial_session_id: null,
    tutorial_run_id: null,
    tutorial_source_data_hash: null,
    updated_at: "2026-05-16T00:00:00Z",
    ...overrides,
  };
}

describe("api/client user composer preferences", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("GET parses the UserComposerPreferences payload", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify(makePayload({ default_mode: "guided" })), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const prefs = await fetchUserComposerPreferences();

    expect(prefs.default_mode).toBe("guided");
    expect(prefs.banner_dismissed_at).toBeNull();
    expect(prefs.tutorial_completed_at).toBeNull();
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/composer-preferences");
    expect(init?.method).toBeUndefined();
  });

  it("PATCH sends only the supplied partial fields", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify(makePayload({ default_mode: "freeform" })), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const result = await updateUserComposerPreferences({
      default_mode: "freeform",
    });

    expect(result.default_mode).toBe("freeform");
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/composer-preferences");
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(init?.body as string)).toEqual({
      default_mode: "freeform",
    });
  });

  it("GET throws an ApiError on non-2xx (5xx server failure)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "boom" }), {
        status: 500,
        statusText: "Internal Server Error",
        headers: { "content-type": "application/json" },
      }),
    );

    await expect(fetchUserComposerPreferences()).rejects.toMatchObject({
      status: 500,
    });
  });

  it("PATCH throws an ApiError on 422 (invalid mode)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "invalid mode" }), {
        status: 422,
        statusText: "Unprocessable Entity",
        headers: { "content-type": "application/json" },
      }),
    );

    await expect(
      // @ts-expect-error -- intentionally invalid mode to exercise the 422 branch
      updateUserComposerPreferences({ default_mode: "kiosk" }),
    ).rejects.toMatchObject({ status: 422 });
  });
  it("PATCH sends the tutorial resume fields when supplied (elspeth-918f4434b3)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify(
          makePayload({
            tutorial_stage: "guided",
            tutorial_session_id: "sess-1",
          }),
        ),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );

    const result = await updateUserComposerPreferences({
      tutorial_stage: "guided",
      tutorial_session_id: "sess-1",
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
    });

    expect(result.tutorial_stage).toBe("guided");
    expect(result.tutorial_session_id).toBe("sess-1");
    const [, init] = fetchSpy.mock.calls[0];
    expect(JSON.parse(init?.body as string)).toEqual({
      tutorial_stage: "guided",
      tutorial_session_id: "sess-1",
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
    });
  });
});
