// Typed REST helpers for the ELSPETH backend. Tests use these for setup
// (creating sessions, querying state) instead of driving the UI, per the
// "Testing Through the UI" anti-pattern in e2e-testing-strategies.md.

import { request, type APIRequestContext } from "@playwright/test";

// Overridable by Playwright configs so the same specs can run against either
// the local-spawned webServer or an already-deployed environment (e.g.
// elspeth.foundryside.dev).
const BACKEND_PORT = process.env.PLAYWRIGHT_BACKEND_PORT ?? "8451";
const BACKEND_BASE_URL =
  process.env.PLAYWRIGHT_BACKEND_BASE_URL ?? `http://127.0.0.1:${BACKEND_PORT}`;

export interface SessionSummary {
  id: string;
  title: string;
}

export type CompositionStateSeed = Record<string, unknown>;

export async function authedContext(token: string): Promise<APIRequestContext> {
  return request.newContext({
    baseURL: BACKEND_BASE_URL,
    extraHTTPHeaders: { Authorization: `Bearer ${token}` },
  });
}

export function tokenFromStorageState(
  storageState: { origins?: { localStorage?: { name: string; value: string }[] }[] } | null,
): string {
  const origin = storageState?.origins?.[0];
  const entry = origin?.localStorage?.find((e) => e.name === "auth_token");
  if (!entry) {
    throw new Error(
      "auth_token missing from storageState — globalSetup did not run or did not write the token",
    );
  }
  return entry.value;
}

export async function createSession(
  ctx: APIRequestContext,
  title: string,
): Promise<SessionSummary> {
  const resp = await ctx.post("/api/sessions", { data: { title } });
  if (!resp.ok()) {
    throw new Error(
      `POST /api/sessions failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
    );
  }
  return (await resp.json()) as SessionSummary;
}

export async function deleteSession(
  ctx: APIRequestContext,
  sessionId: string,
): Promise<void> {
  const resp = await ctx.delete(`/api/sessions/${sessionId}`);
  // 404 is acceptable — session may have been deleted by the test itself.
  if (!resp.ok() && resp.status() !== 404) {
    throw new Error(
      `DELETE /api/sessions/${sessionId} failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
    );
  }
}

export async function seedCompositionState(
  ctx: APIRequestContext,
  sessionId: string,
  state: CompositionStateSeed,
): Promise<Record<string, unknown>> {
  const resp = await ctx.post(`/api/sessions/${sessionId}/state/e2e-seed`, {
    data: { state },
  });
  if (!resp.ok()) {
    throw new Error(
      `POST /api/sessions/${sessionId}/state/e2e-seed failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
    );
  }
  return (await resp.json()) as Record<string, unknown>;
}

// Minimal blob metadata fields used by tests.  The backend returns more fields
// (session_id, mime_type, size_bytes, etc.) — we only surface what tests need
// to avoid coupling the helper to the full BlobMetadata interface in types/index.ts.
export interface BlobMetadata {
  id: string;
  filename: string;
}

/**
 * Upload a text blob to a session via multipart POST.
 *
 * Mirrors the frontend's uploadBlob() in src/api/client.ts — same endpoint,
 * same multipart "file" field name.  Used by guided-mode E2E tests to seed a
 * source CSV before navigating the wizard.
 *
 * Error responses (4xx/5xx) throw with the first 500 chars of the response
 * body so the Playwright report shows a useful failure message.
 */
export async function uploadBlob(
  ctx: APIRequestContext,
  sessionId: string,
  filename: string,
  contents: string,
  mimeType: string = "text/csv",
): Promise<BlobMetadata> {
  const resp = await ctx.post(`/api/sessions/${sessionId}/blobs`, {
    multipart: {
      file: {
        name: filename,
        mimeType,
        buffer: Buffer.from(contents, "utf-8"),
      },
    },
  });
  if (!resp.ok()) {
    throw new Error(
      `POST /api/sessions/${sessionId}/blobs failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
    );
  }
  return (await resp.json()) as BlobMetadata;
}
