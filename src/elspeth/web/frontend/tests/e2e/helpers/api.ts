// Typed REST helpers for the ELSPETH backend. Tests use these for setup
// (creating sessions, querying state) instead of driving the UI, per the
// "Testing Through the UI" anti-pattern in e2e-testing-strategies.md.

import { request, type APIRequestContext } from "@playwright/test";

const BACKEND_BASE_URL = "http://127.0.0.1:8451";

export interface SessionSummary {
  id: string;
  title: string;
}

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
