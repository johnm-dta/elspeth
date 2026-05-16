// Staging-targeted globalSetup. Reads STAGING_BASE_URL, STAGING_USERNAME,
// and STAGING_PASSWORD from the environment, logs in against the deployed
// backend, and writes the same storageState shape that the local global
// setup writes — so every spec sees an auth_token in localStorage at the
// staging frontend origin.
//
// Differences from the local global-setup.ts:
//   - No register-then-fallback dance. Staging runs with
//     ELSPETH_WEB__REGISTRATION_MODE=closed; we expect the operator to
//     have created the user out of band. Bare login only.
//   - The storage state origin matches the staging URL (not localhost),
//     because the frontend reads auth_token via window.localStorage
//     scoped by origin.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { request } from "@playwright/test";

const TOKEN_KEY = "auth_token";

function required(name: string): string {
  const v = process.env[name];
  if (!v) {
    throw new Error(`${name} must be set for the staging Playwright run`);
  }
  return v;
}

async function obtainToken(
  apiBaseURL: string,
  username: string,
  password: string,
): Promise<string> {
  const ctx = await request.newContext({ baseURL: apiBaseURL });
  try {
    const resp = await ctx.post("/api/auth/login", {
      data: { username, password },
    });
    if (!resp.ok()) {
      const detail = await resp.text();
      throw new Error(
        `staging login failed (${resp.status()}): ${detail.slice(0, 500)}`,
      );
    }
    const body = (await resp.json()) as { access_token: string };
    return body.access_token;
  } finally {
    await ctx.dispose();
  }
}

function writeStorageState(path: string, origin: string, token: string): void {
  const state = {
    cookies: [],
    origins: [
      {
        origin,
        localStorage: [{ name: TOKEN_KEY, value: token }],
      },
    ],
  };
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(state, null, 2), { encoding: "utf-8" });
}

const HERE = dirname(fileURLToPath(import.meta.url));
const STORAGE_STATE_PATH = resolve(HERE, "..", ".auth", "staging-user.json");

export default async function stagingGlobalSetup(): Promise<void> {
  const baseURL = required("STAGING_BASE_URL");
  const username = required("STAGING_USERNAME");
  const password = required("STAGING_PASSWORD");
  const token = await obtainToken(baseURL, username, password);
  writeStorageState(STORAGE_STATE_PATH, baseURL, token);
  console.log(
    `[staging-globalSetup] wrote storageState for ${username} @ ${baseURL} → ${STORAGE_STATE_PATH}`,
  );
}
