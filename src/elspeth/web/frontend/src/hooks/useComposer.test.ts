import { describe, expect, it } from "vitest";

import {
  COMPOSE_TIMEOUT_MS,
  COMPOSE_TIMEOUT_ABORT_REASON,
  COMPOSE_USER_CANCEL_ABORT_REASON,
} from "@/config/composer";

describe("COMPOSE_TIMEOUT_MS", () => {
  it("outlives the backend compose wall clock so the server's structured 422 always wins", () => {
    // ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS (deploy/elspeth-web.env) — the
    // backend's own wall-clock budget for a compose turn. Healthy freeform
    // multi-tool turns legitimately run right up to it (11-22 tool calls
    // observed in the 2026-07-10 battery), so the client must not abort
    // first: a client abort in the 90-270s band killed healthy turns 4/4
    // and left sessions wedged behind zombie server turns
    // (elspeth-e08063c3a5). The client cap exists only for a truly dead
    // connection; the grace margin covers response transit after the
    // backend deadline fires.
    const backendWallClockMs = 270_000;

    expect(COMPOSE_TIMEOUT_MS).toBe(295_000);
    expect(COMPOSE_TIMEOUT_MS).toBeGreaterThan(backendWallClockMs);
  });

  it("uses distinct abort reasons for timeout and user cancel paths", () => {
    expect(COMPOSE_TIMEOUT_ABORT_REASON).not.toBe(COMPOSE_USER_CANCEL_ABORT_REASON);
  });
});
