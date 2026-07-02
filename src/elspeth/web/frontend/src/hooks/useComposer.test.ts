import { describe, expect, it } from "vitest";

import {
  COMPOSE_TIMEOUT_MS,
  COMPOSE_TIMEOUT_ABORT_REASON,
  COMPOSE_USER_CANCEL_ABORT_REASON,
} from "@/config/composer";

describe("COMPOSE_TIMEOUT_MS", () => {
  it("is an honest stall ceiling for subphase-era compose turns", () => {
    // Live subphase turns settle in seconds to ~30s; the client abort fires
    // at ~3x that observed worst case so slow-but-alive turns still land
    // while a stalled request does not sit unreported for minutes
    // (elspeth-b189b5b3b8). The old 295s value dated from big-bang compose —
    // see config/composer.ts for the ordering trade against the backend's
    // 270s wall.
    const observedLiveTurnCeilingMs = 30_000;

    expect(COMPOSE_TIMEOUT_MS).toBe(90_000);
    expect(COMPOSE_TIMEOUT_MS).toBeGreaterThanOrEqual(3 * observedLiveTurnCeilingMs);
    // Regression guard: never drift back toward the pre-subphase 295s value.
    expect(COMPOSE_TIMEOUT_MS).toBeLessThan(270_000);
  });

  it("uses distinct abort reasons for timeout and user cancel paths", () => {
    expect(COMPOSE_TIMEOUT_ABORT_REASON).not.toBe(COMPOSE_USER_CANCEL_ABORT_REASON);
  });
});
