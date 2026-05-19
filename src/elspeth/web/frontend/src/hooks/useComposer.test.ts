import { describe, expect, it } from "vitest";

import {
  COMPOSE_BACKEND_TIMEOUT_MS,
  COMPOSE_CLIENT_GRACE_MS,
  COMPOSE_SERVER_TRANSPORT_HEADROOM_MS,
  COMPOSE_TIMEOUT_MS,
  COMPOSE_TIMEOUT_ABORT_REASON,
  COMPOSE_TRANSPORT_IDLE_CEILING_MS,
  COMPOSE_USER_CANCEL_ABORT_REASON,
} from "@/config/composer";

describe("COMPOSE_TIMEOUT_MS", () => {
  it("outlasts the backend deadline while staying below the transport idle ceiling", () => {
    // Backend ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS = 270.0 in the staging
    // deployment. The client must outlast it with grace, but still abort before
    // the roughly 300s browser/proxy idle ceiling turns the request into an
    // opaque transport abort.
    const minimumClientGraceMs = 20_000;

    expect(COMPOSE_BACKEND_TIMEOUT_MS).toBeLessThanOrEqual(
      COMPOSE_TRANSPORT_IDLE_CEILING_MS - COMPOSE_SERVER_TRANSPORT_HEADROOM_MS,
    );
    expect(COMPOSE_CLIENT_GRACE_MS).toBeGreaterThanOrEqual(minimumClientGraceMs);
    expect(COMPOSE_CLIENT_GRACE_MS).toBeLessThan(COMPOSE_SERVER_TRANSPORT_HEADROOM_MS);
    expect(COMPOSE_TIMEOUT_MS).toBe(COMPOSE_BACKEND_TIMEOUT_MS + COMPOSE_CLIENT_GRACE_MS);
    expect(COMPOSE_TIMEOUT_MS).toBeGreaterThanOrEqual(
      COMPOSE_BACKEND_TIMEOUT_MS + minimumClientGraceMs,
    );
    expect(COMPOSE_TIMEOUT_MS).toBeLessThan(COMPOSE_TRANSPORT_IDLE_CEILING_MS);
  });

  it("uses distinct abort reasons for timeout and user cancel paths", () => {
    expect(COMPOSE_TIMEOUT_ABORT_REASON).not.toBe(COMPOSE_USER_CANCEL_ABORT_REASON);
  });
});
