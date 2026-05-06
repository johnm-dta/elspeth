import { describe, expect, it } from "vitest";

import { COMPOSE_TIMEOUT_MS } from "@/config/composer";

describe("COMPOSE_TIMEOUT_MS", () => {
  it("exceeds the deployed backend composer budget so server errors win the race", () => {
    // Backend ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS = 300.0 in
    // deploy/elspeth-web.env. The client must outlast it (with grace)
    // or the AbortController kills the fetch before the structured 422
    // can be emitted.
    expect(COMPOSE_TIMEOUT_MS).toBeGreaterThan(300_000);
    expect(COMPOSE_TIMEOUT_MS).toBe(330_000);
  });
});
