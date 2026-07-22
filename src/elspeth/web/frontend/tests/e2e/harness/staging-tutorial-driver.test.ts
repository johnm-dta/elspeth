import { describe, expect, it } from "vitest";

import {
  GUIDED_STAGE_PRIMARY_ACTION_NAMES,
  STAGED_GUIDED_PHASES,
  isComposeRequest,
  isRunRequest,
} from "../../../scripts/staging-tutorial-driver.mjs";

describe("standalone staging tutorial driver contract", () => {
  const sessionId = "00000000-0000-4000-8000-000000000000";

  it("drives the staged guided flow instead of the retired Build it turn", () => {
    expect(STAGED_GUIDED_PHASES).toEqual(["Source", "Output", "Transforms"]);
    expect(GUIDED_STAGE_PRIMARY_ACTION_NAMES).toEqual([
      "Confirm wiring",
      "Continue",
      "Let source decide (pass all fields through)",
    ]);
    expect(GUIDED_STAGE_PRIMARY_ACTION_NAMES).not.toContain("Build it");
  });

  it("treats guided/respond, not composer messages, as the compose step", () => {
    expect(
      isComposeRequest(
        `https://staging.example/api/sessions/${sessionId}/guided/respond`,
        "POST",
      ),
    ).toBe(true);
    expect(
      isComposeRequest(
        `https://staging.example/api/sessions/${sessionId}/messages`,
        "POST",
      ),
    ).toBe(false);
  });

  it("identifies the tutorial run request independently", () => {
    expect(
      isRunRequest("https://staging.example/api/tutorial/run", "POST"),
    ).toBe(true);
    expect(
      isRunRequest("https://staging.example/api/tutorial/run", "GET"),
    ).toBe(false);
  });
});
