import { describe, expect, it } from "vitest";

import { isAcknowledgementPrimaryActionName } from "./guided-driver";

describe("guided tutorial driver acknowledgement selectors", () => {
  it("covers the prompt-template two-stage primary and normal acknowledgement buttons", () => {
    expect(isAcknowledgementPrimaryActionName("View prompt")).toBe(true);
    expect(
      isAcknowledgementPrimaryActionName("Approve the LLM prompt template"),
    ).toBe(true);
    expect(
      isAcknowledgementPrimaryActionName("Acknowledge the pipeline decision"),
    ).toBe(true);
    expect(isAcknowledgementPrimaryActionName("Acknowledge")).toBe(true);
  });

  it("does not match the retired generic prompt toggle label", () => {
    expect(isAcknowledgementPrimaryActionName("View")).toBe(false);
  });
});
