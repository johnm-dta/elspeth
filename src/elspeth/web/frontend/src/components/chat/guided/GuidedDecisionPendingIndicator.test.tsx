import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import type { ComposerProgressSnapshot } from "@/types/api";
import { GuidedDecisionPendingIndicator } from "./GuidedDecisionPendingIndicator";

function snapshot(
  overrides: Partial<ComposerProgressSnapshot> = {},
): ComposerProgressSnapshot {
  return {
    session_id: "session-1",
    request_id: "req-1",
    phase: "calling_model",
    headline: "Planning the wiring update.",
    evidence: [],
    likely_next: null,
    reason: null,
    updated_at: "2026-07-23T00:00:00Z",
    ...overrides,
  };
}

describe("GuidedDecisionPendingIndicator", () => {
  it("renders the fallback headline and an elapsed readout with no live progress", () => {
    render(
      <p role="status">
        <GuidedDecisionPendingIndicator fallback="Saving decision..." composerProgress={null} />
      </p>,
    );
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Saving decision...");
    expect(status).toHaveTextContent(/00:00/);
  });

  it("renders the live phase headline while a snapshot is non-terminal", () => {
    render(
      <p role="status">
        <GuidedDecisionPendingIndicator
          fallback="Saving decision..."
          composerProgress={snapshot()}
        />
      </p>,
    );
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Planning the wiring update.");
    expect(status).not.toHaveTextContent("Saving decision...");
  });

  it.each(["idle", "complete", "failed", "cancelled"] as const)(
    "ignores a stale %s snapshot and keeps the fallback headline",
    (phase) => {
      render(
        <p role="status">
          <GuidedDecisionPendingIndicator
            fallback="Saving decision..."
            composerProgress={snapshot({ phase })}
          />
        </p>,
      );
      expect(screen.getByRole("status")).toHaveTextContent("Saving decision...");
    },
  );
});
