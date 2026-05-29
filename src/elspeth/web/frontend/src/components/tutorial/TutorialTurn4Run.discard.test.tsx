import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api/client";
import { TutorialTurn4Run } from "./TutorialTurn4Run";

vi.mock("@/api/client", () => ({ runTutorialPipeline: vi.fn() }));

function noop(): void {}

describe("TutorialTurn4Run — source-discarded row surfacing (#28)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("surfaces the discarded row count when the source dropped rows", async () => {
    vi.mocked(api.runTutorialPipeline).mockResolvedValue({
      run_id: "run-discard",
      output: {
        rows: [{ url: "a" }, { url: "b" }],
        source_data_hash: "h",
        discarded_row_count: 3,
      },
      seeded_from_cache: false,
      cache_key: null,
    });

    render(
      <TutorialTurn4Run
        sessionId="sess-discard"
        prompt="discard-prompt"
        onCompleted={noop}
        onCancelled={noop}
        onBack={noop}
      />,
    );

    expect(await screen.findByText(/3 rows were discarded at the source/i)).toBeInTheDocument();
  });

  it("shows no discard notice when nothing was dropped", async () => {
    vi.mocked(api.runTutorialPipeline).mockResolvedValue({
      run_id: "run-clean",
      output: {
        rows: [{ url: "a" }],
        source_data_hash: "h",
        discarded_row_count: 0,
      },
      seeded_from_cache: false,
      cache_key: null,
    });

    render(
      <TutorialTurn4Run
        sessionId="sess-clean"
        prompt="clean-prompt"
        onCompleted={noop}
        onCancelled={noop}
        onBack={noop}
      />,
    );

    expect(await screen.findByText(/rows returned/i)).toBeInTheDocument();
    expect(screen.queryByText(/discarded at the source/i)).not.toBeInTheDocument();
  });
});
