import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProgressView } from "./ProgressView";
import { useWebSocket } from "@/hooks/useWebSocket";

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn(),
}));

describe("ProgressView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders live progress with explicit source and token units", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: {
        source_rows_processed: 1,
        tokens_succeeded: 9_323,
        tokens_failed: 2,
        tokens_quarantined: 1,
        tokens_routed_success: 7,
        tokens_routed_failure: 2,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    });

    render(<ProgressView />);

    expect(screen.getByText("Source Rows")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("Tokens Succeeded")).toBeInTheDocument();
    expect(screen.getByText("9,323")).toBeInTheDocument();
    expect(screen.getByText("Tokens Failed")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("7 routed success")).toBeInTheDocument();
    expect(screen.getByText("2 routed failure")).toBeInTheDocument();
    expect(screen.getByText("1 quarantined")).toBeInTheDocument();
  });
});
