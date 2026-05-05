import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SessionSidebar } from "./SessionSidebar";
import { TERMINAL_RUN_STATUS_VALUES, type RunStatus } from "@/types/index";

const selectSession = vi.fn();
const createSession = vi.fn();
const archiveSession = vi.fn();
const logout = vi.fn();

const executionState: {
  activeRunId: string | null;
  progress: { status: RunStatus } | null;
} = {
  activeRunId: "run-1",
  progress: { status: "running" },
};

vi.mock("@/hooks/useSession", () => ({
  useSession: () => ({
    sessions: [
      {
        id: "session-1",
        title: "Current session",
        updated_at: "2026-05-05T00:00:00.000Z",
        forked_from_session_id: null,
      },
    ],
    activeSessionId: "session-1",
    createSession,
    selectSession,
    archiveSession,
  }),
}));

vi.mock("@/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { username: "operator" },
    logout,
  }),
}));

vi.mock("@/stores/executionStore", () => ({
  useExecutionStore: <T,>(selector: (state: typeof executionState) => T) =>
    selector(executionState),
}));

describe("SessionSidebar active run indicator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    executionState.activeRunId = "run-1";
    executionState.progress = { status: "running" };
  });

  it("shows the active-run marker while a run is non-terminal", () => {
    render(<SessionSidebar />);

    expect(screen.getByLabelText("Pipeline running")).toBeInTheDocument();
  });

  it("hides the active-run marker for every terminal status", () => {
    for (const status of TERMINAL_RUN_STATUS_VALUES) {
      executionState.progress = { status };
      const { unmount } = render(<SessionSidebar />);

      expect(screen.queryByLabelText("Pipeline running")).not.toBeInTheDocument();
      unmount();
    }
  });
});
