import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SessionSidebar } from "./SessionSidebar";
import { TERMINAL_RUN_STATUS_VALUES, type RunStatus } from "@/types/index";

const selectSession = vi.fn();
const createSession = vi.fn();
const archiveSession = vi.fn();
const renameSession = vi.fn();
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
    renameSession,
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

describe("SessionSidebar rename", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    executionState.activeRunId = null;
    executionState.progress = null;
  });

  it("opens inline rename on double-click and saves the trimmed title", async () => {
    const user = userEvent.setup();
    renameSession.mockResolvedValue(undefined);
    render(<SessionSidebar />);

    await user.dblClick(screen.getByText("Current session"));
    const input = screen.getByRole("textbox", { name: "Rename session" });
    await user.clear(input);
    await user.type(input, "  Demo pipeline  ");
    await user.click(screen.getByRole("button", { name: "Save session name" }));

    expect(renameSession).toHaveBeenCalledWith("session-1", "Demo pipeline");
  });

  it("cancels inline rename with Escape", async () => {
    const user = userEvent.setup();
    render(<SessionSidebar />);

    await user.dblClick(screen.getByText("Current session"));
    await user.keyboard("{Escape}");

    expect(screen.queryByRole("textbox", { name: "Rename session" })).not.toBeInTheDocument();
    expect(renameSession).not.toHaveBeenCalled();
  });
});
