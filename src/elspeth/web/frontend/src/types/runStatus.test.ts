import { describe, expect, it } from "vitest";

import {
  RUN_STATUS_VALUES,
  TERMINAL_RUN_STATUS_VALUES,
  isTerminalRunStatus,
  type RunStatus,
} from "./index";

describe("run-status taxonomy", () => {
  it("classifies every declared status through the shared terminal guard", () => {
    const terminal = new Set<RunStatus>(TERMINAL_RUN_STATUS_VALUES);

    for (const status of RUN_STATUS_VALUES) {
      expect(isTerminalRunStatus(status)).toBe(terminal.has(status));
    }
  });
});
