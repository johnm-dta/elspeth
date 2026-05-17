import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SideRailValidationBanner } from "./SideRailValidationBanner";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";
import { makeComposition } from "@/test/composerFixtures";

describe("SideRailValidationBanner", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    useExecutionStore.getState().reset();
  });

  it("renders execution-store errors as side-rail alerts", () => {
    useExecutionStore.setState({ error: "Validation service unavailable" });

    render(<SideRailValidationBanner />);

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Validation service unavailable",
    );
  });

  it("renders validation results from the execution store", () => {
    useSessionStore.setState({
      compositionState: makeComposition(1),
    } as never);
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        summary: "Validation failed",
        checks: [],
        errors: [
          {
            component_id: "select_columns",
            component_type: "transform",
            message: "Bad transform",
            suggestion: "Choose a supported projection.",
          },
        ],
        warnings: [],
      },
    });

    render(<SideRailValidationBanner />);

    expect(screen.getByRole("alert")).toHaveTextContent("Bad transform");
    expect(
      screen.getByRole("button", { name: /transform:select_columns/ }),
    ).toBeInTheDocument();
  });

  it("selects node ids without dispatching retired inspector tab events", async () => {
    const user = userEvent.setup();
    const selectNode = vi.fn();
    const onSwitchTab = vi.fn();
    window.addEventListener("elspeth-switch-tab", onSwitchTab);
    useSessionStore.setState({
      compositionState: makeComposition(1),
      selectNode,
    } as never);
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        summary: "Validation failed",
        checks: [],
        errors: [
          {
            component_id: "select_columns",
            component_type: "transform",
            message: "Bad transform",
            suggestion: null,
          },
        ],
        warnings: [],
      },
    });

    render(<SideRailValidationBanner />);
    await user.click(
      screen.getByRole("button", { name: /transform:select_columns/ }),
    );

    expect(selectNode).toHaveBeenCalledWith("select_columns");
    expect(onSwitchTab).not.toHaveBeenCalled();
    window.removeEventListener("elspeth-switch-tab", onSwitchTab);
  });
});
