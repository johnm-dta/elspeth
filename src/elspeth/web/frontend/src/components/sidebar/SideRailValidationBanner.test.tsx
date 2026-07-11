import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SideRailValidationBanner } from "./SideRailValidationBanner";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";
import { makeComposition } from "@/test/composerFixtures";

const SUGGESTION = {
  component: "csv_source",
  message: "Consider increasing batch size",
  severity: "info",
};
const BLOCKED_READINESS = {
  authoring_valid: false,
  execution_ready: false,
  completion_ready: false,
  blockers: [],
};

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
        readiness: {
          ...BLOCKED_READINESS,
          blockers: [
            {
              code: "validation_error",
              component_id: "select_columns",
              component_type: "transform",
              detail: "select_columns",
            },
          ],
        },
      },
    });

    render(<SideRailValidationBanner />);

    expect(screen.getByRole("alert")).toHaveTextContent("Bad transform");
    expect(
      screen.getByRole("button", { name: /transform:select_columns/ }),
    ).toBeInTheDocument();
  });

  it("selects node ids and opens the graph modal without dispatching retired inspector tab events", async () => {
    const user = userEvent.setup();
    const selectNode = vi.fn();
    const onSwitchTab = vi.fn();
    const onOpenGraph = vi.fn();
    window.addEventListener("elspeth-switch-tab", onSwitchTab);
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, onOpenGraph);
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
        readiness: {
          ...BLOCKED_READINESS,
          blockers: [
            {
              code: "validation_error",
              component_id: "select_columns",
              component_type: "transform",
              detail: "select_columns",
            },
          ],
        },
      },
    });

    render(<SideRailValidationBanner />);
    await user.click(
      screen.getByRole("button", { name: /transform:select_columns/ }),
    );

    expect(selectNode).toHaveBeenCalledWith("select_columns");
    expect(onOpenGraph).toHaveBeenCalledTimes(1);
    expect(onSwitchTab).not.toHaveBeenCalled();
    window.removeEventListener("elspeth-switch-tab", onSwitchTab);
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, onOpenGraph);
  });

  it("selects source validation entries and opens the graph modal", async () => {
    const user = userEvent.setup();
    const selectNode = vi.fn();
    const onOpenGraph = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, onOpenGraph);
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
            component_id: "source",
            component_type: "source",
            message: "Source path is outside the allowlist",
            suggestion: null,
          },
        ],
        warnings: [],
        readiness: BLOCKED_READINESS,
      },
    });

    render(<SideRailValidationBanner />);
    await user.click(screen.getByRole("button", { name: /source:\s*source/ }));

    expect(selectNode).toHaveBeenCalledWith("source");
    expect(onOpenGraph).toHaveBeenCalledTimes(1);
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, onOpenGraph);
  });

  it("selects sink validation entries and opens the graph modal", async () => {
    const user = userEvent.setup();
    const selectNode = vi.fn();
    const onOpenGraph = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, onOpenGraph);
    useSessionStore.setState({
      compositionState: makeComposition(1, {
        outputs: [
          {
            name: "validated_rows",
            plugin: "jsonl_file",
            options: {},
          },
        ],
      }),
      selectNode,
    } as never);
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        summary: "Validation failed",
        checks: [],
        errors: [
          {
            component_id: "validated_rows",
            component_type: "sink",
            message: "Sink output path is outside the allowlist",
            suggestion: null,
          },
        ],
        warnings: [],
        readiness: BLOCKED_READINESS,
      },
    });

    render(<SideRailValidationBanner />);
    await user.click(
      screen.getByRole("button", { name: /sink:\s*validated_rows/ }),
    );

    expect(selectNode).toHaveBeenCalledWith("validated_rows");
    expect(onOpenGraph).toHaveBeenCalledTimes(1);
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, onOpenGraph);
  });

  describe("SuggestionList", () => {
    it("renders suggestion text and Apply button when suggestions are present", () => {
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: [SUGGESTION],
        }),
      } as never);

      render(<SideRailValidationBanner />);

      expect(
        screen.getByText(/Consider increasing batch size/),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Apply" }),
      ).toBeInTheDocument();
    });

    it("banner appears when only suggestions are present (error and validationResult are null)", () => {
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: [SUGGESTION],
        }),
      } as never);
      // error and validationResult remain null (store reset in beforeEach)

      const { container } = render(<SideRailValidationBanner />);

      expect(container.firstChild).not.toBeNull();
      expect(
        screen.getByText(/Consider increasing batch size/),
      ).toBeInTheDocument();
    });

    it("Apply button shows 'Apply' when idle and 'Applying...' when composing, and is disabled while composing", () => {
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: [SUGGESTION],
        }),
        isComposing: false,
        // Post-boot: the backend wall clock has landed, so the readiness gate
        // is open and Apply reflects only the composing state.
        composeTimeoutReady: true,
      } as never);

      const { unmount } = render(<SideRailValidationBanner />);

      expect(
        screen.getByRole("button", { name: "Apply" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Apply" }),
      ).not.toBeDisabled();

      unmount();

      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: [SUGGESTION],
        }),
        isComposing: true,
        composeTimeoutReady: true,
      } as never);

      render(<SideRailValidationBanner />);

      expect(
        screen.getByRole("button", { name: "Applying..." }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Applying..." }),
      ).toBeDisabled();
    });

    it("handleApply sends the correct prompt when Apply is clicked", async () => {
      const user = userEvent.setup();
      const sendMessage = vi.fn().mockResolvedValue(undefined);
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: [SUGGESTION],
        }),
        sendMessage,
        isComposing: false,
        composeTimeoutReady: true,
      } as never);

      render(<SideRailValidationBanner />);
      await user.click(screen.getByRole("button", { name: "Apply" }));

      // useComposer wraps storeSendMessage(content, signal) — assert the exact
      // prompt string; the second arg is an AbortSignal from the hook.
      expect(sendMessage).toHaveBeenCalledWith(
        "Please apply this suggestion to the pipeline:\n\n**csv_source:** Consider increasing batch size",
        expect.any(AbortSignal),
      );
    });

    it("Apply button activates via keyboard (Enter key)", async () => {
      const user = userEvent.setup();
      const sendMessage = vi.fn().mockResolvedValue(undefined);
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: [SUGGESTION],
        }),
        sendMessage,
        isComposing: false,
        composeTimeoutReady: true,
      } as never);

      render(<SideRailValidationBanner />);
      const applyBtn = screen.getByRole("button", { name: "Apply" });
      applyBtn.focus();
      await user.keyboard("{Enter}");

      expect(sendMessage).toHaveBeenCalledTimes(1);
    });

    it("holds Apply closed until the compose timeout is ready (bootstrap race)", async () => {
      // composeTimeoutReady defaults false via resetStore — the boot window.
      // The side-rail Apply is a programmatic freeform sender; it must not
      // start a compose against the stale default ceiling any more than the
      // main Send may.
      const user = userEvent.setup();
      const sendMessage = vi.fn().mockResolvedValue(undefined);
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: [SUGGESTION],
        }),
        sendMessage,
        isComposing: false,
      } as never);

      render(<SideRailValidationBanner />);
      const applyBtn = screen.getByRole("button", { name: "Apply" });
      expect(applyBtn).toBeDisabled();
      await user.click(applyBtn);
      expect(sendMessage).not.toHaveBeenCalled();
    });

    it("collapses by default when more than 2 suggestions are present, expands on header click", async () => {
      const user = userEvent.setup();
      const manySuggestions = [
        SUGGESTION,
        { component: "sink_a", message: "Msg A", severity: "info" },
        { component: "sink_b", message: "Msg B", severity: "info" },
      ];
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: manySuggestions,
        }),
      } as never);

      render(<SideRailValidationBanner />);

      // Default collapsed — list items should not be visible
      expect(
        screen.queryByText(/Consider increasing batch size/),
      ).not.toBeInTheDocument();

      // Click the header to expand
      await user.click(screen.getByRole("button", { name: /Suggestions \(3\)/ }));

      expect(
        screen.getByText(/Consider increasing batch size/),
      ).toBeInTheDocument();
    });

    it("collapsible header toggles on Enter keypress", async () => {
      const user = userEvent.setup();
      const manySuggestions = [
        SUGGESTION,
        { component: "sink_a", message: "Msg A", severity: "info" },
        { component: "sink_b", message: "Msg B", severity: "info" },
      ];
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: manySuggestions,
        }),
      } as never);

      render(<SideRailValidationBanner />);

      const header = screen.getByRole("button", { name: /Suggestions/ });
      expect(header).toHaveAttribute("aria-expanded", "false");

      header.focus();
      await user.keyboard("{Enter}");

      expect(header).toHaveAttribute("aria-expanded", "true");
      expect(
        screen.getByText(/Consider increasing batch size/),
      ).toBeInTheDocument();

      await user.keyboard("{Enter}");

      expect(header).toHaveAttribute("aria-expanded", "false");
    });

    it("collapsible header toggles on Space keypress", async () => {
      const user = userEvent.setup();
      const manySuggestions = [
        SUGGESTION,
        { component: "sink_a", message: "Msg A", severity: "info" },
        { component: "sink_b", message: "Msg B", severity: "info" },
      ];
      useSessionStore.setState({
        compositionState: makeComposition(1, {
          validation_suggestions: manySuggestions,
        }),
      } as never);

      render(<SideRailValidationBanner />);

      const header = screen.getByRole("button", { name: /Suggestions/ });
      expect(header).toHaveAttribute("aria-expanded", "false");

      header.focus();
      await user.keyboard(" ");

      expect(header).toHaveAttribute("aria-expanded", "true");
      expect(
        screen.getByText(/Consider increasing batch size/),
      ).toBeInTheDocument();

      await user.keyboard(" ");

      expect(header).toHaveAttribute("aria-expanded", "false");
    });
  });
});
