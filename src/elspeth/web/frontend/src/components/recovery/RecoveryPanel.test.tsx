import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { RecoveryPanel } from "./RecoveryPanel";
import type { CompositionState, ComposerRecoveryError } from "@/types/api";
import { compositionStateAuthorityFields } from "@/test/composerFixtures";

vi.mock("./RecoveryTranscript", () => ({
  RecoveryTranscript: () => <div>Tool transcript</div>,
}));

function makeState(version = 2): CompositionState {
  return {
    id: `state-${version}`,
    ...compositionStateAuthorityFields,
    version,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

function makeRecoveryError(): ComposerRecoveryError {
  return {
    status: 500,
    detail: "Composer failed after a tool call",
    error_type: "composer_plugin_crash",
    partial_state: makeState(3),
    failed_turn: {
      assistant_message_id: "assistant-1",
      tool_calls_attempted: 2,
      tool_responses_persisted: 1,
      transcript_url: null,
    },
  };
}

function renderPanel(
  overrides: Partial<React.ComponentProps<typeof RecoveryPanel>> = {},
) {
  const props: React.ComponentProps<typeof RecoveryPanel> = {
    activeSessionId: "session-1",
    currentState: makeState(1),
    recoveryError: makeRecoveryError(),
    onApply: vi.fn(() => ({ applied: true, needsConfirmation: false })),
    onDiscard: vi.fn(),
    ...overrides,
  };
  const result = render(<RecoveryPanel {...props} />);
  return { ...result, props };
}

function Harness({
  onApply,
  onDiscard,
}: {
  onApply: React.ComponentProps<typeof RecoveryPanel>["onApply"];
  onDiscard: React.ComponentProps<typeof RecoveryPanel>["onDiscard"];
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Return target
      </button>
      {open ? (
        <RecoveryPanel
          activeSessionId="session-1"
          currentState={makeState(1)}
          recoveryError={makeRecoveryError()}
          onApply={(options) => {
            const result = onApply(options);
            if (result.applied) {
              setOpen(false);
            }
            return result;
          }}
          onDiscard={() => {
            onDiscard();
            setOpen(false);
          }}
        />
      ) : null}
    </>
  );
}

describe("RecoveryPanel", () => {
  it("renders headline reason evidence diff transcript and controls", () => {
    renderPanel();

    expect(
      screen.getByRole("dialog", { name: "Recover partial composer draft" }),
    ).toHaveAttribute("aria-modal", "true");
    expect(screen.getByText("Composer failed after a tool call")).toBeInTheDocument();
    // The reason badge carries a visually-hidden "Recovery reason:" prefix
    // (an aria-label on a role-less span is not exposed to AT — WCAG 1.3.1,
    // elspeth-37293a3b7c).
    const reasonBadge = screen.getByText("Composer plugin crash");
    expect(reasonBadge).toHaveClass("recovery-panel-reason");
    expect(reasonBadge).toHaveTextContent(
      "Recovery reason: Composer plugin crash",
    );
    expect(screen.getByText("2 tool calls attempted")).toBeInTheDocument();
    expect(screen.getByText("1 tool response persisted")).toBeInTheDocument();
    expect(screen.getByText("Pipeline changes")).toBeInTheDocument();
    expect(screen.getByText("Tool transcript")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply partial draft" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Discard recovery" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "View raw transcript controls" })).toBeInTheDocument();
  });

  it("does not auto-apply when Enter is pressed on the dialog", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn(() => ({ applied: true, needsConfirmation: false }));
    renderPanel({ onApply });

    screen.getByRole("dialog").focus();
    await user.keyboard("{Enter}");

    expect(onApply).not.toHaveBeenCalled();
  });

  // elspeth-83eb51334f: this panel was the only role=dialog surface without
  // Escape dismissal. Escape routes through Discard — the panel's safe exit
  // (it drops the recovery OFFER, not composed state).
  it("dismisses via Discard when Escape is pressed", async () => {
    const user = userEvent.setup();
    const onDiscard = vi.fn();
    const onApply = vi.fn(() => ({ applied: false, needsConfirmation: false }));
    renderPanel({ onDiscard, onApply });

    screen.getByRole("dialog").focus();
    await user.keyboard("{Escape}");

    expect(onDiscard).toHaveBeenCalledTimes(1);
    expect(onApply).not.toHaveBeenCalled();
  });

  it("opens inline confirmation when apply reports a concurrent edit", async () => {
    const user = userEvent.setup();
    const onApply = vi
      .fn()
      .mockReturnValueOnce({ applied: false, needsConfirmation: true })
      .mockReturnValueOnce({ applied: true, needsConfirmation: false });
    renderPanel({ onApply });

    await user.click(screen.getByRole("button", { name: "Apply partial draft" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The current pipeline changed after this failed turn started.",
    );

    await user.click(screen.getByRole("button", { name: "Apply anyway" }));
    expect(onApply).toHaveBeenNthCalledWith(2, { confirmed: true });
  });

  it("discard closes without invoking apply", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn(() => ({ applied: true, needsConfirmation: false }));
    const onDiscard = vi.fn();
    renderPanel({ onApply, onDiscard });

    await user.click(screen.getByRole("button", { name: "Discard recovery" }));

    expect(onDiscard).toHaveBeenCalledTimes(1);
    expect(onApply).not.toHaveBeenCalled();
  });

  it("does not discard when the backdrop is clicked", async () => {
    const user = userEvent.setup();
    const onDiscard = vi.fn();
    const { container } = renderPanel({ onDiscard });

    const backdrop = container.querySelector(".recovery-panel-backdrop");
    expect(backdrop).not.toBeNull();
    await user.click(backdrop!);

    expect(onDiscard).not.toHaveBeenCalled();
  });

  it("initial focus lands on the safe apply action", async () => {
    renderPanel();

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Apply partial draft" })).toHaveFocus(),
    );
  });

  it("restores focus after discard apply confirm and cancel confirmation", async () => {
    const user = userEvent.setup();

    const discardApply = vi.fn(() => ({ applied: true, needsConfirmation: false }));
    const discardClose = vi.fn();
    const discardRender = render(
      <Harness onApply={discardApply} onDiscard={discardClose} />,
    );
    const discardReturn = screen.getByRole("button", { name: "Return target" });
    discardReturn.focus();
    await user.click(discardReturn);
    await user.click(screen.getByRole("button", { name: "Discard recovery" }));
    await waitFor(() => expect(discardReturn).toHaveFocus());
    discardRender.unmount();

    const applyFn = vi.fn(() => ({ applied: true, needsConfirmation: false }));
    const applyRender = render(<Harness onApply={applyFn} onDiscard={vi.fn()} />);
    const applyReturn = screen.getByRole("button", { name: "Return target" });
    applyReturn.focus();
    await user.click(applyReturn);
    await user.click(screen.getByRole("button", { name: "Apply partial draft" }));
    await waitFor(() => expect(applyReturn).toHaveFocus());
    applyRender.unmount();

    const confirmFn = vi
      .fn()
      .mockReturnValueOnce({ applied: false, needsConfirmation: true })
      .mockReturnValueOnce({ applied: true, needsConfirmation: false });
    const confirmRender = render(<Harness onApply={confirmFn} onDiscard={vi.fn()} />);
    const confirmReturn = screen.getByRole("button", { name: "Return target" });
    confirmReturn.focus();
    await user.click(confirmReturn);
    await user.click(screen.getByRole("button", { name: "Apply partial draft" }));
    await user.click(screen.getByRole("button", { name: "Apply anyway" }));
    await waitFor(() => expect(confirmReturn).toHaveFocus());
    confirmRender.unmount();

    const cancelFn = vi.fn(() => ({ applied: false, needsConfirmation: true }));
    const cancelRender = render(<Harness onApply={cancelFn} onDiscard={vi.fn()} />);
    const cancelReturn = screen.getByRole("button", { name: "Return target" });
    cancelReturn.focus();
    await user.click(cancelReturn);
    await user.click(screen.getByRole("button", { name: "Apply partial draft" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Discard recovery" }));
    await waitFor(() => expect(cancelReturn).toHaveFocus());
    cancelRender.unmount();
  });
});
