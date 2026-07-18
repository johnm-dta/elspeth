import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import {
  ExportYamlButton,
  EXPORT_YAML_EMPTY_PIPELINE_TITLE,
} from "./ExportYamlButton";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";
import type { CompositionState } from "@/types/index";
import { compositionStateAuthorityFields } from "@/test/composerFixtures";

/** Minimal composition with content (one source) — export is meaningful. */
function nonEmptyState(): CompositionState {
  return {
    id: "state-1",
    ...compositionStateAuthorityFields,
    version: 1,
    sources: { source: { plugin: "csv", options: {} } },
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

/** Composition record exists but holds no sources/nodes/outputs. */
function emptyState(): CompositionState {
  return {
    id: "state-0",
    ...compositionStateAuthorityFields,
    version: 1,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

describe("ExportYamlButton", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: null,
      compositionState: null,
    } as never);
  });

  it("renders nothing when there is no active session", () => {
    const { container } = render(<ExportYamlButton />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the export button when a session is active", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExportYamlButton />);

    expect(
      screen.getByRole("button", { name: /export yaml/i }),
    ).toBeInTheDocument();
  });

  it("dispatches OPEN_YAML_MODAL_EVENT on click when the pipeline has content", () => {
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: nonEmptyState(),
    } as never);
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);

    render(<ExportYamlButton />);
    fireEvent.click(screen.getByRole("button", { name: /export yaml/i }));

    expect(handler).toHaveBeenCalled();
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
  });

  // elspeth-bff8043d33: Export must be disabled — with a stated reason —
  // while the pipeline has no components, matching its Run/Save siblings
  // instead of opening a near-empty modal.
  it("disables export with a reason when no composition exists yet", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);

    render(<ExportYamlButton />);

    const button = screen.getByRole("button", { name: /export yaml/i });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-disabled", "true");
    expect(button).toHaveAttribute("title", EXPORT_YAML_EMPTY_PIPELINE_TITLE);
    // The reason is surfaced to AT via aria-describedby → hidden span
    // (the disabled-with-reason idiom shared with ExecuteButton).
    expect(button).toHaveAccessibleDescription(
      EXPORT_YAML_EMPTY_PIPELINE_TITLE,
    );
  });

  it("disables export when the composition exists but has no components", () => {
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: emptyState(),
    } as never);

    render(<ExportYamlButton />);

    expect(screen.getByRole("button", { name: /export yaml/i })).toBeDisabled();
  });

  it("does not dispatch the open event while disabled", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);

    render(<ExportYamlButton />);
    fireEvent.click(screen.getByRole("button", { name: /export yaml/i }));

    expect(handler).not.toHaveBeenCalled();
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
  });

  it("enables export without a disabled reason once the pipeline has content", () => {
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: nonEmptyState(),
    } as never);

    render(<ExportYamlButton />);

    const button = screen.getByRole("button", { name: /export yaml/i });
    expect(button).toBeEnabled();
    expect(button).not.toHaveAttribute("aria-disabled");
    expect(button).not.toHaveAttribute("title");
    expect(button).not.toHaveAttribute("aria-describedby");
  });
});
