import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { YamlView } from "./YamlView";
import { useSessionStore } from "@/stores/sessionStore";

vi.mock("@/api/client", () => ({
  fetchYaml: vi.fn(),
}));

function makeState(version = 1) {
  return {
    id: "state-1",
    version,
    source: null,
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: "test", description: "" },
  };
}

describe("YamlView", () => {
  beforeEach(async () => {
    useSessionStore.setState({
      activeSessionId: null,
      compositionState: null,
    });

    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockReset();
  });

  it("renders the empty state when there is no composition state", () => {
    render(<YamlView />);

    expect(
      screen.getByText("YAML will appear here once your pipeline has components."),
    ).toBeInTheDocument();
  });

  it("shows a validation-blocked alert when YAML export returns 409", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockRejectedValue({
      status: 409,
      detail: "Current composition state is invalid. Fix validation errors before exporting YAML.",
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
    });

    render(<YamlView />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "YAML export is blocked by validation errors.",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Current composition state is invalid. Fix validation errors before exporting YAML.",
    );
    expect(
      screen.queryByText("YAML will appear here once your pipeline has components."),
    ).not.toBeInTheDocument();
  });

  it("exposes copied state as a data attribute for forced-colors CSS", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockResolvedValue({
      yaml: "source:\n  plugin: text\n",
    });
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
    });

    render(<YamlView />);
    const user = userEvent.setup();

    const copyButton = await screen.findByRole("button", {
      name: "Copy YAML to clipboard",
    });
    expect(copyButton).toHaveAttribute("data-copied", "false");

    await user.click(copyButton);

    await waitFor(() => {
      expect(copyButton).toHaveAttribute("data-copied", "true");
    });
  });

  it("hides visual line numbers from assistive technology", async () => {
    const { fetchYaml } = await import("@/api/client");
    vi.mocked(fetchYaml).mockResolvedValue({
      yaml: "source:\n  plugin: text\n",
    });

    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(),
    });

    const { container } = render(<YamlView />);

    await screen.findByRole("button", { name: "Copy YAML to clipboard" });
    const lineNumbers = container.querySelectorAll(".yaml-view-line-number");

    expect(lineNumbers.length).toBeGreaterThan(0);
    lineNumbers.forEach((lineNumber) => {
      expect(lineNumber).toHaveAttribute("aria-hidden", "true");
    });
  });
});
