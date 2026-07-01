import { beforeEach, describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CatalogDrawer } from "./CatalogDrawer";
import * as api from "@/api/client";

vi.mock("@/api/client", () => ({
  listSources: vi.fn().mockResolvedValue([
    {
      name: "csv",
      plugin_type: "source",
      description: "CSV file source",
      config_fields: [],
      usage_when_to_use: null,
      usage_when_not_to_use: null,
      example_use: null,
      capability_tags: [],
      audit_characteristics: [],
    },
  ]),
  listTransforms: vi.fn().mockResolvedValue([
    {
      name: "uppercase",
      plugin_type: "transform",
      description: "Uppercase transform",
      config_fields: [],
      usage_when_to_use: null,
      usage_when_not_to_use: null,
      example_use: null,
      capability_tags: [],
      audit_characteristics: [],
    },
  ]),
  listSinks: vi.fn().mockResolvedValue([
    {
      name: "json",
      plugin_type: "sink",
      description: "JSON file sink",
      config_fields: [],
      usage_when_to_use: null,
      usage_when_not_to_use: null,
      example_use: null,
      capability_tags: [],
      audit_characteristics: [],
    },
  ]),
  getPluginSchema: vi.fn().mockResolvedValue({
    name: "csv",
    plugin_type: "source",
    description: "CSV file source",
    json_schema: {
      properties: { path: { type: "string", description: "File path" } },
      required: ["path"],
    },
  }),
}));

// Import the mocked client so we can assert on call counts.
import { listSources, listTransforms, listSinks } from "@/api/client";

describe("CatalogDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders nothing when closed", () => {
    render(<CatalogDrawer isOpen={false} onClose={vi.fn()} />);
    expect(screen.queryByText("Plugin Catalog")).not.toBeInTheDocument();
  });

  it("fetches catalog on first open", async () => {
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    await waitFor(() => {
      expect(listSources).toHaveBeenCalled();
      expect(listTransforms).toHaveBeenCalled();
      expect(listSinks).toHaveBeenCalled();
    });
  });

  it("announces the open drawer as a modal dialog", async () => {
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);

    const dialog = screen.getByRole("dialog", { name: "Plugin Catalog" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByText("Reference")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Browse available sources, transforms, and sinks before asking the composer to apply them.",
      ),
    ).toBeInTheDocument();
    await screen.findByText("CSV file source");
  });

  it("shows three tabs", async () => {
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Sources" })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Transforms" })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Sinks" })).toBeInTheDocument();
    });
  });

  it("links tabs to the active tab panel with roving focus metadata", async () => {
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);

    const sourcesTab = await screen.findByRole("tab", { name: "Sources" });
    const transformsTab = screen.getByRole("tab", { name: "Transforms" });
    const panel = screen.getByRole("tabpanel", { name: "Sources" });

    expect(sourcesTab).toHaveAttribute("id", "catalog-tab-sources");
    expect(sourcesTab).toHaveAttribute("aria-controls", "catalog-panel-sources");
    expect(sourcesTab).toHaveAttribute("tabIndex", "0");
    expect(transformsTab).toHaveAttribute("aria-controls", "catalog-panel-transforms");
    expect(transformsTab).toHaveAttribute("tabIndex", "-1");
    expect(panel).toHaveAttribute("id", "catalog-panel-sources");
    expect(panel).toHaveAttribute("aria-labelledby", "catalog-tab-sources");
  });

  it("supports arrow-key tab navigation with wrapping focus", async () => {
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    const user = userEvent.setup();

    const sourcesTab = await screen.findByRole("tab", { name: "Sources" });
    sourcesTab.focus();

    await user.keyboard("{ArrowRight}");
    const transformsTab = screen.getByRole("tab", { name: "Transforms" });
    expect(transformsTab).toHaveFocus();
    expect(transformsTab).toHaveAttribute("aria-selected", "true");
    expect(transformsTab).toHaveAttribute("tabIndex", "0");
    expect(sourcesTab).toHaveAttribute("tabIndex", "-1");
    expect(screen.getByRole("tabpanel", { name: "Transforms" })).toHaveAttribute(
      "id",
      "catalog-panel-transforms",
    );

    await user.keyboard("{ArrowLeft}");
    expect(sourcesTab).toHaveFocus();
    expect(sourcesTab).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{ArrowLeft}");
    const sinksTab = screen.getByRole("tab", { name: "Sinks" });
    expect(sinksTab).toHaveFocus();
    expect(sinksTab).toHaveAttribute("aria-selected", "true");
  });

  it("shows plugin list after fetch", async () => {
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    await waitFor(() => {
      expect(screen.getByText("csv")).toBeInTheDocument();
      expect(screen.getByText("CSV file source")).toBeInTheDocument();
    });
  });

  it("preserves the search query across close and reopen", async () => {
    const { rerender } = render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    const user = userEvent.setup();
    const input = await screen.findByRole("textbox", { name: "Search plugins" });

    await user.type(input, "csv");
    expect(input).toHaveValue("csv");

    rerender(<CatalogDrawer isOpen={false} onClose={vi.fn()} />);
    expect(screen.queryByRole("textbox", { name: "Search plugins" })).not.toBeInTheDocument();

    rerender(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByRole("textbox", { name: "Search plugins" })).toHaveValue("csv");
  });

  it("retries catalog loading inline after an initial fetch failure", async () => {
    vi.mocked(listSources)
      .mockRejectedValueOnce(new Error("catalog unavailable"))
      .mockResolvedValueOnce([
        {
          name: "csv",
          plugin_type: "source",
          description: "CSV file source",
          config_fields: [],
          usage_when_to_use: null,
          usage_when_not_to_use: null,
          example_use: null,
          capability_tags: [],
          audit_characteristics: [],
        },
      ]);
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    const user = userEvent.setup();

    expect(await screen.findByText("Failed to load plugin catalog.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry loading plugin catalog" }));

    await waitFor(() => {
      expect(listSources).toHaveBeenCalledTimes(2);
    });
    expect(await screen.findByText("CSV file source")).toBeInTheDocument();
  });

  it("announces a catalog load failure through an assertive alert (WCAG 4.1.3)", async () => {
    vi.mocked(listSources).mockRejectedValueOnce(new Error("catalog unavailable"));
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Failed to load plugin catalog.");
  });

  it("escape key closes drawer", async () => {
    const onClose = vi.fn();
    render(<CatalogDrawer isOpen={true} onClose={onClose} />);
    const user = userEvent.setup();
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("backdrop click closes drawer", async () => {
    const onClose = vi.fn();
    render(<CatalogDrawer isOpen={true} onClose={onClose} />);
    const user = userEvent.setup();
    const backdrop = screen.getByTestId("catalog-backdrop");
    await user.click(backdrop);
    expect(onClose).toHaveBeenCalled();
  });
});

const csv = {
  name: "csv",
  plugin_type: "source",
  description: "Read CSV files.",
  config_fields: [],
  usage_when_to_use: "When you have a CSV file.",
  usage_when_not_to_use: null,
  example_use: null,
  capability_tags: ["csv", "file"],
  audit_characteristics: ["io_read", "quarantine"],
};
const azure = {
  name: "azure_blob",
  plugin_type: "source",
  description: "Read Azure blobs.",
  config_fields: [],
  usage_when_to_use: null,
  usage_when_not_to_use: null,
  example_use: null,
  capability_tags: ["azure", "blob", "network"],
  audit_characteristics: ["io_read", "external_call"],
};

describe("CatalogDrawer — Phase 7B reshape", () => {
  beforeEach(() => {
    vi.mocked(api.listSources).mockResolvedValue([csv, azure] as never);
    vi.mocked(api.listTransforms).mockResolvedValue([] as never);
    vi.mocked(api.listSinks).mockResolvedValue([] as never);
  });

  it("renders InlineChatSourceEntry as the first row of the Sources tab", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("azure_blob")).toBeInTheDocument());
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
  });

  it("does NOT render InlineChatSourceEntry on the Transforms or Sinks tabs", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("azure_blob")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("tab", { name: /transforms/i }));
    expect(screen.queryByText(/inline data from chat/i)).not.toBeInTheDocument();
  });

  it("renders capability-tag chips derived from the loaded source list", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("azure_blob")).toBeInTheDocument());
    // The chip strip shows tags present in the visible-tab plugins.
    expect(screen.getByRole("button", { name: /^csv$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^azure$/i })).toBeInTheDocument();
  });

  it("filtering by capability tag narrows the visible plugins", async () => {
    // Assert on plugin descriptions (unique to plugin cards) rather than
    // names — once "csv" is also a capability-tag chip, screen.getByText("csv")
    // is ambiguous because the chip strip renders a button with that text.
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Read Azure blobs.")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /^csv$/i }));
    // csv plugin still visible; azure_blob filtered out (no "csv" tag).
    expect(screen.getByText("Read CSV files.")).toBeInTheDocument();
    expect(screen.queryByText("Read Azure blobs.")).not.toBeInTheDocument();
  });

  it("filtering by audit characteristic narrows the visible plugins", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Read Azure blobs.")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /network call/i }));
    // azure has "external_call" (rendered as "Network call"); csv doesn't.
    expect(screen.getByText("Read Azure blobs.")).toBeInTheDocument();
    expect(screen.queryByText("Read CSV files.")).not.toBeInTheDocument();
  });

  it("extends search across the when_to_use prose", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Read Azure blobs.")).toBeInTheDocument());
    const input = screen.getByPlaceholderText(/search plugins/i);
    await userEvent.type(input, "CSV file");
    // csv's usage_when_to_use mentions "CSV file"; azure's doesn't.
    expect(screen.getByText("Read CSV files.")).toBeInTheDocument();
    expect(screen.queryByText("Read Azure blobs.")).not.toBeInTheDocument();
  });

  it("extends search across capability tags", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Read Azure blobs.")).toBeInTheDocument());
    const input = screen.getByPlaceholderText(/search plugins/i);
    await userEvent.type(input, "blob");
    expect(screen.queryByText("Read CSV files.")).not.toBeInTheDocument();
    expect(screen.getByText("Read Azure blobs.")).toBeInTheDocument();
  });

  it("filter state is per-tab — switching tabs does not carry filters over", async () => {
    // Regression guard for the cross-tab UX trap. An active capability
    // filter on Sources must NOT silently filter Transforms on tab switch.
    vi.mocked(api.listTransforms).mockResolvedValue([
      {
        name: "uppercase",
        plugin_type: "transform",
        description: "Uppercase strings.",
        config_fields: [],
        usage_when_to_use: null,
        usage_when_not_to_use: null,
        example_use: null,
        capability_tags: ["string"],
        audit_characteristics: ["deterministic"],
      },
    ] as never);
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("azure_blob")).toBeInTheDocument());

    // Activate "csv" filter on Sources tab.
    await userEvent.click(screen.getByRole("button", { name: /^csv$/i }));
    expect(screen.queryByText("Read Azure blobs.")).not.toBeInTheDocument();

    // Switch to Transforms — uppercase must be visible (its tab has no filter).
    await userEvent.click(screen.getByRole("tab", { name: /transforms/i }));
    await waitFor(() => expect(screen.getByText("uppercase")).toBeInTheDocument());
  });

  it("shows 'No plugins match the active filters.' when filters are non-empty and match nothing", async () => {
    // B3 regression gate: empty-state message must vary by filter state.
    // Targeted scenario: two plugins, one filter that matches only csv,
    // a second filter that matches only azure — AND composition empties the list.
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("azure_blob")).toBeInTheDocument());
    // Click "file" capability chip (only csv has it).
    await userEvent.click(screen.getByRole("button", { name: /^file$/i }));
    // Then click the "Network call" audit chip (only azure has external_call).
    await userEvent.click(screen.getByRole("button", { name: /network call/i }));
    // Plugin list is now empty with active filters.
    const empty = screen.getByText("No plugins match the active filters.");
    expect(empty).toBeInTheDocument();
    // M05 (WCAG 4.1.3): the empty state is a polite live region so a
    // screen-reader user hears it when filters eliminate every plugin.
    expect(empty).toHaveAttribute("role", "status");
    expect(empty).toHaveAttribute("aria-live", "polite");
    // Synthetic entry must STILL be visible — it is a pinned affordance.
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
  });

  it("shows 'No plugins available.' (not the filter variant) when no filters are active and the list is empty", async () => {
    // B3 regression gate: empty-state message with no active filters.
    vi.mocked(api.listSources).mockResolvedValue([] as never);
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    const empty = await screen.findByText("No plugins available.");
    // M05 (WCAG 4.1.3): announced through a polite live region.
    expect(empty).toHaveAttribute("role", "status");
    expect(empty).toHaveAttribute("aria-live", "polite");
    // Synthetic entry must still be visible — no filters are active and
    // the list is empty, but InlineChatSourceEntry is always pinned.
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
  });

  it("announces the visible plugin count through a polite live region (M05)", async () => {
    // Two sources load → "2 plugins"; narrowing to one via a capability
    // filter updates the same live region to the singular "1 plugin".
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("azure_blob")).toBeInTheDocument());

    const count = screen.getByText("2 plugins");
    expect(count).toHaveAttribute("role", "status");
    expect(count).toHaveAttribute("aria-live", "polite");

    // Filter to the single csv plugin — count switches to the singular form.
    await userEvent.click(screen.getByRole("button", { name: /^csv$/i }));
    expect(screen.getByText("1 plugin")).toHaveAttribute("aria-live", "polite");
  });

  it("exposes the plugin cards as a list with one listitem per plugin (M06)", async () => {
    // WCAG 1.3.1: the flat run of cards now carries list semantics so AT
    // announces "list, N items". The pinned InlineChatSourceEntry is a
    // role="region", NOT a listitem, so it does not inflate the count.
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("azure_blob")).toBeInTheDocument());

    expect(screen.getByRole("list")).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    // Each card is an article named by its plugin name (M06 second half).
    expect(screen.getByRole("article", { name: "csv" })).toBeInTheDocument();
    expect(screen.getByRole("article", { name: "azure_blob" })).toBeInTheDocument();
  });

  it("InlineChatSourceEntry and empty-state message are simultaneously visible when filters eliminate all real plugins", async () => {
    // B3 regression gate: the two sibling renders coexist.
    // This test is a red-green gate: if InlineChatSourceEntry is rendered
    // inside the pluginList.length === 0 conditional, one of the two
    // assertions below will fail (the synthetic entry disappears or the
    // empty state is suppressed). With the Step-3 restructure both pass.
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("azure_blob")).toBeInTheDocument());
    // Filter to a combination that eliminates all real plugins (AND logic).
    await userEvent.click(screen.getByRole("button", { name: /^file$/i }));
    await userEvent.click(screen.getByRole("button", { name: /network call/i }));
    // Both must be in the document at the same time.
    expect(screen.getByText("No plugins match the active filters.")).toBeInTheDocument();
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
  });
});

describe("CatalogDrawer — Alt+1-3 tab shortcuts", () => {
  it("Alt+1 switches to the Sources tab", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByRole("tab", { name: /sources/i })).toBeInTheDocument());

    // Start on Sources (default); switch away first to make the Alt+1 test meaningful.
    await userEvent.click(screen.getByRole("tab", { name: /transforms/i }));
    expect(screen.getByRole("tab", { name: /transforms/i })).toHaveAttribute("aria-selected", "true");

    fireEvent.keyDown(document, { key: "1", altKey: true });
    expect(screen.getByRole("tab", { name: /sources/i })).toHaveAttribute("aria-selected", "true");
  });

  it("Alt+2 switches to the Transforms tab", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByRole("tab", { name: /sources/i })).toBeInTheDocument());

    fireEvent.keyDown(document, { key: "2", altKey: true });
    expect(screen.getByRole("tab", { name: /transforms/i })).toHaveAttribute("aria-selected", "true");
  });

  it("Alt+3 switches to the Sinks tab", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByRole("tab", { name: /sources/i })).toBeInTheDocument());

    fireEvent.keyDown(document, { key: "3", altKey: true });
    expect(screen.getByRole("tab", { name: /sinks/i })).toHaveAttribute("aria-selected", "true");
  });

  it("Alt+1-3 has no effect when drawer is closed", () => {
    render(<CatalogDrawer isOpen={false} onClose={() => {}} />);
    // No keydown handler registered when closed — this must not throw.
    fireEvent.keyDown(document, { key: "1", altKey: true });
  });
});
