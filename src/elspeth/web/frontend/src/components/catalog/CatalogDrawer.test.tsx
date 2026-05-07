import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CatalogDrawer } from "./CatalogDrawer";

vi.mock("@/api/client", () => ({
  listSources: vi.fn().mockResolvedValue([
    { name: "csv", plugin_type: "source", description: "CSV file source", config_fields: [] },
  ]),
  listTransforms: vi.fn().mockResolvedValue([
    { name: "uppercase", plugin_type: "transform", description: "Uppercase transform", config_fields: [] },
  ]),
  listSinks: vi.fn().mockResolvedValue([
    { name: "json", plugin_type: "sink", description: "JSON file sink", config_fields: [] },
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

  it("shows three tabs", async () => {
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Sources" })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Transforms" })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Sinks" })).toBeInTheDocument();
    });
  });

  it("shows plugin list after fetch", async () => {
    render(<CatalogDrawer isOpen={true} onClose={vi.fn()} />);
    await waitFor(() => {
      expect(screen.getByText("csv")).toBeInTheDocument();
      expect(screen.getByText("CSV file source")).toBeInTheDocument();
    });
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
