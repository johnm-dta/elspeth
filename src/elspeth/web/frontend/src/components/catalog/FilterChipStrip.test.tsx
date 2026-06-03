import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FilterChipStrip, type CatalogFilters } from "./FilterChipStrip";

const ALL_OFF: CatalogFilters = {
  capabilityTags: new Set(),
  auditCharacteristics: new Set(),
};

describe("FilterChipStrip", () => {
  it("renders one chip per capability tag", () => {
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv", "file", "http"]}
        availableAuditCharacteristics={[]}
        filters={ALL_OFF}
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /csv/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^http/i })).toBeInTheDocument();
  });

  it("emits an updated filter set when a chip is toggled", async () => {
    const onChange = vi.fn();
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableAuditCharacteristics={[]}
        filters={ALL_OFF}
        onChange={onChange}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /csv/i }));
    expect(onChange).toHaveBeenCalled();
    const updated: CatalogFilters = onChange.mock.calls[0][0];
    expect(updated.capabilityTags.has("csv")).toBe(true);
  });

  it("toggling an active chip removes it", async () => {
    const onChange = vi.fn();
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableAuditCharacteristics={[]}
        filters={{ ...ALL_OFF, capabilityTags: new Set(["csv"]) }}
        onChange={onChange}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /csv/i }));
    const updated: CatalogFilters = onChange.mock.calls[0][0];
    expect(updated.capabilityTags.has("csv")).toBe(false);
  });

  it("renders 'Clear filters' when any filter is active", () => {
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableAuditCharacteristics={[]}
        filters={{ ...ALL_OFF, capabilityTags: new Set(["csv"]) }}
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /clear filters/i })).toBeInTheDocument();
  });

  it("does not render 'Clear filters' when no filters are active", () => {
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableAuditCharacteristics={[]}
        filters={ALL_OFF}
        onChange={() => {}}
      />,
    );
    expect(screen.queryByRole("button", { name: /clear filters/i })).not.toBeInTheDocument();
  });
});
