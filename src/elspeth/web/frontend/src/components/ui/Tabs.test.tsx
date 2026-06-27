import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Tabs, type TabItem } from "./Tabs";

const TABS: TabItem[] = [
  { id: "overview", label: "Overview" },
  { id: "rows", label: "Rows", count: 12 },
];

describe("Tabs", () => {
  it("renders a tab button per item inside a tablist", () => {
    render(<Tabs tabs={TABS} value="overview" />);
    expect(screen.getByRole("tablist")).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(2);
  });

  it("marks the active tab as selected", () => {
    render(<Tabs tabs={TABS} value="overview" />);
    const active = screen.getByRole("tab", { name: "Overview" });
    expect(active).toHaveAttribute("aria-selected", "true");
    expect(active).toHaveClass("tab-strip-tab", "tab-strip-tab-active");
    const inactive = screen.getByRole("tab", { name: /Rows/ });
    expect(inactive).toHaveAttribute("aria-selected", "false");
    expect(inactive).not.toHaveClass("tab-strip-tab-active");
  });

  it("calls onChange with the clicked tab id", async () => {
    const onChange = vi.fn();
    render(<Tabs tabs={TABS} value="overview" onChange={onChange} />);
    await userEvent.click(screen.getByRole("tab", { name: /Rows/ }));
    expect(onChange).toHaveBeenCalledWith("rows");
  });

  it("renders a count pill when count is provided", () => {
    render(<Tabs tabs={TABS} value="overview" />);
    expect(screen.getByText("12")).toBeInTheDocument();
  });
});
