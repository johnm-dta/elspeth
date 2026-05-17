import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ShortcutsHelp } from "./ShortcutsHelp";

describe("ShortcutsHelp", () => {
  it("lists the plugin catalog shortcut", () => {
    render(<ShortcutsHelp onClose={vi.fn()} />);

    const entry = screen
      .getByText("Open plugin catalog")
      .closest(".shortcuts-list-item");
    expect(entry).not.toBeNull();
    expect(entry).toHaveTextContent("Ctrl/Cmd+Shift+P");
  });

  it("lists graph and YAML modal shortcuts", () => {
    render(<ShortcutsHelp onClose={vi.fn()} />);

    const graphEntry = screen
      .getByText("Open graph view")
      .closest(".shortcuts-list-item");
    const yamlEntry = screen
      .getByText("Export YAML")
      .closest(".shortcuts-list-item");

    expect(graphEntry).not.toBeNull();
    expect(graphEntry).toHaveTextContent("Ctrl/Cmd+Shift+G");
    expect(yamlEntry).not.toBeNull();
    expect(yamlEntry).toHaveTextContent("Ctrl/Cmd+Shift+Y");
  });

  it("does not list retired inspector tab shortcuts", () => {
    render(<ShortcutsHelp onClose={vi.fn()} />);

    expect(screen.queryByText(/Alt\+1-2/i)).toBeNull();
    expect(screen.queryByText(/Switch inspector tab/i)).toBeNull();
  });
});

describe("ShortcutsHelp — Phase 7B regroup", () => {
  it("renders an 'Actions' subheading", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    expect(screen.getByRole("heading", { name: /actions/i })).toBeInTheDocument();
  });

  it("renders a 'Reference' subheading", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    expect(screen.getByRole("heading", { name: /reference/i })).toBeInTheDocument();
  });

  it("places 'Open plugin catalog' under the Reference subheading", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    const referenceHeading = screen.getByRole("heading", { name: /reference/i });
    // The catalog entry sits in the <dl> that follows the Reference heading.
    const referenceList = referenceHeading.nextElementSibling;
    expect(referenceList).not.toBeNull();
    expect(referenceList?.textContent).toMatch(/open plugin catalog/i);
  });

  it("places 'Validate pipeline' and 'Execute pipeline' under Actions", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    const actionsHeading = screen.getByRole("heading", { name: /actions/i });
    const actionsList = actionsHeading.nextElementSibling;
    expect(actionsList?.textContent).toMatch(/validate pipeline/i);
    expect(actionsList?.textContent).toMatch(/execute pipeline/i);
  });

  it("lists 'Open graph view' and 'Export YAML' under Actions", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    const actionsHeading = screen.getByRole("heading", { name: /actions/i });
    const actionsList = actionsHeading.nextElementSibling;
    expect(actionsList?.textContent).toMatch(/open graph view/i);
    expect(actionsList?.textContent).toMatch(/export yaml/i);
  });

  it("lists Alt+1-3 catalog tab shortcut under Actions", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    const actionsHeading = screen.getByRole("heading", { name: /actions/i });
    const actionsList = actionsHeading.nextElementSibling;
    expect(actionsList?.textContent).toMatch(/alt\+1-3/i);
    expect(actionsList?.textContent).toMatch(/switch.*catalog.*tab|sources.*transforms.*sinks/i);
  });
});
