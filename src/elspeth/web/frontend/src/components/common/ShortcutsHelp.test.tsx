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
