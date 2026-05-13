import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ShortcutsHelp } from "./ShortcutsHelp";

describe("ShortcutsHelp", () => {
  it("lists the plugin catalog shortcut", () => {
    render(<ShortcutsHelp onClose={vi.fn()} />);

    const entry = screen.getByText("Open plugin catalog").closest(".shortcuts-list-item");
    expect(entry).not.toBeNull();
    expect(entry).toHaveTextContent("Ctrl/Cmd+Shift+P");
  });
});
