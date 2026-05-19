import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { CatalogButton } from "./CatalogButton";
import { OPEN_CATALOG_EVENT } from "@/lib/composer-events";

describe("CatalogButton", () => {
  it("renders a catalog-reference button", () => {
    render(<CatalogButton />);

    expect(
      screen.getByRole("button", { name: /catalog \(reference\)/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("Plugin catalog")).toBeInTheDocument();
    expect(screen.getByText("Reference")).toBeInTheDocument();
  });

  it("dispatches OPEN_CATALOG_EVENT on click", () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_CATALOG_EVENT, handler);

    render(<CatalogButton />);
    fireEvent.click(
      screen.getByRole("button", { name: /catalog \(reference\)/i }),
    );

    expect(handler).toHaveBeenCalled();
    window.removeEventListener(OPEN_CATALOG_EVENT, handler);
  });
});
