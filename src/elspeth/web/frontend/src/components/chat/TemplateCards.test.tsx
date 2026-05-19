/**
 * Tests for TemplateCards — the empty-state example-use-case card grid.
 *
 * Covers: card count, accessible labels, SDA dl markup, and the current
 * static-gallery contract. The future user-storable favourites flow may
 * re-enable activation, but generic examples must not send messages today.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TemplateCards } from "./TemplateCards";
import { TEMPLATES } from "./templates_data";

describe("TemplateCards", () => {
  it("renders exactly twelve template cards", () => {
    const { container } = render(<TemplateCards onSelectTemplate={vi.fn()} />);
    const cards = container.querySelectorAll("article.template-card");
    expect(cards).toHaveLength(12);
  });

  it("grid has aria-label 'Example use cases'", () => {
    render(<TemplateCards onSelectTemplate={vi.fn()} />);
    expect(screen.getByRole("group", { name: "Example use cases" })).toBeTruthy();
  });

  it("subtitle mentions 'auditable'", () => {
    render(<TemplateCards onSelectTemplate={vi.fn()} />);
    expect(screen.getByText(/auditable/i)).toBeTruthy();
  });

  it("each card has aria-label of the form 'Domain: description'", () => {
    render(<TemplateCards onSelectTemplate={vi.fn()} />);
    for (const template of TEMPLATES) {
      const expected = `${template.domain}: ${template.description}`;
      expect(
        screen.getByRole("article", { name: expected }),
      ).toBeTruthy();
    }
  });

  it("does not expose the example tiles as buttons", () => {
    render(<TemplateCards onSelectTemplate={vi.fn()} />);
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("each card contains a dl with Sense, Decide, Act dt labels", () => {
    const { container } = render(<TemplateCards onSelectTemplate={vi.fn()} />);
    const dls = container.querySelectorAll("dl.template-card-sda");
    expect(dls).toHaveLength(12);
    for (const dl of dls) {
      const dts = dl.querySelectorAll("dt");
      const labels = Array.from(dts).map((dt) => dt.textContent);
      expect(labels).toContain("Sense");
      expect(labels).toContain("Decide");
      expect(labels).toContain("Act");
    }
  });

  it("keeps the future favourites callback reserved without invoking it", () => {
    const handler = vi.fn();
    render(<TemplateCards onSelectTemplate={handler} />);

    expect(handler).not.toHaveBeenCalled();
  });
});
