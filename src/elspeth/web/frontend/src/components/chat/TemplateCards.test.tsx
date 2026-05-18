/**
 * Tests for TemplateCards — the empty-state example-use-case card grid.
 *
 * Covers: card count, aria-labels, SDA dl markup, onSelectTemplate callback
 * shape (seed_prompt + recommended_starting_point as second arg).
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TemplateCards } from "./TemplateCards";
import { TEMPLATES } from "./templates_data";

describe("TemplateCards", () => {
  it("renders exactly six template cards", () => {
    render(<TemplateCards onSelectTemplate={vi.fn()} />);
    // Each card is a <button>; count via aria-label pattern.
    const cards = screen.getAllByRole("button");
    expect(cards).toHaveLength(6);
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
        screen.getByRole("button", { name: expected }),
      ).toBeTruthy();
    }
  });

  it("each card contains a dl with Sense, Decide, Act dt labels", () => {
    const { container } = render(<TemplateCards onSelectTemplate={vi.fn()} />);
    const dls = container.querySelectorAll("dl.template-card-sda");
    expect(dls).toHaveLength(6);
    for (const dl of dls) {
      const dts = dl.querySelectorAll("dt");
      const labels = Array.from(dts).map((dt) => dt.textContent);
      expect(labels).toContain("Sense");
      expect(labels).toContain("Decide");
      expect(labels).toContain("Act");
    }
  });

  it("clicking a card calls onSelectTemplate with seed_prompt and recommended_starting_point", async () => {
    const user = userEvent.setup();
    const handler = vi.fn();
    render(<TemplateCards onSelectTemplate={handler} />);

    const firstTemplate = TEMPLATES[0];
    const card = screen.getByRole("button", {
      name: `${firstTemplate.domain}: ${firstTemplate.description}`,
    });
    await user.click(card);

    expect(handler).toHaveBeenCalledOnce();
    expect(handler).toHaveBeenCalledWith(
      firstTemplate.seed_prompt,
      firstTemplate.recommended_starting_point,
    );
  });

  it("clicking every card passes the correct seed_prompt", async () => {
    const user = userEvent.setup();
    const handler = vi.fn();
    render(<TemplateCards onSelectTemplate={handler} />);

    for (const template of TEMPLATES) {
      handler.mockClear();
      const card = screen.getByRole("button", {
        name: `${template.domain}: ${template.description}`,
      });
      await user.click(card);
      expect(handler).toHaveBeenCalledWith(
        template.seed_prompt,
        template.recommended_starting_point,
      );
    }
  });
});
