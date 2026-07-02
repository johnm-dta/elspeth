/**
 * Tests for TemplateCards — the empty-state example-use-case card grid.
 *
 * Covers: card count, accessible labels, SDA dl markup, and the per-card
 * "Use this example" action (elspeth-b948756c5a): activating a card's
 * button hands that template's seed prompt and recommended starting point
 * to the parent exactly once.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

  it("exposes one 'Use this example' button per card, disambiguated by domain", () => {
    render(<TemplateCards onSelectTemplate={vi.fn()} />);
    const buttons = screen.getAllByRole("button", { name: /^use this example: /i });
    expect(buttons).toHaveLength(TEMPLATES.length);
    for (const template of TEMPLATES) {
      expect(
        screen.getByRole("button", {
          name: `Use this example: ${template.domain}`,
        }),
      ).toBeTruthy();
    }
  });

  it("clicking a card's action invokes onSelectTemplate with that template's seed prompt and starting point", async () => {
    const user = userEvent.setup();
    const handler = vi.fn();
    render(<TemplateCards onSelectTemplate={handler} />);

    const tender = TEMPLATES.find((t) => t.id === "tender-evaluation");
    if (tender === undefined) throw new Error("tender-evaluation template missing");

    await user.click(
      screen.getByRole("button", { name: `Use this example: ${tender.domain}` }),
    );

    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler).toHaveBeenCalledWith(
      tender.seed_prompt,
      tender.recommended_starting_point,
    );
  });

  it("does not invoke onSelectTemplate on render", () => {
    const handler = vi.fn();
    render(<TemplateCards onSelectTemplate={handler} />);
    expect(handler).not.toHaveBeenCalled();
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
});
