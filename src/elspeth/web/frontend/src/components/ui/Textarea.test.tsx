import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Textarea } from "./Textarea";

describe("Textarea", () => {
  it("renders a textarea element carrying the .textarea class", () => {
    const { container } = render(<Textarea placeholder="notes" />);
    const control = container.querySelector("textarea");
    expect(control).not.toBeNull();
    expect(control).toHaveClass("textarea");
  });

  it("associates the label with the control so getByLabelText resolves it", () => {
    render(<Textarea label="Prompt" />);
    const control = screen.getByLabelText("Prompt");
    expect(control.tagName).toBe("TEXTAREA");
    expect(control).toHaveClass("textarea");
  });

  it("honours an explicit id for the label association", () => {
    render(<Textarea label="Body" id="amend-body" />);
    expect(screen.getByLabelText("Body")).toHaveAttribute("id", "amend-body");
  });

  it("renders the field label and hint chrome", () => {
    const { container } = render(
      <Textarea label="Message" hint="markdown allowed" />,
    );
    expect(container.querySelector(".field-label")).not.toBeNull();
    expect(container.querySelector(".field-hint")).not.toBeNull();
    expect(screen.getByText("markdown allowed")).toHaveClass("field-hint");
  });

  it("defaults rows to 3 and forwards an override", () => {
    const { rerender, container } = render(<Textarea />);
    expect(container.querySelector("textarea")).toHaveAttribute("rows", "3");
    rerender(<Textarea rows={6} />);
    expect(container.querySelector("textarea")).toHaveAttribute("rows", "6");
  });

  it("merges a caller-supplied className with .textarea", () => {
    const { container } = render(<Textarea className="tall" />);
    const control = container.querySelector("textarea");
    expect(control).toHaveClass("textarea");
    expect(control).toHaveClass("tall");
  });

  it("forwards value, required, and onChange", async () => {
    const handleChange = vi.fn();
    render(<Textarea label="Comment" required onChange={handleChange} />);
    const control = screen.getByLabelText("Comment");
    expect(control).toBeRequired();
    await userEvent.type(control, "hi");
    expect(handleChange).toHaveBeenCalled();
  });

  it("renders a bare control when neither label nor hint is given", () => {
    const { container } = render(<Textarea />);
    expect(container.querySelector("div")).toBeNull();
    expect(container.querySelector("textarea")).not.toBeNull();
  });
});
