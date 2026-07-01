import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Input } from "./Input";

describe("Input", () => {
  it("renders an input element carrying the .input class", () => {
    const { container } = render(<Input placeholder="search" />);
    const input = container.querySelector("input");
    expect(input).not.toBeNull();
    expect(input).toHaveClass("input");
  });

  it("associates the label with the control so getByLabelText resolves it", () => {
    render(<Input label="Username" />);
    const control = screen.getByLabelText("Username");
    expect(control.tagName).toBe("INPUT");
    expect(control).toHaveClass("input");
  });

  it("honours an explicit id for the label association", () => {
    render(<Input label="Token" id="api-token" />);
    const control = screen.getByLabelText("Token");
    expect(control).toHaveAttribute("id", "api-token");
  });

  it("renders the field label and hint chrome", () => {
    const { container } = render(<Input label="Path" hint="absolute path" />);
    expect(container.querySelector(".field-label")).not.toBeNull();
    expect(container.querySelector(".field-hint")).not.toBeNull();
    expect(screen.getByText("absolute path")).toHaveClass("field-hint");
  });

  it("applies .input-mono when mono is set", () => {
    const { container } = render(<Input mono />);
    expect(container.querySelector("input")).toHaveClass("input-mono");
  });

  it("does not apply .input-mono by default", () => {
    const { container } = render(<Input />);
    expect(container.querySelector("input")).not.toHaveClass("input-mono");
  });

  it("merges a caller-supplied className with .input", () => {
    const { container } = render(<Input className="wide" />);
    const input = container.querySelector("input");
    expect(input).toHaveClass("input");
    expect(input).toHaveClass("wide");
  });

  it("forwards value, required, and other DOM props", () => {
    render(
      <Input
        label="Email"
        value="user@example.com"
        required
        type="email"
        autoComplete="email"
        onChange={() => {}}
      />,
    );
    const control = screen.getByLabelText("Email");
    expect(control).toHaveValue("user@example.com");
    expect(control).toBeRequired();
    expect(control).toHaveAttribute("type", "email");
    expect(control).toHaveAttribute("autocomplete", "email");
  });

  it("forwards onChange when the user types", async () => {
    const handleChange = vi.fn();
    render(<Input label="Name" onChange={handleChange} />);
    await userEvent.type(screen.getByLabelText("Name"), "ab");
    expect(handleChange).toHaveBeenCalled();
  });

  it("renders a bare control when neither label nor hint is given", () => {
    const { container } = render(<Input />);
    expect(container.querySelector("div")).toBeNull();
    expect(container.querySelector("input")).not.toBeNull();
  });
});
