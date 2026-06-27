import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Card, CardHeader } from "./Card";

describe("Card", () => {
  it("renders a div with the card class", () => {
    render(<Card>body</Card>);
    const el = screen.getByText("body");
    expect(el).toHaveClass("card");
    expect(el).not.toHaveClass("card-paper");
  });

  it("adds card-paper when paper is set", () => {
    render(<Card paper>body</Card>);
    expect(screen.getByText("body")).toHaveClass("card", "card-paper");
  });

  it("zeroes the padding when pad is false", () => {
    render(<Card pad={false}>body</Card>);
    expect(screen.getByText("body")).toHaveStyle({ padding: "0px" });
  });

  it("CardHeader renders the title and right-aligned actions", () => {
    render(<CardHeader title="Settings" actions={<button>edit</button>} eyebrow="Section" />);
    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(screen.getByText("Section")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "edit" })).toBeInTheDocument();
  });
});
