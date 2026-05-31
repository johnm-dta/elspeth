import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReadOnlyProvider, useReadOnly } from "./ReadOnlyContext";

function Probe(): JSX.Element {
  const readOnly = useReadOnly();
  return <span data-testid="probe">{String(readOnly)}</span>;
}

describe("ReadOnlyContext", () => {
  it("returns false when no provider is mounted (default)", () => {
    render(<Probe />);
    expect(screen.getByTestId("probe")).toHaveTextContent("false");
  });

  it("returns true inside a default-value ReadOnlyProvider", () => {
    render(
      <ReadOnlyProvider>
        <Probe />
      </ReadOnlyProvider>,
    );
    expect(screen.getByTestId("probe")).toHaveTextContent("true");
  });

  it("returns the explicit value when ReadOnlyProvider is given one", () => {
    render(
      <ReadOnlyProvider value={false}>
        <Probe />
      </ReadOnlyProvider>,
    );
    expect(screen.getByTestId("probe")).toHaveTextContent("false");
  });
});
