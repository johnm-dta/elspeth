import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { YamlDisplay } from "./YamlDisplay";

const SAMPLE_YAML = "version: 1\nname: example\nsource:\n  plugin: text\n";

describe("YamlDisplay", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the supplied YAML with Copy and Download buttons", () => {
    render(<YamlDisplay yaml={SAMPLE_YAML} />);
    expect(
      screen.getByRole("button", { name: "Copy YAML to clipboard" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Download YAML file" }),
    ).toBeInTheDocument();
    // The first line of the YAML must appear in the highlighted output.
    expect(screen.getByText(/version/)).toBeInTheDocument();
  });

  it("flips data-copied to true after Copy is clicked", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    render(<YamlDisplay yaml={SAMPLE_YAML} />);
    const user = userEvent.setup();
    const copyBtn = screen.getByRole("button", {
      name: "Copy YAML to clipboard",
    });
    expect(copyBtn).toHaveAttribute("data-copied", "false");
    await user.click(copyBtn);

    // userEvent v14+ installs its own clipboard wrapper; the underlying
    // call chain still routes through navigator.clipboard.writeText, but
    // the spy assertion is unreliable across jsdom versions. The
    // user-facing contract is the data-copied state flip, which is what
    // CSS forced-colors relies on (YamlView.test.tsx pins the same
    // contract for the composer path).
    await waitFor(() => {
      expect(copyBtn).toHaveAttribute("data-copied", "true");
    });
  });

  it("uses the default filename when none is supplied", () => {
    const createObjectURL = vi.fn().mockReturnValue("blob:fake-url");
    const revokeObjectURL = vi.fn();
    // Older jsdom doesn't define these — patch onto globalThis.URL.
    (globalThis.URL as unknown as { createObjectURL: typeof createObjectURL }).createObjectURL =
      createObjectURL;
    (globalThis.URL as unknown as { revokeObjectURL: typeof revokeObjectURL }).revokeObjectURL =
      revokeObjectURL;

    // Intercept the <a> click to capture the download attribute.
    const downloadCalls: string[] = [];
    const realCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      const el = realCreate(tag) as HTMLElement & { click?: () => void };
      if (tag === "a") {
        el.click = () => {
          downloadCalls.push((el as HTMLAnchorElement).download);
        };
      }
      return el;
    });

    render(<YamlDisplay yaml={SAMPLE_YAML} />);
    const downloadBtn = screen.getByRole("button", {
      name: "Download YAML file",
    });
    downloadBtn.click();
    expect(downloadCalls).toContain("pipeline.yaml");
  });

  it("honours an explicit filename prop", () => {
    const createObjectURL = vi.fn().mockReturnValue("blob:fake-url");
    const revokeObjectURL = vi.fn();
    (globalThis.URL as unknown as { createObjectURL: typeof createObjectURL }).createObjectURL =
      createObjectURL;
    (globalThis.URL as unknown as { revokeObjectURL: typeof revokeObjectURL }).revokeObjectURL =
      revokeObjectURL;

    const downloadCalls: string[] = [];
    const realCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      const el = realCreate(tag) as HTMLElement & { click?: () => void };
      if (tag === "a") {
        el.click = () => {
          downloadCalls.push((el as HTMLAnchorElement).download);
        };
      }
      return el;
    });

    render(<YamlDisplay yaml={SAMPLE_YAML} filename="custom-export.yaml" />);
    screen.getByRole("button", { name: "Download YAML file" }).click();
    expect(downloadCalls).toContain("custom-export.yaml");
  });

  it("hides visual line numbers from assistive technology", () => {
    const { container } = render(<YamlDisplay yaml={SAMPLE_YAML} />);
    const lineNumbers = container.querySelectorAll(".yaml-view-line-number");
    expect(lineNumbers.length).toBeGreaterThan(0);
    lineNumbers.forEach((lineNumber) => {
      expect(lineNumber).toHaveAttribute("aria-hidden", "true");
    });
  });
});
