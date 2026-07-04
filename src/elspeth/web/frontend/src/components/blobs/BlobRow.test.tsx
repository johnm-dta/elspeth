import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { previewBlobContent, previewBlobContentSnippet } from "@/api/client";
import type { BlobMetadata } from "@/types/api";
import { BlobRow } from "./BlobRow";

vi.mock("@/api/client", () => ({
  previewBlobContent: vi.fn(),
  previewBlobContentSnippet: vi.fn(),
}));

function makeBlob(overrides: Partial<BlobMetadata> = {}): BlobMetadata {
  return {
    id: "blob-1",
    session_id: "session-1",
    filename: "data.csv",
    mime_type: "text/csv",
    size_bytes: 6000,
    content_hash: null,
    created_at: new Date().toISOString(),
    created_by: "user",
    source_description: null,
    status: "ready",
    creation_modality: "verbatim",
    created_from_message_id: null,
    creating_model_identifier: null,
    creating_model_version: null,
    creating_provider: null,
    creating_composer_skill_hash: null,
    creating_arguments_hash: null,
    ...overrides,
  };
}

describe("BlobRow preview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads inline preview through the bounded preview helper", async () => {
    (previewBlobContentSnippet as ReturnType<typeof vi.fn>).mockResolvedValue({
      text: "preview text",
      truncated: false,
      limit: 5000,
    });

    const user = userEvent.setup();
    render(
      <BlobRow
        blob={makeBlob()}
        sessionId="session-1"
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onUseAsInput={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /preview data\.csv/i }));

    await waitFor(() => {
      expect(previewBlobContentSnippet).toHaveBeenCalledWith(
        "session-1",
        "blob-1",
        5000,
      );
    });
    expect(previewBlobContent).not.toHaveBeenCalled();
    expect(screen.getByText("preview text")).toBeInTheDocument();
  });
});

describe("BlobRow structural self-disclosure (T-3)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows row count and column names for a well-formed CSV, as real accessible text", async () => {
    (previewBlobContentSnippet as ReturnType<typeof vi.fn>).mockResolvedValue({
      text: "name,age\nAlice,30\nBob,40\n",
      truncated: false,
      limit: 5000,
    });

    const user = userEvent.setup();
    render(
      <BlobRow
        blob={makeBlob({ mime_type: "text/csv" })}
        sessionId="session-1"
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onUseAsInput={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /preview data\.csv/i }));

    const summary = await screen.findByTestId("blob-row-structure");
    expect(summary).toHaveTextContent("2 rows");
    expect(summary).toHaveTextContent("columns: name, age");
    // Real text in the accessibility tree, not a title-only affordance.
    expect(summary.getAttribute("title")).toBeNull();
  });

  it("shows row count and keys for a JSON array of records", async () => {
    (previewBlobContentSnippet as ReturnType<typeof vi.fn>).mockResolvedValue({
      text: JSON.stringify([{ id: 1, status: "ok" }, { id: 2, status: "error" }]),
      truncated: false,
      limit: 5000,
    });

    const user = userEvent.setup();
    render(
      <BlobRow
        blob={makeBlob({ filename: "rows.json", mime_type: "application/json" })}
        sessionId="session-1"
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onUseAsInput={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /preview rows\.json/i }));

    const summary = await screen.findByTestId("blob-row-structure");
    expect(summary).toHaveTextContent("2 rows");
    expect(summary).toHaveTextContent("keys: id, status");
  });

  it("HONESTY: ragged CSV rows surface a plain failure, never a guessed count", async () => {
    (previewBlobContentSnippet as ReturnType<typeof vi.fn>).mockResolvedValue({
      text: "name,age\nAlice,30\nBob\n",
      truncated: false,
      limit: 5000,
    });

    const user = userEvent.setup();
    render(
      <BlobRow
        blob={makeBlob({ mime_type: "text/csv" })}
        sessionId="session-1"
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onUseAsInput={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /preview data\.csv/i }));

    const summary = await screen.findByTestId("blob-row-structure");
    expect(summary).toHaveTextContent(/couldn't be read/i);
    expect(summary).not.toHaveTextContent(/0 rows/);
    expect(summary).not.toHaveTextContent(/unknown row count/i);
  });

  it("HONESTY: a truncated preview shows known columns but refuses a row count", async () => {
    (previewBlobContentSnippet as ReturnType<typeof vi.fn>).mockResolvedValue({
      text: "name,age\nAlice,30\nBob,4",
      truncated: true,
      limit: 5000,
    });

    const user = userEvent.setup();
    render(
      <BlobRow
        blob={makeBlob({ mime_type: "text/csv" })}
        sessionId="session-1"
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onUseAsInput={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /preview data\.csv/i }));

    const summary = await screen.findByTestId("blob-row-structure");
    expect(summary).toHaveTextContent(/truncated/i);
    expect(summary).toHaveTextContent("columns: name, age");
    expect(summary).not.toHaveTextContent(/\d+ rows?\b/);
  });

  it("does not render a structure block for content types with no structural handling (e.g. text/plain)", async () => {
    (previewBlobContentSnippet as ReturnType<typeof vi.fn>).mockResolvedValue({
      text: "just some free text",
      truncated: false,
      limit: 5000,
    });

    const user = userEvent.setup();
    render(
      <BlobRow
        blob={makeBlob({ filename: "notes.txt", mime_type: "text/plain" })}
        sessionId="session-1"
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onUseAsInput={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /preview notes\.txt/i }));

    await waitFor(() => {
      expect(screen.getByText("just some free text")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("blob-row-structure")).not.toBeInTheDocument();
  });
});

describe("BlobRow status indicator (WCAG 1.4.1 non-colour cue)", () => {
  it("exposes the status as an accessible image with a visible glyph, not colour alone", () => {
    render(
      <BlobRow
        blob={makeBlob({ status: "ready" })}
        sessionId="session-1"
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onUseAsInput={vi.fn()}
      />,
    );

    const dot = screen.getByRole("img", { name: "Ready" });
    // A shape glyph carries the cue, so the status survives colour-vision
    // deficiency rather than relying on hue.
    expect(dot.textContent).not.toBe("");
  });

  it("renders a distinct glyph per status so ready/pending/error differ by shape", () => {
    const glyphFor = (status: BlobMetadata["status"]): string => {
      const { unmount } = render(
        <BlobRow
          blob={makeBlob({ status })}
          sessionId="session-1"
          onDownload={vi.fn()}
          onDelete={vi.fn()}
          onUseAsInput={vi.fn()}
        />,
      );
      const label = status.charAt(0).toUpperCase() + status.slice(1);
      const glyph = screen.getByRole("img", { name: label }).textContent ?? "";
      unmount();
      return glyph;
    };

    const glyphs = ["ready", "pending", "error"].map((s) =>
      glyphFor(s as BlobMetadata["status"]),
    );
    // All three glyphs must be distinct shapes (a colour-blind user can tell
    // them apart without seeing the hue).
    expect(new Set(glyphs).size).toBe(3);
  });
});
