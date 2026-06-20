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
