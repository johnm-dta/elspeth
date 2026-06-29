import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useBlobStore } from "@/stores/blobStore";
import { useSessionStore } from "@/stores/sessionStore";
import { BlobManager } from "./BlobManager";
import type { BlobMetadata } from "@/types/api";

function makeBlob(overrides: Partial<BlobMetadata> = {}): BlobMetadata {
  return {
    id: "blob-1",
    session_id: "session-1",
    filename: "data.csv",
    mime_type: "text/csv",
    size_bytes: 1024,
    content_hash: null,
    created_at: new Date().toISOString(),
    created_by: "user",
    source_description: null,
    status: "ready",
    // Inline-blob provenance defaults (Phase 5a Task 2.5).
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

/** Set up store state with a no-op loadBlobs so the useEffect doesn't clobber isLoading. */
function setBlobState(blobs: BlobMetadata[]) {
  useBlobStore.setState({
    blobs,
    isLoading: false,
    error: null,
    loadBlobs: vi.fn().mockResolvedValue(undefined),
  });
}

describe("BlobManager categorized folders", () => {
  beforeEach(() => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    vi.clearAllMocks();
  });

  it("groups blobs into Source, Output, and Other sections", () => {
    setBlobState([
      makeBlob({ id: "b1", filename: "input.csv", created_by: "user" }),
      makeBlob({ id: "b2", filename: "results.json", created_by: "pipeline" }),
      makeBlob({ id: "b3", filename: "prompt.txt", created_by: "assistant" }),
    ]);

    render(<BlobManager onUseAsInput={vi.fn()} />);

    expect(screen.getByText("Source files")).toBeInTheDocument();
    expect(screen.getByText("Output files")).toBeInTheDocument();
    expect(screen.getByText("Other files")).toBeInTheDocument();
  });

  it("puts user-uploaded files in Source section", () => {
    setBlobState([makeBlob({ id: "b1", filename: "data.csv", created_by: "user" })]);

    render(<BlobManager onUseAsInput={vi.fn()} />);

    expect(screen.getByText("Source files")).toBeInTheDocument();
    expect(screen.getByText("data.csv")).toBeInTheDocument();
  });

  it("puts pipeline-created files in Output section", () => {
    setBlobState([makeBlob({ id: "b2", filename: "results.json", created_by: "pipeline" })]);

    render(<BlobManager onUseAsInput={vi.fn()} />);

    expect(screen.getByText("Output files")).toBeInTheDocument();
    expect(screen.getByText("results.json")).toBeInTheDocument();
  });

  it("shows empty state for empty file list", () => {
    setBlobState([]);

    render(<BlobManager onUseAsInput={vi.fn()} />);

    expect(screen.getByText(/No files yet/)).toBeInTheDocument();
  });

  it("hides empty categories", () => {
    setBlobState([makeBlob({ id: "b1", filename: "data.csv", created_by: "user" })]);

    render(<BlobManager onUseAsInput={vi.fn()} />);

    expect(screen.getByText("Source files")).toBeInTheDocument();
    expect(screen.queryByText("Output files")).not.toBeInTheDocument();
    expect(screen.queryByText("Other files")).not.toBeInTheDocument();
  });
});

describe("BlobManager delete confirmation (WCAG 3.3.4)", () => {
  beforeEach(() => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    vi.clearAllMocks();
  });

  it("requires confirmation before deleting a file, then deletes on confirm", async () => {
    const deleteBlob = vi.fn().mockResolvedValue(undefined);
    setBlobState([makeBlob({ id: "b1", filename: "data.csv", created_by: "user" })]);
    useBlobStore.setState({ deleteBlob });

    const user = userEvent.setup();
    render(<BlobManager onUseAsInput={vi.fn()} />);

    // Requesting deletion opens a danger dialog naming the file — and does NOT
    // delete immediately (irreversible data-loss guard).
    await user.click(screen.getByRole("button", { name: "Delete data.csv" }));
    const dialog = screen.getByRole("alertdialog");
    expect(
      within(dialog).getByText(/Delete "data\.csv"\? This cannot be undone\./),
    ).toBeInTheDocument();
    expect(deleteBlob).not.toHaveBeenCalled();

    // Confirming fires the delete with the active session + blob id.
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));
    expect(deleteBlob).toHaveBeenCalledWith("session-1", "b1");
  });

  it("leaves the file intact when the dialog is cancelled", async () => {
    const deleteBlob = vi.fn().mockResolvedValue(undefined);
    setBlobState([makeBlob({ id: "b1", filename: "data.csv", created_by: "user" })]);
    useBlobStore.setState({ deleteBlob });

    const user = userEvent.setup();
    render(<BlobManager onUseAsInput={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Delete data.csv" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(deleteBlob).not.toHaveBeenCalled();
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });
});
