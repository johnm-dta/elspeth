import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
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
