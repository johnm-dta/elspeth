import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RunOutputsPanel } from "./RunOutputsPanel";
import {
  downloadRunOutputContent,
  fetchRunOutputPreview,
  fetchRunOutputs,
} from "@/api/client";
import type {
  ApiError,
  RunOutputArtifact,
  RunOutputArtifactPreview,
  RunOutputsResponse,
} from "@/types/index";

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>("@/api/client");
  return {
    ...actual,
    fetchRunOutputs: vi.fn(),
    fetchRunOutputPreview: vi.fn(),
    downloadRunOutputContent: vi.fn(),
  };
});

const RUN_ID = "run-abc";

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function fileArtifact(overrides: Partial<RunOutputArtifact> = {}): RunOutputArtifact {
  return {
    artifact_id: "art-1",
    sink_node_id: "results",
    artifact_type: "file",
    path_or_uri: "file:///data/outputs/results.csv",
    content_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    size_bytes: 1024,
    created_at: "2026-05-10T00:00:00Z",
    exists_now: true,
    downloadable: true,
    storage_kind: "sink_file",
    ...overrides,
  };
}

function manifest(artifacts: RunOutputArtifact[]): RunOutputsResponse {
  return {
    run_id: RUN_ID,
    landscape_run_id: RUN_ID,
    artifacts,
  };
}

function csvPreview(overrides: Partial<RunOutputArtifactPreview> = {}): RunOutputArtifactPreview {
  return {
    artifact_id: "art-1",
    content_type: "csv",
    preview_text: "col1,col2\n1,2\n3,4\n",
    truncated: false,
    total_size_bytes: 18,
    row_count_preview: 3,
    ...overrides,
  };
}

describe("RunOutputsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads the manifest on mount and renders artifact rows", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact()]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() => expect(screen.getByText("results.csv")).toBeInTheDocument());
    expect(screen.getByText("file")).toBeInTheDocument();
    expect(screen.getByText("1.0 KiB")).toBeInTheDocument();
    expect(screen.getByText("abcdef012345…")).toBeInTheDocument();
    expect(screen.getByText("Download")).toBeInTheDocument();
    // Run timestamp: rendered as a compact local YYYY-MM-DD HH:MM:SS
    // alongside the file size. The exact text varies by host timezone
    // (the fixture is "2026-05-10T00:00:00Z"), so assert the *shape*
    // and verify the unmodified ISO survives in the tooltip.
    expect(screen.getByText(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/)).toBeInTheDocument();
    expect(
      screen.getByTitle("2026-05-10T00:00:00Z"),
    ).toBeInTheDocument();
  });

  it("shows an empty state when the run produced no outputs", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(manifest([]));

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() =>
      expect(screen.getByText(/produced no outputs/i)).toBeInTheDocument(),
    );
  });

  it("renders the 'no longer available on disk' state for purged artifacts", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact({ exists_now: false, downloadable: false })]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() =>
      expect(screen.getByText(/no longer available on disk/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText("Download")).not.toBeInTheDocument();
    expect(screen.queryByText("Preview")).not.toBeInTheDocument();
  });

  it("does not leak raw internal storage paths in unavailable-row tooltips", async () => {
    // Regression for elspeth-52af16f9ae: the frontend previously guessed
    // "internal storage" from a path-shape regex that no repo code
    // actually produces. The backend now classifies storage_kind
    // structurally; a purged blob-store artifact (exists_now=false)
    // must still mask its raw path in the tooltip.
    const rawPath = "/var/lib/elspeth/data/blobs/session-1/blob-abc_report.json";
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([
        fileArtifact({
          sink_node_id: "final_report",
          path_or_uri: rawPath,
          exists_now: false,
          downloadable: false,
          storage_kind: "blob",
        }),
      ]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    const unavailable = await screen.findByText(/no longer available on disk/i);
    expect(unavailable).toHaveAttribute(
      "title",
      "Recorded artifact for final_report",
    );
    expect(unavailable.getAttribute("title")).not.toContain(rawPath);
  });

  it("shows the 'outside allowed sink directories' tooltip when not downloadable", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact({ exists_now: true, downloadable: false })]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() =>
      expect(
        screen.getByText(/outside allowed sink directories/i),
      ).toBeInTheDocument(),
    );
    expect(screen.queryByText("Download")).not.toBeInTheDocument();
  });

  it("renders non-file artifacts as metadata-only with no action buttons", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([
        fileArtifact({
          artifact_id: "art-db",
          artifact_type: "database",
          path_or_uri: "postgres://prod/results.outputs",
          exists_now: false,
          downloadable: false,
        }),
      ]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() =>
      expect(
        screen.getByText("postgres://prod/results.outputs"),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText("database")).toBeInTheDocument();
    expect(screen.queryByText("Download")).not.toBeInTheDocument();
    expect(screen.queryByText("Preview")).not.toBeInTheDocument();
    // Non-file artifacts also do NOT show the "no longer available" message
    // (that's a file-only concept).
    expect(screen.queryByText(/no longer available on disk/i)).not.toBeInTheDocument();
  });

  it("lazy-fetches preview when Preview is clicked and shows truncation footer", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact()]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      csvPreview({ truncated: true, total_size_bytes: 5_000_000 }),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    const previewBtn = await screen.findByRole("button", { name: /^Preview$/ });
    expect(fetchRunOutputPreview).not.toHaveBeenCalled();
    fireEvent.click(previewBtn);

    await waitFor(() =>
      expect(fetchRunOutputPreview).toHaveBeenCalledWith(RUN_ID, "art-1"),
    );
    // CSV renders as a table — assert on cell content.
    await waitFor(() => expect(screen.getByText("col1")).toBeInTheDocument());
    expect(screen.getByText("col2")).toBeInTheDocument();
    // Truncation footer is present when truncated=true.
    expect(screen.getByText(/preview truncated/i)).toBeInTheDocument();
    // The "download for full file" element MUST be a button, not an
    // anchor. A plain `<a href download>` would 401 because Bearer
    // auth doesn't ride on top-level navigation. Same regression
    // guard as the row-level Download button — see the dedicated
    // download-via-button test below.
    const fullFileLink = screen.getByText(/download for full file/i);
    expect(fullFileLink.tagName).toBe("BUTTON");
    expect(fullFileLink.hasAttribute("href")).toBe(false);
  });

  it("does not refetch preview when the row is collapsed and re-expanded", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact()]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(csvPreview());

    render(<RunOutputsPanel runId={RUN_ID} />);

    const previewBtn = await screen.findByRole("button", { name: /^Preview$/ });
    fireEvent.click(previewBtn);
    await waitFor(() => expect(fetchRunOutputPreview).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /Hide preview/ }));
    fireEvent.click(screen.getByRole("button", { name: /^Preview$/ }));

    // Cached — no second fetch.
    expect(fetchRunOutputPreview).toHaveBeenCalledTimes(1);
  });

  it("renders the binary 'no inline preview' message for binary content", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact({ path_or_uri: "file:///data/outputs/blob.bin" })]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      csvPreview({
        content_type: "binary",
        preview_text: "",
        row_count_preview: null,
      }),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);
    fireEvent.click(await screen.findByRole("button", { name: /^Preview$/ }));

    await waitFor(() =>
      expect(screen.getByText(/no inline preview available/i)).toBeInTheDocument(),
    );
  });

  it("renders jsonl preview with each line in a single cell (no comma fragmentation)", async () => {
    // Regression for Codex P2 finding: TabularPreview previously called
    // `line.split(",")` for both csv and jsonl. For jsonl, that fragments
    // each JSON object across cells (e.g. `{"a":1,"b":2}` splits into
    // `{"a":1` / `"b":2}`). Each line must remain intact in a single cell.
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact({ path_or_uri: "file:///data/outputs/results.jsonl" })]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      csvPreview({
        content_type: "jsonl",
        preview_text: '{"a":1,"b":2}\n{"a":3,"b":4}\n',
      }),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);
    fireEvent.click(await screen.findByRole("button", { name: /^Preview$/ }));

    // The full JSON object is present in the rendered output rather than
    // any of its fragments. If the comma-split bug had survived,
    // {"a":1 and "b":2} would appear as separate cell text and
    // {"a":1,"b":2} (the whole object) would not be queryable as a unit.
    await waitFor(() =>
      expect(screen.getByText('{"a":1,"b":2}')).toBeInTheDocument(),
    );
    expect(screen.getByText('{"a":3,"b":4}')).toBeInTheDocument();
  });

  it("renders TSV preview (content_type=csv, tab-separated) with tab delimiter sniffing", async () => {
    // Regression for Codex P2 finding: backend tags both `.csv` and
    // `.tsv` files as content_type "csv" (web/execution/preview.py
    // _CSV_EXTENSIONS), so a tab-separated blob arrives at TabularPreview
    // with content_type="csv". A literal `split(",")` collapses the
    // entire row into one cell. The renderer must sniff the first line
    // for tab vs comma and use whichever is more frequent.
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact({ path_or_uri: "file:///data/outputs/results.tsv" })]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      csvPreview({
        content_type: "csv",
        preview_text: "id\tname\tvalue\n1\tAlice\t9.99\n2\tBob\t19.95\n",
      }),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);
    fireEvent.click(await screen.findByRole("button", { name: /^Preview$/ }));

    // Each header and data cell is its own queryable element. Before the
    // fix, "id\tname\tvalue" rendered as a single cell.
    await waitFor(() => expect(screen.getByText("id")).toBeInTheDocument());
    expect(screen.getByText("name")).toBeInTheDocument();
    expect(screen.getByText("value")).toBeInTheDocument();
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("19.95")).toBeInTheDocument();
  });

  it("pretty-prints JSON previews and offers a table view for record arrays", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact({ path_or_uri: "file:///data/outputs/results.json" })]),
    );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      csvPreview({
        content_type: "json",
        preview_text: '[{"id":1,"status":"ok"},{"id":2,"status":"error"}]',
        row_count_preview: null,
      }),
    );

    const { container } = render(<RunOutputsPanel runId={RUN_ID} />);
    fireEvent.click(await screen.findByRole("button", { name: /^Preview$/ }));

    await waitFor(() =>
      expect(container.querySelector('[data-codeblock-format="json"]')?.textContent).toContain(
        '"id": 1,',
      ),
    );

    fireEvent.click(screen.getByRole("button", { name: "Table view" }));

    expect(screen.getByText("id")).toBeInTheDocument();
    expect(screen.getByText("status")).toBeInTheDocument();
    expect(screen.getByText("error")).toBeInTheDocument();
  });

  it("derives a friendly artifact label instead of showing opaque blob-store paths", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([
        fileArtifact({
          sink_node_id: "final_report",
          path_or_uri:
            "/home/john/.local/share/elspeth/data/blobs/session-1/blob-abc_report.json",
          storage_kind: "blob",
        }),
      ]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() =>
      expect(screen.getByText("final_report")).toBeInTheDocument(),
    );
    expect(
      screen.queryByText(
        "/home/john/.local/share/elspeth/data/blobs/session-1/blob-abc_report.json",
      ),
    ).not.toBeInTheDocument();
  });

  it("derives a friendly artifact label for payload-store paths too", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([
        fileArtifact({
          sink_node_id: "final_report",
          path_or_uri:
            "/home/john/.local/share/elspeth/data/payloads/ab/abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
          storage_kind: "payload",
        }),
      ]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() =>
      expect(screen.getByText("final_report")).toBeInTheDocument(),
    );
    expect(
      screen.queryByText(
        "/home/john/.local/share/elspeth/data/payloads/ab/abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
      ),
    ).not.toBeInTheDocument();
  });

  it("preserves legitimate hash-like filenames when storage_kind is not internal storage", async () => {
    // Regression for elspeth-52af16f9ae: this filename would have matched
    // the old sha256-basename regex heuristic, but the backend classifies
    // it as "unknown" (it's not under the blob or payload store layout),
    // so the raw name must render as-is.
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([
        fileArtifact({
          sink_node_id: "final_report",
          path_or_uri:
            "file:///data/outputs/sha256-abcdef0123456789abcdef0123456789.json",
          storage_kind: "unknown",
        }),
      ]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() =>
      expect(
        screen.getByText("sha256-abcdef0123456789abcdef0123456789.json"),
      ).toBeInTheDocument(),
    );
    expect(screen.queryByText("final_report")).not.toBeInTheDocument();
  });

  it("preserves user output paths that include a blob-storage-looking directory name", async () => {
    // Regression for elspeth-52af16f9ae: a user-configured sink path that
    // happens to contain a "blob-storage" segment is classified
    // storage_kind="sink_file" by the backend (it's not under the real
    // blob-store layout), so the raw path must render as-is — the old
    // regex heuristic would have masked this incorrectly.
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([
        fileArtifact({
          sink_node_id: "final_report",
          path_or_uri: "file:///data/outputs/blob-storage/report.json",
          storage_kind: "sink_file",
        }),
      ]),
    );

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() => expect(screen.getByText("report.json")).toBeInTheDocument());
    expect(screen.getByTitle("file:///data/outputs/blob-storage/report.json")).toBeInTheDocument();
    expect(screen.queryByText("final_report")).not.toBeInTheDocument();
  });

  it("transitions to 'no longer available on disk' on artifact_purged_or_moved race", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact()]),
    );
    // parseResponse throws an ApiError-shaped object, NOT an Error
    // instance. The race detection matches on the structured
    // error_type field, not on a substring of detail.
    const apiError: ApiError = {
      status: 410,
      detail: "Resource gone",
      error_type: "artifact_purged_or_moved",
    };
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockRejectedValue(apiError);

    render(<RunOutputsPanel runId={RUN_ID} />);
    fireEvent.click(await screen.findByRole("button", { name: /^Preview$/ }));

    await waitFor(() =>
      expect(
        screen.getByText(
          /file is no longer available on disk \(purged or moved/i,
        ),
      ).toBeInTheDocument(),
    );
  });

  it("renders a generic error using ApiError.detail for non-purged preview failures", async () => {
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact()]),
    );
    const apiError: ApiError = { status: 500, detail: "server boom" };
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockRejectedValue(apiError);

    render(<RunOutputsPanel runId={RUN_ID} />);
    fireEvent.click(await screen.findByRole("button", { name: /^Preview$/ }));

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByText(/server boom/)).toBeInTheDocument();
  });

  it("surfaces a manifest-load error and offers Refresh", async () => {
    const apiError: ApiError = { status: 503, detail: "Manifest unavailable" };
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockRejectedValue(apiError);

    render(<RunOutputsPanel runId={RUN_ID} />);

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByText(/Manifest unavailable/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Refresh/ })).toBeInTheDocument();
  });

  it("clears prior manifest and preview state as soon as runId changes", async () => {
    const runBLoad = deferred<RunOutputsResponse>();
    (fetchRunOutputs as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(manifest([fileArtifact()]))
      .mockReturnValueOnce(runBLoad.promise);
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockResolvedValue(
      csvPreview({ preview_text: "old-run-preview\n" }),
    );

    const { rerender } = render(<RunOutputsPanel runId="run-a" />);
    fireEvent.click(await screen.findByRole("button", { name: /^Preview$/ }));
    await waitFor(() =>
      expect(screen.getByText("old-run-preview")).toBeInTheDocument(),
    );

    rerender(<RunOutputsPanel runId="run-b" />);

    expect(screen.queryByText("results.csv")).not.toBeInTheDocument();
    expect(screen.queryByText("old-run-preview")).not.toBeInTheDocument();
    expect(screen.getByText(/loading outputs/i)).toBeInTheDocument();

    const apiError: ApiError = { status: 503, detail: "run B manifest failed" };
    runBLoad.reject(apiError);

    await waitFor(() =>
      expect(screen.getByText(/run B manifest failed/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText("results.csv")).not.toBeInTheDocument();
    expect(screen.queryByText("old-run-preview")).not.toBeInTheDocument();
  });

  it("ignores a late manifest response for a previous runId", async () => {
    const runALoad = deferred<RunOutputsResponse>();
    (fetchRunOutputs as ReturnType<typeof vi.fn>)
      .mockReturnValueOnce(runALoad.promise)
      .mockResolvedValueOnce(
        manifest([
          fileArtifact({
            artifact_id: "run-b-artifact",
            path_or_uri: "file:///data/outputs/run-b.csv",
          }),
        ]),
      );

    const { rerender } = render(<RunOutputsPanel runId="run-a" />);
    rerender(<RunOutputsPanel runId="run-b" />);

    await waitFor(() => expect(screen.getByText("run-b.csv")).toBeInTheDocument());

    runALoad.resolve(
      manifest([
        fileArtifact({
          artifact_id: "run-a-artifact",
          path_or_uri: "file:///data/outputs/run-a.csv",
        }),
      ]),
    );

    await waitFor(() =>
      expect(fetchRunOutputs).toHaveBeenCalledWith("run-a"),
    );
    expect(screen.queryByText("run-a.csv")).not.toBeInTheDocument();
    expect(screen.getByText("run-b.csv")).toBeInTheDocument();
  });

  it("ignores a late preview response for a previous runId", async () => {
    const oldPreview = deferred<RunOutputArtifactPreview>();
    (fetchRunOutputs as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(manifest([fileArtifact()]))
      .mockResolvedValueOnce(
        manifest([
          fileArtifact({
            path_or_uri: "file:///data/outputs/run-b-results.csv",
          }),
        ]),
      );
    (fetchRunOutputPreview as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      oldPreview.promise,
    );

    const { rerender } = render(<RunOutputsPanel runId="run-a" />);
    fireEvent.click(await screen.findByRole("button", { name: /^Preview$/ }));
    await waitFor(() =>
      expect(fetchRunOutputPreview).toHaveBeenCalledWith("run-a", "art-1"),
    );

    rerender(<RunOutputsPanel runId="run-b" />);
    await waitFor(() =>
      expect(screen.getByText("run-b-results.csv")).toBeInTheDocument(),
    );

    oldPreview.resolve(
      csvPreview({ preview_text: "stale-run-a-preview\n", artifact_id: "art-1" }),
    );

    await waitFor(() =>
      expect(fetchRunOutputs).toHaveBeenCalledWith("run-b"),
    );
    expect(screen.queryByText("stale-run-a-preview")).not.toBeInTheDocument();
  });

  it("download button calls downloadRunOutputContent with auth (not a plain anchor)", async () => {
    // The /content endpoint requires Authorization headers — a plain
    // `<a download>` would 401 on top-level navigation. Verify the
    // Download UI element is a button that calls the auth-attached
    // downloader, not an anchor with an href.
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue(
      manifest([fileArtifact({ artifact_id: "art with spaces/and slash" })]),
    );
    const blob = new Blob(["downloaded bytes"], { type: "text/csv" });
    (downloadRunOutputContent as ReturnType<typeof vi.fn>).mockResolvedValue({
      data: blob,
      filename: "results.csv",
    });
    // Stub objectURL primitives so triggerBrowserDownload doesn't break.
    const createSpy = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
    const revokeSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    render(<RunOutputsPanel runId={RUN_ID} />);

    const dlBtn = await screen.findByRole("button", { name: /Download/ });
    expect(dlBtn.tagName).toBe("BUTTON");
    expect(dlBtn.hasAttribute("href")).toBe(false);
    fireEvent.click(dlBtn);

    await waitFor(() =>
      expect(downloadRunOutputContent).toHaveBeenCalledWith(
        RUN_ID,
        "art with spaces/and slash",
      ),
    );
    expect(createSpy).toHaveBeenCalledWith(blob);
    expect(revokeSpy).toHaveBeenCalledWith("blob:mock");

    createSpy.mockRestore();
    revokeSpy.mockRestore();
  });
});
