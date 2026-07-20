import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { parseDocument } from "yaml";
import {
  ImportYamlModal,
  ImportYamlModalHost,
  analyseImportYamlDraft,
  buildImportConfirmMessage,
  buildImportSuccessMessage,
  findImportYamlSourceBindingCandidates,
  IMPORT_YAML_NOT_RUNNABLE_INTRO,
  IMPORT_YAML_422_MESSAGE,
  IMPORT_YAML_SECTIONS_REQUIRED_MESSAGE,
} from "./ImportYamlModal";
import { OPEN_IMPORT_YAML_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import * as api from "@/api/client";
import type { BlobMetadata } from "@/types/api";
import type { CompositionState } from "@/types/index";
import { compositionStateAuthorityFields } from "@/test/composerFixtures";

vi.mock("yaml", async (importOriginal) => {
  const actual = await importOriginal<typeof import("yaml")>();
  return {
    ...actual,
    parseDocument: vi.fn(actual.parseDocument),
  };
});

vi.mock("@/api/client", () => ({
  importCompositionYaml: vi.fn(),
  listBlobs: vi.fn(),
  uploadBlob: vi.fn(),
  getPluginSchema: vi.fn(),
}));

function nonEmptyState(): CompositionState {
  return {
    id: "state-1",
    ...compositionStateAuthorityFields,
    version: 1,
    sources: { source: { plugin: "csv", options: {} } },
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

function emptyState(): CompositionState {
  return {
    id: "state-0",
    ...compositionStateAuthorityFields,
    version: 1,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

const PIPELINE_YAML =
  "source:\n" +
  "  plugin: csv\n" +
  "  on_success: result\n" +
  "sinks:\n" +
  "  result:\n" +
  "    plugin: json\n" +
  "    on_write_failure: fail\n";

const ADVANCED_EXPERIMENT_YAML =
  "sources:\n" +
  "  primary:\n" +
  "    plugin: csv\n" +
  "    on_success: experiment_in\n" +
  "    options:\n" +
  "      path: examples/statistical_batch_plugins/experiment_scores.csv\n" +
  "      schema:\n" +
  "        mode: observed\n" +
  "    on_validation_failure: discard\n" +
  "aggregations:\n" +
  "  - name: prompt_experiment\n" +
  "    plugin: batch_experiment_compare\n" +
  "    input: experiment_in\n" +
  "    on_success: output\n" +
  "    on_error: discard\n" +
  "    trigger:\n" +
  "      count: 8\n" +
  "    output_mode: transform\n" +
  "    options:\n" +
  "      variant_field: prompt_variant\n" +
  "      score_field: score\n" +
  "      baseline_variant: control\n" +
  "sinks:\n" +
  "  output:\n" +
  "    plugin: json\n" +
  "    on_write_failure: discard\n" +
  "    options:\n" +
  "      path: outputs/experiment_compare.jsonl\n" +
  "      format: jsonl\n";

// Two named sources publish into a declared `queue` named `inbound`; a
// downstream transform consumes it. Exercises queue import preflight
// recognition and step counting.
const QUEUE_PIPELINE_YAML =
  "sources:\n" +
  "  orders:\n" +
  "    plugin: csv\n" +
  "    on_success: inbound\n" +
  "  refunds:\n" +
  "    plugin: csv\n" +
  "    on_success: inbound\n" +
  "queues:\n" +
  "  inbound:\n" +
  "    description: Orders and refunds interleave here\n" +
  "transforms:\n" +
  "  - name: summarize\n" +
  "    plugin: llm\n" +
  "    input: inbound\n" +
  "    on_success: result\n" +
  "sinks:\n" +
  "  result:\n" +
  "    plugin: json\n";

// The top-level `queues` value is a mapping (so preflight counts it), but the
// `inbound` entry is a bare scalar rather than a mapping — an entry-level
// defect the SERVER validation path must catch, not one the client silently
// discards.
const QUEUE_MALFORMED_ENTRY_YAML =
  "sources:\n" +
  "  orders:\n" +
  "    plugin: csv\n" +
  "    on_success: inbound\n" +
  "queues:\n" +
  "  inbound: not-a-mapping\n" +
  "sinks:\n" +
  "  result:\n" +
  "    plugin: json\n";

function makeBlob(overrides: Partial<BlobMetadata> = {}): BlobMetadata {
  return {
    id: "22222222-2222-2222-2222-222222222222",
    session_id: "sess-1",
    filename: "experiment_scores.csv",
    mime_type: "text/csv",
    size_bytes: 128,
    content_hash: "hash",
    created_at: "2026-07-10T00:00:00Z",
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

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

describe("ImportYamlModal", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    onClose.mockReset();
    vi.mocked(api.importCompositionYaml).mockReset();
    vi.mocked(api.listBlobs).mockReset();
    vi.mocked(api.listBlobs).mockResolvedValue([]);
    vi.mocked(api.uploadBlob).mockReset();
    vi.mocked(parseDocument).mockClear();
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
      // Loaded-and-empty is the default here; an unloaded state (false) makes
      // the confirm step appear even for a null composition (its own test).
      compositionStateLoaded: true,
      // No exported sidecar by default: only the round-trip tests set one, so
      // the common path stays a two-arg importCompositionYaml call. setState
      // merges, so reset it here or a prior test's binding leaks forward.
      exportedYamlBlobBinding: null,
      guidedSession: null,
      // Stubbed rather than exercised: selectSession's own refetch fan-out
      // (messages/proposals/preferences/blobs/interpretation-events) is
      // sessionStore's concern, already covered by sessionStore's own tests.
      // This test only asserts the import flow calls it with the right id.
      selectSession: vi.fn().mockResolvedValue(undefined),
    } as never);
    useExecutionStore.setState({
      validate: vi.fn().mockResolvedValue(true),
    } as never);
  });

  afterEach(() => {
    useSessionStore.setState({
      activeSessionId: null,
      compositionState: null,
      guidedSession: null,
    } as never);
  });

  function typeYaml(text = PIPELINE_YAML): void {
    fireEvent.change(screen.getByLabelText(/pipeline yaml/i), {
      target: { value: text },
    });
  }

  // Clicks that trigger the async import call resolve/reject on a
  // microtask outside the synchronous fireEvent dispatch -- wrapping in
  // act() keeps the resulting state update inside a tracked act boundary
  // (avoiding a spurious "not wrapped in act" warning) without needing a
  // real user-event pointer sequence for a plain click.
  async function clickImport(): Promise<void> {
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^import$/i }));
    });
  }

  async function clickConfirmReplace(): Promise<void> {
    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", { name: /replace pipeline/i }),
      );
    });
  }

  // ── Open / close / focus ──────────────────────────────────────────────────

  it("renders a labelled textarea and moves focus to it on open", () => {
    render(<ImportYamlModal onClose={onClose} />);

    const textarea = screen.getByLabelText(/pipeline yaml/i);
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveFocus();
  });

  it("host opens on the app-level import event and closes through the modal action", () => {
    render(<ImportYamlModalHost />);

    expect(screen.queryByRole("dialog", { name: /import yaml/i })).toBeNull();

    fireEvent(window, new CustomEvent(OPEN_IMPORT_YAML_MODAL_EVENT));

    expect(
      screen.getByRole("dialog", { name: /import yaml/i }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));

    expect(screen.queryByRole("dialog", { name: /import yaml/i })).toBeNull();
  });

  it("host ignores the app-level import event without an active session", () => {
    useSessionStore.setState({
      activeSessionId: null,
    } as never);
    render(<ImportYamlModalHost />);

    fireEvent(window, new CustomEvent(OPEN_IMPORT_YAML_MODAL_EVENT));

    expect(screen.queryByRole("dialog", { name: /import yaml/i })).toBeNull();
  });

  it("host closes when the active session is cleared", () => {
    render(<ImportYamlModalHost />);
    fireEvent(window, new CustomEvent(OPEN_IMPORT_YAML_MODAL_EVENT));
    expect(
      screen.getByRole("dialog", { name: /import yaml/i }),
    ).toBeInTheDocument();

    act(() => {
      useSessionStore.setState({
        activeSessionId: null,
      } as never);
    });

    expect(screen.queryByRole("dialog", { name: /import yaml/i })).toBeNull();
  });

  it("closes on Escape while drafting", () => {
    render(<ImportYamlModal onClose={onClose} />);

    fireEvent.keyDown(document, { key: "Escape" });

    expect(onClose).toHaveBeenCalled();
  });

  it("closes when the backdrop is clicked", () => {
    render(<ImportYamlModal onClose={onClose} />);

    fireEvent.click(screen.getByTestId("import-yaml-modal-backdrop"));

    expect(onClose).toHaveBeenCalled();
  });

  it("closes when the close button is clicked", () => {
    render(<ImportYamlModal onClose={onClose} />);

    fireEvent.click(
      screen.getByRole("button", { name: /close import yaml/i }),
    );

    expect(onClose).toHaveBeenCalled();
  });

  it("disables Import while the textarea is empty", () => {
    render(<ImportYamlModal onClose={onClose} />);

    expect(screen.getByRole("button", { name: /^import$/i })).toBeDisabled();
  });

  it("disables Import with inline validation when YAML has no pipeline sections", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml("not: a pipeline document\n");

    expect(screen.getByRole("button", { name: /^import$/i })).toBeDisabled();
    expect(screen.getByText("Validation summary")).toBeInTheDocument();
    expect(screen.getByText(IMPORT_YAML_SECTIONS_REQUIRED_MESSAGE)).toBeInTheDocument();
    expect(screen.queryByText("Parsed preview")).toBeNull();
  });

  it("disables Import with inline validation for a nested non-runtime pipeline wrapper", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      "pipeline:\n" +
        "  sources:\n" +
        "    source:\n" +
        "      plugin: csv\n",
    );

    expect(screen.getByRole("button", { name: /^import$/i })).toBeDisabled();
    expect(screen.getByText(IMPORT_YAML_SECTIONS_REQUIRED_MESSAGE)).toBeInTheDocument();
  });

  it("previews and enables Import for flow-style YAML the backend accepts", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      '{"source": {"plugin": "csv", "on_success": "result"}, ' +
        '"sinks": {"result": {"plugin": "json", "on_write_failure": "fail"}}}',
    );

    expect(screen.getByRole("button", { name: /^import$/i })).not.toBeDisabled();
    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    expect(screen.getByText("1 source, 0 processing steps, 1 output")).toBeInTheDocument();
    expect(screen.getByText(/Ready for server validation/i)).toBeInTheDocument();
  });

  it("disables Import with inline validation when YAML syntax is malformed", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml("sources: [this is: not: valid");

    expect(screen.getByRole("button", { name: /^import$/i })).toBeDisabled();
    expect(screen.getByText("Validation summary")).toBeInTheDocument();
    expect(screen.getByText(/YAML parse failed near line 1/i)).toBeInTheDocument();
    expect(screen.queryByText("Parsed preview")).toBeNull();
  });

  it("accepts uniformly indented top-level runtime YAML", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      "  source:\n" +
        "    plugin: csv\n" +
        "    on_success: result\n" +
        "  sinks:\n" +
        "    result:\n" +
        "      plugin: json\n" +
        "      on_write_failure: fail\n",
    );

    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    expect(screen.getByText("1 source, 0 processing steps, 1 output")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^import$/i })).not.toBeDisabled();
  });

  it("accepts top-level runtime keys with whitespace before the YAML colon", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      "sources :\n" +
        "  source:\n" +
        "    plugin: csv\n" +
        "    on_success: result\n" +
        "sinks :\n" +
        "  result:\n" +
        "    plugin: json\n" +
        "    on_write_failure: fail\n",
    );

    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    expect(screen.getByText("1 source, 0 processing steps, 1 output")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^import$/i })).not.toBeDisabled();
  });

  it("accepts quoted top-level runtime keys", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      '"sources":\n' +
        "  source:\n" +
        "    plugin: csv\n" +
        "    on_success: result\n" +
        '"sinks":\n' +
        "  result:\n" +
        "    plugin: json\n" +
        "    on_write_failure: fail\n",
    );

    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    expect(screen.getByText("1 source, 0 processing steps, 1 output")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^import$/i })).not.toBeDisabled();
  });

  it("shows a parsed preview and validation summary before Import is enabled", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      "source:\n" +
        "  plugin: csv\n" +
        "  on_success: summarize\n" +
        "transforms:\n" +
        "  - name: summarize\n" +
        "    plugin: llm\n" +
        "    input: source\n" +
        "    on_success: result\n" +
        "    on_error: fail\n" +
        "sinks:\n" +
        "  result:\n" +
        "    plugin: json\n" +
        "    on_write_failure: fail\n",
    );

    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    expect(
      screen.getByText("1 source, 1 processing step, 1 output"),
    ).toBeInTheDocument();
    expect(screen.getByText("Validation summary")).toBeInTheDocument();
    expect(screen.getByText(/Ready for server validation/i)).toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "Import YAML preflight" }),
    ).toHaveAttribute("aria-live", "polite");
    expect(screen.getByLabelText(/pipeline yaml/i)).toHaveAccessibleDescription(
      /Ready for server validation/i,
    );
    expect(screen.getByRole("button", { name: /^import$/i })).not.toBeDisabled();
  });

  it("counts section body entries when the section header has an inline comment", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      "source:\n" +
        "  plugin: csv\n" +
        "  on_success: summarize\n" +
        "transforms: # generated steps\n" +
        "  - name: summarize\n" +
        "    plugin: llm\n" +
        "  - name: normalize\n" +
        "    plugin: lowercase\n" +
        "sinks:\n" +
        "  result:\n" +
        "    plugin: json\n",
    );

    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    expect(
      screen.getByText("1 source, 2 processing steps, 1 output"),
    ).toBeInTheDocument();
  });

  it("counts indentless block-sequence entries (PyYAML's default emitter style)", () => {
    // The '-' marker of a YAML block sequence may sit at the SAME indent as
    // its parent key ("transforms:" and "- name: a" both at column 0) -- a
    // valid, common emitter style. countYamlSectionEntries used to break the
    // scan the instant it saw indent <= start.indent, undercounting to 0.
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      "source:\n" +
        "  plugin: csv\n" +
        "  on_success: a\n" +
        "transforms:\n" +
        "- name: a\n" +
        "  plugin: uppercase\n" +
        "- name: b\n" +
        "  plugin: lowercase\n" +
        "sinks:\n" +
        "  result:\n" +
        "    plugin: json\n",
    );

    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    expect(
      screen.getByText("1 source, 2 processing steps, 1 output"),
    ).toBeInTheDocument();
  });

  it("recognises a top-level queues section and counts queue nodes as processing steps", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(QUEUE_PIPELINE_YAML);

    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    // Two sources; the queue node AND the transform each count as one
    // processing step; one sink. A queue must not be flagged as an unknown
    // section, nor counted as a source/output.
    expect(
      screen.getByText("2 sources, 2 processing steps, 1 output"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Ready for server validation/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^import$/i })).not.toBeDisabled();
  });

  it("preflights a very large paste without throwing", () => {
    // Import preflight runs synchronously in render. Keep a large-paste guard
    // so scanner/parser changes do not reintroduce RangeError or stack issues.
    const bigBody = Array.from(
      { length: 5000 },
      (_, i) => `  field_${i}: value`,
    ).join("\n");
    const bigYaml = `source:\n  plugin: csv\n${bigBody}\nsinks:\n  result:\n    plugin: json\n`;

    render(<ImportYamlModal onClose={onClose} />);

    expect(() => typeYaml(bigYaml)).not.toThrow();
    expect(screen.getByRole("button", { name: /^import$/i })).not.toBeDisabled();
  });

  it("does not close on Escape while the confirm step owns the keyboard", () => {
    useSessionStore.setState({ compositionState: nonEmptyState() } as never);
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    fireEvent.click(screen.getByRole("button", { name: /^import$/i }));
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });

    // ConfirmDialog's own Escape handler cancels back to draft; the outer
    // modal must not ALSO close (that would drop both dialogs on one key).
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(screen.getByRole("dialog", { name: /import yaml/i })).toBeInTheDocument();
  });

  // ── File picker ────────────────────────────────────────────────────────────

  it("reads a chosen .yaml file into the textarea client-side (no network call)", async () => {
    render(<ImportYamlModal onClose={onClose} />);
    const file = new File([PIPELINE_YAML], "pipeline.yaml", {
      type: "text/yaml",
    });
    const input = screen.getByLabelText(
      /choose a \.yaml file/i,
    ) as HTMLInputElement;

    await userEvent.upload(input, file);

    await waitFor(() =>
      expect(screen.getByLabelText(/pipeline yaml/i)).toHaveValue(
        PIPELINE_YAML,
      ),
    );
  });

  it("finds file-backed sources that can be rebound to uploaded blobs", () => {
    expect(
      findImportYamlSourceBindingCandidates(ADVANCED_EXPERIMENT_YAML),
    ).toEqual([
      {
        sourceName: "primary",
        optionKey: "path",
        path: "examples/statistical_batch_plugins/experiment_scores.csv",
      },
    ]);
  });

  it("defers draft YAML analysis through one shared parse", () => {
    const componentSource = readFileSync(
      "src/components/sidebar/ImportYamlModal.tsx",
      "utf8",
    );
    expect(componentSource).toContain("useDeferredValue");

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(ADVANCED_EXPERIMENT_YAML);

    expect(screen.getByText("Parsed preview")).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: /uploaded file for source primary/i }),
    ).toBeInTheDocument();
    expect(parseDocument).toHaveBeenCalledTimes(1);
    expect(parseDocument).toHaveBeenCalledWith(
      ADVANCED_EXPERIMENT_YAML,
      { prettyErrors: false },
    );
  });

  it("imports a pasted advanced example with an explicit uploaded source binding", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    const blob = makeBlob();
    vi.mocked(api.listBlobs).mockResolvedValue([blob]);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-advanced",
      version: 9,
      is_valid: true,
      validation_errors: null,
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(ADVANCED_EXPERIMENT_YAML);

    const bindingSelect = await screen.findByRole("combobox", {
      name: /uploaded file for source primary/i,
    });
    fireEvent.change(bindingSelect, { target: { value: blob.id } });
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText(buildImportSuccessMessage(9))).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith(
      "sess-1",
      ADVANCED_EXPERIMENT_YAML,
      { primary: blob.id },
    );
  });

  it("clears a selected source binding when the YAML source path changes", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    const blob = makeBlob();
    vi.mocked(api.listBlobs).mockResolvedValue([blob]);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-path-edited",
      version: 11,
      is_valid: true,
      validation_errors: null,
    });
    const editedYaml = ADVANCED_EXPERIMENT_YAML.replace(
      "examples/statistical_batch_plugins/experiment_scores.csv",
      "examples/statistical_batch_plugins/other_scores.csv",
    );

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(ADVANCED_EXPERIMENT_YAML);
    const bindingSelect = await screen.findByRole("combobox", {
      name: /uploaded file for source primary/i,
    });
    fireEvent.change(bindingSelect, { target: { value: blob.id } });
    expect(bindingSelect).toHaveValue(blob.id);

    typeYaml(editedYaml);

    await waitFor(() =>
      expect(
        screen.getByRole("combobox", {
          name: /uploaded file for source primary/i,
        }),
      ).toHaveValue(""),
    );
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText(buildImportSuccessMessage(11))).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith("sess-1", editedYaml);
  });

  it("uploads a source file from the import modal and uses it as source_blob_ids", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    const blob = makeBlob({ id: "33333333-3333-3333-3333-333333333333" });
    vi.mocked(api.uploadBlob).mockResolvedValue(blob);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-uploaded",
      version: 10,
      is_valid: true,
      validation_errors: null,
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(ADVANCED_EXPERIMENT_YAML);

    const sourceFileInput = await screen.findByLabelText(
      /upload file for source primary/i,
    );
    await userEvent.upload(
      sourceFileInput,
      new File(["id,prompt_variant,score\n1,control,0.5\n"], "experiment_scores.csv", {
        type: "text/csv",
      }),
    );

    await waitFor(() =>
      expect(api.uploadBlob).toHaveBeenCalledWith(
        "sess-1",
        expect.objectContaining({ name: "experiment_scores.csv" }),
      ),
    );
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText(buildImportSuccessMessage(10))).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith(
      "sess-1",
      ADVANCED_EXPERIMENT_YAML,
      { primary: blob.id },
    );
  });

  it("does not bind an upload that finishes after the YAML source path changes", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    const upload = deferred<BlobMetadata>();
    const blob = makeBlob({ id: "44444444-4444-4444-4444-444444444444" });
    vi.mocked(api.uploadBlob).mockReturnValue(upload.promise);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-stale-upload",
      version: 12,
      is_valid: true,
      validation_errors: null,
    });
    const editedYaml = ADVANCED_EXPERIMENT_YAML.replace(
      "examples/statistical_batch_plugins/experiment_scores.csv",
      "examples/statistical_batch_plugins/late_scores.csv",
    );

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(ADVANCED_EXPERIMENT_YAML);
    const sourceFileInput = await screen.findByLabelText(
      /upload file for source primary/i,
    );
    await userEvent.upload(
      sourceFileInput,
      new File(["id,prompt_variant,score\n1,control,0.5\n"], "experiment_scores.csv", {
        type: "text/csv",
      }),
    );
    await waitFor(() => expect(api.uploadBlob).toHaveBeenCalledOnce());

    typeYaml(editedYaml);
    await act(async () => {
      upload.resolve(blob);
      await upload.promise;
    });
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText(buildImportSuccessMessage(12))).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith("sess-1", editedYaml);
  });

  it("does not keep Import disabled for an upload tied to a removed source path", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    const upload = deferred<BlobMetadata>();
    vi.mocked(api.uploadBlob).mockReturnValue(upload.promise);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-stale-pending",
      version: 13,
      is_valid: true,
      validation_errors: null,
    });
    const editedYaml = ADVANCED_EXPERIMENT_YAML.replace(
      "examples/statistical_batch_plugins/experiment_scores.csv",
      "examples/statistical_batch_plugins/current_scores.csv",
    );

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(ADVANCED_EXPERIMENT_YAML);
    const sourceFileInput = await screen.findByLabelText(
      /upload file for source primary/i,
    );
    await userEvent.upload(
      sourceFileInput,
      new File(["id,prompt_variant,score\n1,control,0.5\n"], "experiment_scores.csv", {
        type: "text/csv",
      }),
    );
    expect(screen.getByRole("button", { name: /^import$/i })).toBeDisabled();

    typeYaml(editedYaml);

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^import$/i })).not.toBeDisabled(),
    );
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText(buildImportSuccessMessage(13))).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith("sess-1", editedYaml);
    await act(async () => {
      upload.resolve(makeBlob({ id: "55555555-5555-5555-5555-555555555555" }));
      await upload.promise;
    });
  });

  // ── Happy path (no confirm needed — current composition is empty) ─────────

  it("submits directly and shows the version on success when the current pipeline is empty", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-2",
      version: 2,
      is_valid: true,
      validation_errors: null,
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    expect(screen.queryByRole("alertdialog")).toBeNull();
    await waitFor(() =>
      expect(
        screen.getByText(buildImportSuccessMessage(2)),
      ).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith(
      "sess-1",
      PIPELINE_YAML,
    );
    expect(useSessionStore.getState().selectSession).toHaveBeenCalledWith(
      "sess-1",
    );
    expect(useExecutionStore.getState().validate).toHaveBeenCalledWith(
      "sess-1",
    );
  });

  it("confirms before importing when composition state is not yet loaded (unknown != empty)", () => {
    // A transient session-load failure leaves compositionState null with
    // compositionStateLoaded false. Treating that as "empty" would replace a
    // real server-side pipeline with no confirmation — so the confirm step
    // must appear.
    useSessionStore.setState({
      compositionState: null,
      compositionStateLoaded: false,
    } as never);
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    fireEvent.click(screen.getByRole("button", { name: /^import$/i }));

    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    expect(api.importCompositionYaml).not.toHaveBeenCalled();
  });

  // ── Confirm step (replacing a non-trivial pipeline) ────────────────────────

  it("shows a confirm step before submitting when the current pipeline is non-trivial", () => {
    useSessionStore.setState({
      compositionState: nonEmptyState(),
      guidedSession: null,
    } as never);

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    fireEvent.click(screen.getByRole("button", { name: /^import$/i }));

    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    expect(
      screen.getByText(buildImportConfirmMessage(false)),
    ).toBeInTheDocument();
    expect(api.importCompositionYaml).not.toHaveBeenCalled();
  });

  it("mentions the guided-mode reset in the confirm copy while guided is genuinely active", () => {
    useSessionStore.setState({
      compositionState: nonEmptyState(),
      // terminal: null is load-bearing here -- this must match
      // isGuidedBuildActive's own non-terminal check, not just non-null.
      guidedSession: { id: "g1", step: 1, terminal: null } as never,
    } as never);

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    fireEvent.click(screen.getByRole("button", { name: /^import$/i }));

    expect(
      screen.getByText(buildImportConfirmMessage(true)),
    ).toBeInTheDocument();
    expect(screen.getByText(/switch it to freeform/i)).toBeInTheDocument();
  });

  it("does NOT claim a guided switch when the guided session is already terminal", () => {
    // A terminal guided_session (completed / exited_to_freeform) means the
    // user has already left guided -- SideRail (and this modal) is only
    // reachable in that state or the null state, never mid-active-guided
    // (isGuidedBuildActive suppresses the rail then). Saying "will switch
    // it to freeform" here would be affirmatively wrong: it already has.
    useSessionStore.setState({
      compositionState: nonEmptyState(),
      guidedSession: {
        id: "g1",
        step: 1,
        terminal: { kind: "exited_to_freeform" },
      } as never,
    } as never);

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    fireEvent.click(screen.getByRole("button", { name: /^import$/i }));

    expect(
      screen.getByText(buildImportConfirmMessage(false)),
    ).toBeInTheDocument();
    expect(screen.queryByText(/switch it to freeform/i)).toBeNull();
  });

  it("submits after the confirm step is accepted", async () => {
    useSessionStore.setState({ compositionState: nonEmptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-3",
      version: 3,
      is_valid: true,
      validation_errors: null,
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    fireEvent.click(screen.getByRole("button", { name: /^import$/i }));
    await clickConfirmReplace();

    await waitFor(() =>
      expect(
        screen.getByText(buildImportSuccessMessage(3)),
      ).toBeInTheDocument(),
    );
  });

  it("returns to the draft form without submitting when the confirm step is cancelled", () => {
    useSessionStore.setState({ compositionState: nonEmptyState() } as never);

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    fireEvent.click(screen.getByRole("button", { name: /^import$/i }));
    fireEvent.click(
      screen.getByRole("button", { name: /keep current pipeline/i }),
    );

    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(api.importCompositionYaml).not.toHaveBeenCalled();
    expect(screen.getByLabelText(/pipeline yaml/i)).toHaveValue(
      PIPELINE_YAML,
    );
  });

  // ── is_valid:false outcome ──────────────────────────────────────────────────

  it("shows the not-runnable framing and the validation errors when is_valid is false", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-4",
      version: 4,
      is_valid: false,
      validation_errors: ["sinks: field required"],
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    await waitFor(() =>
      expect(
        screen.getByText(buildImportSuccessMessage(4)),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText(IMPORT_YAML_NOT_RUNNABLE_INTRO)).toBeInTheDocument();
    expect(screen.getByText("sinks: field required")).toBeInTheDocument();
    // Tier-1 honesty: a 200 with is_valid:false must be framed as NOT ready
    // to run -- never as an unqualified success.
    expect(screen.getByText(/is not ready to run/i)).toBeInTheDocument();
  });

  it("does not render a dangling colon when is_valid is false with no validation_errors", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-5",
      version: 5,
      is_valid: false,
      validation_errors: null,
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    await waitFor(() =>
      expect(
        screen.getByText(buildImportSuccessMessage(5)),
      ).toBeInTheDocument(),
    );
    expect(screen.queryByText(IMPORT_YAML_NOT_RUNNABLE_INTRO)).toBeNull();
    expect(
      screen.getByText(/is not ready to run\.$/),
    ).toBeInTheDocument();
  });

  it("renders sanitized disabled-component repair actions without fetching private schema", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-disabled",
      version: 6,
      is_valid: false,
      validation_errors: ["A saved component is unavailable."],
      plugin_policy_findings: [
        {
          component_id: "legacy_output",
          plugin_id: "sink:database",
          reason_code: "credential_unavailable",
          snapshot_fingerprint: "current-snapshot",
        },
      ],
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    const repairRegion = await screen.findByRole("region", {
      name: /unavailable saved components/i,
    });
    expect(repairRegion).toHaveTextContent("legacy_output");
    expect(repairRegion).toHaveTextContent("sink:database");
    expect(repairRegion).toHaveTextContent("Credential unavailable");
    expect(
      screen.getByRole("button", {
        name: /remove disabled component legacy_output.*sink:database/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: /replace disabled component legacy_output.*available sink/i,
      }),
    ).toBeInTheDocument();
    expect(api.getPluginSchema).not.toHaveBeenCalled();
  });

  // ── Error classes ────────────────────────────────────────────────────────

  it("maps a 422 to friendly copy naming the 256 KB limit", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockRejectedValue({
      status: 422,
      detail: "Unprocessable Entity",
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByText(IMPORT_YAML_422_MESSAGE)).toBeInTheDocument();
  });

  it("names the affected component and policy reason for a plugin-policy 422", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockRejectedValue({
      status: 422,
      detail: "Unprocessable Entity",
      error_type: "plugin_not_enabled",
      component_id: "main",
      plugin_id: "sink:database",
      snapshot_fingerprint: "snapshot-a",
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "Component main (sink:database) is not enabled under the current plugin policy.",
    );
    expect(screen.queryByText(IMPORT_YAML_422_MESSAGE)).not.toBeInTheDocument();
  });

  it("renders the backend detail verbatim for a 400 (structural defect)", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockRejectedValue({
      status: 400,
      detail: "sources.source.plugin must be a non-empty string",
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    await waitFor(() =>
      expect(
        screen.getByText("sources.source.plugin must be a non-empty string"),
      ).toBeInTheDocument(),
    );
    // Form content survives the failure so the user can fix and retry.
    expect(screen.getByLabelText(/pipeline yaml/i)).toHaveValue(
      PIPELINE_YAML,
    );
  });

  it("renders the backend detail verbatim for a 404 (blob not found)", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockRejectedValue({
      status: 404,
      detail: "Blob not found",
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText("Blob not found")).toBeInTheDocument(),
    );
  });

  it("renders the backend detail verbatim for a 409 (blob service unavailable)", async () => {
    useSessionStore.setState({ compositionState: emptyState() } as never);
    vi.mocked(api.importCompositionYaml).mockRejectedValue({
      status: 409,
      detail: "Blob service unavailable for YAML import",
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    await waitFor(() =>
      expect(
        screen.getByText("Blob service unavailable for YAML import"),
      ).toBeInTheDocument(),
    );
  });

  // ── source_blob_ids sidecar replay (blob-backed export round-trip) ──────────

  const BLOB_UUID = "11111111-1111-1111-1111-111111111111";

  it("replays the source_blob_ids sidecar when the pasted YAML matches the export verbatim", async () => {
    // The export fetch stashed this binding for the current session. Pasting
    // the exact exported YAML must re-supply the sidecar so a blob-backed
    // source rebinds instead of 400-ing as unbound blob storage.
    useSessionStore.setState({
      compositionState: emptyState(),
      exportedYamlBlobBinding: {
        sessionId: "sess-1",
        yaml: PIPELINE_YAML,
        sourceBlobIds: { source: BLOB_UUID },
      },
    } as never);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-6",
      version: 6,
      is_valid: true,
      validation_errors: null,
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText(buildImportSuccessMessage(6))).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith("sess-1", PIPELINE_YAML, {
      source: BLOB_UUID,
    });
  });

  it("does NOT replay the sidecar when the pasted YAML differs from the export", async () => {
    // A hand-edited paste can no longer be trusted to match the sidecar's
    // source names; sending it would risk an unknown-source 400. Degrade to a
    // plain two-arg import (the backend then asks the user to re-provide).
    useSessionStore.setState({
      compositionState: emptyState(),
      exportedYamlBlobBinding: {
        sessionId: "sess-1",
        yaml: PIPELINE_YAML,
        sourceBlobIds: { source: BLOB_UUID },
      },
    } as never);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-7",
      version: 7,
      is_valid: true,
      validation_errors: null,
    });
    const edited = `${PIPELINE_YAML}# hand edited\n`;

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(edited);
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText(buildImportSuccessMessage(7))).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith("sess-1", edited);
  });

  it("does NOT replay a sidecar that belongs to a different session", async () => {
    // Blob refs are session-scoped; replaying another session's binding would
    // 404 on the foreign blob. The sessionId guard drops it.
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: emptyState(),
      exportedYamlBlobBinding: {
        sessionId: "sess-other",
        yaml: PIPELINE_YAML,
        sourceBlobIds: { source: BLOB_UUID },
      },
    } as never);
    vi.mocked(api.importCompositionYaml).mockResolvedValue({
      id: "state-8",
      version: 8,
      is_valid: true,
      validation_errors: null,
    });

    render(<ImportYamlModal onClose={onClose} />);
    typeYaml();
    await clickImport();

    await waitFor(() =>
      expect(screen.getByText(buildImportSuccessMessage(8))).toBeInTheDocument(),
    );
    expect(api.importCompositionYaml).toHaveBeenCalledWith("sess-1", PIPELINE_YAML);
  });
});

describe("analyseImportYamlDraft queue recognition", () => {
  it("counts a queue section as a processing step, not a source or output", () => {
    const analysis = analyseImportYamlDraft(QUEUE_PIPELINE_YAML);
    expect(analysis.sectionsParsed).toBe(true);
    expect(analysis.canImport).toBe(true);
    expect(analysis.sourceCount).toBe(2);
    // queue `inbound` + transform `summarize`.
    expect(analysis.stepCount).toBe(2);
    expect(analysis.outputCount).toBe(1);
  });

  it("keeps a malformed queue entry importable so the server validates it", () => {
    const analysis = analyseImportYamlDraft(QUEUE_MALFORMED_ENTRY_YAML);
    // The `queues` section is a mapping (so the queue still counts as a
    // structural step); the bad entry value is left for the server rather than
    // being silently discarded client-side.
    expect(analysis.canImport).toBe(true);
    expect(analysis.stepCount).toBe(1);
  });

  it("leaves queue-free import preflight unchanged", () => {
    const analysis = analyseImportYamlDraft(PIPELINE_YAML);
    expect(analysis.sourceCount).toBe(1);
    expect(analysis.stepCount).toBe(0);
    expect(analysis.outputCount).toBe(1);
    expect(analysis.canImport).toBe(true);
  });

  it("names queues in the required-section message", () => {
    expect(IMPORT_YAML_SECTIONS_REQUIRED_MESSAGE).toContain("queues");
  });
});
