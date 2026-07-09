import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  ImportYamlModal,
  buildImportConfirmMessage,
  buildImportSuccessMessage,
  IMPORT_YAML_NOT_RUNNABLE_INTRO,
  IMPORT_YAML_422_MESSAGE,
} from "./ImportYamlModal";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import * as api from "@/api/client";
import type { CompositionState } from "@/types/index";

vi.mock("@/api/client", () => ({
  importCompositionYaml: vi.fn(),
}));

function nonEmptyState(): CompositionState {
  return {
    id: "state-1",
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

describe("ImportYamlModal", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    onClose.mockReset();
    vi.mocked(api.importCompositionYaml).mockReset();
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

  it("keeps Import disabled until the paste looks like pipeline YAML", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml("not: a pipeline document\n");

    expect(screen.getByRole("button", { name: /^import$/i })).toBeDisabled();
    expect(screen.getByText("Validation summary")).toBeInTheDocument();
    expect(screen.getByText(/No pipeline sections found/i)).toBeInTheDocument();
  });

  it("does not treat a nested pipeline wrapper as importable runtime YAML", () => {
    render(<ImportYamlModal onClose={onClose} />);
    typeYaml(
      "pipeline:\n" +
        "  sources:\n" +
        "    source:\n" +
        "      plugin: csv\n",
    );

    expect(screen.getByRole("button", { name: /^import$/i })).toBeDisabled();
    expect(screen.getByText(/No pipeline sections found/i)).toBeInTheDocument();
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
