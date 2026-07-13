import { useDeferredValue, useEffect, useId, useMemo, useRef, useState } from "react";
import { parseDocument } from "yaml";
import * as api from "@/api/client";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import {
  OPEN_CATALOG_EVENT,
  OPEN_IMPORT_YAML_MODAL_EVENT,
} from "@/lib/composer-events";
import { PREFILL_CHAT_INPUT_EVENT } from "@/components/catalog/PluginCard";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { BlobMetadata } from "@/types/api";
import type { ApiError, PluginPolicyFinding } from "@/types/index";
import { hasCompositionContent } from "@/utils/compositionState";
import { plural } from "@/utils/plural";

export const IMPORT_YAML_CONFIRM_TITLE = "Replace the current pipeline?";
export const IMPORT_YAML_CONFIRM_CONFIRM_LABEL = "Replace pipeline";
export const IMPORT_YAML_CONFIRM_CANCEL_LABEL = "Keep current pipeline";

/**
 * Confirm-step message. The history note always applies (the backend
 * always creates a new version rather than overwriting); the guided note
 * is appended only while guided is genuinely ACTIVE (a guided_session
 * exists and its own `.terminal` is null) -- the same predicate
 * `isGuidedBuildActive` (components/chat/guided/guidedBuildActive.ts) uses
 * to decide whether SideRail itself is on screen. A terminal guided_session
 * (completed / exited_to_freeform) means the user already left guided; the
 * sentence must not claim a switch that already happened.
 */
export function buildImportConfirmMessage(guidedActive: boolean): string {
  const base =
    "Importing will replace the current pipeline with the one in this YAML. " +
    "The version you have now stays in your version history, and you can " +
    "revert to it afterwards.";
  return guidedActive
    ? `${base} This session is in guided mode; importing will switch it to freeform.`
    : base;
}

/** Success copy -- pinned as a helper so tests assert the exact string. */
export function buildImportSuccessMessage(version: number): string {
  return `Imported as version ${version}.`;
}

export const IMPORT_YAML_NOT_RUNNABLE_INTRO =
  "Validation found problems with this pipeline. It has been saved, but is not ready to run:";
// Used when is_valid is false but the backend sent no validation_errors
// entries -- IMPORT_YAML_NOT_RUNNABLE_INTRO's trailing colon promises a list
// that would otherwise render empty.
export const IMPORT_YAML_NOT_RUNNABLE_INTRO_NO_DETAIL =
  "Validation found problems with this pipeline. It has been saved, but is not ready to run.";

export const IMPORT_YAML_422_MESSAGE =
  "This paste could not be imported: it is empty, or larger than the 256 KB limit.";

const IMPORT_YAML_GENERIC_ERROR_DETAIL = "Failed to import YAML. Please try again.";

// Client-side import preflight mirrors only the backend's first hard gates:
// syntactically valid YAML, mapping root, and at least one runtime pipeline
// section. Plugin/schema validation remains server-owned.
export const IMPORT_YAML_SECTIONS_REQUIRED_MESSAGE =
  "Pipeline YAML must define at least one pipeline section: sources, source, transforms, gates, aggregations, coalesce, queues, or sinks.";

interface ImportErrorInfo {
  title: string;
  detail: string;
}

/**
 * Map an import failure to display copy.
 *
 * 400s (structural defects, anchors/aliases, unbound blob-storage paths,
 * literal credentials) and 404/409 (blob lookup) all carry a plain,
 * user-language HTTPException detail string from the backend -- rendered
 * verbatim. 422 is the one case worth overriding: it fires from the
 * request body's own Pydantic field constraint (empty/oversized yaml, or
 * an unknown field), and FastAPI's default validation-error body is an
 * array (not a string), so `parseResponse` cannot extract a usable
 * `detail` from it -- it falls back to `response.statusText`
 * ("Unprocessable Entity"), which is not useful to a user.
 */
function describeImportError(apiErr: ApiError): ImportErrorInfo {
  if (apiErr.status === 422) {
    return { title: "Could not import this paste.", detail: IMPORT_YAML_422_MESSAGE };
  }
  return {
    title: "Could not import this pipeline.",
    detail: apiErr.detail && apiErr.detail.trim().length > 0
      ? apiErr.detail
      : IMPORT_YAML_GENERIC_ERROR_DETAIL,
  };
}

interface ImportSuccessInfo {
  version: number;
  isValid: boolean;
  validationErrors: string[];
  pluginPolicyFindings: PluginPolicyFinding[];
}

function unavailableReasonLabel(reason: PluginPolicyFinding["reason_code"]): string {
  return {
    plugin_not_enabled: "Not enabled",
    plugin_not_installed: "Not installed",
    plugin_unavailable: "Unavailable",
    credential_unavailable: "Credential unavailable",
    profile_unavailable: "Profile unavailable",
  }[reason];
}

function pluginKind(pluginId: string): string {
  return pluginId.split(":", 1)[0] || "plugin";
}

type ImportPhase = "draft" | "confirming" | "submitting" | "success";

interface ImportYamlModalProps {
  onClose: () => void;
}

interface ImportYamlDraftAnalysis {
  hasText: boolean;
  canImport: boolean;
  // Whether client-side YAML preflight recognised pipeline sections and
  // produced a real preview. Import stays disabled for syntax/root-section
  // failures; deeper plugin/schema validation remains server-owned.
  sectionsParsed: boolean;
  sourceCount: number;
  stepCount: number;
  outputCount: number;
  validationMessage: string;
}

export interface ImportYamlSourceBindingCandidate {
  sourceName: string;
  optionKey: "path" | "file";
  path: string;
}

type ImportYamlSection =
  | "aggregations"
  | "coalesce"
  | "gates"
  | "queues"
  | "sinks"
  | "source"
  | "sources"
  | "transforms";

const IMPORT_YAML_SECTION_ALIASES: Record<string, ImportYamlSection> = {
  aggregations: "aggregations",
  coalesce: "coalesce",
  gates: "gates",
  queues: "queues",
  sinks: "sinks",
  source: "source",
  sources: "sources",
  transforms: "transforms",
};

const IMPORT_YAML_SOURCE_PATH_KEYS = ["path", "file"] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function offsetToLineColumn(text: string, offset: number): { line: number; column: number } {
  let line = 1;
  let column = 1;
  const cappedOffset = Math.max(0, Math.min(offset, text.length));
  for (let index = 0; index < cappedOffset; index += 1) {
    if (text[index] === "\n") {
      line += 1;
      column = 1;
    } else {
      column += 1;
    }
  }
  return { line, column };
}

function yamlParseFailureMessage(yamlText: string, error: unknown): string {
  const detail =
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof error.message === "string"
      ? error.message
      : "Invalid YAML syntax.";
  const offset =
    typeof error === "object" &&
    error !== null &&
    "pos" in error &&
    Array.isArray(error.pos) &&
    typeof error.pos[0] === "number"
      ? error.pos[0]
      : null;
  if (offset === null) {
    return `YAML parse failed: ${detail}`;
  }
  const location = offsetToLineColumn(yamlText, offset);
  return `YAML parse failed near line ${location.line}, column ${location.column}: ${detail}`;
}

function countRecordEntries(value: unknown, sectionName: string): number | string {
  if (!isRecord(value)) {
    return `Section "${sectionName}" must be a mapping.`;
  }
  return Object.keys(value).length;
}

function countSequenceEntries(value: unknown, sectionName: string): number | string {
  if (!Array.isArray(value)) {
    return `Section "${sectionName}" must be a list.`;
  }
  return value.length;
}

function countParsedSectionEntries(
  section: ImportYamlSection,
  value: unknown,
): number | string {
  if (section === "source") {
    return isRecord(value) ? 1 : 'Section "source" must be a mapping.';
  }
  if (section === "sources" || section === "queues" || section === "sinks") {
    return countRecordEntries(value, section);
  }
  return countSequenceEntries(value, section);
}

type ImportYamlParsedDocument = ReturnType<typeof parseDocument>;

interface ParsedImportYamlDraft {
  hasText: boolean;
  document: ImportYamlParsedDocument | null;
  root: unknown;
}

function parseImportYamlDraft(yamlText: string): ParsedImportYamlDraft {
  if (yamlText.trim().length === 0) {
    return { hasText: false, document: null, root: null };
  }
  const document = parseDocument(yamlText, { prettyErrors: false });
  return {
    hasText: true,
    document,
    root: document.errors.length > 0 ? null : document.toJS({}),
  };
}

function findImportYamlSourceBindingCandidatesFromParsed(
  parsedDraft: ParsedImportYamlDraft,
): ImportYamlSourceBindingCandidate[] {
  if (!parsedDraft.hasText || parsedDraft.document === null) return [];
  if (parsedDraft.document.errors.length > 0) return [];

  const parsedRoot = parsedDraft.root;
  if (!isRecord(parsedRoot)) return [];

  let rawSources = parsedRoot.sources;
  if (rawSources === undefined && parsedRoot.source !== undefined) {
    rawSources = { source: parsedRoot.source };
  }
  if (!isRecord(rawSources)) return [];

  const candidates: ImportYamlSourceBindingCandidate[] = [];
  for (const [sourceName, rawSource] of Object.entries(rawSources)) {
    if (!sourceName || !isRecord(rawSource)) continue;
    const options = rawSource.options;
    if (!isRecord(options)) continue;
    for (const optionKey of IMPORT_YAML_SOURCE_PATH_KEYS) {
      const path = options[optionKey];
      if (typeof path === "string" && path.trim().length > 0) {
        candidates.push({ sourceName, optionKey, path });
        break;
      }
    }
  }
  return candidates;
}

export function findImportYamlSourceBindingCandidates(
  yamlText: string,
): ImportYamlSourceBindingCandidate[] {
  return findImportYamlSourceBindingCandidatesFromParsed(
    parseImportYamlDraft(yamlText),
  );
}

function analyseImportYamlDraftFromParsed(
  yamlText: string,
  parsedDraft: ParsedImportYamlDraft,
): ImportYamlDraftAnalysis {
  if (!parsedDraft.hasText) {
    return {
      hasText: false,
      canImport: false,
      sectionsParsed: false,
      sourceCount: 0,
      stepCount: 0,
      outputCount: 0,
      validationMessage: "Paste YAML to preview it.",
    };
  }

  const parsed = parsedDraft.document;
  if (parsed === null) {
    return {
      hasText: false,
      canImport: false,
      sectionsParsed: false,
      sourceCount: 0,
      stepCount: 0,
      outputCount: 0,
      validationMessage: "Paste YAML to preview it.",
    };
  }
  if (parsed.errors.length > 0) {
    return {
      hasText: true,
      canImport: false,
      sectionsParsed: false,
      sourceCount: 0,
      stepCount: 0,
      outputCount: 0,
      validationMessage: yamlParseFailureMessage(yamlText, parsed.errors[0]),
    };
  }

  const parsedRoot = parsedDraft.root;
  if (!isRecord(parsedRoot)) {
    return {
      hasText: true,
      canImport: false,
      sectionsParsed: false,
      sourceCount: 0,
      stepCount: 0,
      outputCount: 0,
      validationMessage: "Pipeline YAML must be a mapping.",
    };
  }

  const parsedSectionKeys = Object.keys(parsedRoot)
    .map((key) => IMPORT_YAML_SECTION_ALIASES[key])
    .filter((section): section is ImportYamlSection => section !== undefined);
  if (parsedSectionKeys.length === 0) {
    return {
      hasText: true,
      canImport: false,
      sectionsParsed: false,
      sourceCount: 0,
      stepCount: 0,
      outputCount: 0,
      validationMessage: IMPORT_YAML_SECTIONS_REQUIRED_MESSAGE,
    };
  }

  let parsedSourceCount = 0;
  let parsedStepCount = 0;
  let parsedOutputCount = 0;
  for (const [key, value] of Object.entries(parsedRoot)) {
    const section = IMPORT_YAML_SECTION_ALIASES[key];
    if (section === undefined) continue;
    const count = countParsedSectionEntries(section, value);
    if (typeof count === "string") {
      return {
        hasText: true,
        canImport: false,
        sectionsParsed: false,
        sourceCount: 0,
        stepCount: 0,
        outputCount: 0,
        validationMessage: count,
      };
    }
    if (section === "source" || section === "sources") {
      parsedSourceCount += count;
    } else if (section === "sinks") {
      parsedOutputCount += count;
    } else {
      parsedStepCount += count;
    }
  }

  return {
    hasText: true,
    canImport: true,
    sectionsParsed: true,
    sourceCount: parsedSourceCount,
    stepCount: parsedStepCount,
    outputCount: parsedOutputCount,
    validationMessage: "Ready for server validation.",
  };
}

export function analyseImportYamlDraft(yamlText: string): ImportYamlDraftAnalysis {
  return analyseImportYamlDraftFromParsed(
    yamlText,
    parseImportYamlDraft(yamlText),
  );
}

function importYamlCountLine(analysis: ImportYamlDraftAnalysis): string {
  return [
    plural(analysis.sourceCount, "source"),
    plural(analysis.stepCount, "processing step"),
    plural(analysis.outputCount, "output"),
  ].join(", ");
}

function ImportYamlDraftPreview({
  analysis,
  id,
}: {
  analysis: ImportYamlDraftAnalysis;
  id: string;
}): JSX.Element | null {
  if (!analysis.hasText) return null;
  return (
    <section
      id={id}
      className="import-yaml-preview"
      role="status"
      aria-live="polite"
      aria-label="Import YAML preflight"
    >
      {analysis.sectionsParsed && (
        <div className="import-yaml-preview-section">
          <div className="import-yaml-preview-heading">Parsed preview</div>
          <p className="import-yaml-preview-counts">
            {importYamlCountLine(analysis)}
          </p>
        </div>
      )}
      <div className="import-yaml-preview-section">
        <div className="import-yaml-preview-heading">Validation summary</div>
        <p className={analysis.sectionsParsed ? "import-yaml-preview-ok" : "import-yaml-preview-warning"}>
          {analysis.validationMessage}
        </p>
      </div>
    </section>
  );
}

function sourceBlobBindingLabel(candidate: ImportYamlSourceBindingCandidate): string {
  return `${candidate.sourceName} ${candidate.optionKey}: ${candidate.path}`;
}

function sourceBlobBindingKey(candidate: ImportYamlSourceBindingCandidate): string {
  return `${candidate.sourceName}\u0000${candidate.optionKey}\u0000${candidate.path}`;
}

function ImportYamlSourceBindings({
  candidates,
  readyBlobs,
  selectedBlobIds,
  isLoading,
  loadError,
  uploadPending,
  uploadErrors,
  onSelectBlob,
  onUploadFile,
}: {
  candidates: ImportYamlSourceBindingCandidate[];
  readyBlobs: BlobMetadata[];
  selectedBlobIds: Record<string, string>;
  isLoading: boolean;
  loadError: string | null;
  uploadPending: Record<string, boolean>;
  uploadErrors: Record<string, string>;
  onSelectBlob: (candidate: ImportYamlSourceBindingCandidate, blobId: string) => void;
  onUploadFile: (
    candidate: ImportYamlSourceBindingCandidate,
    event: React.ChangeEvent<HTMLInputElement>,
  ) => void;
}): JSX.Element | null {
  if (candidates.length === 0) return null;

  return (
    <section
      className="import-yaml-source-bindings"
      aria-label="Source file bindings"
    >
      <div className="import-yaml-source-bindings-heading">
        Source file bindings
      </div>
      {isLoading && (
        <p className="field-hint" role="status">
          Loading uploaded files...
        </p>
      )}
      {loadError && (
        <div role="alert" className="validation-banner validation-banner-fail">
          {loadError}
        </div>
      )}
      <div className="import-yaml-source-bindings-list">
        {candidates.map((candidate) => {
          const candidateKey = sourceBlobBindingKey(candidate);
          const selectedBlobId = selectedBlobIds[candidateKey] ?? "";
          const pending = Boolean(uploadPending[candidateKey]);
          const uploadError = uploadErrors[candidateKey];
          return (
            <div
              key={candidateKey}
              className="import-yaml-source-binding-row"
            >
              <div className="import-yaml-source-binding-detail">
                <div className="import-yaml-source-binding-name">
                  {candidate.sourceName}
                </div>
                <code title={sourceBlobBindingLabel(candidate)}>
                  {candidate.optionKey}: {candidate.path}
                </code>
              </div>
              <select
                className="input"
                aria-label={`Uploaded file for source ${candidate.sourceName}`}
                value={selectedBlobId}
                onChange={(event) =>
                  onSelectBlob(candidate, event.target.value)
                }
                disabled={isLoading || pending}
              >
                <option value="">No uploaded file</option>
                {readyBlobs.map((blob) => (
                  <option key={blob.id} value={blob.id}>
                    {blob.filename}
                  </option>
                ))}
              </select>
              <input
                type="file"
                className="input"
                aria-label={`Upload file for source ${candidate.sourceName}`}
                onChange={(event) => onUploadFile(candidate, event)}
                disabled={pending}
              />
              {pending && (
                <p className="field-hint" role="status">
                  Uploading...
                </p>
              )}
              {uploadError && (
                <p className="validation-banner-fail-title" role="alert">
                  {uploadError}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

/**
 * Import-YAML modal: paste or load-from-file, confirm when replacing a
 * non-trivial pipeline, submit, and render the outcome. Mounted only while
 * open (by ImportYamlModalHost), so `useFocusTrap`'s `active` is always true:
 * mount IS open, matching ExportYamlModal's focus/Escape/backdrop idiom.
 */
export function ImportYamlModal({ onClose }: ImportYamlModalProps): JSX.Element {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const compositionStateLoaded = useSessionStore((s) => s.compositionStateLoaded);
  const guidedSession = useSessionStore((s) => s.guidedSession);
  const selectSession = useSessionStore((s) => s.selectSession);
  // Mirrors isGuidedBuildActive's own null-session / null-terminal check --
  // see buildImportConfirmMessage's doc comment for why terminal excludes.
  const guidedActive = guidedSession !== null && guidedSession.terminal === null;

  const [yamlText, setYamlText] = useState("");
  const [phase, setPhase] = useState<ImportPhase>("draft");
  const [error, setError] = useState<ImportErrorInfo | null>(null);
  const [successInfo, setSuccessInfo] = useState<ImportSuccessInfo | null>(null);
  const [availableBlobs, setAvailableBlobs] = useState<BlobMetadata[]>([]);
  const [blobsLoading, setBlobsLoading] = useState(false);
  const [blobsLoadError, setBlobsLoadError] = useState<string | null>(null);
  const [sourceBlobBindings, setSourceBlobBindings] = useState<Record<string, string>>({});
  const [sourceUploadPending, setSourceUploadPending] = useState<Record<string, boolean>>({});
  const [sourceUploadErrors, setSourceUploadErrors] = useState<Record<string, string>>({});

  const dialogRef = useRef<HTMLDivElement>(null);
  const successCloseRef = useRef<HTMLButtonElement>(null);
  const currentSourceBindingKeysRef = useRef<Set<string>>(new Set());
  const titleId = useId();
  const textareaId = useId();
  const fileInputId = useId();
  const errorId = useId();
  const draftPreviewId = useId();

  useFocusTrap(dialogRef, true, ".import-yaml-textarea");

  const isSubmitting = phase === "submitting";
  // Parse the deferred text once per settled draft. The preview and source
  // binding discovery both consume this shared result; submit still sends the
  // live textarea value from yamlText.
  const deferredYamlText = useDeferredValue(yamlText);
  const isDraftAnalysisPending = yamlText !== deferredYamlText;
  const parsedDraft = useMemo(
    () => parseImportYamlDraft(deferredYamlText),
    [deferredYamlText],
  );
  const draftAnalysis = useMemo(
    () => analyseImportYamlDraftFromParsed(deferredYamlText, parsedDraft),
    [deferredYamlText, parsedDraft],
  );
  const sourceBindingCandidates = useMemo(
    () => findImportYamlSourceBindingCandidatesFromParsed(parsedDraft),
    [parsedDraft],
  );
  const sourceBindingCandidateKeys = useMemo(
    () => new Set(sourceBindingCandidates.map(sourceBlobBindingKey)),
    [sourceBindingCandidates],
  );
  currentSourceBindingKeysRef.current = sourceBindingCandidateKeys;
  const readyBlobs = useMemo(
    () => availableBlobs.filter((blob) => blob.status === "ready"),
    [availableBlobs],
  );
  const canSubmitYaml = !isDraftAnalysisPending && draftAnalysis.canImport;
  const hasSourceBindingCandidates = sourceBindingCandidates.length > 0;
  const hasPendingSourceUpload = sourceBindingCandidates.some((candidate) =>
    Boolean(sourceUploadPending[sourceBlobBindingKey(candidate)]),
  );
  // Escape/backdrop close while drafting or after a result; NOT while the
  // nested ConfirmDialog owns the keyboard (its own Escape handler applies),
  // and not mid-submit (avoid abandoning the request with no feedback).
  const canClose = phase === "draft" || phase === "success";

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && canClose) {
        e.preventDefault();
        onClose();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [canClose, onClose]);

  // On the draft→success transition the draft body (and any ConfirmDialog)
  // unmounts; ConfirmDialog's focus-trap cleanup would otherwise restore focus
  // to the now-disabled Import button, dropping focus to <body>. Move focus to
  // the success Close button so keyboard/AT users land on a real control and
  // the success status is in reading order.
  useEffect(() => {
    if (phase === "success") {
      successCloseRef.current?.focus();
    }
  }, [phase]);

  useEffect(() => {
    if (!activeSessionId || !hasSourceBindingCandidates) {
      setAvailableBlobs([]);
      setBlobsLoadError(null);
      setBlobsLoading(false);
      return;
    }

    let cancelled = false;
    setBlobsLoading(true);
    setBlobsLoadError(null);
    void api
      .listBlobs(activeSessionId)
      .then((blobs) => {
        if (!cancelled) {
          setAvailableBlobs(blobs);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const message =
            err instanceof Error ? err.message : "Could not load uploaded files.";
          setBlobsLoadError(message);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setBlobsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeSessionId, hasSourceBindingCandidates]);

  useEffect(() => {
    setSourceBlobBindings((current) => {
      const next: Record<string, string> = {};
      let changed = false;
      for (const [candidateKey, blobId] of Object.entries(current)) {
        if (sourceBindingCandidateKeys.has(candidateKey)) {
          next[candidateKey] = blobId;
        } else {
          changed = true;
        }
      }
      return changed ? next : current;
    });
    setSourceUploadPending((current) => {
      const next: Record<string, boolean> = {};
      let changed = false;
      for (const [candidateKey, pending] of Object.entries(current)) {
        if (sourceBindingCandidateKeys.has(candidateKey)) {
          next[candidateKey] = pending;
        } else {
          changed = true;
        }
      }
      return changed ? next : current;
    });
    setSourceUploadErrors((current) => {
      const next: Record<string, string> = {};
      let changed = false;
      for (const [candidateKey, message] of Object.entries(current)) {
        if (sourceBindingCandidateKeys.has(candidateKey)) {
          next[candidateKey] = message;
        } else {
          changed = true;
        }
      }
      return changed ? next : current;
    });
  }, [sourceBindingCandidateKeys]);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0];
    // Reset the input so re-selecting the same file still fires onChange.
    event.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setYamlText(reader.result);
        setError(null);
      }
    };
    reader.onerror = () => {
      setError({
        title: "Could not read the selected file.",
        detail: "Try choosing the file again, or paste the YAML directly.",
      });
    };
    reader.readAsText(file);
  }

  function handleSourceBlobSelect(
    candidate: ImportYamlSourceBindingCandidate,
    blobId: string,
  ): void {
    const candidateKey = sourceBlobBindingKey(candidate);
    setSourceBlobBindings((current) => {
      const next = { ...current };
      if (blobId) {
        next[candidateKey] = blobId;
      } else {
        delete next[candidateKey];
      }
      return next;
    });
  }

  async function handleSourceFileUpload(
    candidate: ImportYamlSourceBindingCandidate,
    event: React.ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !activeSessionId) return;
    const candidateKey = sourceBlobBindingKey(candidate);

    setSourceUploadPending((current) => ({ ...current, [candidateKey]: true }));
    setSourceUploadErrors((current) => {
      const next = { ...current };
      delete next[candidateKey];
      return next;
    });
    try {
      const blob = await api.uploadBlob(activeSessionId, file);
      setAvailableBlobs((current) => [
        blob,
        ...current.filter((candidate) => candidate.id !== blob.id),
      ]);
      if (currentSourceBindingKeysRef.current.has(candidateKey)) {
        setSourceBlobBindings((current) => ({ ...current, [candidateKey]: blob.id }));
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Could not upload source file.";
      if (currentSourceBindingKeysRef.current.has(candidateKey)) {
        setSourceUploadErrors((current) => ({ ...current, [candidateKey]: message }));
      }
    } finally {
      setSourceUploadPending((current) => {
        const next = { ...current };
        delete next[candidateKey];
        return next;
      });
    }
  }

  async function doImport(): Promise<void> {
    if (!activeSessionId) return;
    setError(null);
    setPhase("submitting");
    try {
      // Re-supply the export's source_blob_ids sidecar so a blob-backed source
      // rebinds instead of 400-ing as unbound blob storage. Gate on BOTH the
      // session and an exact YAML match: blob refs are session-scoped, and a
      // sidecar naming a source absent from an edited paste would 400. On any
      // mismatch we send no sidecar and the backend cleanly asks the user to
      // re-provide the blob. Read at submit time (not subscribed) — it only
      // matters at the moment of import.
      const binding = useSessionStore.getState().exportedYamlBlobBinding;
      const sourceBlobIds =
        binding &&
        binding.sessionId === activeSessionId &&
        binding.yaml.trim() === yamlText.trim()
          ? binding.sourceBlobIds
          : undefined;
      const explicitSourceBlobIds =
        sourceBindingCandidates.reduce<Record<string, string>>((acc, candidate) => {
          const blobId = sourceBlobBindings[sourceBlobBindingKey(candidate)];
          if (blobId) {
            acc[candidate.sourceName] = blobId;
          }
          return acc;
        }, {});
      const importSourceBlobIds = {
        ...(sourceBlobIds ?? {}),
        ...explicitSourceBlobIds,
      };
      const result = Object.keys(importSourceBlobIds).length > 0
        ? await api.importCompositionYaml(activeSessionId, yamlText, importSourceBlobIds)
        : await api.importCompositionYaml(activeSessionId, yamlText);
      setSuccessInfo({
        version: result.version,
        isValid: result.is_valid,
        validationErrors: result.validation_errors ?? [],
        pluginPolicyFindings: result.plugin_policy_findings ?? [],
      });
      setPhase("success");
      // Reuse the existing session-load refetch to sync the full canonical
      // state (messages, compositionState, guided reset to null, proposals,
      // versions) -- the same mechanism selectSession already provides for
      // any session-context change. There is no dedicated "refresh current
      // session" action to call instead; growing one was out of this
      // change's ownership (sessionStore.ts).
      void selectSession(activeSessionId);
      // Re-run validation explicitly so the rail's existing validation
      // banner actually reflects the imported state -- selectSession clears
      // validationResult (R4-H3) and nothing else repopulates it on a plain
      // refetch.
      void useExecutionStore.getState().validate(activeSessionId);
    } catch (err) {
      setError(describeImportError(err as ApiError));
      setPhase("draft");
    }
  }

  function requestDisabledComponentRemoval(finding: PluginPolicyFinding): void {
    window.dispatchEvent(
      new CustomEvent(PREFILL_CHAT_INPUT_EVENT, {
        detail: `Remove disabled component ${finding.component_id} (${finding.plugin_id}) from this pipeline.`,
      }),
    );
    onClose();
  }

  function requestDisabledComponentReplacement(): void {
    onClose();
    window.dispatchEvent(new CustomEvent(OPEN_CATALOG_EVENT));
  }

  function handleSubmitClick(): void {
    if (!canSubmitYaml) return;
    // Confirm before a destructive replace when the session HAS content, and
    // also when its state is not yet known (mid-refetch, or a failed load
    // that left compositionState null): treating "unknown" as "empty" would
    // silently replace a real pipeline with no confirmation. Only a loaded,
    // confirmed-empty composition skips the confirm step.
    if (!compositionStateLoaded || hasCompositionContent(compositionState)) {
      setPhase("confirming");
    } else {
      void doImport();
    }
  }

  return (
    <>
      <div
        className="yaml-modal-backdrop"
        data-testid="import-yaml-modal-backdrop"
        onClick={() => canClose && onClose()}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="yaml-modal"
      >
        <header className="yaml-modal-header">
          <h2 id={titleId}>Import YAML</h2>
          <button
            type="button"
            className="yaml-modal-close"
            onClick={onClose}
            disabled={!canClose}
            aria-disabled={!canClose ? true : undefined}
            aria-label="Close import YAML"
          >
            ×
          </button>
        </header>
        <div className="yaml-modal-body">
          {phase !== "success" && (
            <div className="import-yaml-body">
              {error && (
                <div id={errorId} role="alert" className="validation-banner validation-banner-fail">
                  <div className="validation-banner-fail-title">{error.title}</div>
                  <div>{error.detail}</div>
                </div>
              )}
              <div>
                <label htmlFor={textareaId} className="field-label">
                  Pipeline YAML
                </label>
                <textarea
                  id={textareaId}
                  className="textarea input-mono import-yaml-textarea"
                  value={yamlText}
                  onChange={(e) => setYamlText(e.target.value)}
                  aria-describedby={
                    [
                      error ? errorId : null,
                      draftAnalysis.hasText ? draftPreviewId : null,
                    ]
                      .filter((id): id is string => id !== null)
                      .join(" ") || undefined
                  }
                  disabled={isSubmitting}
                  placeholder="Paste exported pipeline YAML here"
                  rows={14}
                />
              </div>
              <div className="import-yaml-file-row">
                <label htmlFor={fileInputId} className="field-label">
                  Or choose a .yaml file
                </label>
                <input
                  id={fileInputId}
                  type="file"
                  accept=".yaml,.yml,text/yaml,application/x-yaml"
                  className="input"
                  onChange={handleFileChange}
                  disabled={isSubmitting}
                />
                <p className="field-hint">
                  The file is read in your browser only — nothing is
                  uploaded until you click Import.
                </p>
              </div>
              <ImportYamlDraftPreview analysis={draftAnalysis} id={draftPreviewId} />
              <ImportYamlSourceBindings
                candidates={sourceBindingCandidates}
                readyBlobs={readyBlobs}
                selectedBlobIds={sourceBlobBindings}
                isLoading={blobsLoading}
                loadError={blobsLoadError}
                uploadPending={sourceUploadPending}
                uploadErrors={sourceUploadErrors}
                onSelectBlob={handleSourceBlobSelect}
                onUploadFile={(candidate, event) => {
                  void handleSourceFileUpload(candidate, event);
                }}
              />
              <div className="import-yaml-actions">
                <button
                  type="button"
                  className="btn"
                  onClick={onClose}
                  disabled={isSubmitting}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!canSubmitYaml || isSubmitting || hasPendingSourceUpload}
                  onClick={handleSubmitClick}
                >
                  {isSubmitting ? "Importing…" : "Import"}
                </button>
              </div>
              {isSubmitting && (
                <div role="status" aria-live="polite" className="sr-only">
                  Importing your pipeline YAML.
                </div>
              )}
            </div>
          )}
          {phase === "confirming" && (
            <ConfirmDialog
              title={IMPORT_YAML_CONFIRM_TITLE}
              message={buildImportConfirmMessage(guidedActive)}
              confirmLabel={IMPORT_YAML_CONFIRM_CONFIRM_LABEL}
              cancelLabel={IMPORT_YAML_CONFIRM_CANCEL_LABEL}
              onConfirm={() => void doImport()}
              onCancel={() => setPhase("draft")}
            />
          )}
          {phase === "success" && successInfo && (
            <div className="import-yaml-body">
              <div className="validation-banner validation-banner-pass" role="status">
                {buildImportSuccessMessage(successInfo.version)}
              </div>
              {!successInfo.isValid && (
                <div className="validation-banner validation-banner-fail" role="alert">
                  <div className="validation-banner-fail-title">
                    {successInfo.validationErrors.length > 0
                      ? IMPORT_YAML_NOT_RUNNABLE_INTRO
                      : IMPORT_YAML_NOT_RUNNABLE_INTRO_NO_DETAIL}
                  </div>
                  {successInfo.validationErrors.length > 0 && (
                    <ul className="validation-banner-fail-list">
                      {successInfo.validationErrors.map((message, index) => (
                        <li key={index} className="validation-banner-error-item">
                          {message}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
              {successInfo.pluginPolicyFindings.length > 0 && (
                <section
                  role="region"
                  aria-labelledby="import-disabled-components-title"
                  className="validation-banner validation-banner-fail"
                >
                  <div
                    id="import-disabled-components-title"
                    className="validation-banner-fail-title"
                  >
                    Unavailable saved components
                  </div>
                  <ul className="validation-banner-fail-list">
                    {successInfo.pluginPolicyFindings.map((finding) => (
                      <li
                        key={`${finding.component_id}:${finding.plugin_id}`}
                        className="validation-banner-error-item"
                      >
                        <div>
                          <strong>{finding.component_id}</strong>{" "}
                          <code>{finding.plugin_id}</code> —{" "}
                          {unavailableReasonLabel(finding.reason_code)}
                        </div>
                        <div className="import-yaml-actions">
                          <button
                            type="button"
                            className="btn btn-small"
                            aria-label={`Remove disabled component ${finding.component_id} (${finding.plugin_id})`}
                            onClick={() => requestDisabledComponentRemoval(finding)}
                          >
                            Remove
                          </button>
                          <button
                            type="button"
                            className="btn btn-small"
                            aria-label={`Replace disabled component ${finding.component_id} (${finding.plugin_id}) with an available ${pluginKind(finding.plugin_id)}`}
                            onClick={requestDisabledComponentReplacement}
                          >
                            Replace
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
              <div className="import-yaml-actions">
                <button ref={successCloseRef} type="button" className="btn btn-primary" onClick={onClose}>
                  Close
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export function ImportYamlModalHost(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    function handleOpen() {
      if (useSessionStore.getState().activeSessionId) {
        setIsOpen(true);
      }
    }

    window.addEventListener(OPEN_IMPORT_YAML_MODAL_EVENT, handleOpen);
    return () =>
      window.removeEventListener(OPEN_IMPORT_YAML_MODAL_EVENT, handleOpen);
  }, []);

  useEffect(() => {
    if (!activeSessionId) {
      setIsOpen(false);
    }
  }, [activeSessionId]);

  if (!activeSessionId || !isOpen) return null;

  return <ImportYamlModal onClose={() => setIsOpen(false)} />;
}
