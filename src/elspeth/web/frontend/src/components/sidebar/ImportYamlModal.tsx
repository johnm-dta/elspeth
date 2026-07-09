import { useEffect, useId, useMemo, useRef, useState } from "react";
import * as api from "@/api/client";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { ApiError } from "@/types/index";
import { hasCompositionContent } from "@/utils/compositionState";

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
}

type ImportPhase = "draft" | "confirming" | "submitting" | "success";

interface ImportYamlModalProps {
  onClose: () => void;
}

interface ImportYamlDraftAnalysis {
  hasText: boolean;
  canImport: boolean;
  sourceCount: number;
  stepCount: number;
  outputCount: number;
  validationMessage: string;
}

type ImportYamlSection =
  | "aggregations"
  | "coalesce"
  | "gates"
  | "sinks"
  | "source"
  | "sources"
  | "transforms";

const IMPORT_YAML_SECTION_ALIASES: Record<string, ImportYamlSection> = {
  aggregations: "aggregations",
  coalesce: "coalesce",
  gates: "gates",
  sinks: "sinks",
  source: "source",
  sources: "sources",
  transforms: "transforms",
};

export function analyseImportYamlDraft(yamlText: string): ImportYamlDraftAnalysis {
  const hasText = yamlText.trim().length > 0;
  if (!hasText) {
    return {
      hasText: false,
      canImport: false,
      sourceCount: 0,
      stepCount: 0,
      outputCount: 0,
      validationMessage: "Paste YAML to preview it.",
    };
  }

  const lines = normaliseYamlPreviewIndent(yamlText.replace(/\r\n/g, "\n").split("\n"));
  const sectionStarts: Array<{
    section: ImportYamlSection;
    lineIndex: number;
    indent: number;
    inlineValue: string;
  }> = [];

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index];
    if (leadingSpaces(rawLine) !== 0) continue;
    const trimmed = rawLine.trim();
    if (trimmed.length === 0 || trimmed.startsWith("#")) continue;
    const match = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(trimmed);
    if (!match) continue;
    const section = IMPORT_YAML_SECTION_ALIASES[match[1]];
    if (section === undefined) continue;
    sectionStarts.push({
      section,
      lineIndex: index,
      indent: leadingSpaces(rawLine),
      inlineValue: match[2].trim(),
    });
  }

  let sourceCount = 0;
  let stepCount = 0;
  let outputCount = 0;
  const foundSections = new Set<ImportYamlSection>();
  for (const start of sectionStarts) {
    foundSections.add(start.section);
    if (start.section === "source") {
      sourceCount += 1;
    } else if (start.section === "sources") {
      sourceCount += countYamlSectionEntries(lines, start);
    } else if (start.section === "sinks") {
      outputCount += countYamlSectionEntries(lines, start);
    } else {
      stepCount += countYamlSectionEntries(lines, start);
    }
  }

  if (foundSections.size === 0) {
    return {
      hasText: true,
      canImport: false,
      sourceCount: 0,
      stepCount: 0,
      outputCount: 0,
      validationMessage:
        "No pipeline sections found. Add source, transforms, gates, aggregations, coalesce, or sinks.",
    };
  }

  return {
    hasText: true,
    canImport: true,
    sourceCount,
    stepCount,
    outputCount,
    validationMessage: "Ready for server validation.",
  };
}

function leadingSpaces(value: string): number {
  return value.length - value.trimStart().length;
}

function normaliseYamlPreviewIndent(lines: string[]): string[] {
  const significantIndents = lines
    .map((line) => ({ line, trimmed: line.trim() }))
    .filter(
      ({ trimmed }) =>
        trimmed.length > 0 &&
        !trimmed.startsWith("#") &&
        trimmed !== "---" &&
        trimmed !== "...",
    )
    .map(({ line }) => leadingSpaces(line));
  if (significantIndents.length === 0) {
    return lines;
  }
  const commonIndent = Math.min(...significantIndents);
  if (commonIndent === 0) {
    return lines;
  }
  return lines.map((line) => {
    if (line.trim().length === 0) {
      return line;
    }
    return line.slice(Math.min(commonIndent, leadingSpaces(line)));
  });
}

function countYamlSectionEntries(
  lines: string[],
  start: {
    lineIndex: number;
    indent: number;
    inlineValue: string;
    section: ImportYamlSection;
  },
): number {
  if (start.inlineValue.length > 0) {
    if (start.inlineValue === "{}" || start.inlineValue === "[]") {
      return 0;
    }
    return 1;
  }

  let count = 0;
  let childIndent: number | null = null;
  for (let index = start.lineIndex + 1; index < lines.length; index += 1) {
    const rawLine = lines[index];
    const trimmed = rawLine.trim();
    if (trimmed.length === 0 || trimmed.startsWith("#")) continue;
    const indent = leadingSpaces(rawLine);
    if (indent <= start.indent) break;
    if (childIndent !== null && indent !== childIndent) continue;

    if (isRuntimeListSection(start.section)) {
      if (trimmed.startsWith("- ")) {
        childIndent = indent;
        count += 1;
      } else if (childIndent === null && /^[A-Za-z0-9_.-]+:\s*/.test(trimmed)) {
        childIndent = indent;
        count += 1;
      }
      continue;
    }

    if (/^[A-Za-z0-9_.-]+:\s*/.test(trimmed)) {
      childIndent = indent;
      count += 1;
    }
  }
  return count;
}

function isRuntimeListSection(section: ImportYamlSection): boolean {
  return (
    section === "aggregations" ||
    section === "coalesce" ||
    section === "gates" ||
    section === "transforms"
  );
}

function importYamlCountLine(analysis: ImportYamlDraftAnalysis): string {
  return [
    pluraliseCount(analysis.sourceCount, "source"),
    pluraliseCount(analysis.stepCount, "processing step"),
    pluraliseCount(analysis.outputCount, "output"),
  ].join(", ");
}

function pluraliseCount(count: number, singular: string): string {
  return `${count} ${count === 1 ? singular : `${singular}s`}`;
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
      {analysis.canImport && (
        <div className="import-yaml-preview-section">
          <div className="import-yaml-preview-heading">Parsed preview</div>
          <p className="import-yaml-preview-counts">
            {importYamlCountLine(analysis)}
          </p>
        </div>
      )}
      <div className="import-yaml-preview-section">
        <div className="import-yaml-preview-heading">Validation summary</div>
        <p className={analysis.canImport ? "import-yaml-preview-ok" : "import-yaml-preview-warning"}>
          {analysis.validationMessage}
        </p>
      </div>
    </section>
  );
}

/**
 * Import-YAML modal: paste or load-from-file, confirm when replacing a
 * non-trivial pipeline, submit, and render the outcome. Mounted only while
 * open (by ImportYamlButton), so `useFocusTrap`'s `active` is always true --
 * mount IS open, matching ExportYamlModal's focus/Escape/backdrop idiom but
 * without the always-mounted + internal-isOpen shape (see ImportYamlButton
 * for why).
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

  const dialogRef = useRef<HTMLDivElement>(null);
  const successCloseRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const textareaId = useId();
  const fileInputId = useId();
  const errorId = useId();
  const draftPreviewId = useId();

  useFocusTrap(dialogRef, true, ".import-yaml-textarea");

  const isSubmitting = phase === "submitting";
  const draftAnalysis = useMemo(
    () => analyseImportYamlDraft(yamlText),
    [yamlText],
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
      const result = sourceBlobIds
        ? await api.importCompositionYaml(activeSessionId, yamlText, sourceBlobIds)
        : await api.importCompositionYaml(activeSessionId, yamlText);
      setSuccessInfo({
        version: result.version,
        isValid: result.is_valid,
        validationErrors: result.validation_errors ?? [],
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

  function handleSubmitClick(): void {
    if (!draftAnalysis.canImport) return;
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
                  disabled={!draftAnalysis.canImport || isSubmitting}
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
