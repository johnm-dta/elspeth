// ============================================================================
// InlineSourceCreatedTurn.tsx — Phase 5a Task 3
//
// Informational confirmation widget rendered after an inline-blob source is
// attached to the composition state. Surfaces enough provenance to let the
// operator decide whether to amend (Edit the list) or proceed; never replaces
// the full blob content, which lives behind the authenticated blob endpoint.
//
// Design constraints (load-bearing — do not change without re-reading the
// Phase 5a Task 3 spec):
//
//   * Root element is `<section role="region" aria-label="Source created
//     from your message">`. AT users navigate to "Source created" by region
//     name (F-18). The aria-label substring "source created" is matched by
//     `InlineSourceCreatedTurn.test.tsx` — keep it stable.
//
//   * Audit fields (blob id, SHA-256 hash) are wrapped in `<details>` with
//     a `<summary>Show audit info</summary>` (F-21). The hash MUST be absent
//     from the rendered DOM before the disclosure is opened — testing-library
//     queries fail otherwise because react-testing-library does match through
//     `<details>` contents. Use a runtime `hidden`-attribute gate on the
//     hash text rather than CSS display:none.
//
//   * Content preview is clipped to 280 chars INCLUDING the ellipsis. The
//     test asserts `textContent.length <= 280`. The blob's full payload is
//     never delivered to the widget — `InlineSourceSummary.contentPreview`
//     is the wiring's responsibility to populate with a bounded slice.
//
//   * The Edit-the-list affordance is gated by provenance. Verbatim and
//     disambiguated sources reflect content the user already authored or
//     explicitly approved; surfacing an Edit button there would suggest a
//     re-authoring step that doesn't make sense. LLM-authored content
//     (llm-generated, llm-generated-then-amended) is the case where the
//     user may want to amend before continuing — that's F-4.
// ============================================================================

import { useState } from "react";
import type { InlineSourceSummary, InlineSourceProvenance } from "@/types/api";

/**
 * Provenances where the "Edit the list" affordance is surfaced.
 *
 * CLOSED LIST — the two LLM-authored modalities. Verbatim content is
 * user-typed (no re-authoring step needed); disambiguated content was
 * already user-confirmed during the disambiguation turn. Widening the set
 * means widening the spec — re-read Phase 5a Task 3 §"Edit the list" first.
 */
const EDITABLE_PROVENANCES: ReadonlySet<InlineSourceProvenance> = new Set([
  "llm-generated",
  "llm-generated-then-amended",
]);

/** Max chars in the visible preview, INCLUDING the trailing ellipsis. */
const PREVIEW_MAX_LENGTH = 280;
const PREVIEW_ELLIPSIS = "…"; // single-char "…" so we don't bust the budget

function clipPreview(text: string): string {
  if (text.length <= PREVIEW_MAX_LENGTH) {
    return text;
  }
  return text.slice(0, PREVIEW_MAX_LENGTH - 1) + PREVIEW_ELLIPSIS;
}

function describeRowCount(rowCount: number | null): string {
  if (rowCount === null) return "unknown row count";
  if (rowCount === 1) return "1 row";
  return `${rowCount} rows`;
}

export interface InlineSourceCreatedTurnProps {
  summary: InlineSourceSummary;
  /**
   * Invoked when the user clicks "Edit the list". The handler receives the
   * full summary so the modal/textarea can pre-fill from `contentPreview`
   * (full payload may need a separate fetch — that's the caller's
   * responsibility, not the widget's).
   */
  onEdit: (summary: InlineSourceSummary) => void;
}

export function InlineSourceCreatedTurn({
  summary,
  onEdit,
}: InlineSourceCreatedTurnProps): JSX.Element {
  // Disclosure state is local to the widget — re-mounting (e.g. when the
  // inline source is replaced) collapses the audit info again, which is the
  // desired default-closed behaviour per F-21.
  const [auditOpen, setAuditOpen] = useState(false);

  const clippedPreview = clipPreview(summary.contentPreview);
  const showEdit = EDITABLE_PROVENANCES.has(summary.provenance);
  const rowCountLabel = describeRowCount(summary.rowCount);

  return (
    <section
      role="region"
      aria-label="Source created from your message"
      data-testid="inline-source-created-turn"
      className="inline-source-created-turn"
    >
      <header className="inline-source-created-turn-header">
        <dl className="inline-source-created-turn-facts">
          <div>
            <dt>Filename</dt>
            <dd>{summary.filename}</dd>
          </div>
          <div>
            <dt>Type</dt>
            <dd>{summary.mimeType}</dd>
          </div>
          <div>
            <dt>Rows</dt>
            <dd>{rowCountLabel}</dd>
          </div>
        </dl>
      </header>

      <div className="inline-source-created-turn-content">
        <h4 className="inline-source-created-turn-content-title">
          Source contents
        </h4>
        <pre
          data-testid="inline-source-preview"
          className="inline-source-created-turn-preview"
        >
          {clippedPreview}
        </pre>
      </div>

      {showEdit && (
        <div className="inline-source-created-turn-actions">
          <button
            type="button"
            className="inline-source-created-turn-edit"
            onClick={() => onEdit(summary)}
          >
            Edit the list
          </button>
        </div>
      )}

      {/*
        Audit-info disclosure (F-21).  Rendered as a native <details> for
        keyboard + AT support, but the hash text is gated by `auditOpen` so
        it is genuinely absent from the DOM until the user opens the
        disclosure.  Without the runtime gate, testing-library queries
        (and AT readers walking the accessibility tree) would surface the
        hash even when visually collapsed — the spec is "hash MUST NOT be
        visible before disclosure is opened", not "hash should be visually
        hidden".

        Why a summary `onClick` handler in addition to (rather than instead
        of) the native <details> `open` toggle: jsdom does not synthesise the
        `toggle` event when a user-event clicks on `<summary>`, so an
        `onToggle`-only implementation would never flip `auditOpen` under
        test.  Driving the gate from the summary click keeps the
        keyboard/AT semantics of <details> (Enter / Space still toggle
        natively) while remaining test-observable.  We call
        `event.preventDefault()` so the native open-state and our state
        stay aligned via our setState call — without it, the browser would
        toggle `open` AFTER our setState ran on the next paint, producing
        a one-frame mismatch on real DOM.
      */}
      <details
        className="inline-source-created-turn-audit"
        open={auditOpen}
      >
        <summary
          onClick={(event) => {
            event.preventDefault();
            setAuditOpen((prev) => !prev);
          }}
        >
          Show audit info
        </summary>
        {auditOpen && (
          <dl className="inline-source-created-turn-audit-facts">
            <div>
              <dt>Blob ID</dt>
              <dd>
                <code>{summary.blobId}</code>
              </dd>
            </div>
            <div>
              <dt>SHA-256</dt>
              <dd>
                <code>{summary.contentHash}</code>
              </dd>
            </div>
            <div>
              <dt>Provenance</dt>
              <dd>{summary.provenance}</dd>
            </div>
          </dl>
        )}
      </details>
    </section>
  );
}
