/**
 * ExplainDialog (Phase 2C)
 *
 * Modal dialog rendering the narrative explanation of what the current
 * pipeline will record. The narrative is fetched lazily on first open
 * and cached by composition_version in the auditReadinessStore.
 *
 * Design spec: docs/composer/ux-redesign-2026-05/07-audit-readiness-panel.md
 * §"The Explain view".
 *
 * Modal pattern matches SecretsPanel: backdrop and dialog are siblings
 * inside a fragment, Escape is handled at the document level via a
 * useEffect-registered listener (more robust than onKeyDown on the
 * dialog div — fires even if focus escapes the trap).
 */
import { useEffect, useId, useRef } from "react";

import { useAuditReadinessStore } from "../../stores/auditReadinessStore";
import { useFocusTrap } from "@/hooks/useFocusTrap";

export interface ExplainDialogProps {
  sessionId: string;
  compositionVersion: number;
  onClose: () => void;
}

export function ExplainDialog({
  sessionId,
  compositionVersion,
  onClose,
}: ExplainDialogProps) {
  // Store fields are per-session-keyed maps, NOT flat. Reading
  // `s.isLoadingExplain` / `s.explainError` would evaluate to `undefined`
  // at runtime — the dialog would never show loading or error. The correct
  // accessors key by sessionId.
  const explain = useAuditReadinessStore((s) => {
    const cached = s.explainsBySession[sessionId];
    return cached?.composition_version === compositionVersion ? cached : undefined;
  });
  const isLoading = useAuditReadinessStore(
    (s) => s.isLoadingExplainBySession[sessionId] ?? false,
  );
  const error = useAuditReadinessStore(
    (s) => s.explainErrorBySession[sessionId] ?? null,
  );
  const loadExplain = useAuditReadinessStore((s) => s.loadExplain);
  const titleId = useId();

  // Focus contract: trap focus inside the dialog, restore to opener on close.
  // useFocusTrap handles: Tab-wrap, initial focus (Close button), and focus
  // restoration on unmount. active=true unconditionally because
  // ExplainDialog renders only while open.
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, true, ".explain-dialog-close");

  useEffect(() => {
    void loadExplain(sessionId, compositionVersion);
  }, [sessionId, compositionVersion, loadExplain]);

  // Escape closes the dialog. Registered on `document` (not on the
  // dialog div via onKeyDown) so it fires even if focus has escaped
  // the trap — defence against focus drift bugs in nested content.
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <>
      {/* Backdrop — sibling of the dialog, NOT nested inside it.
          Nesting confuses the modal's a11y tree and prevents some
          screen readers from recognising the dialog boundary. */}
      <div
        className="explain-dialog-backdrop"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="explain-dialog"
      >
        <div className="explain-dialog-content">
          <header className="explain-dialog-header">
            <h2 id={titleId} className="explain-dialog-title">
              What this pipeline will record
            </h2>
            <button
              type="button"
              className="explain-dialog-close"
              onClick={onClose}
              aria-label="Close"
            >
              ×
            </button>
          </header>

          {isLoading && !explain && (
            <p className="explain-dialog-loading">Generating explanation…</p>
          )}

          {error && !explain && (
            <div role="alert" className="explain-dialog-error">
              {error}
            </div>
          )}

          {explain && (
            <pre className="explain-dialog-narrative">{explain.narrative}</pre>
          )}
        </div>
      </div>
    </>
  );
}
