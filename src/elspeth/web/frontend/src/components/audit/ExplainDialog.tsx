/**
 * ExplainDialog (Phase 2C)
 *
 * Modal dialog rendering the narrative explanation of what the current
 * pipeline will record. The narrative is fetched lazily on first open
 * and cached by composition_version in the auditReadinessStore.
 *
 * Design spec: docs/composer/ux-redesign-2026-05/07-audit-readiness-panel.md
 * §"The Explain view".
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
  const explain = useAuditReadinessStore((s) => s.explainsBySession[sessionId]);
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
  // restoration on unmount. Escape is handled by a separate onKeyDown because
  // useFocusTrap does not register an Escape listener (matches CommandPalette
  // pattern: onKeyDown Escape → onClose). active=true unconditionally because
  // ExplainDialog renders only while open.
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, true, ".explain-dialog-close");

  useEffect(() => {
    void loadExplain(sessionId, compositionVersion);
  }, [sessionId, compositionVersion, loadExplain]);

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      className="explain-dialog"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          onClose();
        }
      }}
    >
      <div className="explain-dialog-backdrop" onClick={onClose} aria-hidden="true" />
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
  );
}
