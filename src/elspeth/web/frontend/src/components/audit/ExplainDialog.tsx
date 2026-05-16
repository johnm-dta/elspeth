/**
 * Placeholder shipped by 14b. 14c replaces this with the full implementation
 * (modal narrative view with proper dialog semantics: focus trap, aria-modal,
 * initial focus, focus restoration, escape-to-close, backdrop dismiss).
 *
 * This placeholder is intentionally NOT a dialog. It renders a minimal stub
 * tagged with `data-testid="explaindialog-placeholder"` so the panel's tests
 * can assert the component mounted, lint/typecheck pass, and the W2
 * accessibility defect (role="dialog" without focus management) does not ship.
 * The loadExplain side-effect is wired here so the parent's lazy-fetch flow
 * is already exercised — 14c only needs to add the modal semantics.
 *
 * DO NOT extend the placeholder. Extensions belong in 14c.
 */
import { useEffect } from "react";

import { useAuditReadinessStore } from "../../stores/auditReadinessStore";

export interface ExplainDialogProps {
  sessionId: string;
  compositionVersion: number;
  onClose: () => void;
}

export function ExplainDialog({ sessionId, compositionVersion, onClose }: ExplainDialogProps) {
  const explain = useAuditReadinessStore((s) => s.explainsBySession[sessionId]);
  const loadExplain = useAuditReadinessStore((s) => s.loadExplain);

  useEffect(() => {
    void loadExplain(sessionId, compositionVersion);
  }, [sessionId, compositionVersion, loadExplain]);

  return (
    <div
      data-testid="explaindialog-placeholder"
      aria-label="What this pipeline will record"
    >
      {explain && <pre>{explain.narrative}</pre>}
      <button type="button" onClick={onClose}>
        Close
      </button>
    </div>
  );
}
