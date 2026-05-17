// ============================================================================
// SideRail
//
// Right-column scaffold that hosts the Phase 3 side-rail affordances. Slots are
// explicit ReactNode props supplied by the caller; SideRail only places them.
// ============================================================================

import { type ReactNode } from "react";

interface SideRailProps {
  auditReadinessSlot?: ReactNode | null;
  graphMiniSlot?: ReactNode | null;
  catalogSlot?: ReactNode | null;
  exportYamlSlot?: ReactNode | null;
  executeButtonSlot?: ReactNode | null;
  completionBarSlot?: ReactNode | null;
  children?: ReactNode;
}

export function SideRail({
  auditReadinessSlot = null,
  graphMiniSlot = null,
  catalogSlot = null,
  exportYamlSlot = null,
  executeButtonSlot = null,
  completionBarSlot = null,
  children,
}: SideRailProps): JSX.Element {
  return (
    <aside className="side-rail" aria-label="Composer side rail">
      <div data-testid="siderail-slot-audit-readiness" className="side-rail-slot">
        {auditReadinessSlot}
      </div>
      <div data-testid="siderail-slot-graph-mini" className="side-rail-slot">
        {graphMiniSlot}
      </div>
      <div data-testid="siderail-slot-catalog" className="side-rail-slot">
        {catalogSlot}
      </div>
      <div data-testid="siderail-slot-export-yaml" className="side-rail-slot">
        {exportYamlSlot}
      </div>
      <div data-testid="siderail-slot-execute-button" className="side-rail-slot">
        {executeButtonSlot}
      </div>
      {children && <div className="side-rail-transitional">{children}</div>}
      <div data-testid="siderail-slot-completion-bar" className="side-rail-slot">
        {completionBarSlot}
      </div>
    </aside>
  );
}
