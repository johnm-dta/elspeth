// ============================================================================
// SideRail
//
// Right-column scaffold that hosts the Phase 3 side-rail affordances. Slots are
// explicit ReactNode props supplied by the caller; SideRail only places them.
// ============================================================================

import { type ReactNode } from "react";

interface SideRailProps {
  auditReadinessSlot?: ReactNode | null;
  validationBannerSlot?: ReactNode | null;
  graphMiniSlot?: ReactNode | null;
  catalogSlot?: ReactNode | null;
  completionBarSlot?: ReactNode | null;
}

export function SideRail({
  auditReadinessSlot = null,
  validationBannerSlot = null,
  graphMiniSlot = null,
  catalogSlot = null,
  completionBarSlot = null,
}: SideRailProps): JSX.Element {
  return (
    <aside className="side-rail" aria-label="Composer side rail">
      <div data-testid="siderail-slot-audit-readiness" className="side-rail-slot">
        {auditReadinessSlot}
      </div>
      <div data-testid="siderail-slot-validation-banner" className="side-rail-slot">
        {validationBannerSlot}
      </div>
      <div data-testid="siderail-slot-graph-mini" className="side-rail-slot">
        {graphMiniSlot}
      </div>
      <div data-testid="siderail-slot-completion-bar" className="side-rail-slot">
        {completionBarSlot}
      </div>
      <div data-testid="siderail-slot-catalog" className="side-rail-slot">
        {catalogSlot}
      </div>
    </aside>
  );
}
