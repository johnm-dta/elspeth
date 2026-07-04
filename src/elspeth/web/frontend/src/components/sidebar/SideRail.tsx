// ============================================================================
// SideRail
//
// Right-column scaffold that hosts the Phase 3 side-rail affordances. Slots are
// explicit ReactNode props supplied by the caller; SideRail only places them.
// ============================================================================

import { type ReactNode } from "react";
import { ImportYamlButton } from "./ImportYamlButton";

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
      {/* elspeth-24c56585f9 T-1: mounted directly (not as a caller-fed slot)
          because its natural home -- alongside Export YAML inside
          CompletionBar -- and CompletionBar's own call site in App.tsx are
          both outside this change's file ownership. Placed adjacent to the
          completion-bar slot (where Export actually lives) and above
          Catalog, so it reads as part of the same compose/export/import
          cluster rather than the reference-browsing group below it. */}
      <div data-testid="siderail-slot-import-yaml" className="side-rail-slot">
        <ImportYamlButton />
      </div>
      <div data-testid="siderail-slot-catalog" className="side-rail-slot">
        {catalogSlot}
      </div>
    </aside>
  );
}
