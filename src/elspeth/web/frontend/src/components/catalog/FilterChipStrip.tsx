// ============================================================================
// FilterChipStrip
//
// Two groups of filter chips (capability tags and audit characteristics)
// at the top of each catalog tab. Filters compose with each other and
// with search via AND: a plugin must match the search query AND have a
// tag in every active group it appears in. Empty group = "all."
//
// Per design doc 08-§Filters: "The filter strip lets users narrow the
// catalog to 'what works for my sensitive-data pipeline' or 'what
// doesn't make a network call' in one click."
// ============================================================================

import { useCallback } from "react";
import { lookupAuditCharacteristic } from "./auditCharacteristics";

// Trust tier is kind-derived internal metadata (see 16b OD-C rationale):
// it is not surfaced as a filter dimension in the catalog UI.
export interface CatalogFilters {
  capabilityTags: Set<string>;
  auditCharacteristics: Set<string>;
}

interface FilterChipStripProps {
  availableCapabilityTags: string[];
  availableAuditCharacteristics: string[];
  filters: CatalogFilters;
  onChange: (next: CatalogFilters) => void;
}

function toggle<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

export function FilterChipStrip({
  availableCapabilityTags,
  availableAuditCharacteristics,
  filters,
  onChange,
}: FilterChipStripProps) {
  const anyActive =
    filters.capabilityTags.size > 0 ||
    filters.auditCharacteristics.size > 0;

  const toggleTag = useCallback(
    (tag: string) => onChange({ ...filters, capabilityTags: toggle(filters.capabilityTags, tag) }),
    [filters, onChange],
  );
  const toggleAudit = useCallback(
    (flag: string) => onChange({ ...filters, auditCharacteristics: toggle(filters.auditCharacteristics, flag) }),
    [filters, onChange],
  );
  const clear = useCallback(
    () =>
      onChange({
        capabilityTags: new Set(),
        auditCharacteristics: new Set(),
      }),
    [onChange],
  );

  return (
    // role="group" so "Catalog filters" is exposed as the strip's accessible
    // name — aria-label on a role-less div (role=generic) is ignored by AT
    // (WCAG 1.3.1, elspeth-37293a3b7c).
    <div className="filter-chip-strip" role="group" aria-label="Catalog filters">
      {availableCapabilityTags.length > 0 && (
        <ChipGroup label="Capability">
          {availableCapabilityTags.map((tag) => (
            <Chip
              key={tag}
              active={filters.capabilityTags.has(tag)}
              onToggle={() => toggleTag(tag)}
              label={tag}
            />
          ))}
        </ChipGroup>
      )}
      {availableAuditCharacteristics.length > 0 && (
        <ChipGroup label="Audit">
          {availableAuditCharacteristics.map((flag) => {
            const meta = lookupAuditCharacteristic(flag);
            const label = meta?.label ?? flag;
            return (
              <Chip
                key={flag}
                active={filters.auditCharacteristics.has(flag)}
                onToggle={() => toggleAudit(flag)}
                label={label}
              />
            );
          })}
        </ChipGroup>
      )}
      {anyActive && (
        <button
          type="button"
          className="filter-chip-clear"
          onClick={clear}
          aria-label="Clear filters"
        >
          Clear filters
        </button>
      )}
    </div>
  );
}

function ChipGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="filter-chip-group">
      <span className="filter-chip-group-label">{label}:</span>
      <div className="filter-chip-row">{children}</div>
    </div>
  );
}

function Chip({ active, onToggle, label }: { active: boolean; onToggle: () => void; label: string }) {
  return (
    <button
      type="button"
      className={`filter-chip ${active ? "filter-chip-active" : ""}`}
      aria-pressed={active}
      onClick={onToggle}
    >
      {label}
    </button>
  );
}
