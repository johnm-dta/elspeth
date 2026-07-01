// ============================================================================
// PluginCard (Phase 7B — reference, not toolkit)
//
// Renders one plugin as a reference card per design doc
// 08-§"Plugin card content design":
//
//   ┌──────────────────────────────────────────────────┐
//   │  csv                                             │
//   │  Read rows from a CSV file. ...                  │
//   │                                                  │
//   │  Audit: ✓ reads I/O  ✓ quarantines  ✓ coerces   │
//   │                                                  │
//   │  When you'd use this:   ...                      │
//   │  When you wouldn't:     ...                      │
//   │  Example use:           <pre>...</pre>           │
//   │                                                  │
//   │  [ Schema → ]                                    │
//   └──────────────────────────────────────────────────┘
//
// The "Use in pipeline" button and supporting machinery (from the
// pre-Phase-7B toolkit framing) are deliberately removed. The
// PREFILL_CHAT_INPUT_EVENT export stays — InlineChatSourceEntry uses it
// for its prefill action, and ChatInput.tsx remains the receiver.
// ============================================================================

import { useState, type MouseEvent } from "react";
import type { PluginSummary, PluginSchemaInfo } from "@/types/index";
import { AuditCharacteristicIcon } from "./AuditCharacteristicIcon";

/** Event name dispatched by InlineChatSourceEntry and consumed by
 *  ChatInput.tsx. Re-exported here for backwards compatibility with
 *  the existing import chain in ChatInput.tsx; do not rename. */
export const PREFILL_CHAT_INPUT_EVENT = "composer:prefill-chat-input";

interface JsonSchemaField {
  type?: string;
  description?: string;
}
interface JsonSchemaObject {
  properties?: Record<string, JsonSchemaField>;
  required?: string[];
}
interface DiscriminatedSchema {
  oneOf?: Array<{ $ref?: string }>;
  discriminator?: { propertyName?: string; mapping?: Record<string, string> };
  $defs?: Record<string, JsonSchemaObject>;
}

const DEFS_REF_PREFIX = "#/$defs/";
function pluginCardIdSegment(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "unnamed";
}

interface PluginCardProps {
  plugin: PluginSummary;
  schema: PluginSchemaInfo | null;
  schemaError?: boolean;
  onExpand: () => void;
  onRetrySchema?: () => void;
  /** Test-only: start in the expanded state to assert schema rendering. */
  initialExpanded?: boolean;
}

function isDiscriminated(s: DiscriminatedSchema & JsonSchemaObject): boolean {
  return Array.isArray(s.oneOf) && s.$defs !== undefined;
}

function resolveVariants(s: DiscriminatedSchema): Array<{ label: string; def: JsonSchemaObject }> {
  const defs = s.$defs ?? {};
  const mapping = s.discriminator?.mapping ?? {};
  const refToValue = new Map<string, string>();
  for (const [discValue, ref] of Object.entries(mapping)) {
    if (ref.startsWith(DEFS_REF_PREFIX)) {
      refToValue.set(ref.slice(DEFS_REF_PREFIX.length), discValue);
    }
  }
  const discProp = s.discriminator?.propertyName ?? "variant";
  const out: Array<{ label: string; def: JsonSchemaObject }> = [];
  for (const entry of s.oneOf ?? []) {
    const ref = entry.$ref ?? "";
    if (!ref.startsWith(DEFS_REF_PREFIX)) continue;
    const defName = ref.slice(DEFS_REF_PREFIX.length);
    const def = defs[defName];
    if (def === undefined) continue;
    out.push({ label: `${discProp}: ${refToValue.get(defName) ?? defName}`, def });
  }
  return out;
}

function variantKindLabel(s: DiscriminatedSchema): string {
  const name = s.discriminator?.propertyName?.trim() || "variant";
  return name.endsWith("s") ? name : `${name}s`;
}

function renderFields(properties: Record<string, JsonSchemaField>, required: string[] | undefined): JSX.Element[] {
  const req = new Set(required ?? []);
  return Object.entries(properties).map(([name, field]) => (
    <div key={name}>
      <span className="plugin-card-field-name">{name}</span>
      <span className="plugin-card-field-type">{field.type ?? "any"}</span>
      {req.has(name) && <span className="plugin-card-field-required">required</span>}
      {field.description && <div className="plugin-card-field-desc">{field.description}</div>}
    </div>
  ));
}

const PROSE_FALLBACK = "See the technical description above.";

export function PluginCard({
  plugin,
  schema,
  schemaError,
  onExpand,
  onRetrySchema,
  initialExpanded = false,
}: PluginCardProps) {
  const [expanded, setExpanded] = useState(initialExpanded);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const cardId = `${pluginCardIdSegment(plugin.plugin_type)}-${pluginCardIdSegment(plugin.name)}`;
  const nameId = `plugin-card-name-${cardId}`;
  const detailsPanelId = `plugin-card-details-panel-${cardId}`;
  const schemaPanelId = `plugin-card-schema-panel-${cardId}`;

  function handleDisclosureClick(e: MouseEvent<HTMLButtonElement>) {
    e.preventDefault();
    if (!expanded) onExpand();
    setExpanded((p) => !p);
  }

  function handleRetry(e: MouseEvent<HTMLButtonElement>) {
    e.stopPropagation();
    (onRetrySchema ?? onExpand)();
  }

  const configSchema = schema?.json_schema as (DiscriminatedSchema & JsonSchemaObject) | undefined;

  const allFallback =
    plugin.usage_when_to_use === null &&
    plugin.usage_when_not_to_use === null &&
    plugin.example_use === null;

  return (
    // role="article" + aria-labelledby promotes the card to a named region
    // (WCAG 1.3.1): the plugin name reads as the card's accessible name
    // instead of the name being an undifferentiated bold span.
    <div className="plugin-card" role="article" aria-labelledby={nameId}>
      <div className="plugin-card-header-row">
        <span id={nameId} className="plugin-card-name">{plugin.name}</span>
        <span className="plugin-card-kind">{plugin.plugin_type}</span>
      </div>

      <div className="plugin-card-desc" title={plugin.description}>
        {plugin.description}
      </div>

      {plugin.audit_characteristics.length > 0 && (
        <div className="plugin-card-audit-strip" aria-label="Audit characteristics">
          {[...plugin.audit_characteristics].sort().map((flag) => (
            <AuditCharacteristicIcon key={flag} flag={flag} />
          ))}
        </div>
      )}

      <div className="plugin-card-actions">
        <button
          type="button"
          className="btn btn-small plugin-card-detail-toggle"
          onClick={() => setDetailsOpen((open) => !open)}
          aria-expanded={detailsOpen}
          aria-controls={detailsPanelId}
          aria-label={`Reference details for ${plugin.name}`}
        >
          Details
        </button>
        <button
          type="button"
          className="btn btn-small plugin-card-disclosure"
          onClick={handleDisclosureClick}
          aria-expanded={expanded}
          aria-controls={schemaPanelId}
          aria-label={`Schema for ${plugin.name}`}
        >
          Schema
        </button>
      </div>

      {detailsOpen && (
        <div id={detailsPanelId} className="plugin-card-details">
          {allFallback ? (
            <div className="plugin-card-prose-fallback">{PROSE_FALLBACK}</div>
          ) : (
            <>
              <ProseSection label="Use when" body={plugin.usage_when_to_use} />
              <ProseSection label="Avoid when" body={plugin.usage_when_not_to_use} />
              {plugin.example_use !== null && (
                <div className="plugin-card-example">
                  <div className="plugin-card-example-label">Example</div>
                  <pre className="plugin-card-example-code">{plugin.example_use}</pre>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {expanded && (
        <div id={schemaPanelId} className="plugin-card-expanded">
          {schemaError ? (
            <div className="plugin-card-schema-error">
              <span>Failed to load schema.</span>
              <button type="button" className="btn btn-small" onClick={handleRetry} aria-label="Retry loading schema">
                Retry
              </button>
            </div>
          ) : schema === null || configSchema === undefined ? (
            <div role="status" aria-live="polite" className="plugin-card-schema-loading">
              <span className="spinner" aria-hidden="true" /> Loading schema...
            </div>
          ) : isDiscriminated(configSchema) ? (
            <div className="plugin-card-variants">
              <div className="plugin-card-variants-hint">
                This plugin supports multiple {variantKindLabel(configSchema)}. Configure exactly one:
              </div>
              {resolveVariants(configSchema).map((v) => (
                <div key={v.label} className="plugin-card-variant">
                  <div className="plugin-card-variant-label">{v.label}</div>
                  {v.def.properties ? (
                    <div className="plugin-card-fields">{renderFields(v.def.properties, v.def.required)}</div>
                  ) : (
                    <span className="plugin-card-no-fields">No configuration fields.</span>
                  )}
                </div>
              ))}
            </div>
          ) : configSchema.properties ? (
            <div className="plugin-card-fields">{renderFields(configSchema.properties, configSchema.required)}</div>
          ) : (
            <span className="plugin-card-no-fields">No configuration fields.</span>
          )}
        </div>
      )}
    </div>
  );
}

function ProseSection({ label, body }: { label: string; body: string | null }) {
  if (body === null) return null;
  return (
    <div className="plugin-card-prose-section">
      <div className="plugin-card-prose-label">{label}:</div>
      <div className="plugin-card-prose-body">{body}</div>
    </div>
  );
}
