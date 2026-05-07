// ============================================================================
// PluginCard
//
// Collapsible card showing plugin name, description, and config schema.
// Collapsed: name + one-line description. Expanded: config schema fields.
//
// Handles two JSON-Schema shapes:
//   1. Flat single-model schema: { properties, required } at the top level.
//   2. Pydantic discriminated union: { oneOf: [{$ref}, ...], discriminator,
//      $defs } — rendered as one section per variant, labelled by its
//      discriminator mapping value (e.g., "provider: azure" / "openrouter").
// ============================================================================

import { useState } from "react";
import type { PluginSummary, PluginSchemaInfo } from "@/types/index";

interface PluginCardProps {
  plugin: PluginSummary;
  schema: PluginSchemaInfo | null;
  schemaError?: boolean;
  onExpand: () => void;
}

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
  discriminator?: {
    propertyName?: string;
    mapping?: Record<string, string>;
  };
  $defs?: Record<string, JsonSchemaObject>;
}

const DEFS_REF_PREFIX = "#/$defs/";

function isDiscriminated(
  schema: DiscriminatedSchema & JsonSchemaObject,
): boolean {
  return Array.isArray(schema.oneOf) && schema.$defs !== undefined;
}

function resolveVariants(
  schema: DiscriminatedSchema,
): Array<{ label: string; def: JsonSchemaObject }> {
  const defs = schema.$defs ?? {};
  const mapping = schema.discriminator?.mapping ?? {};
  // Invert the mapping: "azure" -> "#/$defs/AzureOpenAIConfig" becomes
  // "AzureOpenAIConfig" -> "azure", so each $defs entry can be labeled.
  const refToValue = new Map<string, string>();
  for (const [discValue, ref] of Object.entries(mapping)) {
    if (ref.startsWith(DEFS_REF_PREFIX)) {
      refToValue.set(ref.slice(DEFS_REF_PREFIX.length), discValue);
    }
  }
  const discProp = schema.discriminator?.propertyName ?? "variant";
  const variants: Array<{ label: string; def: JsonSchemaObject }> = [];
  for (const entry of schema.oneOf ?? []) {
    const ref = entry.$ref ?? "";
    if (!ref.startsWith(DEFS_REF_PREFIX)) continue;
    const defName = ref.slice(DEFS_REF_PREFIX.length);
    const def = defs[defName];
    if (def === undefined) continue;
    const discValue = refToValue.get(defName) ?? defName;
    variants.push({ label: `${discProp}: ${discValue}`, def });
  }
  return variants;
}

function renderFields(
  properties: Record<string, JsonSchemaField>,
  required: string[] | undefined,
): JSX.Element[] {
  const requiredSet = new Set(required ?? []);
  return Object.entries(properties).map(([name, field]) => (
    <div key={name}>
      <span className="plugin-card-field-name">{name}</span>
      <span className="plugin-card-field-type">{field.type ?? "any"}</span>
      {requiredSet.has(name) && (
        <span className="plugin-card-field-required">required</span>
      )}
      {field.description && (
        <div className="plugin-card-field-desc">{field.description}</div>
      )}
    </div>
  ));
}

export function PluginCard({ plugin, schema, schemaError, onExpand }: PluginCardProps) {
  const [expanded, setExpanded] = useState(false);

  function handleClick() {
    if (!expanded) {
      onExpand();
    }
    setExpanded((prev) => !prev);
  }

  const configSchema = schema?.json_schema as
    | (DiscriminatedSchema & JsonSchemaObject)
    | undefined;

  return (
    <div
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      }}
      tabIndex={0}
      role="button"
      aria-expanded={expanded}
      className="plugin-card"
    >
      {/* Header — always visible */}
      <div className="plugin-card-header">
        <span className="plugin-card-name">{plugin.name}</span>
        <span className="plugin-card-desc">{plugin.description}</span>
      </div>

      {/* Expanded content */}
      {expanded && (
        <div
          className="plugin-card-expanded"
          onClick={(e) => e.stopPropagation()}
        >
          {schemaError ? (
            <span className="plugin-card-schema-error">
              Failed to load schema. Collapse and expand to retry.
            </span>
          ) : schema === null || configSchema === undefined ? (
            <span className="plugin-card-schema-loading">Loading...</span>
          ) : isDiscriminated(configSchema) ? (
            <div className="plugin-card-variants">
              {resolveVariants(configSchema).map((variant) => (
                <div key={variant.label} className="plugin-card-variant">
                  <div className="plugin-card-variant-label">
                    {variant.label}
                  </div>
                  {variant.def.properties ? (
                    <div className="plugin-card-fields">
                      {renderFields(variant.def.properties, variant.def.required)}
                    </div>
                  ) : (
                    <span className="plugin-card-no-fields">
                      No configuration fields.
                    </span>
                  )}
                </div>
              ))}
            </div>
          ) : configSchema.properties ? (
            <div className="plugin-card-fields">
              {renderFields(configSchema.properties, configSchema.required)}
            </div>
          ) : (
            <span className="plugin-card-no-fields">
              No configuration fields.
            </span>
          )}
        </div>
      )}
    </div>
  );
}
