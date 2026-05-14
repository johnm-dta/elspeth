// src/components/chat/guided/SchemaFormTurn.tsx
//
// Guided-mode widget for the schema_form turn type (Task 7.5).
// Conventions inherited from SingleSelectTurn (Task 7.2 template),
// InspectAndConfirmTurn (Task 7.3), and MultiSelectWithCustomTurn (Task 7.4):
//   - Props: { payload: SchemaFormPayload; onSubmit: (body: GuidedRespondRequest) => void }
//   - onSubmit is SYNC -- the widget constructs the body; the store awaits the round-trip
//   - All 6 GuidedRespondRequest fields set explicitly; unused ones = null (no omission)
//   - <button type="button"> (never <div onClick>)
//   - DOM IDs prefixed with React 18 useId() -- multiple turn instances coexist in
//     GuidedHistory (Task 7.9) and element IDs would collide without per-instance scoping
//   - Visible labels (htmlFor / button text) ARE the accessible name; do not add
//     redundant aria-label that overrides what sighted users see
//   - CSS via App.css class names with design tokens; no hardcoded colours
//
// SHAPE NOTE (differs from chip-group widgets):
// This widget does NOT use <fieldset>+<legend> or chip buttons (per the Task 7.2
// SHAPE NOTE: "schema_form establishes its own structure"). It uses form semantics:
//   - One <label>+control row per schema property
//   - A "Show advanced (N)" toggle button for optional fields
//   - A Continue button at the bottom
//
// SCOPE -- supported JSON Schema subset (MVP, Task 7.5):
//   SUPPORTED: string (no enum) -> text input
//              string + enum    -> select dropdown
//              integer          -> number input (step="1")
//              number           -> number input (step="any")
//              boolean          -> checkbox
//   FALLBACK:  array / object / $ref / anyOf / allOf / oneOf -> JSON textarea
//              User types/edits JSON literal; parse errors show inline and
//              disable Continue until fixed.
//
// The JSON-textarea fallback is a deliberate MVP scope choice -- not a contract
// conflict. Full JSON Schema support (e.g. Optional[str] detection for anyOf,
// nested object flattening) is deferred and does NOT have a Filigree issue.
// Source of truth: protocol.py:53-56 (SchemaFormPayload wire shape).
//
// CONTINUE INVARIANT:
//   Disabled when any required field's value is "empty":
//     - text/number: empty string
//     - boolean:     NEVER empty (always true/false -- a required bool is
//                    always satisfied regardless of its current value)
//     - enum select: empty string (no option selected)
//     - JSON fallback: empty string OR parse error present
//   Continue fires with edited_values = EVERY property from schema_block.properties,
//   using the user's current input or the prefilled/default for untouched fields.
//
// JSON-FALLBACK PARSE CHOICE:
//   Continue is disabled when any JSON-fallback field has a parse error. The per-field
//   error string is stored in formState.errors (keyed by property name). This matches
//   the "disabled = invariant violated" pattern from Task 7.4 (canAddPending) and the
//   "single source of truth" convention -- the same error set drives both the inline
//   error message and the continue-disabled predicate.
//
// EMPTY OPTIONAL NUMERIC:
//   Submitted as null per the Tier 2 type contract -- an empty string in a numeric
//   slot would violate the schema-declared type at the trust boundary. Null preserves
//   the "edited_values includes EVERY property" invariant while letting the server
//   distinguish "user explicitly cleared" from "field was never offered." Required
//   numeric fields are blocked by canSubmit so they never reach this branch.
//
// STATE SPLIT:
//   formState: { values, errors } -- owned by the form; submitted on Continue.
//   advancedExpanded: boolean     -- owned separately; orthogonal to form values.
//   Keeping them separate is correct here (unlike InspectAndConfirmTurn's nullable
//   struct, which encoded mode + data together because the two views hold DIFFERENT
//   data). Toggle state does not affect what gets submitted; merging would be
//   misleading.
//
// FOCUS MANAGEMENT:
//   firstRunRef: skips initial-mount effect so we don't steal focus on widget appear.
//   Show-advanced click: focuses the first revealed field after the re-render.
//   Hide-advanced click: focuses the toggle button after the re-render.
//   Same pattern as InspectAndConfirmTurn (Task 7.3).
//
// Wire-response shape:
//   Continue: { chosen: null, custom_inputs: null, edited_values: <structured dict>,
//               accepted_step_index: null, edit_step_index: null, control_signal: null }
//   edited_values has the shape { plugin, options, observed_columns, sample_rows }:
//     - plugin: payload.plugin (the plugin name chosen in the preceding SINGLE_SELECT)
//     - options: { ...allFormValues } — every property from schema_block.properties
//     - observed_columns: [] (empty at source step; populated by backend after run)
//     - sample_rows: [] (empty at source step; populated by backend after run)
//   The backend's SCHEMA_FORM dispatcher (_dispatch_guided_respond, SCHEMA_FORM branch
//   at STEP_1_SOURCE) requires "plugin" in edited_values to construct SourceResolved.
//   The backend's SCHEMA_FORM dispatcher at STEP_2_SINK currently ignores edited_values
//   (it just emits MULTI_SELECT_WITH_CUSTOM), but the same wire shape is used for
//   shape conformance and future compatibility.

import { useEffect, useId, useRef, useState } from "react";
import type { GuidedRespondRequest, SchemaFormPayload } from "@/types/guided";

// ── JSON Schema sub-types used for rendering decisions ────────────────────────

// A single property definition from schema_block.properties. We use unknown
// and narrow explicitly rather than typing the full JSON Schema spec.
type PropSchema = Record<string, unknown>;

// The field type we dispatch to a control. Single source of truth for
// inferFieldType() -- no duplicated if/else ladders in JSX.
type FieldType = "text" | "number-int" | "number-float" | "checkbox" | "enum" | "json-fallback";

// ── Helper functions ───────────────────────────────────────────────────────────

/**
 * Returns the JSON Schema type string for a property schema, or null if absent.
 * Reads schema["type"] only if it is a string.
 */
function schemaType(prop: PropSchema): string | null {
  const t = prop["type"];
  return typeof t === "string" ? t : null;
}

/**
 * Returns the enum values if the property is a string enum, or null otherwise.
 */
function schemaEnum(prop: PropSchema): string[] | null {
  const e = prop["enum"];
  if (!Array.isArray(e)) return null;
  // Narrow: each element must be a string (Pydantic always emits string enums
  // for StrEnum fields; if a non-string slips through, fall back to json-fallback).
  if (!e.every((v) => typeof v === "string")) return null;
  return e as string[];
}

/**
 * Derives the FieldType for a single property schema.
 *
 * Rules (in order):
 *   1. anyOf / allOf / oneOf present            -> json-fallback
 *   2. $ref present                             -> json-fallback
 *   3. type === "array" or type === "object"    -> json-fallback
 *   4. type === "boolean"                       -> checkbox
 *   5. type === "integer"                       -> number-int
 *   6. type === "number"                        -> number-float
 *   7. type === "string" AND enum present       -> enum
 *   8. type === "string"                        -> text
 *   9. anything else (missing / exotic type)    -> json-fallback
 */
export function inferFieldType(prop: PropSchema): FieldType {
  // Rule 1: composite types always fall back
  if ("anyOf" in prop || "allOf" in prop || "oneOf" in prop) return "json-fallback";
  // Rule 2: $ref
  if ("$ref" in prop) return "json-fallback";

  const t = schemaType(prop);
  // Rule 3: container types
  if (t === "array" || t === "object") return "json-fallback";
  // Rule 4: boolean
  if (t === "boolean") return "checkbox";
  // Rule 5: integer
  if (t === "integer") return "number-int";
  // Rule 6: number (float)
  if (t === "number") return "number-float";
  // Rules 7-8: string
  if (t === "string") {
    return schemaEnum(prop) !== null ? "enum" : "text";
  }
  // Rule 9: fallback for missing or exotic type
  return "json-fallback";
}

/**
 * Derives the initial display value for a form field, applying precedence:
 *   prefilled[name] > schema default > empty (type-appropriate sentinel).
 *
 * For JSON-fallback fields, the return value is the JSON string representation
 * so it can be placed directly in the textarea.
 */
function initialValueFor(
  name: string,
  prop: PropSchema,
  prefilled: Record<string, unknown>,
  fieldType: FieldType,
): string | boolean {
  // Prefilled takes priority
  if (Object.prototype.hasOwnProperty.call(prefilled, name)) {
    const v = prefilled[name];
    if (fieldType === "checkbox") return Boolean(v);
    if (fieldType === "json-fallback") return JSON.stringify(v, null, 2);
    // text, enum, number-int, number-float: coerce to string for input state
    return String(v);
  }

  // Schema default next
  const def = prop["default"];
  if (def !== undefined) {
    if (fieldType === "checkbox") return Boolean(def);
    if (fieldType === "json-fallback") return JSON.stringify(def, null, 2);
    return String(def);
  }

  // Empty sentinel
  if (fieldType === "checkbox") return false;
  // For json-fallback array fields, the natural empty state is "[]" --
  // for object/$ref/anyOf we use "null" (a valid JSON null literal) so the
  // textarea has a parseable initial value rather than an empty string
  // that would immediately trigger a parse error.
  if (fieldType === "json-fallback") {
    const t = schemaType(prop);
    if (t === "array") return "[]";
    return "null";
  }
  return "";
}

/**
 * Returns the label text for a property: title if present, else the property
 * name itself (fallback for schemas where Pydantic omits title).
 */
function labelFor(name: string, prop: PropSchema): string {
  const commonLabels: Record<string, string> = {
    path: "Input file path",
    output_path: "Output file path",
    has_header: "Has header row",
    delimiter: "Column delimiter",
    encoding: "Text encoding",
    mode: "Mode",
    batch_size: "Batch size",
    timeout: "Timeout seconds",
  };
  if (Object.prototype.hasOwnProperty.call(commonLabels, name)) {
    return commonLabels[name];
  }
  const t = prop["title"];
  return typeof t === "string" && t.length > 0 ? t : name;
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface SchemaFormTurnProps {
  payload: SchemaFormPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
  disabled?: boolean;
}

// Form state holds current field values (string or boolean) and per-field
// JSON parse error messages (only set for json-fallback fields with invalid JSON).
interface FormState {
  values: Record<string, string | boolean>;
  errors: Record<string, string>;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function SchemaFormTurn({
  payload,
  onSubmit,
  disabled = false,
}: SchemaFormTurnProps) {
  // Derive the ordered list of properties from the schema once.
  // schema_block.properties is a plain object; we preserve declaration order
  // (JS spec guarantees string-keyed insertion order on plain objects).
  // Pydantic always emits the "properties" key -- even for models with no fields
  // it emits `"properties": {}`. We do NOT defensively ?? {} here: if the wire
  // somehow omits "properties", Object.keys(undefined) crashes loudly and
  // attributably (per CLAUDE.md offensive programming). The EMPTY_SCHEMA_PAYLOAD
  // test covers the legitimate `properties: {}` case.
  const properties = payload.schema_block["properties"] as Record<string, PropSchema>;
  const propertyNames = Object.keys(properties);

  // Required fields: those listed in schema_block.required (may be absent).
  // We derive this once here rather than scattering ?? [] throughout the JSX.
  const requiredRaw = payload.schema_block["required"];
  const requiredFields: ReadonlySet<string> = new Set(
    Array.isArray(requiredRaw) ? (requiredRaw as string[]) : [],
  );

  // Derive initial form state from schema defaults and prefilled values.
  const initialFormState = (): FormState => {
    const values: Record<string, string | boolean> = {};
    for (const name of propertyNames) {
      const prop = properties[name];
      const ft = inferFieldType(prop);
      values[name] = initialValueFor(name, prop, payload.prefilled, ft);
    }
    return { values, errors: {} };
  };

  const [formState, setFormState] = useState<FormState>(initialFormState);

  // advancedExpanded is orthogonal to form state (see STATE SPLIT note above).
  const [advancedExpanded, setAdvancedExpanded] = useState(false);

  // useId scopes DOM IDs per-instance so multiple SchemaFormTurns rendered
  // simultaneously (e.g. active turn + GuidedHistory replay in Task 7.9)
  // don't produce id collisions when field names recur across turns.
  const reactId = useId();
  const fieldInputId = (name: string) => `${reactId}-field-${name}`;
  const fieldHintId = (name: string) => `${reactId}-hint-${name}`;
  const fieldErrorId = (name: string) => `${reactId}-error-${name}`;
  // Optional-fields region id: paired with the toggle button's aria-controls so
  // assistive tech announces "expanded/collapsed <Optional fields>" not just
  // "expanded/collapsed". Convention-setting for Task 7.9 (GuidedHistory) which
  // is also a collapsible region -- both widgets land the WAI-ARIA pattern
  // together rather than copying a gap.
  const optionalSectionId = `${reactId}-optional-section`;

  // Partition properties into required / optional based on schema.required.
  const requiredNames = propertyNames.filter((n) => requiredFields.has(n));
  const optionalNames = propertyNames.filter((n) => !requiredFields.has(n));

  // ── Focus management ──────────────────────────────────────────────────────

  // Refs for the first advanced (optional) field and the toggle button.
  // Set via ref callbacks in JSX so they track whatever control is first.
  const firstAdvancedRef = useRef<
    HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement | null
  >(null);
  const toggleBtnRef = useRef<HTMLButtonElement | null>(null);

  // Skip the first effect run: on initial widget mount the user did NOT toggle
  // the view. Auto-focusing on mount would steal focus from wherever the user
  // actually was. Same convention as InspectAndConfirmTurn (Task 7.3).
  const firstRunRef = useRef(true);
  useEffect(() => {
    if (firstRunRef.current) {
      firstRunRef.current = false;
      return;
    }
    if (advancedExpanded) {
      firstAdvancedRef.current?.focus();
    } else {
      toggleBtnRef.current?.focus();
    }
  }, [advancedExpanded]);

  // ── Field value updaters ──────────────────────────────────────────────────

  function handleTextChange(name: string, value: string) {
    setFormState((prev) => ({
      ...prev,
      values: { ...prev.values, [name]: value },
    }));
  }

  function handleCheckboxChange(name: string, checked: boolean) {
    setFormState((prev) => ({
      ...prev,
      values: { ...prev.values, [name]: checked },
    }));
  }

  function handleJsonChange(name: string, raw: string) {
    // Attempt to parse immediately so errors show inline as the user types.
    // The parsed value is not stored in state -- on submit we re-parse from
    // the raw string. This keeps the state type simple (string) and avoids
    // a separate "parsed value" slot that would need synchronizing.
    let error = "";
    try {
      JSON.parse(raw);
    } catch {
      error = "Invalid JSON";
    }
    setFormState((prev) => ({
      values: { ...prev.values, [name]: raw },
      errors: { ...prev.errors, [name]: error },
    }));
  }

  // ── Submit predicate ──────────────────────────────────────────────────────

  // A required field is "empty" when:
  //   - text/enum: its string value is "" (trimmed for text, exact for enum)
  //   - number-int/number-float: its string value is ""
  //   - boolean: NEVER empty (required bool is always satisfied)
  //   - json-fallback: value is "" OR there is a parse error
  //
  // Optional fields never block Continue regardless of their value.
  //
  // Additionally, ANY json-fallback field with a parse error blocks Continue,
  // even optional ones -- bad JSON in an optional field would corrupt the
  // edited_values dict sent to the server.
  function isFieldEmpty(name: string, fieldType: FieldType): boolean {
    const val = formState.values[name];
    if (fieldType === "checkbox") return false; // boolean always satisfied
    if (typeof val === "string") {
      if (fieldType === "text" || fieldType === "number-int" || fieldType === "number-float") {
        return val.trim() === "";
      }
      if (fieldType === "enum") return val === "";
      if (fieldType === "json-fallback") return val.trim() === "";
    }
    return false;
  }

  const hasParseErrors = Object.values(formState.errors).some((e) => e !== "");

  const canSubmit = (() => {
    if (hasParseErrors) return false;
    for (const name of requiredNames) {
      const ft = inferFieldType(properties[name]);
      if (isFieldEmpty(name, ft)) return false;
    }
    return true;
  })();

  // ── Submit handler ────────────────────────────────────────────────────────

  function handleContinue() {
    // Narrow because TS doesn't see the cross-handler invariant: canSubmit is
    // the disabled predicate and this guard -- same predicate, deduped, not
    // defensive programming.
    if (!canSubmit) return;

    // Build the options dict: EVERY property from the schema, using the
    // current field state (prefilled value, schema default, or user input).
    // JSON-fallback fields are re-parsed here; any parse error would have
    // already been caught by hasParseErrors and canSubmit would be false, so
    // this parse is guaranteed to succeed at submit time.
    //
    // EMPTY OPTIONAL NUMERIC: an empty string in a number-int / number-float
    // field is submitted as null, NOT as "" (which would violate the Tier 2
    // type contract -- the schema declares int/float and the widget owes the
    // declared type at the trust boundary). Required-empty numerics are
    // already blocked by canSubmit and never reach this branch.
    const options: Record<string, unknown> = {};
    for (const name of propertyNames) {
      const ft = inferFieldType(properties[name]);
      const raw = formState.values[name];
      if (ft === "json-fallback") {
        // raw is the textarea string; parse it back to the structured value.
        // canSubmit guarantees no parse errors here.
        options[name] = JSON.parse(raw as string);
      } else if (ft === "number-int" || ft === "number-float") {
        // Empty -> null (Tier 2 type contract). Otherwise parse to number.
        // canSubmit guarantees required numerics are non-empty.
        if ((raw as string) === "") {
          options[name] = null;
        } else {
          const n =
            ft === "number-int"
              ? parseInt(raw as string, 10)
              : parseFloat(raw as string);
          // NaN means the user typed something the input accepted but isn't a
          // number (e.g. "1e" mid-typing). Submit null to honour the type
          // contract -- an unparseable string would corrupt the audit trail.
          options[name] = isNaN(n) ? null : n;
        }
      } else if (ft === "checkbox") {
        options[name] = raw as boolean;
      } else {
        // text / enum
        options[name] = raw as string;
      }
    }

    // The backend's SCHEMA_FORM dispatcher requires the structured shape:
    // { plugin, options, observed_columns, sample_rows }. "plugin" is the key
    // the backend uses to resolve the plugin and construct SourceResolved (or
    // SinkResolved). observed_columns and sample_rows are populated by the
    // backend after the source runs; this widget submits empty lists as
    // shape-conformant placeholders.
    onSubmit({
      chosen: null,
      edited_values: {
        plugin: payload.plugin,
        options,
        observed_columns: [],
        sample_rows: [],
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  // ── Field rendering helper ────────────────────────────────────────────────

  function renderField(name: string, isFirst: boolean) {
    const prop = properties[name];
    const fieldType = inferFieldType(prop);
    const label = labelFor(name, prop);
    const description = typeof prop["description"] === "string" ? prop["description"] : null;
    const inputId = fieldInputId(name);
    const hintId = description !== null ? fieldHintId(name) : undefined;
    const errorId = fieldErrorId(name);
    const hasError = (formState.errors[name] ?? "") !== "";
    const val = formState.values[name];

    // Build ref callback for focus restoration: the first optional field gets
    // the ref so Show-advanced click can focus it.
    const firstAdvancedCallback = (
      el: HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement | null,
    ) => {
      if (isFirst) firstAdvancedRef.current = el;
    };

    // aria-describedby: wire hint, error, or both -- only wire IDs that exist
    const describedByParts: string[] = [];
    if (hintId) describedByParts.push(hintId);
    if (hasError) describedByParts.push(errorId);
    const describedBy =
      describedByParts.length > 0 ? describedByParts.join(" ") : undefined;

    return (
      <div key={name} className="guided-schema-field-row">
        {fieldType === "checkbox" ? (
          // Checkbox: label wraps the input for larger click target. We still use
          // htmlFor so the accessible name is always the visible text.
          <>
            <div className="guided-schema-checkbox-row">
              <input
                id={inputId}
                ref={isFirst ? firstAdvancedCallback : undefined}
                type="checkbox"
                className="guided-schema-checkbox"
                checked={val as boolean}
                disabled={disabled}
                aria-describedby={describedBy}
                onChange={(e) => handleCheckboxChange(name, e.target.checked)}
              />
              <label htmlFor={inputId} className="guided-schema-label">
                {label}
              </label>
            </div>
            {description !== null && (
              <p id={hintId} className="guided-schema-hint">
                {description}
              </p>
            )}
          </>
        ) : fieldType === "enum" ? (
          <>
            <label htmlFor={inputId} className="guided-schema-label">
              {label}
            </label>
            <select
              id={inputId}
              ref={isFirst ? firstAdvancedCallback as (el: HTMLSelectElement | null) => void : undefined}
              className="guided-schema-select"
              value={val as string}
              disabled={disabled}
              aria-describedby={describedBy}
              onChange={(e) => handleTextChange(name, e.target.value)}
            >
              {(schemaEnum(prop) ?? []).map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
            {description !== null && (
              <p id={hintId} className="guided-schema-hint">
                {description}
              </p>
            )}
          </>
        ) : fieldType === "json-fallback" ? (
          <>
            <label htmlFor={inputId} className="guided-schema-label">
              {label}
            </label>
            <textarea
              id={inputId}
              ref={isFirst ? firstAdvancedCallback as (el: HTMLTextAreaElement | null) => void : undefined}
              className={`guided-schema-textarea${hasError ? " guided-schema-textarea--error" : ""}`}
              value={val as string}
              disabled={disabled}
              aria-describedby={describedBy}
              onChange={(e) => handleJsonChange(name, e.target.value)}
              rows={4}
              spellCheck={false}
            />
            {description !== null && (
              <p id={hintId} className="guided-schema-hint">
                {description}
              </p>
            )}
            {hasError && (
              <p id={errorId} className="guided-schema-error" role="alert">
                {formState.errors[name]}
              </p>
            )}
          </>
        ) : (
          // text / number-int / number-float
          <>
            <label htmlFor={inputId} className="guided-schema-label">
              {label}
            </label>
            <input
              id={inputId}
              ref={isFirst ? firstAdvancedCallback as (el: HTMLInputElement | null) => void : undefined}
              type={fieldType === "text" ? "text" : "number"}
              step={
                fieldType === "number-int"
                  ? "1"
                  : fieldType === "number-float"
                    ? "any"
                    : undefined
              }
              className="guided-schema-input"
              value={val as string}
              disabled={disabled}
              aria-describedby={describedBy}
              onChange={(e) => handleTextChange(name, e.target.value)}
            />
            {description !== null && (
              <p id={hintId} className="guided-schema-hint">
                {description}
              </p>
            )}
          </>
        )}
      </div>
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="guided-turn guided-schema-form">
      {/* Required fields -- always visible */}
      {requiredNames.length > 0 && (
        <div className="guided-schema-required-section">
          {requiredNames.map((name) => renderField(name, false))}
        </div>
      )}

      {/* Optional fields -- collapsed until user clicks toggle */}
      {optionalNames.length > 0 && (
        <>
          <button
            ref={toggleBtnRef}
            type="button"
            className="guided-schema-advanced-toggle"
            onClick={() => setAdvancedExpanded((prev) => !prev)}
            aria-expanded={advancedExpanded}
            aria-controls={optionalSectionId}
            disabled={disabled}
          >
            {advancedExpanded
              ? "Hide advanced"
              : `Show advanced (${optionalNames.length})`}
          </button>
          {/* Container is rendered UNCONDITIONALLY so the toggle's
              aria-controls reference resolves on initial render (before first
              expansion). When collapsed, the `hidden` attribute removes it
              from the accessibility tree AND hides it visually
              (`[hidden] { display: none; }` is the browser default). The
              CHILDREN are still gated on `advancedExpanded` so optional
              fields don't run their ref callbacks until the user opts in. */}
          <div
            id={optionalSectionId}
            role="region"
            aria-label="Optional fields"
            className="guided-schema-optional-section"
            hidden={!advancedExpanded}
          >
            {advancedExpanded &&
              optionalNames.map((name, idx) => renderField(name, idx === 0))}
          </div>
        </>
      )}

      {/* Continue action.
          aria-disabled mirrors disabled because some screen readers skip
          disabled buttons entirely; aria-disabled keeps the announcement in
          the AT tree while still preventing click activation. */}
      <div className="guided-schema-actions">
        <button
          type="button"
          className="guided-schema-continue-btn"
          onClick={handleContinue}
          disabled={disabled || !canSubmit}
          aria-disabled={disabled || !canSubmit}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
