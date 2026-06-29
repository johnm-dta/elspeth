import { useId, useState } from "react";
import type { GuidedRespondRequest, KnobField, SchemaFormPayload } from "@/types/guided";
import { TUTORIAL_VALIDATION_FAILURE_CAVEAT } from "@/components/tutorial/copy";
import { RecipeContextHeader } from "./RecipeContextHeader";

interface SchemaFormTurnProps {
  payload: SchemaFormPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
  disabled?: boolean;
  /**
   * Tutorial mode: surfaces the worked-example teaching copy for prefilled
   * required-no-default knobs the passive learner cannot type (e.g. the source's
   * on_validation_failure="discard"). Off for the normal composer flow.
   */
  isTutorial?: boolean;
}

type FormValues = Record<string, unknown>;

export function SchemaFormTurn({ payload, onSubmit, disabled = false, isTutorial = false }: SchemaFormTurnProps) {
  const reactId = useId();
  const [values, setValues] = useState<FormValues>(() =>
    initialValues(payload.knobs.fields, payload.prefilled),
  );

  function isVisible(field: KnobField, state: FormValues = values): boolean {
    if (!field.visible_when) return true;
    return state[field.visible_when.field] === field.visible_when.equals;
  }

  function visibleFields(state: FormValues = values): KnobField[] {
    return payload.knobs.fields.filter((field) => isVisible(field, state));
  }

  function onChange(name: string, value: unknown) {
    setValues((prev) => {
      const next = { ...prev, [name]: value };
      for (const field of payload.knobs.fields) {
        if (field.visible_when?.field === name && field.visible_when.equals !== value) {
          delete next[field.name];
        }
      }
      return next;
    });
  }

  function canSubmit(): boolean {
    for (const field of visibleFields()) {
      const value = values[field.name];
      // Block on any field the form already knows holds an invalid value
      // (broken JSON / non-numeric number), required or not. Previously such a
      // value rode silently through to submit because canSubmit only inspected
      // required fields and never checked validity.
      if (fieldHasError(field, value)) return false;
      if (!field.required) continue;
      if (field.kind === "checkbox") continue;
      if (value === undefined || value === null || value === "") return false;
      if (Array.isArray(value) && value.length === 0) return false;
    }
    return true;
  }

  function handleContinue() {
    if (!canSubmit()) return;
    const submitted: Record<string, unknown> =
      payload.mode === "recipe_decision" ? { ...payload.prefilled } : {};
    for (const field of visibleFields()) {
      submitted[field.name] = submittedValue(field, values[field.name]);
    }

    if (payload.mode === "plugin_options") {
      onSubmit({
        chosen: null,
        edited_values: {
          plugin: payload.plugin,
          options: submitted,
          observed_columns: [],
          sample_rows: [],
        },
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      });
      return;
    }

    onSubmit({
      chosen: ["accept"],
      edited_values: {
        recipe_name: payload.recipe_context.recipe_name,
        slots: submitted,
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  function handleBuildManually() {
    onSubmit({
      chosen: ["build_manually"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  }

  const showValidationFailureTeaching =
    isTutorial &&
    payload.mode === "plugin_options" &&
    payload.knobs.fields.some((field) => field.name === "on_validation_failure");

  return (
    <div className="guided-turn guided-schema-form">
      {payload.mode === "recipe_decision" && (
        <RecipeContextHeader context={payload.recipe_context} />
      )}
      <div className="guided-schema-fields">
        {visibleFields().map((field) => (
          <KnobFieldRenderer
            key={field.name}
            field={field}
            value={values[field.name]}
            onChange={(value) => onChange(field.name, value)}
            idPrefix={reactId}
            disabled={disabled}
            isTutorial={isTutorial}
          />
        ))}
      </div>
      {showValidationFailureTeaching && (
        <p className="guided-schema-hint guided-schema-teaching" role="note">
          {TUTORIAL_VALIDATION_FAILURE_CAVEAT}
        </p>
      )}
      <div className="guided-schema-actions">
        <button
          type="button"
          className="guided-turn-primary"
          onClick={handleContinue}
          disabled={disabled || !canSubmit()}
        >
          {payload.mode === "recipe_decision" ? "Apply recipe" : "Continue"}
        </button>
        {payload.mode === "recipe_decision" && payload.recipe_context.alternatives.includes("build_manually") && (
          <button
            type="button"
            className="guided-turn-secondary"
            onClick={handleBuildManually}
            disabled={disabled}
          >
            Build manually
          </button>
        )}
      </div>
    </div>
  );
}

function initialValues(fields: KnobField[], prefilled: Record<string, unknown>): FormValues {
  const values: FormValues = {};
  for (const field of fields) {
    if (Object.prototype.hasOwnProperty.call(prefilled, field.name)) {
      values[field.name] = prefilled[field.name];
    } else if (field.default !== undefined) {
      values[field.name] = field.default;
    } else {
      values[field.name] = emptyForKind(field.kind);
    }
  }
  return values;
}

function emptyForKind(kind: KnobField["kind"]): unknown {
  switch (kind) {
    case "checkbox":
      return false;
    case "string-list":
      return [];
    case "json-object":
      return {};
    case "json-array":
      return [];
    case "json-value":
      return null;
    case "number-int":
    case "number-float":
      return "";
    case "text":
    case "enum":
    case "blob-ref":
      return "";
  }
  const _exhaustive: never = kind;
  return _exhaustive;
}

function submittedValue(field: KnobField, value: unknown): unknown {
  if (value === undefined) return field.default ?? null;
  if (field.kind === "string-list") {
    if (typeof value === "string") return value.split("\n").filter((line) => line !== "");
    if (Array.isArray(value)) return value;
    return [];
  }
  return value;
}

// Whether a field participates in the required-marker / aria-required treatment.
// Mirrors canSubmit's required predicate: a checkbox always carries a boolean
// value, so the form never treats one as unmet — marking it required (visibly or
// programmatically) would contradict that, so the marker tracks exactly the
// predicate the gate enforces.
function isRequiredField(field: KnobField): boolean {
  return field.required && field.kind !== "checkbox";
}

// True when the field's current value is in a state the form already knows is
// invalid: a JSON object/array whose text failed to parse, or a number field
// holding raw text that isn't a usable number. The onChange handlers keep that
// raw text (instead of silently blanking/coercing it) precisely so it can be
// surfaced here — both to gate Continue and to render the inline error.
//
// json-value is excluded by design: its parse is lossy (a bare `x` and a parsed
// `"x"` both end up as the string "x"), so the form genuinely cannot tell a
// valid scalar from a broken one and must not guess.
function fieldHasError(field: KnobField, value: unknown): boolean {
  switch (field.kind) {
    case "number-int":
    case "number-float":
      return typeof value === "string" && value.trim() !== "";
    case "json-object":
    case "json-array":
      return typeof value === "string" && value.trim() !== "";
    default:
      return false;
  }
}

// Join the description / error ids into a single aria-describedby token list,
// dropping the undefined ones. Returns undefined when nothing is described so
// the attribute is omitted rather than rendered empty.
function describedBy(...ids: Array<string | undefined>): string | undefined {
  const joined = ids.filter((id): id is string => Boolean(id)).join(" ");
  return joined === "" ? undefined : joined;
}

// Field label with the required marker. The asterisk is aria-hidden (so AT does
// not read "star") while the screen-reader-only "(required)" carries the cue in
// the accessible name; aria-required on the control (set per branch) conveys the
// state programmatically. Non-required fields render the label alone.
function FieldLabel({ field, htmlFor }: { field: KnobField; htmlFor: string }) {
  return (
    <label htmlFor={htmlFor} className="guided-schema-label">
      {field.label}
      {isRequiredField(field) && (
        <>
          <span className="guided-schema-required-marker" aria-hidden="true">
            {" *"}
          </span>
          <span className="visually-hidden">{" (required)"}</span>
        </>
      )}
    </label>
  );
}

function KnobFieldRenderer({
  field,
  value,
  onChange,
  idPrefix,
  disabled,
  isTutorial = false,
}: {
  field: KnobField;
  value: unknown;
  onChange: (value: unknown) => void;
  idPrefix: string;
  disabled: boolean;
  isTutorial?: boolean;
}) {
  const id = `${idPrefix}-${field.name}`;
  const descriptionId = field.description ? `${id}-description` : undefined;
  const required = isRequiredField(field);

  switch (field.kind) {
    case "text":
    case "blob-ref": {
      const rawString = value === null ? "" : String(value ?? "");
      // Tutorial path-leak mask (audience-sensitive: this is the public
      // Composer-screenshot surface). A blob-backed source/sink commits its
      // `path` knob as the server's ABSOLUTE blob storage_path
      // (/home/<user>/.../data/blobs/<session>/<blob_id>_name.json — see
      // web/blobs/service.py _storage_path), leaking the deploy dir + OS
      // username. Show the friendly basename to the passive learner; the real
      // path stays in form state and flows to submit unchanged (handleContinue
      // reads `values`, never this DOM string). readOnly so the masked string
      // can never overwrite the real value.
      const maskPathLeak =
        isTutorial && field.name === "path" && rawString.startsWith("/");
      const displayString = maskPathLeak ? friendlyBlobRef(rawString) : rawString;
      return (
        <div className="guided-schema-field-row">
          <FieldLabel field={field} htmlFor={id} />
          <input
            id={id}
            type="text"
            className="guided-schema-input"
            value={displayString}
            placeholder={field.kind === "blob-ref" ? "blob UUID" : undefined}
            required={required}
            aria-required={required || undefined}
            aria-describedby={descriptionId}
            onChange={(event) => onChange(event.target.value)}
            disabled={disabled}
            readOnly={maskPathLeak}
          />
          {field.description && (
            <p id={descriptionId} className="guided-schema-hint">
              {field.description}
            </p>
          )}
          {field.nullable && value !== null && (
            <button
              type="button"
              className="guided-turn-secondary"
              onClick={() => onChange(null)}
              disabled={disabled}
            >
              Clear {field.label}
            </button>
          )}
        </div>
      );
    }
    case "number-int":
    case "number-float": {
      const hasError = fieldHasError(field, value);
      const errorId = hasError ? `${id}-error` : undefined;
      return (
        <div className="guided-schema-field-row">
          <FieldLabel field={field} htmlFor={id} />
          <input
            id={id}
            type="number"
            className={`guided-schema-input${hasError ? " guided-schema-input--error" : ""}`}
            step={field.kind === "number-int" ? "1" : "any"}
            value={value === null ? "" : String(value ?? "")}
            required={required}
            aria-required={required || undefined}
            aria-invalid={hasError || undefined}
            aria-describedby={describedBy(descriptionId, errorId)}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw === "") {
                onChange(field.nullable ? null : "");
                return;
              }
              const parsed = field.kind === "number-int" ? parseInt(raw, 10) : parseFloat(raw);
              // Keep the raw text (instead of silently blanking it) when it
              // isn't a usable number, so fieldHasError can flag it inline. For
              // an integer field a fractional entry is surfaced too (parseInt
              // truncates "1.5" → 1) rather than quietly dropping the decimal.
              const usable =
                !Number.isNaN(parsed) &&
                (field.kind === "number-float" || Number(raw) === parsed);
              onChange(usable ? parsed : raw);
            }}
            disabled={disabled}
          />
          {field.description && (
            <p id={descriptionId} className="guided-schema-hint">
              {field.description}
            </p>
          )}
          {hasError && (
            <p id={errorId} className="guided-schema-error" role="alert">
              {field.kind === "number-int" ? "Enter a whole number." : "Enter a valid number."}
            </p>
          )}
        </div>
      );
    }
    case "checkbox":
      return (
        <div className="guided-schema-field-row guided-schema-checkbox-row">
          <input
            id={id}
            type="checkbox"
            className="guided-schema-checkbox"
            checked={Boolean(value)}
            aria-describedby={descriptionId}
            onChange={(event) => onChange(event.target.checked)}
            disabled={disabled}
          />
          <FieldLabel field={field} htmlFor={id} />
          {field.description && (
            <p id={descriptionId} className="guided-schema-hint">
              {field.description}
            </p>
          )}
        </div>
      );
    case "enum":
      return (
        <div className="guided-schema-field-row">
          <FieldLabel field={field} htmlFor={id} />
          <select
            id={id}
            className="guided-schema-select"
            value={String(value ?? "")}
            required={required}
            aria-required={required || undefined}
            aria-describedby={descriptionId}
            onChange={(event) => onChange(event.target.value)}
            disabled={disabled}
          >
            <option value="" disabled={field.required}>
              Select...
            </option>
            {(field.enum ?? []).map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
          {field.description && (
            <p id={descriptionId} className="guided-schema-hint">
              {field.description}
            </p>
          )}
        </div>
      );
    case "string-list":
      return (
        <div className="guided-schema-field-row">
          <FieldLabel field={field} htmlFor={id} />
          <textarea
            id={id}
            className="guided-schema-textarea"
            value={typeof value === "string" ? value : Array.isArray(value) ? value.join("\n") : ""}
            required={required}
            aria-required={required || undefined}
            aria-describedby={descriptionId}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                onChange(`${event.currentTarget.value}\n`);
              }
            }}
            disabled={disabled}
          />
          {field.description && (
            <p id={descriptionId} className="guided-schema-hint">
              {field.description}
            </p>
          )}
        </div>
      );
    case "json-object":
    case "json-array":
    case "json-value": {
      const hasError = fieldHasError(field, value);
      const errorId = hasError ? `${id}-error` : undefined;
      return (
        <div className="guided-schema-field-row">
          <FieldLabel field={field} htmlFor={id} />
          <textarea
            id={id}
            className={`guided-schema-textarea${hasError ? " guided-schema-textarea--error" : ""}`}
            value={jsonText(value, field.kind)}
            required={required}
            aria-required={required || undefined}
            aria-invalid={hasError || undefined}
            aria-describedby={describedBy(descriptionId, errorId)}
            onChange={(event) => {
              try {
                onChange(JSON.parse(event.target.value));
              } catch {
                onChange(event.target.value);
              }
            }}
            disabled={disabled}
            // Tutorial: the passive learner authors nothing. Show the prefilled
            // raw-JSON value read-only (transparency) instead of an editable,
            // intimidating JSON editor.
            readOnly={isTutorial}
          />
          {field.description && (
            <p id={descriptionId} className="guided-schema-hint">
              {field.description}
            </p>
          )}
          {hasError && (
            <p id={errorId} className="guided-schema-error" role="alert">
              Invalid JSON — check brackets, quotes, and commas.
            </p>
          )}
        </div>
      );
    }
  }
  const _exhaustive: never = field.kind;
  return _exhaustive;
}

// A blob storage_path basename is "<blob_uuid>_<original_filename>"
// (web/blobs/service.py _storage_path). Strip the uuid_ prefix to recover the
// friendly filename; if no uuid prefix is present the basename is returned
// unchanged. Never returns the directory portion, so the deploy path + OS
// username are dropped.
const BLOB_ID_PREFIX_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_/i;
function friendlyBlobRef(absPath: string): string {
  const base = absPath.split("/").pop() ?? absPath;
  return base.replace(BLOB_ID_PREFIX_RE, "");
}

function jsonText(value: unknown, kind: "json-object" | "json-array" | "json-value"): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? emptyForKind(kind), null, 2);
}
