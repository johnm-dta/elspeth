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
      if (!field.required) continue;
      const value = values[field.name];
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

  switch (field.kind) {
    case "text":
    case "blob-ref":
      return (
        <div className="guided-schema-field-row">
          <label htmlFor={id} className="guided-schema-label">
            {field.label}
          </label>
          <input
            id={id}
            type="text"
            className="guided-schema-input"
            value={value === null ? "" : String(value ?? "")}
            placeholder={field.kind === "blob-ref" ? "blob UUID" : undefined}
            aria-describedby={descriptionId}
            onChange={(event) => onChange(event.target.value)}
            disabled={disabled}
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
    case "number-int":
    case "number-float":
      return (
        <div className="guided-schema-field-row">
          <label htmlFor={id} className="guided-schema-label">
            {field.label}
          </label>
          <input
            id={id}
            type="number"
            className="guided-schema-input"
            step={field.kind === "number-int" ? "1" : "any"}
            value={value === null ? "" : String(value ?? "")}
            aria-describedby={descriptionId}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw === "") {
                onChange(field.nullable ? null : "");
                return;
              }
              const parsed = field.kind === "number-int" ? parseInt(raw, 10) : parseFloat(raw);
              onChange(Number.isNaN(parsed) ? "" : parsed);
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
          <label htmlFor={id} className="guided-schema-label">
            {field.label}
          </label>
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
          <label htmlFor={id} className="guided-schema-label">
            {field.label}
          </label>
          <select
            id={id}
            className="guided-schema-select"
            value={String(value ?? "")}
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
          <label htmlFor={id} className="guided-schema-label">
            {field.label}
          </label>
          <textarea
            id={id}
            className="guided-schema-textarea"
            value={typeof value === "string" ? value : Array.isArray(value) ? value.join("\n") : ""}
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
    case "json-value":
      return (
        <div className="guided-schema-field-row">
          <label htmlFor={id} className="guided-schema-label">
            {field.label}
          </label>
          <textarea
            id={id}
            className="guided-schema-textarea"
            value={jsonText(value, field.kind)}
            aria-describedby={descriptionId}
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
        </div>
      );
  }
  const _exhaustive: never = field.kind;
  return _exhaustive;
}

function jsonText(value: unknown, kind: "json-object" | "json-array" | "json-value"): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? emptyForKind(kind), null, 2);
}
