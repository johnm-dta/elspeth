import { useId } from "react";
import type { ReactNode, TextareaHTMLAttributes } from "react";

/**
 * Multiline text input (vertical resize) matching the ELSPETH input styling.
 * Composes `.textarea`. Optional label + hint. Used by the chat composer,
 * prompt editing, and the "I meant…" amend forms.
 *
 * The label is associated with the control via the passed `id`; when no `id`
 * is supplied one is generated with React `useId()` so the label/control pair
 * is always programmatically linked.
 */
export interface TextareaProps
  extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Field label rendered above the control. */
  label?: ReactNode;
  /** Helper text rendered below the control. */
  hint?: ReactNode;
}

export function Textarea({
  label,
  hint,
  id,
  className = "",
  rows = 3,
  ...rest
}: TextareaProps) {
  const generatedId = useId();
  const taId = id ?? generatedId;
  const cls = ["textarea", className].filter(Boolean).join(" ");
  const control = <textarea id={taId} className={cls} rows={rows} {...rest} />;
  if (!label && !hint) return control;
  return (
    <div>
      {label ? (
        <label className="field-label" htmlFor={taId}>
          {label}
        </label>
      ) : null}
      {control}
      {hint ? <div className="field-hint">{hint}</div> : null}
    </div>
  );
}
